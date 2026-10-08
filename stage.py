import os
import argparse
import getpass
from typing import Optional
from datetime import datetime, timezone
from dataclasses import dataclass

import mlflow
from prefect import flow, task, get_run_logger
from prefect.runtime import flow_run
from mlflow.entities import ViewType
from mlflow.tracking import MlflowClient
from mlflow.exceptions import MlflowException

MODEL_NAME = os.getenv("MODEL_NAME", "EmotionRecognitionModel")
MODEL_ALIAS = os.getenv("MODEL_ALIAS", "staging")
# Picks the candidate among runs (validation split only)
SELECTION_METRIC = os.getenv("SELECTION_METRIC", "micro_f1_valid")
# Decides whether the candidate replaces the incumbent (held-out test split)
GATE_METRIC = os.getenv("GATE_METRIC", "micro_f1_test")
MIN_GATE_SCORE = float(os.getenv("MIN_GATE_SCORE", "0.30"))
MIN_IMPROVEMENT = float(os.getenv("MIN_IMPROVEMENT", "0.005"))


@dataclass
class GateDecision:
    approved: bool
    reason: str
    candidate_run_id: str
    candidate_score: Optional[float]
    incumbent_version: Optional[str] = None
    incumbent_score: Optional[float] = None

    @property
    def improvement(self):
        if self.candidate_score is None or self.incumbent_score is None:
            return None
        return self.candidate_score - self.incumbent_score


def actor():
    """Who is acting: the Prefect flow run when inside one, else the OS user."""
    if flow_run.id:
        return f"prefect:{flow_run.name}"
    return getpass.getuser()


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def select_best_run(client, experiment_name, metric=SELECTION_METRIC):
    """Return the finished run with the highest value of `metric`."""
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"Experiment '{experiment_name}' not found.")

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"metrics.{metric} >= 0 and attributes.status = 'FINISHED'",
        run_view_type=ViewType.ACTIVE_ONLY,
        order_by=[f"metrics.{metric} DESC"],
        max_results=1,
    )
    if not runs:
        raise ValueError(f"No finished runs with metric '{metric}' in '{experiment_name}'.")
    return runs[0]


def alias_version(client, model_name, alias):
    """ModelVersion behind `alias`, or None if the alias is not set."""
    try:
        return client.get_model_version_by_alias(model_name, alias)
    except MlflowException:
        return None


def evaluate_gate(
    candidate_run,
    incumbent_version,
    incumbent_run,
    gate_metric=GATE_METRIC,
    min_score=MIN_GATE_SCORE,
    min_improvement=MIN_IMPROVEMENT,
):
    """Decide whether `candidate_run` may replace the incumbent. Fails closed."""
    run_id = candidate_run.info.run_id
    score = candidate_run.data.metrics.get(gate_metric)
    decision = GateDecision(False, "", run_id, score)

    if score is None:
        decision.reason = f"candidate has no {gate_metric}"
        return decision
    if score < min_score:
        decision.reason = f"{gate_metric} {score:.4f} is below the floor {min_score:.4f}"
        return decision
    if incumbent_version is None:
        decision.approved = True
        decision.reason = f"no incumbent; {gate_metric} {score:.4f} meets the floor"
        return decision

    decision.incumbent_version = str(incumbent_version.version)
    if incumbent_version.run_id == run_id:
        decision.reason = f"candidate is already v{incumbent_version.version}"
        return decision

    decision.incumbent_score = incumbent_run.data.metrics.get(gate_metric)
    if decision.incumbent_score is None:
        decision.reason = (
            f"incumbent v{incumbent_version.version} has no {gate_metric}; "
            "retrain it or promote manually"
        )
        return decision
    if decision.improvement < min_improvement:
        decision.reason = (
            f"{gate_metric} improves by {decision.improvement:+.4f} over "
            f"v{incumbent_version.version}; at least {min_improvement:+.4f} required"
        )
        return decision

    decision.approved = True
    decision.reason = (
        f"{gate_metric} improves by {decision.improvement:+.4f} over "
        f"v{incumbent_version.version}"
    )
    return decision


def audit_tags(decision, experiment_name, gate_metric, min_score, min_improvement):
    tags = {
        "promotion.decision": "approved" if decision.approved else "rejected",
        "promotion.reason": decision.reason,
        "promotion.evaluated_at": utc_now(),
        "promotion.evaluated_by": actor(),
        "promotion.experiment": experiment_name,
        "promotion.gate_metric": gate_metric,
        "promotion.min_score": str(min_score),
        "promotion.min_improvement": str(min_improvement),
        "promotion.candidate_score": str(decision.candidate_score),
    }
    if decision.incumbent_version is not None:
        tags["promotion.incumbent_version"] = decision.incumbent_version
        tags["promotion.incumbent_score"] = str(decision.incumbent_score)
    return tags


def promote_best_model(
    tracking_uri,
    experiment_name,
    model_name=MODEL_NAME,
    alias=MODEL_ALIAS,
    selection_metric=SELECTION_METRIC,
    gate_metric=GATE_METRIC,
    min_score=MIN_GATE_SCORE,
    min_improvement=MIN_IMPROVEMENT,
):
    """Run the promotion gate for the best run and record the decision.

    Returns (GateDecision, ModelVersion or None). Every decision is tagged on
    the candidate run; an approved candidate is registered, tagged with the
    same audit record and given `alias`.
    """
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)

    candidate = select_best_run(client, experiment_name, selection_metric)
    incumbent = alias_version(client, model_name, alias)
    incumbent_run = client.get_run(incumbent.run_id) if incumbent else None

    decision = evaluate_gate(
        candidate, incumbent, incumbent_run, gate_metric, min_score, min_improvement
    )
    tags = audit_tags(decision, experiment_name, gate_metric, min_score, min_improvement)
    for key, value in tags.items():
        client.set_tag(candidate.info.run_id, key, value)

    if not decision.approved:
        return decision, None

    version = mlflow.register_model(
        model_uri=f"runs:/{candidate.info.run_id}/model", name=model_name
    )
    for key, value in tags.items():
        client.set_model_version_tag(model_name, version.version, key, value)
    client.update_model_version(
        name=model_name,
        version=version.version,
        description=f"[{tags['promotion.evaluated_at']}] Promoted to '{alias}': {decision.reason}.",
    )
    client.set_registered_model_alias(model_name, alias, version.version)
    return decision, version


@task(name="Register and stage best emotion model")
def run_gate(tracking_uri, experiment_name):
    log = get_run_logger()
    log.info(
        f"Selecting best run from '{experiment_name}' by {SELECTION_METRIC}; "
        f"gating on {GATE_METRIC} (floor {MIN_GATE_SCORE}, min improvement {MIN_IMPROVEMENT})"
    )
    decision, version = promote_best_model(tracking_uri, experiment_name)
    if version is not None:
        log.info(f"{MODEL_NAME} v{version.version} is now '@{MODEL_ALIAS}': {decision.reason}")
    else:
        log.info(f"Not promoted (run {decision.candidate_run_id}): {decision.reason}")


@flow(name="mlflow_staging")
def mlflow_staging(tracking_uri, experiment_name):
    run_gate(tracking_uri, experiment_name)


def cli(argv=None):
    """Run the promotion gate once from the command line."""
    parser = argparse.ArgumentParser(
        description="Evaluate the best run against the current model and record the decision in MLflow."
    )
    for flag, env, fallback in (
        ("--tracking_uri", "MLFLOW_TRACKING_URI", "http://localhost:5000"),
        ("--experiment_name", "MLFLOW_EXPERIMENT_NAME", "goemotions-classification"),
    ):
        parser.add_argument(flag, default=os.getenv(env, fallback), help=f"default: ${env} or {fallback}")
    options = parser.parse_args(argv)
    mlflow_staging(options.tracking_uri, options.experiment_name)


if __name__ == "__main__":
    cli()
