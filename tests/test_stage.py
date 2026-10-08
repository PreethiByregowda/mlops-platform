import mlflow
import mlflow.sklearn
import pytest
from mlflow.tracking import MlflowClient

from stage import promote_best_model

EXPERIMENT = "test-experiment"
MODEL = "TestEmotionModel"


def log_run(model, valid, test=None):
    """Log a candidate run with validation (selection) and test (gate) scores."""
    mlflow.set_experiment(EXPERIMENT)
    metrics = {"micro_f1_valid": valid}
    if test is not None:
        metrics["micro_f1_test"] = test
    with mlflow.start_run() as run:
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, name="model")
    return run.info.run_id


def promote(tracking_uri):
    return promote_best_model(
        tracking_uri,
        EXPERIMENT,
        model_name=MODEL,
        alias="staging",
        min_score=0.30,
        min_improvement=0.005,
    )


def staging(tracking_uri):
    return MlflowClient(tracking_uri).get_model_version_by_alias(MODEL, "staging")


def run_tags(tracking_uri, run_id):
    return MlflowClient(tracking_uri).get_run(run_id).data.tags


def versions(tracking_uri):
    return MlflowClient(tracking_uri).search_model_versions(f"name='{MODEL}'")


# ---------- candidate selection ----------

def test_candidate_is_selected_by_validation_not_test(tracking_uri, tiny_model):
    log_run(tiny_model, valid=0.30, test=0.60)  # better on test, worse on validation
    best_valid = log_run(tiny_model, valid=0.45, test=0.44)

    decision, version = promote(tracking_uri)

    assert decision.approved
    assert staging(tracking_uri).run_id == best_valid


def test_selected_candidate_without_test_score_is_rejected(tracking_uri, tiny_model):
    log_run(tiny_model, valid=0.40, test=0.44)
    log_run(tiny_model, valid=0.50)  # best on validation, no gate metric

    decision, version = promote(tracking_uri)

    assert version is None and not versions(tracking_uri)  # no fallback to 2nd best
    assert "has no micro_f1_test" in decision.reason


def test_first_promotion_with_no_incumbent(tracking_uri, tiny_model):
    best = log_run(tiny_model, valid=0.45, test=0.44)

    decision, version = promote(tracking_uri)

    assert decision.approved and int(version.version) == 1
    assert staging(tracking_uri).run_id == best
    assert "no incumbent" in decision.reason


# ---------- gate rules (fail closed) ----------

def test_rejects_candidate_below_floor(tracking_uri, tiny_model):
    run = log_run(tiny_model, valid=0.45, test=0.10)

    decision, version = promote(tracking_uri)

    assert version is None and not versions(tracking_uri)
    assert "below the floor" in decision.reason
    assert run_tags(tracking_uri, run)["promotion.decision"] == "rejected"


def test_rejects_improvement_smaller_than_margin(tracking_uri, tiny_model):
    log_run(tiny_model, valid=0.40, test=0.440)
    promote(tracking_uri)
    log_run(tiny_model, valid=0.50, test=0.442)  # better on valid, +0.002 on test

    decision, version = promote(tracking_uri)

    assert version is None
    assert "at least +0.0050 required" in decision.reason
    assert int(staging(tracking_uri).version) == 1


def test_never_demotes_a_better_incumbent(tracking_uri, tiny_model):
    log_run(tiny_model, valid=0.40, test=0.46)
    promote(tracking_uri)
    log_run(tiny_model, valid=0.50, test=0.41)  # better on valid, worse on test

    decision, version = promote(tracking_uri)

    assert version is None and int(staging(tracking_uri).version) == 1
    assert decision.improvement == pytest.approx(-0.05)


def test_promotes_when_margin_is_met(tracking_uri, tiny_model):
    log_run(tiny_model, valid=0.40, test=0.44)
    promote(tracking_uri)
    better = log_run(tiny_model, valid=0.50, test=0.46)

    decision, version = promote(tracking_uri)

    assert decision.approved and int(version.version) == 2
    assert staging(tracking_uri).run_id == better
    assert decision.incumbent_version == "1"


def test_rerun_is_a_noop(tracking_uri, tiny_model):
    log_run(tiny_model, valid=0.45, test=0.44)
    promote(tracking_uri)

    decision, version = promote(tracking_uri)

    assert version is None and len(versions(tracking_uri)) == 1
    assert "already v1" in decision.reason


def test_rejects_when_incumbent_has_no_gate_metric(tracking_uri, tiny_model):
    legacy = log_run(tiny_model, valid=0.40)  # trained before test scores existed
    v = mlflow.register_model(f"runs:/{legacy}/model", MODEL)
    MlflowClient(tracking_uri).set_registered_model_alias(MODEL, "staging", v.version)
    log_run(tiny_model, valid=0.50, test=0.46)

    decision, version = promote(tracking_uri)

    assert version is None
    assert "incumbent v1 has no micro_f1_test" in decision.reason


# ---------- audit records ----------

def test_approved_version_carries_audit_record(tracking_uri, tiny_model):
    log_run(tiny_model, valid=0.40, test=0.44)
    promote(tracking_uri)
    log_run(tiny_model, valid=0.50, test=0.46)

    _, version = promote(tracking_uri)
    tags = MlflowClient(tracking_uri).get_model_version(MODEL, version.version).tags

    assert tags["promotion.decision"] == "approved"
    assert tags["promotion.gate_metric"] == "micro_f1_test"
    assert float(tags["promotion.candidate_score"]) == pytest.approx(0.46)
    assert tags["promotion.incumbent_version"] == "1"
    assert float(tags["promotion.incumbent_score"]) == pytest.approx(0.44)
    assert tags["promotion.min_improvement"] == "0.005"
    assert tags["promotion.evaluated_by"]  # OS user outside Prefect
    assert tags["promotion.evaluated_at"].endswith("+00:00")


def test_rejected_candidate_run_records_decision(tracking_uri, tiny_model):
    log_run(tiny_model, valid=0.40, test=0.44)
    promote(tracking_uri)
    rejected = log_run(tiny_model, valid=0.50, test=0.30)

    promote(tracking_uri)
    tags = run_tags(tracking_uri, rejected)

    assert tags["promotion.decision"] == "rejected"
    assert "at least +0.0050 required" in tags["promotion.reason"]


# ---------- infrastructure ----------

def test_registered_alias_loads_as_sklearn_pipeline(tracking_uri, tiny_model):
    log_run(tiny_model, valid=0.45, test=0.44)
    promote(tracking_uri)

    loaded = mlflow.sklearn.load_model(f"models:/{MODEL}@staging")

    assert loaded.predict_proba(["so happy", "very sad"]).shape == (2, 28)


def test_missing_experiment_raises(tracking_uri):
    with pytest.raises(ValueError, match="not found"):
        promote_best_model(tracking_uri, "does-not-exist", model_name=MODEL)
