"""Human approval step: move a gated model version to the production alias.

Only a version that passed the promotion gate (stage.py) can reach production:
by default it must be the current @staging version; with rollback=True it may
be any version the gate approved earlier.
"""
import os
import argparse

from prefect import flow, task, get_run_logger
from mlflow.tracking import MlflowClient

from stage import MODEL_NAME, actor, utc_now, alias_version

STAGING_ALIAS = os.getenv("MODEL_ALIAS", "staging")
PRODUCTION_ALIAS = os.getenv("PRODUCTION_ALIAS", "production")


def approve_for_production(
    tracking_uri,
    version,
    approved_by,
    reason,
    rollback=False,
    model_name=MODEL_NAME,
    staging_alias=STAGING_ALIAS,
    production_alias=PRODUCTION_ALIAS,
):
    """Point `production_alias` at `version` and record who approved it and why.

    Returns the previous production version (or None). Raises ValueError when
    the request is not allowed; nothing is changed in that case.
    """
    version = str(version)
    if not approved_by or not approved_by.strip():
        raise ValueError("approved_by is required")
    if not reason or not reason.strip():
        raise ValueError("reason is required")

    client = MlflowClient(tracking_uri=tracking_uri)
    target = client.get_model_version(model_name, version)  # raises if missing

    current = alias_version(client, model_name, production_alias)
    if current is not None and str(current.version) == version:
        raise ValueError(f"v{version} is already @{production_alias}")

    staging = alias_version(client, model_name, staging_alias)
    is_staging = staging is not None and str(staging.version) == version
    gate_approved = target.tags.get("promotion.decision") == "approved"
    if not is_staging:
        if not rollback:
            raise ValueError(
                f"v{version} is not @{staging_alias}; only the current staging "
                "version can be approved (use rollback for an earlier gated version)"
            )
        if not gate_approved:
            raise ValueError(f"v{version} never passed the promotion gate")

    tags = {
        "production.approved_at": utc_now(),
        "production.approved_by": approved_by.strip(),
        "production.recorded_by": actor(),
        "production.reason": reason.strip(),
        "production.rollback": str(bool(rollback and not is_staging)),
        "production.previous_version": str(current.version) if current else "none",
    }
    for key, value in tags.items():
        client.set_model_version_tag(model_name, version, key, value)
    client.set_registered_model_alias(model_name, production_alias, version)
    return str(current.version) if current else None


@task(name="Approve model version for production")
def approve_task(tracking_uri, version, approved_by, reason, rollback):
    logger = get_run_logger()
    previous = approve_for_production(tracking_uri, version, approved_by, reason, rollback)
    logger.info(
        f"{MODEL_NAME} v{version} is now '@{PRODUCTION_ALIAS}' "
        f"(previous: {previous or 'none'}), approved by {approved_by}: {reason}"
    )


@flow(name="mlflow_production_approval")
def mlflow_production_approval(
    tracking_uri: str, version: int, approved_by: str, reason: str, rollback: bool = False
):
    approve_task(tracking_uri, version, approved_by, reason, rollback)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracking_uri", default=os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    parser.add_argument("--version", required=True, type=int, help="Model version to approve.")
    parser.add_argument("--approved_by", required=True, help="Person approving the release.")
    parser.add_argument("--reason", required=True, help="Why this version is being released.")
    parser.add_argument("--rollback", action="store_true", help="Allow an earlier gated version.")
    args = parser.parse_args()

    mlflow_production_approval(
        args.tracking_uri, args.version, args.approved_by, args.reason, args.rollback
    )
