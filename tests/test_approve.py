import mlflow
import mlflow.sklearn
import pytest
from mlflow.tracking import MlflowClient
from mlflow.exceptions import MlflowException

from approve import approve_for_production
from stage import promote_best_model

EXPERIMENT = "test-experiment"
MODEL = "TestEmotionModel"


def gated_version(tracking_uri, model, valid, test):
    """Log a run and push it through the promotion gate; return its version."""
    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run():
        mlflow.log_metrics({"micro_f1_valid": valid, "micro_f1_test": test})
        mlflow.sklearn.log_model(model, name="model")
    decision, version = promote_best_model(
        tracking_uri, EXPERIMENT, model_name=MODEL, alias="staging",
        min_score=0.30, min_improvement=0.005,
    )
    assert decision.approved, decision.reason
    return str(version.version)


def approve(tracking_uri, version, **kwargs):
    kwargs.setdefault("approved_by", "Jane Doe")
    kwargs.setdefault("reason", "Passed review")
    return approve_for_production(tracking_uri, version, model_name=MODEL, **kwargs)


def alias(tracking_uri, name):
    return str(MlflowClient(tracking_uri).get_model_version_by_alias(MODEL, name).version)


def test_approves_current_staging_version(tracking_uri, tiny_model):
    v1 = gated_version(tracking_uri, tiny_model, 0.40, 0.44)

    previous = approve(tracking_uri, v1)

    assert previous is None
    assert alias(tracking_uri, "production") == v1
    tags = MlflowClient(tracking_uri).get_model_version(MODEL, v1).tags
    assert tags["production.approved_by"] == "Jane Doe"
    assert tags["production.reason"] == "Passed review"
    assert tags["production.previous_version"] == "none"
    assert tags["production.rollback"] == "False"


def test_records_previous_production_version(tracking_uri, tiny_model):
    v1 = gated_version(tracking_uri, tiny_model, 0.40, 0.44)
    approve(tracking_uri, v1)
    v2 = gated_version(tracking_uri, tiny_model, 0.50, 0.46)

    previous = approve(tracking_uri, v2)

    assert previous == v1 and alias(tracking_uri, "production") == v2


def test_refuses_version_that_is_not_staging(tracking_uri, tiny_model):
    v1 = gated_version(tracking_uri, tiny_model, 0.40, 0.44)
    gated_version(tracking_uri, tiny_model, 0.50, 0.46)  # v2 is now @staging

    with pytest.raises(ValueError, match="not @staging"):
        approve(tracking_uri, v1)


def test_rollback_to_an_earlier_gated_version(tracking_uri, tiny_model):
    v1 = gated_version(tracking_uri, tiny_model, 0.40, 0.44)
    approve(tracking_uri, v1)
    v2 = gated_version(tracking_uri, tiny_model, 0.50, 0.46)
    approve(tracking_uri, v2)

    previous = approve(tracking_uri, v1, reason="Regression in v2", rollback=True)

    assert previous == v2 and alias(tracking_uri, "production") == v1
    tags = MlflowClient(tracking_uri).get_model_version(MODEL, v1).tags
    assert tags["production.rollback"] == "True"


def test_rollback_refuses_version_that_never_passed_the_gate(tracking_uri, tiny_model):
    gated_version(tracking_uri, tiny_model, 0.40, 0.44)
    with mlflow.start_run() as run:  # registered by hand, bypassing the gate
        mlflow.sklearn.log_model(tiny_model, name="model")
    ungated = str(mlflow.register_model(f"runs:/{run.info.run_id}/model", MODEL).version)

    with pytest.raises(ValueError, match="never passed the promotion gate"):
        approve(tracking_uri, ungated, rollback=True)


@pytest.mark.parametrize("field", ["approved_by", "reason"])
def test_requires_approver_and_reason(tracking_uri, tiny_model, field):
    v1 = gated_version(tracking_uri, tiny_model, 0.40, 0.44)

    with pytest.raises(ValueError, match=f"{field} is required"):
        approve(tracking_uri, v1, **{field: "  "})

    with pytest.raises(MlflowException):  # alias was not set
        alias(tracking_uri, "production")


def test_refuses_to_reapprove_current_production(tracking_uri, tiny_model):
    v1 = gated_version(tracking_uri, tiny_model, 0.40, 0.44)
    approve(tracking_uri, v1)

    with pytest.raises(ValueError, match="already @production"):
        approve(tracking_uri, v1)
