# 0004. Promotion gate on held-out test data, and human approval for production

- Status: Accepted
- Date: 2026-10-08

## Context

The staging flow promoted the run with the best validation score whenever it beat the current `@staging` model on that same validation score. Three weaknesses:

1. **Selection and acceptance used the same data.** The validation split chose the best candidate *and* decided whether it beat the incumbent, so an optimistic validation score could promote a model that is not actually better.
2. **No safety margin or floor.** Any positive difference, even noise, promoted; and a first model was promoted however poor it was.
3. **No trail and no production step.** Decisions were visible only in Prefect logs, and there was a single alias, so anything that passed the automated check was immediately "the" model.

## Decision

**Held-out test evaluation.** The training flow scores every candidate on the test split as well as the validation split and logs both (`micro_f1_test`, `macro_f1_test`, ...). Candidate selection still uses validation only (`SELECTION_METRIC`, default `micro_f1_valid`); test scores are read only by the gate.

**Promotion gate** (`stage.py`, `mlflow_staging`). The best run by `SELECTION_METRIC` is the only candidate; there is no fallback to the next run. It is promoted to `@staging` only if all hold, otherwise it is rejected (fail closed):

| Rule | Default |
|---|---|
| Candidate has `GATE_METRIC` | `micro_f1_test` |
| Candidate score ≥ `MIN_GATE_SCORE` | 0.30 |
| No incumbent, or candidate − incumbent ≥ `MIN_IMPROVEMENT` on `GATE_METRIC` | 0.005 |
| The incumbent has `GATE_METRIC` (models trained before this change are not compared on a guess) | |
| The candidate is not already the incumbent | |

The decision logic is a pure function (`evaluate_gate`) returning the decision and its reason, so every rule is unit-tested without MLflow.

**Audit record.** Every evaluation, approved or rejected, writes `promotion.*` tags to the candidate run: decision, reason, timestamp (UTC), actor (Prefect flow run or OS user), gate metric, thresholds, candidate score, incumbent version and score. An approved candidate's new model version carries the same tags, and its description states the reason.

**Production approval** (`approve.py`, `mlflow_production_approval`). A separate flow, deployed but never scheduled, sets the `@production` alias. It requires `version`, `approved_by` and `reason`, and only accepts the current `@staging` version, so nothing reaches production without passing the gate. With `rollback=true` it accepts an earlier version whose tags show the gate approved it. It writes `production.*` tags: approver, reason, time, actor, whether it was a rollback, and the previous production version. The inference service serves whichever alias `MODEL_ALIAS` names (default `staging`); a production deployment sets `MODEL_ALIAS=production`.

## Consequences

- A model reaches `@staging` only with a measurable, recorded improvement on data it was not selected on, and reaches `@production` only with a named approver and reason.
- Both automated rejections and human approvals are queryable in MLflow (run and model-version tags) without reading logs.
- **Test-set reuse.** The same test set judges every promotion, so over many promotions decisions slowly adapt to it. Acceptable here; the remedy is refreshing the held-out set periodically, which requires re-scoring the incumbent.
- **Fixed test set assumption.** Test scores are computed at training time, so comparisons are only valid while the test split is unchanged. Changing it requires re-scoring the incumbent before the gate can compare.
- **Migration.** A registry whose `@staging` model predates this change has no `micro_f1_test`; the gate rejects every candidate until that model is retrained or the alias is moved deliberately.
- Thresholds (0.30, 0.005) are illustrative defaults set by environment variable, chosen to reject the near-useless Naive Bayes candidate and noise-level gains, not derived from a product requirement.
- Approval is a parameterised flow run, not a UI button; the approver's identity is what they type, not an authenticated user. Authentication belongs with the deployment platform.

## Alternatives considered

- **Re-evaluate candidate and incumbent on the test set inside the gate.** Robust to a changing test set and to models trained elsewhere, but the gate container would load two models and the test data on every run; logging test scores at training time is simpler while the test set is fixed.
- **Prefect pause-for-input (`pause_flow_run(wait_for_input=...)`).** A UI-driven approval inside the staging flow. Rejected for now: a long-paused flow run holds the decision in Prefect state rather than in the registry, and is harder to test than a plain function.
- **Statistical significance test instead of a fixed margin** (for example a bootstrap confidence interval on micro-F1). More principled; a good follow-up once the evaluation produces per-example predictions.
- **MLflow stage transitions (Staging/Production).** Deprecated in MLflow 3; aliases plus tags express the same workflow.
