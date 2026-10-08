# Architecture Decision Records

Each record captures one significant decision: the context, what was decided, its consequences, and the alternatives that were rejected. Records are immutable once accepted; a later decision that changes course gets a new record that supersedes the old one.

| ADR | Title | Status |
|---|---|---|
| [0001](0001-record-architecture-decisions.md) | Record architecture decisions | Accepted |
| [0002](0002-continuous-integration.md) | Continuous integration on GitHub Actions | Accepted |
| [0003](0003-python-3-12-runtime.md) | Python 3.12 runtime and pinned server stack | Accepted |
| [0004](0004-promotion-gate-and-production-approval.md) | Promotion gate on held-out test data, and human approval for production | Accepted |
| [0005](0005-inference-reliability.md) | Inference service reliability: health probes, registry failure handling and live model refresh | Accepted |

## Template

```markdown
# NNNN. Title

- Status: Proposed | Accepted | Superseded by NNNN
- Date: YYYY-MM-DD

## Context
What forces are at play, and why a decision is needed now.

## Decision
What we will do.

## Consequences
What becomes easier or harder, including costs and risks we accept.

## Alternatives considered
Options rejected, and why.
```
