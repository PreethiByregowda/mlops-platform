# 0002. Continuous integration on GitHub Actions

- Status: Accepted
- Date: 2026-10-08

## Context

Until now every check (tests, lockfile consistency, image builds) ran by hand. The repository has two separately locked Python environments (pipeline and inference) and two Docker images, so it is easy to change one Pipfile without relocking, or break an image that nobody rebuilt. The README's claims about tests and builds need to be continuously checked to stay credible.

## Decision

One workflow, `.github/workflows/ci.yml`, runs on every push and pull request with four independent jobs:

| Job | Check |
|---|---|
| `lint` | `ruff check` with rules `E`, `F`, `W`, `B` (errors and likely bugs; line length excluded) |
| `lockfiles` | `pipenv verify` for both environments: each `Pipfile.lock` matches its `Pipfile` |
| `test` | Matrix over the two environments: `pipenv sync --dev` from the lockfile, then `pytest` |
| `docker` | Matrix over the two images: build with Buildx (GitHub Actions cache), then import the application code inside the image |

Tool versions are pinned (`pipenv`, `ruff`); the ruff pin is duplicated in the workflow and in `Pipfile` so `make lint` matches CI. The workflow has read-only repository permissions and cancels superseded runs on the same ref.

## Consequences

- Broken tests, stale lockfiles, lint errors and unbuildable images are caught before merge.
- Lint covers correctness, not style. black and isort remain the formatters and are not enforced in CI, so formatting can drift; enforcing it later means a one-time reformat.
- `data/` is not in git, so CI builds the training image without data and only checks that its code imports. Training itself is verified by the end-to-end runs documented in the README, not in CI.
- Actions are referenced by major-version tag (for example `actions/checkout@v4`), which is simpler to maintain than commit-SHA pinning but trusts the action publishers' tags.
- The ruff version must be bumped in two places.

## Alternatives considered

- **Run lint inside the pipenv environment.** Keeps one version source, but every lint run would install the full training stack (Prefect, MLflow, datasets) just to run ruff.
- **Full end-to-end job in CI** (MLflow and Prefect servers, a training run, promotion, the inference container). The strongest signal, but it needs the dataset download and roughly 15 minutes per run; deferred to Tier 2.
- **Pin actions by commit SHA.** Stronger supply-chain guarantee; worth adopting together with automated update tooling (for example Dependabot), not by hand.
