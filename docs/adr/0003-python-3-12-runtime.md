# 0003. Python 3.12 runtime and pinned server stack

- Status: Accepted
- Date: 2026-10-08

## Context

Both environments and both images ran Python 3.9, which reached end of life in October 2025 and no longer receives security fixes. The upgrade had to keep the trained model and its metrics comparable, and keep the Prefect 3.4.11 and MLflow 3.1.4 versions that the Docker work-pool deployment was verified with.

Two things shaped the decision:

- **numpy.** numpy 1.24.4 publishes no Python 3.12 wheels. numpy 2.x would require a newer scikit-learn (1.3.2 is built against the numpy 1.x ABI), which changes the pickled pipelines and possibly the metrics.
- **Indirect dependencies.** Under Python 3.9, environment markers held many indirect dependencies at older releases. Relocking for 3.12 resolved them to the newest releases, and two of those are incompatible with the pinned frameworks: FastAPI 0.142 breaks Prefect 3.4.11's API router (`'PrefectRouter' object has no attribute 'routes'`, every request returns 500), and SQLAlchemy 2.1 removed `FallbackAsyncAdaptedQueuePool`, which MLflow 3.1.4 imports.

## Decision

- Run on **Python 3.12**: `python_version = "3.12"` in both Pipfiles, `python:3.12.7-slim-bookworm` in both Dockerfiles, Python 3.12 in CI, and `py312` as the ruff and black target.
- Change only the pins that must change: **numpy 1.24.4 → 1.26.4** (the last 1.x release). scikit-learn stays at **1.3.2**, which has 3.12 wheels.
- Pin **pandas 2.3.3** in both environments. Unpinned, the inference environment resolved to pandas 3.0, a major release, while the training environment stayed on 2.3.
- Pin the **web and database stack** that Prefect's server and MLflow run on to the versions verified end to end: FastAPI 0.128.8, Starlette 0.49.3, uvicorn 0.39.0, SQLAlchemy 2.0.54, Alembic 1.16.5, anyio 4.12.1, pydantic-settings 2.11.0. The Pipfile comment explains why.
- Move formatters to versions that support 3.12 (black 24.8.0, isort 5.13.2). They are not enforced in CI, so no code is reformatted.

## Consequences

- The runtime is supported again, and the same model code produces the same validation metrics (verified end to end; see the README's verified results).
- The root Pipfile now carries explicit pins for packages it does not import. They must be revisited together when Prefect or MLflow is upgraded: the pins and the framework versions move as a set.
- Other indirect dependencies moved to current releases (for example SciPy 1.13 to 1.17). They are fixed by the lockfile until the next relock.
- scikit-learn 1.3.2 now triggers a SciPy deprecation warning on every logistic-regression fit (an L-BFGS-B option). It is harmless with the locked SciPy, but a future relock could pick a SciPy that removes the option. Upgrading scikit-learn (and with it numpy 2 and model retraining) is the follow-up.
- Models registered before the upgrade were pickled under numpy 1.24; they should be retrained rather than served from the new images.

## Alternatives considered

- **Python 3.11.** Supported until October 2027 and would keep the same pins, but numpy 1.24.4 has 3.11 wheels while the rest of the ecosystem is moving to 3.12+; 3.12 buys more support time for the same work.
- **numpy 2 with a current scikit-learn.** The better long-term state, but it changes the model artifacts and possibly the metrics in the same change as the runtime upgrade. Doing them separately keeps each change verifiable.
- **Relock without extra pins and fix breakages individually.** Two breakages surfaced immediately in the server stack; pinning the whole stack to the verified set is cheaper and more predictable than discovering the next one in production.
