# 0001. Record architecture decisions

- Status: Accepted
- Date: 2026-10-08

## Context

The project has already made several non-obvious choices, such as serving by MLflow registry alias, running flows on a Prefect Docker work pool with bridge networking, and building the training image from an allowlist. Those reasons currently live only in commit messages and the README, where they are hard to find and easy to lose when the code changes. The next changes (CI, a Python upgrade, a stricter promotion gate, and inference reliability work) each involve trade-offs that a reviewer or future maintainer will want to understand.

## Decision

Record each significant architectural decision as a short Markdown file in `docs/adr/`, numbered sequentially, using the template in [`README.md`](README.md). A decision is "significant" if it changes a component boundary, a deployment or runtime contract, a dependency baseline, or a safety property of the model lifecycle.

ADRs are written in the same change that implements the decision and are not edited afterwards except to update their status.

## Consequences

- Reasons for a design are reviewable next to the code, and survive refactors.
- Each significant change carries a small writing cost.
- Earlier decisions are documented only where a new ADR builds on them; they are not all back-filled.

## Alternatives considered

- **README sections only.** The README describes what the system does now; it is the wrong place for why, and for options that were rejected.
- **Commit messages only.** Good for small changes, but scattered and hard to discover.
