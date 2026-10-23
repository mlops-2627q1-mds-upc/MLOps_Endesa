# EDN-002: Explicit MLflow run records

- **Date:** 2026-10-02.
- **Milestone:** M2 - Reproducibility.
- **Activity / Topic:** Experiment tracking for #12.
- **Decision Participants:** Dídac Cayuela (task scope and implementation constraints); Pablo Pérez Cano (implementation review).
- **Status:** Tracking implementation approved by Pablo Pérez Cano and merged in PR #26.

## Decision

Use the existing DagsHub MLflow server and explicit client/run IDs. Record the
experiment configuration, Git commit/dirty state, supplied data/model versions,
environment, and lockfile for each run. Keep a separate synthetic smoke experiment
and test logging against a temporary local SQLite store.

## Rationale

The course prescribes MLflow and demonstrates DagsHub configuration. Our
forecasting scripts and notebooks need the same version/configuration records;
framework autologging alone would not describe split boundaries, forecast
origins, or the DVC reference. An explicit client avoids changing a notebook's
active run or global tracking URI. Synchronous metric batches make failures
visible before a run is marked finished. Local tests keep CI independent of
credentials and model/data downloads. The caller remains responsible for the
evaluation inputs and model execution; this adds a small logging interface
rather than a new pipeline framework.

## AI Involvement

Information seeking; Alternative assessment; Recommendation; Solution generation.
OpenAI Codex inspected the repository and course guide, proposed the tracking
plan, and implemented the helper, smoke command, tests, and documentation.

## Response to AI

Accepted with modifications. Dídac took ownership of #12 and required a small,
maintainable implementation without unnecessary abstractions. He kept tracking
work separate from the evolving cards in #5. Codex developed the explicit-client
design and implementation within that scope. Pablo approved the resulting
implementation on 2026-10-03; [PR #26](https://github.com/mlops-2627q1-mds-upc/MLOps_Endesa/pull/26)
was merged on 2026-10-05.

## Assessment of the AI Contribution

Dídac required maintainable code without unnecessary abstractions. The resulting
helper provides the logging interface used by scripts and notebooks. Local tests
and the shared smoke run verified parameter, metric and artifact round trips; these checks establish
tracking behavior, not forecast quality. Pablo's
[review](https://github.com/mlops-2627q1-mds-upc/MLOps_Endesa/pull/26#pullrequestreview-5401489108)
records acceptance of the implementation.

## Other Evidence

- [Issue #12](https://github.com/mlops-2627q1-mds-upc/MLOps_Endesa/issues/12).
- [Tracking setup and reproduction commands](../mlflow.md).
- [Course MLflow demo](https://github.com/mlops-2627q1-mds-upc/MLOps-2627q1-demos/blob/main/docs/mlflow-demo.md).
- Source: `src/tracking.py`; verification: `tests/test_tracking.py` and the shared
  smoke run linked from the implementation PR.
