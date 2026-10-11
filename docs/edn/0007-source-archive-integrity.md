# EDN-007: Verify the source archive before ingestion

- **Date:** 2026-10-11
- **Milestone:** M3 - Quality assurance; related to M2 reproducibility.
- **Activity / Topic:** Source integrity and ingestion validation.
- **Decision Participants:** Dídac Cayuela.
- **Status:** Accepted for planning by Dídac; implementation and teammate review pending.

## Decision

Create [issue #46](https://github.com/mlops-2627q1-mds-upc/MLOps_Endesa/issues/46)
to verify the downloaded archive against its documented SHA-256 and require the
expected series IDs before writing raw Parquet files. Keep the current source
values, float32 storage, state mapping, and positional indexes.

## Rationale

The downloader uses the selected archive URL but does not check its documented
digest. DVC identifies the stored artifacts, and Great Expectations checks the
resulting tables; neither check in the current downloader verifies the incoming
archive bytes. Reuse the existing source pin and add small offline fixtures
instead of requiring shared data or model execution for these checks.

## AI Involvement

Recommendation; Solution generation.

Codex identified the missing integrity check and proposed this bounded robustness
task, its acceptance criteria, and offline tests. It drafted the issue and this
decision record.

## Response to AI

Accepted. Dídac authorized creating the issue and requested that Codex's proposal
be acknowledged in the EDN specifically for this task.

## Assessment of the AI Contribution

Dídac accepted the proposal as a focused addition to ingestion robustness. This
records his acceptance of the task; implementation results and teammate review
remain pending.

## Other Evidence

- [Issue #46: source archive integrity and TSF ingestion](https://github.com/mlops-2627q1-mds-upc/MLOps_Endesa/issues/46).
- [Documented source archive and digest](../DATASET_CARD.md#dataset-structure-and-provenance).
- [Existing ingestion implementation](../../src/dataset.py).
