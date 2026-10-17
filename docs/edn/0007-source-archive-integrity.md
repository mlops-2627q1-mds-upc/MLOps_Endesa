# EDN-007: Verify the source archive before ingestion

- **Date:** 2026-10-11
- **Milestone:** M3 - Quality assurance; related to M2 reproducibility.
- **Activity / Topic:** Source integrity and ingestion validation.
- **Decision Participants:** Dídac Cayuela (decision owner); Sindri Másson (planning-record review); Pablo Pérez Cano (rationale and documentation review).
- **Status:** Decision record approved in PR #47 and implementation in PR #51; the combined conflict resolution awaits current-version review.

## Decision

Create [issue #46](https://github.com/mlops-2627q1-mds-upc/MLOps_Endesa/issues/46)
to verify the downloaded archive against its documented SHA-256 and require the
expected series IDs before writing raw Parquet files. Keep the current source
values, float32 storage, state mapping, and positional indexes. The implementation
and offline checks are available in
[PR #51](https://github.com/mlops-2627q1-mds-upc/MLOps_Endesa/pull/51).

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
decision record, then supported the implementation and validation in PR #51.

## Response to AI

Accepted. Dídac authorized creating the issue and requested that Codex's proposal
be acknowledged in the EDN specifically for this task.

## Assessment of the AI Contribution

Dídac accepted the proposal as a focused addition to ingestion robustness. Sindri
approved the original planning record on 2026-10-12. In his
[2026-10-16 review comment](https://github.com/mlops-2627q1-mds-upc/MLOps_Endesa/pull/47#issuecomment-6213640019),
Pablo confirmed the distinction between archive integrity, DVC identity and GX
validation, and requested this implementation follow-up. PR #51 supplies checksum
verification before ZIP decoding and complete-series validation before writes.
Its 16 offline cases cover LF/CRLF input, series ordering, and rejection without
creating or changing outputs; implementation approval and merge are tracked in
that PR.

On 2026-10-16, `uv run --locked pytest tests/test_dataset.py -q` passed all
16 offline cases. The full default suite passed 87 tests and skipped nine
shared-data checks; `make lint` also passed. No shared-data retrieval or model
execution was needed for these checks.

## Other Evidence

- [Issue #46: source archive integrity and TSF ingestion](https://github.com/mlops-2627q1-mds-upc/MLOps_Endesa/issues/46).
- [Documented source archive and digest](../DATASET_CARD.md#dataset-structure-and-provenance).
- [Ingestion implementation](../../src/dataset.py) and [implementation PR #51](https://github.com/mlops-2627q1-mds-upc/MLOps_Endesa/pull/51).
- [Sindri's approval of the planning record](https://github.com/mlops-2627q1-mds-upc/MLOps_Endesa/pull/47).

- [Offline ingestion cases](../../tests/test_dataset.py).

On 2026-10-17, Pablo Pérez Cano approved `92a56f4` in [PR #47](https://github.com/mlops-2627q1-mds-upc/MLOps_Endesa/pull/47).
