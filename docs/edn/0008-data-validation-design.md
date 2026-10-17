# EDN-008: Data validation as DVC stages with Great Expectations

- **Date:** 2026-10-13
- **Milestone:** M3 - Quality assurance.
- **Activity / Topic:** Implementation of issue #36 (data validation before training).
- **Decision Participants:** Pau Adal implemented the stage. Dídac Cayuela reviewed the PR.
- **Status:** Sindri approved final revision `9e9b230`; PR #37 merged on 2026-10-14.

## Decision

Run data validation as two explicit DVC stages (`configure_gx` and `validate-data`)
using Great Expectations (GX) with a file data context, rather than as ad-hoc checks
inside the EDA notebook. One expectation suite is defined per table (five raw state
files, three processed blocks, and the origins table). The checkpoint runs all nine
suites and exits with code 1 when any check fails, stopping the DVC pipeline at the
`validate-data` stage. Four additional Python functions cover checks that GX cannot
express directly.

The 22 known TAS negative values are preserved as-is and verified explicitly: the
raw TAS series must contain negatives at exactly the 22 positions documented in the
EDA (§5) and nowhere else.

GX Data Docs expectation pages (`gx/data_docs/expectations/`) are committed to Git
so they can be reviewed without re-running the pipeline. Per-run validation HTML
reports (`gx/data_docs/validations/`) are excluded via `.gitignore` to prevent
unbounded repository growth.

## Rationale

Keeping checks only in the EDA notebook means they are never re-run when the data
or `params.yaml` changes. A truncated series or a NaN would propagate silently to
the model. Encoding the checks as DVC stages ties validation to the data and
parameter versions tracked by DVC: any change that could break a check triggers a
re-run automatically.

The course demo provided the structural pattern for a GX file data context with a
single checkpoint. The specific suites, the four Python checks (infinity detection,
exact TAS positions, complete block index sequences, and the forecast origin grid),
and the DVC wiring are specific to this project and derived from the EDA findings
in [PR #31](https://github.com/mlops-2627q1-mds-upc/MLOps_Endesa/pull/31).

GX's `ExpectColumnValuesToBeBetween(-inf, inf)` always passes and does not detect
`np.inf`; a project-specific `_check_finite` function is used instead. Similarly,
GX cannot verify that a DataFrame's index is a contiguous `RangeIndex`, that all
expected forecast windows are present without duplicates, or that TAS negatives fall
at exact positions — all four are handled with Python functions tested by synthetic
fixtures that run in CI without DVC data.

## AI Involvement

Solution generation; Recommendation. Claude (Anthropic) generated the initial
implementation of `gx_context_configuration.py`, `validate_data.py`,
`tests/test_data.py`, `tests/test_validation_checks.py`, and
`docs/data-validation.md`, and proposed the four Python check functions after the
GX limitation with infinity was identified. Claude also proposed the DVC stamp-file
approach to enforce stage ordering.

## Response to AI

Accepted with modifications. The AI-generated code was reviewed through two rounds
of automated review (Codex), which identified five correctness issues: portable GX
paths, complete index validation, complete origins grid validation, checking TAS
negatives against the full raw series rather than only the training block, and the
`result.suite_name` API fix for GX 1.24.0. All five were corrected before approval.

## Assessment of the AI Contribution

The structural approach and the four Python check functions were sound. The initial
implementation missed edge cases that only became apparent when the reviewer tested
with synthetic corrupted data. The iterative review process caught them. The
final revision was approved by Sindri and merged in PR #37 on 2026-10-14. A separate
assessment of the AI contribution by the owner has not been recorded.

## Other Evidence

- [PR #37](https://github.com/mlops-2627q1-mds-upc/MLOps_Endesa/pull/37) — implementation and review history.
- [Issue #36](https://github.com/mlops-2627q1-mds-upc/MLOps_Endesa/issues/36) — original task and acceptance criteria.
- [docs/data-validation.md](../data-validation.md) — what is checked and how to run it.
- [EDA source](https://github.com/mlops-2627q1-mds-upc/MLOps_Endesa/pull/31) — Pablo's source investigation that informed the TAS and range checks.
