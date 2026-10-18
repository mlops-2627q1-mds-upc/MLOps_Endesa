# Engineering Decision Notebook

The EDN records selected engineering choices: what was decided, why, who took
part, and how AI contributed where applicable.

| Entry | Date | Milestone | Status |
| --- | --- | --- | --- |
| [EDN-001: Task planning and review](0001-team-workflow.md) | 2026-09-22 | M1 | Working agreement reviewed and merged in PR #4 |
| [EDN-002: Explicit MLflow run records](0002-mlflow-tracking.md) | 2026-10-02 | M2 | Implementation reviewed by Pablo and merged in PR #26 |
| [EDN-003: Bounded validation comparison](0003-validation-comparison.md) | 2026-10-05 | M2 | Windows corrections reviewed; PR #28 merged |
| [EDN-004: Dataset source limits](0004-dataset-source-limits.md) | 2026-10-10 | M1/M2 | Source-scale and positional limits accepted in PR #40 |
| [EDN-005: Forecast energy measurement](0005-energy-measurement.md) | 2026-10-11 | M3 | Implementation reviewed and merged in PR #41 |
| [EDN-006: Required CI checks](0006-required-ci-checks.md) | 2026-10-11 | M5 | Controls verified; documentation merged in PR #43 on 2026-10-17 |
| [EDN-007: Source archive integrity](0007-source-archive-integrity.md) | 2026-10-11 | M3/M2 | Decision and implementation reviewed in PRs #47/#51; combined conflict resolution approved and merged in PR #69 |
| [EDN-008: Data validation](0008-data-validation-design.md) | 2026-10-13 | M3 | GX implementation reviewed and merged in PR #37; repeatable-setup amendment included in approved and merged PR #69 |
| [EDN-009: Minimum functionality](0009-minimum-functionality.md) | 2026-10-14 | M3 | Base-model cases reviewed in PR #57; local-checkpoint extension included in approved and merged PR #69 |
| [EDN-010: Forecast invariance](0010-forecast-invariance.md) | 2026-10-14 | M3 | Base-model cases reviewed in PR #59; local-checkpoint extension included in approved and merged PR #69 |
| [EDN-011: Directional expectations](0011-directional-expectations.md) | 2026-10-14 | M3 | Base-model cases reviewed in PR #58; local-checkpoint extension included in approved and merged PR #69 |
| [EDN-012: Bounded fine-tuning](0012-chronos-fine-tuning.md) | 2026-10-12 | M2/M3 | Experiment records, integration and provenance repair included in approved and merged PR #69 |

## Adding an entry

Use the [template](TEMPLATE.md) for a choice with meaningful consequences that
could reasonably have been made differently. Using a prescribed course tool,
finishing a task, or asking AI a routine question does not by itself need an entry.

Keep the rationale specific to this project: the constraint, alternatives, and
trade-off that explain the choice. List the people who took part. For AI-assisted
decisions, name the tool's contribution, the human response, and the assessment.
Keep that account brief and accurate; the decision belongs to the team members.

Conversation links and prompt excerpts are optional. Include them only when they
help explain the decision, and remove personal or sensitive material. A link to
the relevant PR, experiment, or design is often enough supporting evidence.
Keep setup logs and task progress in their issues.

If a decision changes, add a dated amendment or a new entry and link the previous
one. Add each entry to the table above.

## Source

This notebook follows *Engineering Decision Notebook for the MLOps Project*,
v2026.0.2, sections 2-4: [course instructions](https://atenea.upc.edu/pluginfile.php/7268356/mod_resource/content/1/Instruction_EDN_MLOps_v2026.pdf)
and [entry template](https://atenea.upc.edu/mod/folder/view.php?id=5643133).
Atenea may require a course login.
