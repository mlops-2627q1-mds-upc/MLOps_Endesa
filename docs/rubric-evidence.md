# Course evidence

Use this register to find the work supporting each assessed practice. Points are
the course rubric weights. Add links with the version, result, reviewer, and date
as work is completed.

**Updated: 2026-10-05.** The working agreement was merged in
[PR #4](https://github.com/mlops-2627q1-mds-upc/MLOps_Endesa/pull/4).
Remaining repository settings are tracked in
[issue #2](https://github.com/mlops-2627q1-mds-upc/MLOps_Endesa/issues/2).

| Milestone | Assessed practice | Points / 100 | Current evidence | Evidence still needed |
| --- | --- | ---: | --- | --- |
| M1 | Problem selection and requirements engineering | 6 | [Project overview](../README.md); [agreed evaluation contract](problem-definition.md#evaluation-contract-for-m2), [model-card extension](MODEL_CARD.md) in [#5](https://github.com/mlops-2627q1-mds-upc/MLOps_Endesa/issues/5); pinned archive inspection in the [dataset-card extension](DATASET_CARD.md) for [#6](https://github.com/mlops-2627q1-mds-upc/MLOps_Endesa/issues/6) | Continuing requirements/card updates and review; verified source units and calendar semantics if obtainable; trained-model evidence in [#7](https://github.com/mlops-2627q1-mds-upc/MLOps_Endesa/issues/7). |
| M1 | Project coordination and communication | 4 | [Working agreement](../CONTRIBUTING.md), [EDN-001](edn/0001-team-workflow.md), [Project](https://github.com/orgs/mlops-2627q1-mds-upc/projects/3), [issue #1](https://github.com/mlops-2627q1-mds-upc/MLOps_Endesa/issues/1), [verified setup](project-setup.md) | Pending access invitations, Project access confirmation, documentation PR review, weekly updates, and feedback follow-up. |
| M2 | Project structure | 5 | Adapted CCDS layout and locked uv environment in [PR #14](https://github.com/mlops-2627q1-mds-upc/MLOps_Endesa/pull/14) for [#10](https://github.com/mlops-2627q1-mds-upc/MLOps_Endesa/issues/10); [setup guide and adaptations](setup.md) | A teammate's clean-checkout setup check recorded in the PR review. |
| M2 | Code and data versioning | 15 | [Workflow](../CONTRIBUTING.md#github-flow-and-review), [verified branch controls](project-setup.md#repository-settings), merged [split PR #27](https://github.com/mlops-2627q1-mds-upc/MLOps_Endesa/pull/27), and [fresh-cache DVC retrieval on 2026-10-05](experiments/001-baseline-chronos.md#verification-and-reproduction) | Teammate clean-checkout retrieval evidence and continued reviewed PR history. |
| M2 | Experiment tracking | 5 | Tracking setup reviewed by Pablo and merged in [PR #26](https://github.com/mlops-2627q1-mds-upc/MLOps_Endesa/pull/26); [three completed and verified real runs](experiments/001-baseline-chronos.md) on 2026-10-05 at clean `3bcc772`, with per-state/aggregate metrics, predictions, versions and timings; [reproduction commands](evaluation.md); [EDN-002](edn/0002-mlflow-tracking.md), [EDN-003](edn/0003-validation-comparison.md). | Teammate reproduction and interpretation review, followed by reviewed merge of [PR #28](https://github.com/mlops-2627q1-mds-upc/MLOps_Endesa/pull/28) for [#12](https://github.com/mlops-2627q1-mds-upc/MLOps_Endesa/issues/12). |
| M3 | Energy efficiency awareness | 5 | No implementation evidence recorded. | CodeCarbon measurements, measurement limitations, and interpretation of efficiency trade-offs. |
| M3 | Quality assurance for ML | 10 | [PR #28](https://github.com/mlops-2627q1-mds-upc/MLOps_Endesa/pull/28): local lint and 40 tests passed; [CI on executed source `3bcc772`](https://github.com/mlops-2627q1-mds-upc/MLOps_Endesa/actions/runs/37356431172) passed lint/test. Tests cover causal validation windows, training-only scales, sample medians, metrics, DVC guards and MLflow round trips. | Peer review of the new evaluator; further data/model quality checks and API tests as those components develop. |
| M4 | ML system design | 15 | No implementation evidence recorded. | Architecture, component responsibilities, constraints, and justified design decisions. |
| M4 | APIs for ML | 10 | No implementation evidence recorded. | API contract, examples, endpoint tests, and verification of the deployed component. |
| M5 | Containers and orchestration | 5 | No implementation evidence recorded. | Reproducible container build, runtime verification, and portability evidence. |
| M5 | CI/CD for ML | 10 | [Review and check policy](../CONTRIBUTING.md#github-flow-and-review); merged [CI PR #21](https://github.com/mlops-2627q1-mds-upc/MLOps_Endesa/pull/21); [green main run at `b5a3114`](https://github.com/mlops-2627q1-mds-upc/MLOps_Endesa/actions/runs/37351872194) on 2026-10-05. Expensive forecasting remains outside CI. | Required checks on main (#20); further data/model/API checks, model promotion criteria, and verified deployment. |
| M6 | Resource monitoring | 5 | No implementation evidence recorded. | Resource/service metrics, dashboards, alert behavior, and an investigation example. |
| M6 | Model performance monitoring | 5 | No implementation evidence recorded. | Data/model monitoring signals, delayed ground-truth handling, and a demonstrated response to degradation. |
| | **Total** | **100** | | |

The last column is a planning guide for this project, not an additional grading
rubric. Record outcomes and limitations in the linked work.

Source: course laboratory slides, *Machine Learning Systems in Production
(MLOps)*, 2026-27, slides 22-32 and 35-41. The supplied course materials in Atenea
remain authoritative for grading and subsequent updates.
