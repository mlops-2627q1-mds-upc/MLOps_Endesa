# Course evidence

Use this register to find the work supporting each assessed practice. Points are
the course rubric weights. Add links with the version, result, reviewer, and date
as work is completed.

**Updated: 2026-10-02.** The working agreement was merged in
[PR #4](https://github.com/mlops-2627q1-mds-upc/MLOps_Endesa/pull/4).
Remaining repository settings are tracked in
[issue #2](https://github.com/mlops-2627q1-mds-upc/MLOps_Endesa/issues/2).

| Milestone | Assessed practice | Points / 100 | Current evidence | Evidence still needed |
| --- | --- | ---: | --- | --- |
| M1 | Problem selection and requirements engineering | 6 | [Project overview](../README.md); [problem and evaluation proposal](problem-definition.md), [model-card extension](MODEL_CARD.md) in [#5](https://github.com/mlops-2627q1-mds-upc/MLOps_Endesa/issues/5); pinned archive inspection in the [dataset-card extension](DATASET_CARD.md) for [#6](https://github.com/mlops-2627q1-mds-upc/MLOps_Endesa/issues/6) | Peer review; agreed horizon, splits, metrics, and success criterion; verified source units and calendar semantics if obtainable; trained-model evidence in [#7](https://github.com/mlops-2627q1-mds-upc/MLOps_Endesa/issues/7). |
| M1 | Project coordination and communication | 4 | [Working agreement](../CONTRIBUTING.md), [EDN-001](edn/0001-team-workflow.md), [Project](https://github.com/orgs/mlops-2627q1-mds-upc/projects/3), [issue #1](https://github.com/mlops-2627q1-mds-upc/MLOps_Endesa/issues/1), [verified setup](project-setup.md) | Pending access invitations, Project access confirmation, documentation PR review, weekly updates, and feedback follow-up. |
| M2 | Project structure | 5 | Adapted CCDS layout and locked uv environment in [PR #14](https://github.com/mlops-2627q1-mds-upc/MLOps_Endesa/pull/14) for [#10](https://github.com/mlops-2627q1-mds-upc/MLOps_Endesa/issues/10); [setup guide and adaptations](setup.md) | A teammate's clean-checkout setup check recorded in the PR review. |
| M2 | Code and data versioning | 15 | [Workflow](../CONTRIBUTING.md#github-flow-and-review) and [verified branch controls](project-setup.md#repository-settings) | Reviewed PR history, DVC versions, and artifact retrieval from a clean checkout. |
| M2 | Experiment tracking | 5 | Tracking helper, local tests, and a synthetic verification command on the task branch for [#12](https://github.com/mlops-2627q1-mds-upc/MLOps_Endesa/issues/12); [setup and reproduction commands](mlflow.md); [EDN-002](edn/0002-mlflow-tracking.md). | Reviewed merge; shared verification evidence linked from #12; comparable baseline/Chronos runs after #17, with configuration, code/data/model versions, metrics, conclusions, and teammate reproduction. |
| M3 | Energy efficiency awareness | 5 | No implementation evidence recorded. | CodeCarbon measurements, measurement limitations, and interpretation of efficiency trade-offs. |
| M3 | Quality assurance for ML | 10 | No implementation evidence recorded. | Static analysis and meaningful code/data/model tests, including failure cases and data validation results. |
| M4 | ML system design | 15 | No implementation evidence recorded. | Architecture, component responsibilities, constraints, and justified design decisions. |
| M4 | APIs for ML | 10 | No implementation evidence recorded. | API contract, examples, endpoint tests, and verification of the deployed component. |
| M5 | Containers and orchestration | 5 | No implementation evidence recorded. | Reproducible container build, runtime verification, and portability evidence. |
| M5 | CI/CD for ML | 10 | [Review and check policy](../CONTRIBUTING.md#github-flow-and-review); merged [CI PR #21](https://github.com/mlops-2627q1-mds-upc/MLOps_Endesa/pull/21), with a [passing PR run](https://github.com/mlops-2627q1-mds-upc/MLOps_Endesa/actions/runs/36479959754) on 2026-09-28 and a [failing first main run](https://github.com/mlops-2627q1-mds-upc/MLOps_Endesa/actions/runs/36858074938) on 2026-10-01; repair tracked in [#24](https://github.com/mlops-2627q1-mds-upc/MLOps_Endesa/issues/24) | Green main run after repair; required checks on main (#20); meaningful data/model/API tests, model promotion criteria, and verified deployment. |
| M6 | Resource monitoring | 5 | No implementation evidence recorded. | Resource/service metrics, dashboards, alert behavior, and an investigation example. |
| M6 | Model performance monitoring | 5 | No implementation evidence recorded. | Data/model monitoring signals, delayed ground-truth handling, and a demonstrated response to degradation. |
| | **Total** | **100** | | |

The last column is a planning guide for this project, not an additional grading
rubric. Record outcomes and limitations in the linked work.

Source: course laboratory slides, *Machine Learning Systems in Production
(MLOps)*, 2026-27, slides 22-32 and 35-41. The supplied course materials in Atenea
remain authoritative for grading and subsequent updates.
