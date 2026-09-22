# MLOps_Endesa: Course Practice Evidence

This register links the 13 assessed practices to project evidence. The points are
the rubric weights, not a self-awarded score. Update the corresponding row when
evidence is available, including the issue/PR, artifact or run, version, reviewer,
and validation date. Detailed outcomes belong in the linked artifacts.

**Updated: 2026-09-22.** The repository documentation contains a project overview and the
initial workflow documentation. The organization Project and documentation issue
are live. The rows below distinguish verified setup from work still to be
reviewed or implemented. Publication and peer review are tracked in
[issue #1](https://github.com/mlops-2627q1-mds-upc/MLOps_Endesa/issues/1);
[issue #2](https://github.com/mlops-2627q1-mds-upc/MLOps_Endesa/issues/2) holds the
outstanding repository setup checklist.

| Milestone | Assessed practice | Points / 100 | Current evidence | Evidence still needed |
| --- | --- | ---: | --- | --- |
| M1 | Problem selection and requirements engineering | 6 | [Project overview](../README.md) | Reviewed success criteria, dataset card, and model card. |
| M1 | Project coordination and communication | 4 | [Working agreement](../CONTRIBUTING.md), [EDN-001](edn/0001-team-workflow.md), [Project](https://github.com/orgs/mlops-2627q1-mds-upc/projects/3), [issue #1](https://github.com/mlops-2627q1-mds-upc/MLOps_Endesa/issues/1), [verified setup](project-setup.md) | Native repository link, access/assignment setup, team review, weekly updates, and feedback follow-up. |
| M2 | Project structure | 5 | No implementation evidence recorded. | Adapted project structure, environment setup, and a teammate's setup check. |
| M2 | Code and data versioning | 15 | [Documented workflow](../CONTRIBUTING.md#github-flow-and-review) | Reviewed PR history, verified branch controls, DVC versions, and artifact retrieval from a clean checkout. |
| M2 | Experiment tracking | 5 | No implementation evidence recorded. | Comparable MLflow runs with configuration, code/data/model versions, metrics, and conclusions. |
| M3 | Energy efficiency awareness | 5 | No implementation evidence recorded. | CodeCarbon measurements, measurement limitations, and interpretation of efficiency trade-offs. |
| M3 | Quality assurance for ML | 10 | No implementation evidence recorded. | Static analysis and meaningful code/data/model tests, including failure cases and data validation results. |
| M4 | ML system design | 15 | No implementation evidence recorded. | Architecture, component responsibilities, constraints, and justified design decisions. |
| M4 | APIs for ML | 10 | No implementation evidence recorded. | API contract, examples, endpoint tests, and verification of the deployed component. |
| M5 | Containers and orchestration | 5 | No implementation evidence recorded. | Reproducible container build, runtime verification, and portability evidence. |
| M5 | CI/CD for ML | 10 | [Review and check policy](../CONTRIBUTING.md#github-flow-and-review) | Successful and failing workflow evidence, model promotion criteria, and deployment verification. |
| M6 | Resource monitoring | 5 | No implementation evidence recorded. | Resource/service metrics, dashboards, alert behavior, and an investigation example. |
| M6 | Model performance monitoring | 5 | No implementation evidence recorded. | Data/model monitoring signals, delayed ground-truth handling, and a demonstrated response to degradation. |
| | **Total** | **100** | | |

The evidence descriptions explain what could substantiate each practice for this
project; they do not introduce additional official grading requirements. Record
the actual outcome and limitations rather than claiming success from a completed
issue or an installed tool. Keep the EDN focused on selected decisions.

Source: course laboratory slides, *Machine Learning Systems in Production
(MLOps)*, 2026-27, slides 22-32 and 35-41. The supplied course materials in Atenea
remain authoritative for grading and subsequent updates.
