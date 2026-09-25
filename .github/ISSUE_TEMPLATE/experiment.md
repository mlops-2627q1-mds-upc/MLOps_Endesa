---
name: Experiment or investigation
about: Define a bounded research question and its completion evidence.
title: ''
labels: ''
assignees: ''
---

## Question or hypothesis

State what the experiment should establish and what decision it will inform.

## Planning and budget

Set the native assignee and milestone, Project Priority and Size, and
`type:experiment` plus the relevant area labels. State the course practice,
dependencies, time limit, and available compute budget.

## Method and comparison

Define the baseline, data/split versions, model revision, configurations, metrics,
and controls for a fair comparison. State how temporal leakage will be prevented
where relevant. Agree the evaluation method before inspecting the final test set.

## Completion criteria

- [ ] Method, configuration, and reproduction instructions are recorded.
- [ ] Results and limitations, including unsuccessful outcomes, are available.
- [ ] A teammate has reviewed the interpretation and next action.

## Results and conclusion

Add MLflow run IDs or other evidence, Git/data/model versions, relevant resource
measurements, and the conclusion. Distinguish planned work from actual results.
Improving the metric is not a prerequisite for completing an investigation.

## Decision and follow-up

Link the relevant EDN entry and any implementation or follow-up issues. Record
whether the findings support changing the model, method, or system.
