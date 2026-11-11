# EDN-020: Keep drift assessment finite and separate from serving

Date: 2026-11-11 (recorded).

Milestone: M6

Activity / Topic: Demand drift (#82)

Decision Participants: Dídac Cayuela

Status: Recorded; peer review pending.

## Decision

I chose a fixed training-only reference and six disjoint eight-week validation windows,
evaluated separately for each state. Alibi runs offline in its own locked environment.
The detector supports investigation; operational notification thresholds remain open.

## Rationale

Overlapping forecast histories would inflate sample counts. Weekly residual summaries
retain level/variability information, while a finite schedule bounds repeated testing.
Alibi requires NumPy below 2, so isolation avoids downgrading the serving environment.
The eight-week windows also impose a substantial detection delay.

## AI Involvement

Solution generation; Other. I used Codex for the detailed detector design and
implementation of the reference/window approach I selected, including features, test
scheduling, dependency isolation and assessment.

## Response to AI

Other. I used detector design and assessment support.

## Assessment of the AI Contribution

I kept the assessment as an investigation tool because it produced 8/30
historical-control alerts and shift-detection delays of 56-112 nominal days. Those
results need further calibration before operational paging; input drift alone does not
establish forecast degradation.

## Other Evidence

[PR #91](https://github.com/mlops-2627q1-mds-upc/MLOps_Endesa/pull/91). The shorter
selected-model replay correctly returned insufficient_data. No final-test targets were
used.
