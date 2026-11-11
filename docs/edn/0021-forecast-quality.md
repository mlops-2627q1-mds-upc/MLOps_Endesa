# EDN-021: Score only released observations and suppress stale quality values

Date: 2026-11-11 (recorded).

Milestone: M6

Activity / Topic: Forecast-quality monitoring (#83)

Decision Participants: Dídac Cayuela

Status: Recorded; peer review pending.

## Decision

I chose to score the selected model and lag48 baseline on identical released
observations, using training-only MASE scales. A separate exporter omits quality values
when reports are missing, invalid or stale.

## Rationale

Reading targets directly would bypass delayed-label semantics, and unmatched rows would
make the comparison inconsistent. I kept source-bound scales and a display minimum of 30
matched rows per state/horizon. Report refresh runs separately from inference and metric
scrapes.

## AI Involvement

Solution generation; Other. Codex implemented scoring, reports, the exporter, dashboard
and tests. Native agent QA found and corrected an aggregate horizon-label error.

## Response to AI

Other. I used scoring and exporter implementation support.

## Assessment of the AI Contribution

I retained explicit availability states because a reachable exporter does not mean its
quality report is valid. Missing samples stay undefined, and the observed interval
coverage remains descriptive rather than becoming a coverage target.

## Other Evidence

[PR #92](https://github.com/mlops-2627q1-mds-upc/MLOps_Endesa/pull/92). The subset had
150 forecasts, with state-level p10-p90 coverage of 61.25-74.93%. Browser/Compose checks
and Virtech monitoring deployment remain pending.
