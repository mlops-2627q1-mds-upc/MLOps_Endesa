# EDN-004: Retain source-scale, positional data with explicit source limits

- **Date:** 2026-10-10.
- **Milestone:** M1 - Inception; related to M2 - Reproducibility.
- **Activity / Topic:** Dataset-card completion for #6.
- **Decision Participants:** Dídac Cayuela authorized the investigation and card update; Pau Adal reviewed and accepted the documented limits.
- **Status:** Accepted by Pau in the 2026-10-10 review; merged in PR #40 on 2026-10-11.

## Decision

Retain the current DVC artifacts and the positional evaluation protocol agreed in
#5 and implemented in #17. Report targets and MAE in stored source scale. Document
MW as supported by the pinned extraction package, with the archive's unit and
civil-clock limits explicit. Pau accepted these limits in PR #40; #6 closed after merge.
Do not rescale values, change precision, or add civil timestamps in this update.

## Rationale

Pablo's source investigation led to the five-state `tsibbledata` files. Their
pinned documentation states MW, and all five training prefixes match
`OperationalLessIndustrial` after `float32` casting. The TSF itself has no unit
or per-observation timestamps. The package script labels period 1 at 00:30,
while TSF starts at 00:00; it also omits an explicit time zone.

Treating the observations as verified local times, or converting their scale,
would add assumptions to the existing comparison. Keeping positions preserves
the reviewed split and current DVC version. The trade-off is that physical-unit
claims and civil-hour, holiday or external-feature joins need further evidence
and a reviewed mapping. This does not change the agreed forecasting protocol.

## AI Involvement

Information seeking; Alternative assessment; Recommendation; Solution generation.
OpenAI Codex checked source-package documentation and AEMO's general clock
convention, verified the raw DVC hashes, compared training values and source
calendar columns, and drafted the card, evidence links and this entry. Pablo's
earlier EDA and source evidence are credited separately.

## Response to AI

Dídac authorized resolving or explicitly documenting the unit/clock limitations,
adding the current DVC version and obtaining review. This authorizes the work;
it is not approval of the drafted interpretation or evidence of team consensus.

## Assessment of the AI Contribution

In his [2026-10-10 review of `e86fac2`](https://github.com/mlops-2627q1-mds-upc/MLOps_Endesa/pull/40#pullrequestreview-5481087502),
Pau accepted the source-scale and positional-only interpretation with its stated
limits, and confirmed that row counts, split boundaries and the 22 TAS negatives
match the project. A separate owner assessment of the AI contribution has not
been recorded. This acceptance does not verify the archive's physical unit or
resolve the interval-start/end mapping.

## Other Evidence

- [Dataset card and recorded versions](../DATASET_CARD.md).
- [Issue #6 and Pablo's source evidence](https://github.com/mlops-2627q1-mds-upc/MLOps_Endesa/issues/6#issuecomment-6013062180).
- [Reviewed EDA, PR #31](https://github.com/mlops-2627q1-mds-upc/MLOps_Endesa/pull/31).
- [Agreed evaluation contract](../problem-definition.md#evaluation-contract-for-m2).
