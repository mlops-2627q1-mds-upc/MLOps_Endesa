# Minimum functionality checks for demand forecasting

[Issue #54](https://github.com/mlops-2627q1-mds-upc/MLOps_Endesa/issues/54)
checks whether the real forecaster continues simple series whose future values
are known. These checks address the M3 model-testing feedback reported by Sindri
on 2026-10-14. They use synthetic demand units and do not estimate performance
on Australian electricity data.

## Cases and limits declared before inference

Every case contains 512 historical values and 48 future values. Demand levels
are 100, 1,000 and 10,000. Repeating patterns use a sine wave with period 48 and
amplitudes of 10% and 50% of each level. There are ten cases: three constant,
one zero, and six repeating patterns. The future continues from position 512,
at phase 32 of the cycle; it does not restart at phase zero.

Both limits below must hold for the median point forecast. MAE is the mean of
the 48 absolute errors; maximum error is the largest absolute error at any step.

| Behaviour | MAE limit | Maximum error limit | Reason |
| --- | ---: | ---: | --- |
| Constant demand | 5% of the demand level | 10% of the demand level | Small average drift is allowed; a single large deviation still fails. |
| Zero demand | 0.01 synthetic units | 0.02 synthetic units | Relative error is undefined at zero; these limits allow small quantization effects without accepting persistent demand. |
| Repeating pattern | 20% of the pattern amplitude | 50% of the pattern amplitude | Error is measured against the variation to be recovered, so a large demand level cannot hide a flat forecast. |

These are proposed regression limits for simple synthetic behaviour, rather than
operational demand thresholds. They are deliberately fixed before the first run.
An exact lag-48 continuation has zero error. A flat forecast at the pattern's
mean fails the pattern limits. MAE prevents positive and negative errors from
cancelling; maximum error prevents one bad step from being hidden by the average.
MASE is unsuitable here because a constant or exactly repeated seasonal history
has a zero seasonal error scale. MAPE also divides by zero for the zero case.

The pinned checkpoint uses uniform token bins over [-15, 15], with spacing about
0.0073 at unit scale. The [Chronos tokenizer](https://github.com/amazon-science/chronos-forecasting/blob/v2.3.2/src/chronos/chronos.py)
substitutes scale 1 for an all-zero context. The zero limits allow roughly one
bin of average error and fewer than three bins at any step; exact floating-point
zero is not required. This is a rationale for the declared limits, not a result
of running the model. [EDN-009](edn/0009-minimum-functionality.md) records the choice
and its pending human review.

## Bounded reproduction

```sh
uv sync --locked --dev --extra forecast
uv run --locked --extra forecast python -m src.modeling.minimum_functionality
```

The runner uses the pinned checkpoint and seed from `params.yaml`, CPU float32,
two CPU threads, batch size 1, and 20 sampled trajectories. Sampling uses
temperature 1.0, top-k 50 and top-p 1.0. The seed is set once for the complete
ordered case set. The existing evaluator loads the checkpoint once and computes
the median using `sample_median`; the known targets never enter the model.

The five-minute cap covers the child process, including pytest startup, checkpoint
retrieval/loading, inference and assertions. Installation is outside that cap.
The process is terminated if it exceeds the cap, and the report records a timeout.
Use `--max-seconds 120` for a smaller cap. Fresh machines may need to retrieve
the checkpoint; a timeout is incomplete evidence and does not imply a model pass.

Reports go to a new ignored `reports/minimum-functionality/<run-id>.json` file.
Use `--output-path <new-file.json>` to choose a destination; existing files are
never overwritten. The report saves the declared protocol and its hash before
starting inference, then records Git/dependency/checkpoint versions, seed,
sampling settings, timings, every target and prediction, both errors, limits,
and case outcomes. All ten results are saved before pytest assertions, so failed
expectations do not prevent later cases from being reported. A timeout or model
loading/inference error leaves uncompleted cases marked `not_run`.

Exit codes are 0 for passing checks, 1 for failed pytest checks, 124 for a timeout,
or another nonzero pytest/error code for incomplete execution. The report's
`status: completed` means inference and scoring completed; individual cases may
still fail. `within_budget` records the observed duration against the cap.

`uv run --locked pytest` and CI keep these ten real-model tests skipped.
Fast tests check case construction, scoring, invalid outputs, and report/timeout
handling using controlled values. They validate the test harness only. Do not
present them as evidence that Chronos meets the behavioural limits.

Case generation and scoring are independent of model loading, so an accepted
fine-tuned checkpoint can use the same cases and limits later. Its model version
and results must be recorded separately. This work adds no training or test-set
tuning and does not depend on the unmerged inference interface in PR #52.
