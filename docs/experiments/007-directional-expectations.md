# Experiment 007: Directional expectations for synthetic demand

Issue [#56](https://github.com/mlops-2627q1-mds-upc/MLOps_Endesa/issues/56)
asks whether sustained changes in recent demand move the next forecast's average
level in the expected direction. These are controlled hypotheses for the pinned
untuned checkpoint, not a requirement that Chronos always responds monotonically.
[EDN-011](../edn/0011-directional-expectations.md) records the proposed acceptance
rules. Pau approved the original
[PR #58](https://github.com/mlops-2627q1-mds-upc/MLOps_Endesa/pull/58) on
2026-10-17. The combined integration and selected-checkpoint extension require
a new peer review.

## Protocol declared before inference

Each pair starts with the same 512 positive float32 values:
`1000 + 150 sin(2πt/48) + 50 cos(4πt/48)`, for positions `t = 0,…,511`.
The values are synthetic units, without a physical-unit or civil-time claim.
Add a constant offset to the final observations, preserving their daily pattern
and all earlier values. The 12 cases combine both directions, both amplitudes
and all three recent windows below.

| Factor | Values | Reason for the expectation |
| --- | --- | --- |
| Direction | Increase; decrease | A sustained new level is hypothesized to persist into the next cycle in the same direction. |
| Amplitude | 5%; 15% of the 1000-unit reference level | Offsets of 50 and 150 units test two distinct positive-demand changes without changing the daily variation. |
| Recent window | 48; 96; 192 observations | One, two and four complete daily cycles separate a sustained level change from an isolated spike. Each window has the same persistence hypothesis. |

For each history, take the median across 20 sampled trajectories at each of the
48 forecast steps, using the evaluator's interpolated median. Let `S` be the
arithmetic mean of those 48 medians and `Δ = S(changed) − S(original)`.
An increase passes when `Δ ≥ 0.10 × offset`; a decrease passes when
`−Δ ≥ 0.10 × |offset|`. Thus the required signed changes are 5 and 15 synthetic
units. This proposed 10% margin requires a measurable response in the expected
direction; an unchanged forecast does not pass. It is an engineering hypothesis,
not a confidence interval or a tolerance estimated from the results.
Individual steps may move in the opposite direction.

An unchanged copy of the reference history is forecast independently. Its maximum
absolute difference from the reference forecast must be at most 0.01 units across
all 48 steps. This stricter comparison detects drift that a horizon mean could
hide. The reference forecast is reused for each paired comparison; the control
therefore checks the same baseline used by all cases.

Use `amazon/chronos-t5-small` at revision
`a971ba21945c4f1796b17a91fe69214b5f4ad472`, seed 7, 20 samples, float32, CPU with
two threads, batch size 1, temperature 1, top-k 50 and top-p 1. Reset the seed
before each of the 14 calls: reference, unchanged control and 12 interventions.
One checkpoint is loaded for the whole run. Matching the seed and sampling setup
reduces an avoidable difference between pairs; it does not require bitwise
agreement across platforms, as the [PyTorch reproducibility guidance](https://docs.pytorch.org/docs/2.14/notes/randomness.html)
explains. The inference adapter follows the project's existing evaluator and the
[upstream Chronos interface](https://github.com/amazon-science/chronos-forecasting).

The parent process writes `plan.json` before starting inference and terminates the
worker if the invocation exceeds 180 seconds, including Python startup, checkpoint
loading and inference. A timeout or model error returns failure and retains the
plan. A completed run retains every expectation result, including failures, in
`report.json`; it does not stop after the first unsupported hypothesis.

## Reproduction

From this checkout, install the locked environment and run the declared protocol:

```sh
uv sync --locked --dev --extra forecast
uv run --extra forecast python -m src.modeling.directional --max-seconds 180
```

The command prints each result and the report path under
`reports/directional/<unique-id>/`. It returns 1 for an unsupported expectation,
failed control, timeout or error. Each invocation gets a new directory. No data
retrieval, training, MLflow upload or final-test access occurs.

Use an available local checkpoint without changing the declared protocol:

```sh
uv run --locked --extra forecast python -m src.modeling.directional \
  --checkpoint models/chronos-t5-small-fine-tuned --max-seconds 180
```

The opt-in pytest route accepts `--model-checkpoint <directory>`. The plan's `model`
field records the executed Hub revision or resolved local path and SHA-256 of
`model.safetensors`, which the worker verifies before inference. `settings` retains
the base configuration; its model fields do not identify local executed weights.
Each run keeps its own results with the same cases, ordering, limits and sampling.

The same checks are available as parameterized pytest cases. Their session fixture
runs the real checkpoint once and shares its report across all 13 assertions:

```sh
uv run --extra forecast pytest tests/test_directional.py --run-directional -k real -v -s
```

Ordinary `make test` skips these 13 checks. It validates case construction, the
comparison rule, control drift, retained failures, timeout handling and matched
seeding with small test doubles. Those checks validate the implementation; only
the opt-in run is evidence about the real model. A local checkpoint can use the
same cases and comparison rules through the explicit route above; its measured
results must be recorded separately.

## Results and limitations

On 2026-10-14, all 12 directional cases and the unchanged control passed on clean
source commit [`57236a624eed310c694c7840c5f6ce96f948f136`](https://github.com/mlops-2627q1-mds-upc/MLOps_Endesa/commit/57236a624eed310c694c7840c5f6ce96f948f136).
The protocol and comparison rules were committed before inference. No cases or
margins were changed after observing the results. The unchanged control's maximum
absolute difference was **0.0**. No counterexamples were observed in this run.

The [complete evidence report](../evidence/issue-56-directional-expectations.json)
retains the base history, case definitions, input hashes, all 14 median forecasts,
declared margins, results, source/checkpoint versions, environment and timings.
Its SHA-256 is
`943b4b99e9b1697950bc3c18a9a778bda772ab8f1944969e25e94d931056a55a`.
The report was copied without modification from the runner's output. All input
hashes and decision statistics were independently recalculated from its arrays.

| Direction | Offset, synthetic units | Recent observations | Observed mean change, Δ | Minimum signed change | Result |
| --- | ---: | ---: | ---: | ---: | --- |
| Increase | +50 | 48 | +20.146267 | 5 | Pass |
| Increase | +50 | 96 | +49.644725 | 5 | Pass |
| Increase | +50 | 192 | +48.882449 | 5 | Pass |
| Increase | +150 | 48 | +101.061283 | 15 | Pass |
| Increase | +150 | 96 | +155.979874 | 15 | Pass |
| Increase | +150 | 192 | +146.895826 | 15 | Pass |
| Decrease | −50 | 48 | −60.673697 | 5 | Pass |
| Decrease | −50 | 96 | −55.885988 | 5 | Pass |
| Decrease | −50 | 192 | −56.200214 | 5 | Pass |
| Decrease | −150 | 48 | −181.729560 | 15 | Pass |
| Decrease | −150 | 96 | −156.231700 | 15 | Pass |
| Decrease | −150 | 192 | −154.688459 | 15 | Pass |

The bounded invocation took **31.127 seconds**, below its 180-second cap.
Checkpoint loading took 0.090 seconds from an existing cache and the forecast loop
took 11.531 seconds. The total also includes worker startup, imports and report
preparation; this is a single observed run, not a cold-start or latency benchmark.
The host was an Apple M4 with ten logical CPUs, macOS 26.2 arm64. Recorded versions
were Python 3.11.15, Chronos 2.3.2, PyTorch 2.14.1, Transformers 5.18.0 and NumPy
2.4.6. The report records the full lockfile hash and unchanged checkpoint/sampling
settings declared above.

The real pytest reproduction command above returned **13 passed, 27 deselected**.
The separate `make test` run returned **98 passed, 22 skipped**: nine checks need
the absent DVC data and the 13 real-checkpoint assertions are opt-in. `make lint`
passed Ruff and the Pylint gate at 9.50/10. These are the original component's
checks, separate from the final combined checks.

One pattern, one phase and one seed test a narrow persistence hypothesis. Even
passing all cases would not establish forecast accuracy, causal demand effects,
universal monotonicity, weather/calendar sensitivity or behavior of the fine-tuned
checkpoint. There are no forecast targets in this experiment and no model
selection or promotion decision.

The selected-checkpoint run is recorded separately in
[Experiment 008](008-selected-checkpoint-behaviour.md). Review of that extension
and the combined change remains pending.
