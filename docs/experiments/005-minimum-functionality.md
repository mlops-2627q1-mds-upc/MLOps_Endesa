# Minimum functionality results: pinned Chronos checkpoint

On 2026-10-14, all ten real-model cases passed the limits declared in
[the protocol](../minimum-functionality.md) before inference. No limits were
changed after seeing the results. Pau approved the original
[PR #57](https://github.com/mlops-2627q1-mds-upc/MLOps_Endesa/pull/57) on
2026-10-17. The combined integration and selected-checkpoint extension require
a new peer review.

## Recorded conditions

| Item | Value |
| --- | --- |
| Source revision | `25fafd00f856e4dbc9e398cb049ca495b935d26b`, clean worktree |
| Checkpoint | `amazon/chronos-t5-small` at `a971ba21945c4f1796b17a91fe69214b5f4ad472` |
| Dependency lock SHA-256 | `1c02f6c9c4604bdda7028dda3c1ae620bd6ef112dab6dc6082abaacf7f004145` |
| Software | Python 3.11.15; Chronos 2.3.2; Torch 2.14.1; Transformers 5.18.0; NumPy 2.4.6; pytest 9.1.1 |
| Hardware and device | macOS 26.2, arm64; CPU float32, two CPU threads |
| Forecast settings | 512 context values, 48 targets, 20 samples, batch size 1; median point forecast |
| Randomness | Seed 7, set once; constant cases, zero case, then patterns in the declared order |
| Sampling | Temperature 1.0, top-k 50, top-p 1.0 |
| Runtime cap | 300 seconds for the child pytest process, including startup/loading/inference/assertions |
| Observed capped-process runtime | 31.774 seconds |
| Checkpoint loading / forecast loop | 0.074 / 8.965 seconds; checkpoint already cached |
| Result | 10 passed, 0 failed, 0 invalid; exit code 0 |
| Protocol SHA-256 | `ce691fec5f19368eb0fafa10b88e9ba28d1b745f25538dedbb3553c848c3cd1f` |

The [complete JSON report](../evidence/issue-54-minimum-functionality.json)
contains every target, prediction, limit, metric and outcome. Its SHA-256 is
`c4379cc39c7acf4e10b16b5c478585a39ad0b1e9686ec9c81d59c06e59738eb7`. The evidence commit only adds documentation and this report; the
executed code and declared tolerances remain at the source revision above.

## Per-case errors

All values below use arbitrary synthetic demand units. Each cell lists
**observed error / declared limit**. Both error limits must pass.

| Pytest case ID | MAE / limit | Maximum error / limit | Outcome |
| --- | ---: | ---: | --- |
| `constant-level-100` | 0.293259 / 5 | 0.293259 / 10 | Pass |
| `constant-level-1000` | 2.93256 / 50 | 2.93256 / 100 | Pass |
| `constant-level-10000` | 29.3252 / 500 | 29.3252 / 1000 | Pass |
| `zero-demand` | 1.39698e-08 / 0.01 | 1.39698e-08 / 0.02 | Pass |
| `pattern-level-100-amplitude-10` | 0.385233 / 2 | 1.25504 / 5 | Pass |
| `pattern-level-100-amplitude-50` | 0.196422 / 10 | 0.367027 / 25 | Pass |
| `pattern-level-1000-amplitude-100` | 3.6794 / 20 | 13.7902 / 50 | Pass |
| `pattern-level-1000-amplitude-500` | 2.19598 / 100 | 10.2877 / 250 | Pass |
| `pattern-level-10000-amplitude-1000` | 42.6583 / 200 | 132.275 / 500 | Pass |
| `pattern-level-10000-amplitude-5000` | 18.8854 / 1000 | 36.4846 / 2500 | Pass |

Constant forecasts have a small systematic offset of about 0.293% of the demand
level. The zero forecast is approximately `1.40e-8` rather than exactly zero.
Pattern errors vary across demand levels under the recorded sampling sequence.
These deviations remain visible in the report even though the cases pass.

## Reproduction and checks

The protocol and implementation were committed before running:

```sh
uv sync --locked --dev --extra forecast
uv run --locked --extra forecast python -m src.modeling.minimum_functionality --output-path reports/minimum-functionality/initial-chronos.json
```

Use a new output filename when repeating the command. The checkpoint is loaded
once for all cases. The real-model pytest run reported `10 passed in 29.32s`;
the runner's longer timing includes the full child-process lifecycle.

Local `make lint` passed (Pylint 9.50/10), and `make test` reported 101 passed,
19 skipped and one existing MLflow/SQLAlchemy deprecation warning. The skips
cover nine data-dependent checks without local DVC data and ten opt-in real-model
checks; the separate command above executed all ten real-model checks. Thirty
fast tests exercise case construction, known continuation, scoring, invalid
outputs, failure preservation and timeout handling. Those controlled forecasts
validate the harness, rather than Chronos itself.

## Interpretation and pending review

This is one untuned checkpoint, one seed, ten synthetic series and one CPU
software environment. The cases establish only the stated behaviours within
the declared limits. They do not measure Australian demand accuracy, interval
coverage, robustness to noise or missing values, or performance after fine-tuning.
No training, final-test evaluation, DVC data access or shared MLflow write occurred.
Bitwise equality across devices and dependency versions has not been checked.

Keep the same limits when assessing a fine-tuned checkpoint; record its version
and outcomes separately. The selected-checkpoint run is recorded in
[Experiment 008](008-selected-checkpoint-behaviour.md); review of that extension
and the combined change remains pending.
