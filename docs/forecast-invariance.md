# Paired forecast behaviour checks

[Issue #55](https://github.com/mlops-2627q1-mds-upc/MLOps_Endesa/issues/55)
checks whether a change to a 512-value history causes an unexpected change in
its complete 48-step forecast. These comparisons measure behaviour, without
training a model or evaluating accuracy against future targets.
[EDN-010](edn/0010-forecast-invariance.md) records the design choice.

## Relationships and thresholds

Let `S = mean(abs(history))`. After undoing any scale factor, compute the absolute
forecast difference at every step, divide by `S`, and report both its mean and
maximum. Both thresholds must hold. The maximum prevents a large change at one
step from disappearing in an average. This normalization supports signed values
and individual zeros; an entirely zero history has no positive scale and is
excluded from this protocol.

| Change | Expected relationship | Mean / maximum thresholds |
| --- | --- | --- |
| Multiply every input by 0.5, 2 or 10 | Divide the changed forecast by the same factor before comparison | `1e-5` / `1e-4` of `S` |
| Add fixed uniform noise bounded by `0.001 * S` | A small input change should produce a limited forecast change | `0.05` / `0.15` of `S` |
| Add fixed uniform noise bounded by `0.01 * S` | Same robustness hypothesis at a larger perturbation | `0.05` / `0.15` of `S` |

Positive scaling preserves the shape and order of a history. Seasonal naive
forecasts satisfy `f(c*x)/c = f(x)` in exact arithmetic. The pinned Chronos
checkpoint uses mean-absolute scaling and uniform token bins, as specified by
its [configuration](https://huggingface.co/amazon/chronos-t5-small/blob/a971ba21945c4f1796b17a91fe69214b5f4ad472/config.json)
and [upstream implementation](https://github.com/amazon-science/chronos-forecasting/blob/v2.3.2/src/chronos/chronos.py).
In exact arithmetic, positive rescaling leaves the normalized inputs unchanged.
With identical tokens and random draws, the decoded forecasts rescale together.
Float32 rounding near a token boundary can break that condition. The scale
thresholds allow small numerical differences; larger differences remain
counterexamples rather than being explained away as rounding.

Noise robustness is a hypothesis, not an exact property of Chronos. The 5% mean
and 15% maximum limits are initial investigation thresholds, declared before
inference. They flag forecast changes substantially larger than the 0.1% or 1%
input bound. They are not operational acceptance limits or measured guarantees.
Neither thresholds nor cases should be adjusted to hide an observed failure.

Noise uses NumPy's generator with seed 7. Each history gets the same ordered
uniform noise vector, scaled by its own `S`; the two noise levels use the same
direction. Changed inputs are rounded to float32. Evidence records the actual
mean and maximum input changes relative to `S`, including that rounding.
No history is shuffled, clipped or converted to another physical unit.

## Cases and execution

The synthetic cases are constant, seasonal, trending with a cycle, and signed.
Their cycles describe positional patterns, without assigning civil dates.
With stored data, each of the five states also contributes:

- A training history ending immediately before `train_end - horizon`.
- A validation history ending at the midpoint origin rounded down to the
  configured origin stride.

Stored files must match their `dvc.lock` checksums and configured column/index
sequences. Only `train.parquet` and `validation.parquet` are opened. Every window
ends strictly before its stated origin. Window hashes identify the inputs in
the report; source observations remain in ignored data files.

The default run has 14 histories, five transformations, 70 pairs and 140
48-step forecasts. It reuses the [validation evaluator](evaluation.md), with
batch size 1 and seed 7 reset before **each** original or changed history. This
matches the sampling stream within a pair. Resetting once before a batch of
two histories would give the two rows different draws. The checkpoint loads
once; all calls use the same 20 samples, float32 dtype, temperature 1, top-k 50
and top-p 1. No changes are made to ordinary evaluation's random stream.

The runner saves the cases, thresholds, settings and versions before inference.
Its report retains every pair, both full forecasts, metrics and counterexample
IDs. A completed run can contain failed relationships. An interrupted run is
marked failed and must not be interpreted as a complete comparison.

## Reproduction

1. Install the locked optional forecasting environment:
   `uv sync --locked --dev --extra forecast`.
2. Restore the two input files with your own configured DVC access:
   `uv run dvc pull data/processed/train.parquet data/processed/validation.parquet -r origin`.
3. Run the paired checks:

   ```sh
   uv run --extra forecast python -m src.modeling.invariance --max-seconds 300
   ```

Run an available local checkpoint with the unchanged protocol:

```sh
uv run --locked --extra forecast python -m src.modeling.invariance \
  --checkpoint models/selected-chronos-small --max-seconds 300
```

The opt-in pytest route also accepts `--model-checkpoint <directory>`. The report's
`model` field records the executed Hub revision or the resolved local checkpoint
path and SHA-256 of `model.safetensors`; the worker verifies that hash before
inference. `settings` retains the base configuration, so its model fields describe
the default rather than local executed weights. A local run produces separate
evidence and uses the existing cases, order, thresholds and sampling settings.

Each invocation creates a new ignored `reports/invariance/<id>/report.json`.
Exit status 1 means at least one declared threshold was exceeded; all completed
outcomes are saved before that exit. `--synthetic-only` explicitly limits a run
to the four synthetic histories. `--params-path` selects another pinned
checkpoint configuration; retain the same cases and thresholds for comparison.
The case and metric functions are independent of the checkpoint.

The 300-second budget includes input preparation and checkpoint loading. It is
checked between forecasts and before recording a complete result. It is a
cooperative deadline: a single loading or prediction call can overrun it.
Use an external process timeout when a hard wall-clock limit is needed.

Routine `make test` checks the transformations, metrics, causal windows,
version guards, failure recording and seed handling without model downloads.
For readable pytest case IDs and an independent bounded model run:

```sh
uv run --extra forecast pytest tests/test_invariance.py \
  -k real_model --run-forecast-invariance -q
```

The module fixture runs the model once and saves the entire report before the
70 assertions. Counterexamples fail those assertions; they are not skipped or
marked as expected failures. These optional checks stay outside ordinary CI.
The [experiment record](experiments/006-forecast-invariance.md) links the
observed results; peer review is pending. One checkpoint, seed and perturbation
direction cannot establish general robustness, accuracy or behaviour after fine-tuning.
