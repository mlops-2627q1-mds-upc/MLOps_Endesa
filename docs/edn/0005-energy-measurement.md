# EDN-005: Energy measurement of the forecast step

- **Date:** 2026-10-11.
- **Milestone:** M3 - Energy efficiency awareness.
- **Activity / Topic:** CodeCarbon measurement for #39.
- **Decision Participants:** Sindri Másson.
- **Status:** Measurement implementation reviewed by Dídac and merged in PR #41 on 2026-10-11.

## Decision

Measure only the forecast step with CodeCarbon's offline tracker, using Spain's
energy mix (`ESP`) and `process` tracking mode, and record per run how each
component's energy was obtained. Where NVML does not support the GPU energy
counter but reports power samples, integrate those samples as the GPU energy.
Where neither is available, record the GPU as not measured instead of 0 kWh.

## Rationale

Measuring only the forecast step keeps model loading and MLflow uploads, which
depend on caching and the network, out of the comparison between methods. Runs
execute locally in Barcelona, so the offline tracker with a fixed country avoids
a network lookup. In `process` mode CPU energy is attributed to this process
rather than to everything running on the machine.

The team decided to fine-tune Chronos (#13) on the GPU of Sindri's machine, so
the GPU energy must be measurable there. Its HP OEM RTX 4060 does not support
NVML's energy counter or current power reading with drivers 610.47 and 617.42, at
idle or under load, with or without administrator rights; HP offers no graphics
firmware update. CodeCarbon then records 0 kWh, which reads as a measurement.
NVML's power samples follow the load (about 50 W idle, about 110 W at full load,
limit 115 W), so they are used. A separate sampler was chosen over overriding
CodeCarbon's internal GPU classes, which would depend on undocumented behavior.

## AI Involvement

Information seeking; Alternative generation; Alternative assessment;
Recommendation; Solution generation. Claude Code (Anthropic) diagnosed the
unsupported NVML readings and proposed the hardware checks. It proposed overriding
CodeCarbon's GPU classes, then a separate sampler, implemented the sampler with
tests, and drafted this entry and the documentation.

## Response to AI

Accepted with modifications. Sindri ran the hardware checks and required the
standard CodeCarbon path to be exhausted before any workaround. Sindri questioned
the robustness of overriding CodeCarbon's GPU classes; Claude Code then
recommended the separate sampler, which Sindri accepted.

## Assessment of the AI Contribution

Sindri committed the sampler after a 10 s check under full load, which read 497
samples without loss and gave 3.03 × 10⁻⁴ kWh at a mean of 109 W.
The measured comparison is recorded in [Experiment 002](../experiments/002-energy-measurement.md);
Dídac reviewed the NVML integration and error handling, approved PR #41, and
the implementation merged on 2026-10-11.

## Other Evidence

- [Issue #39](https://github.com/mlops-2627q1-mds-upc/MLOps_Endesa/issues/39).
- [Energy measurement and observed limitations](../evaluation.md#energy-measurement).
- Source: `src/tracking.py`, `src/modeling/evaluate.py`; tests: `tests/test_tracking.py`.
