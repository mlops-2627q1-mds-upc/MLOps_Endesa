# Monitoring scenarios

Issue [#84](https://github.com/mlops-2627q1-mds-upc/MLOps_Endesa/issues/84)
checks whether a declared fault produces a useful signal and whether that signal
changes after recovery. The runner uses the production API/metrics, public replay
journal, quality report/exporter and Alibi drift APIs. Its predictors and demand
inputs are explicitly synthetic. It never loads the selected checkpoint.

## Declared method

The versioned [scenario declarations](../monitoring/scenarios/scenarios.json)
specify each budget and expected signal before execution. A supervising process
limits each invocation and reserves time for stopping owned resources. These
budgets include setup and cleanup and are below the 30-minute issue limit.

| Scenario | Budget | Normal control and intervention | Observation and recovery |
| --- | --- | --- | --- |
| Normal | 180 seconds | Successful synthetic forecasts through the production handlers | HTTP 200, completed request counts, response duration and readiness |
| Service unavailable | 240 seconds | A passing forecast precedes removal of the fixture predictor, or stopping only the runner's Docker API | Fixture: HTTP 503 and ready=0. Docker: scrape up=0, actual `ForecastApiDown` firing after its unchanged one-minute hold, restart, ready=1 and alert clearing, then another HTTP 200 |
| Slow request | 180 seconds | A passing forecast precedes three requests with a fixed 0.25-second predictor delay | Completed HTTP durations include the delay; removing it permits another successful request. No numerical latency alert is configured |
| Shifted demand | 180 seconds | Seed-7 unchanged demand, one training-standard-deviation level shift in the second complete window, restored demand in the third | Fresh Alibi KS results for every state/window, sample counts, missed detections and control/recovery alarms; next scheduled result records recovery |
| Delayed observation | 120 seconds | Four declared forecasts from the shared #81 fixture; observation delay is two positions | At clock 516, six joins and 186 pending steps; release through 563 produces 192 joins and zero pending steps. The actual quality exporter reads both reports |

Service durations and Prometheus transitions use elapsed monotonic seconds and
UTC observations. Drift and observation release use positional replay time.
One positional step represents a nominal 30-minute observation, without a civil
timestamp. A 2,688-position drift window therefore cannot establish detection
within 30 wall-clock minutes on a live stream. Its compressed synthetic execution
can finish within the declared scenario budget.

The drift reference has 104 weekly cycles and each comparison needs eight complete
weekly cycles per state. The declared configuration retains its six-window/five-state
multiple-testing budget even though this demonstration executes only three windows.
Thresholds are inherited unchanged. Missing samples remain insufficient evidence.
Each control or recovery alarm and each missed shifted window is retained; a
completed experiment can have a negative detection result.

The delayed fixture uses a display minimum of two matches per state/horizon to
exercise report transitions. The report is also scored with the default minimum
30. Final horizon groups remain insufficient under that policy. Overlapping
forecasts share targets and are dependent. Synthetic perfect intervals do not
estimate the selected model's accuracy or establish a coverage SLO.

## Local execution

Run from this checkout with its locked Python 3.11 environment. Every output
directory must be absent. The default engine is an in-process fixture and needs
no Docker, network target, dataset, model download or model inference.

```bash
uv sync --offline --locked --dev
.venv/bin/python -m src.monitoring_scenarios list
.venv/bin/python -m src.monitoring_scenarios run --scenario normal \
  --directory temporary/issue-84/control-01
.venv/bin/python -m src.monitoring_scenarios run --scenario service-unavailable \
  --directory temporary/issue-84/unavailable-fixture-01
.venv/bin/python -m src.monitoring_scenarios run --scenario slow-request \
  --directory temporary/issue-84/slow-fixture-01
.venv/bin/python -m src.monitoring_scenarios run --scenario delayed-observation \
  --directory temporary/issue-84/delayed-01
```

Use the locked [drift environment](../monitoring/drift/pyproject.toml) for the
fresh synthetic KS scenario. On a machine without that environment, preparing it
is a separate dependency installation. Do not silently download packages during
an evidence run or substitute recorded historical playback for this command.

```bash
monitoring/drift/.venv/bin/python -m src.monitoring_scenarios run \
  --scenario shifted-demand --engine alibi \
  --directory temporary/issue-84/shift-01
```

The Mac execution may reuse a compatible drift environment from another checkout
by calling its absolute Python path while keeping this checkout as the working
directory. Only this checkout receives reports. The request records the Python
source bytes and the result records Alibi/numpy/scipy/pandas versions.

For actual HTTP scrapes and alert transitions, supply a unique `endesa-84-*`
target identity and the immutable ID of an inspected cached fixture image. The
runner refuses remote Docker daemons, mutable image tags and dependency/parameter
bytes that differ from the image's source revision. It mounts this checkout's
source read-only over the cached fixture base. That overlay does not create a
final-source release image. Pulling and building images are unsupported.

```bash
FIXTURE_IMAGE=sha256:673b9b4c5c37fd20452fe9662a070e56201bd4459911290dd643f4a590cf7df3
.venv/bin/python -m src.monitoring_scenarios run --scenario normal --engine docker \
  --target-id endesa-84-control-01 --fixture-image "$FIXTURE_IMAGE" \
  --directory temporary/issue-84/control-docker-01
.venv/bin/python -m src.monitoring_scenarios run --scenario service-unavailable --engine docker \
  --target-id endesa-84-down-01 --fixture-image "$FIXTURE_IMAGE" \
  --directory temporary/issue-84/down-docker-01
.venv/bin/python -m src.monitoring_scenarios run --scenario slow-request --engine docker \
  --target-id endesa-84-slow-01 --fixture-image "$FIXTURE_IMAGE" \
  --directory temporary/issue-84/slow-docker-01
```

The example image is the retained #83 Linux amd64 fixture base at
`ae7a9151ee945292fff8d4a18343754f43ddeb12`. Its execution on this arm64 Mac is
emulated. The API has one worker/two numerical threads; each of the two owned
containers is limited to two CPUs/768 MiB. A unique ordinary bridge connects
only those containers. API and Prometheus use dynamically assigned ports bound
explicitly to 127.0.0.1. Outbound traffic is possible. No production checkpoint,
Docker socket, credential or shared data volume is mounted.

The original service rules are mounted unchanged. Only the isolated scrape and
evaluation interval is shortened to two seconds; the one-minute alert hold is
preserved. No Alertmanager or notification receiver is configured, so observed
Prometheus firing/clearing never proves notification delivery.

## Evidence and response

`request.json` retains source HEAD/tree, dirty paths, source/config hashes,
target identity, engine and wall budget. `execution.log` retains child failures.
`result.json` contains expected/observed signals, applicable clock, sample
sufficiency, recovery and missed detections. `summary.json` records exit status,
elapsed time and verified owned cleanup, including after a child crash. A
nonzero exit is a failed procedure and remains a failure after recovery.

Docker runs additionally retain exact image inspection, container IDs, bindings
and network identity in `runtime.json`, actual scrape/alert/target observations
in `observations.jsonl`, raw API metrics and container logs. Incomplete runs keep
their partial evidence. Cleanup verifies each full container ID and ownership
label before stopping it. It retains stopped containers, networks and evidence;
it never removes or prunes resources. If cleanup cannot be verified, stop using
the run and inspect only the IDs named in its runtime receipt. Re-run cleanup
from this checkout after resolving the local daemon error:

```bash
.venv/bin/python -c 'from pathlib import Path; from src.monitoring_scenarios.service import cleanup; print(cleanup(Path("temporary/issue-84/down-docker-01")))'
```

1. For failed scrapes, read the actual target error, API log and readiness before
   interpreting workload panels. Restore only the owned API, observe a successful
   scrape and forecast, then record alert clearing. A missing target is different
   from a failed scrape.
2. For slow requests, inspect completed-request durations and in-flight work.
   Compare matching request/settings/resource identities with the
   [serving benchmark](serving-benchmark.md). Queueing alone establishes neither
   input drift nor forecast degradation. Any numerical alert policy needs a
   separate human decision and validation.
3. For drift, retain the unchanged control, reference/config/sample identities,
   actual KS results and positional delay. Check completeness and source scale
   before investigating a shift. Record a later restored-distribution result;
   it is not an operational alert reset or a reason to retrain automatically.
4. For delayed observations, inspect the replay clock, pending joins and per-horizon
   sample status. Advance only the declared synthetic observation release clock.
   Retain insufficient groups; never fill missing truth from unreleased source
   rows or replace absence with zero error.

Source/fixture CI, fresh synthetic Alibi execution, owned HTTP/Prometheus
execution, selected-model inference, registry publication and deployment are
separate evidence categories. The [historical drift playback](drift.md) remains
recorded evidence: it computes no fresh KS result or recovery. Historical drift
alarms/delays, subset coverage and the benchmark's forced stop with exit 137 and
`OOMKilled=false` are unchanged and do not become adopted operating targets.

## Recorded local results, 12 November 2026

The [compact evidence record](../monitoring/scenarios/local-evidence-2026-11-12.json)
retains exact run source/tree, declaration and report hashes, image/container and
network identities, actual bindings, observed signals and stopped-container
states. Full logs/reports remain in ignored `temporary/issue-84/runs/` under the
recorded run names. The final executable scenario source is
`aa33efba345c5f9e81f0bac322d28291f1b2eaaa`; in-process fixture runs also retain
their earlier exact source identities. No registry image was built or published.

| Evidence | Observed result |
| --- | --- |
| In-process normal, unavailable and slow controls | Passed through production handlers/metrics; unavailable returned 503 with ready=0; restored predictor returned 200 |
| Owned Docker normal | Passed in 9.9 seconds; actual Prometheus alert list was empty |
| Owned Docker outage/recovery | Passed in 78.0 seconds; `ForecastApiDown` fired 63.1 seconds after fault onset, cleared 6.3 seconds after firing, followed by HTTP 200 |
| Owned Docker slow request | Passed in 10.2 seconds; three HTTP 200 responses took 0.257-0.265 seconds; recovery took 0.004 seconds. Completed forecast duration sum was 0.836 seconds across five requests including control/recovery |
| Fresh synthetic Alibi | All five states: unchanged control `no_drift`, shifted window `drift`, restored window `no_drift`; no missed shifted windows or control/recovery alarms in this declared run |
| Delayed observations and exporter | Six matches/186 pending at position 516, then 192/0 at 563; default minimum-30 horizon groups remained insufficient |
| Pinned promtool | Service alert, service dashboard and quality dashboard suites passed using the retained cached Prometheus image |

Two failed runtime procedures remain in the record. `docker-down-01` observed
firing/clearing but its HTTP recovery request failed with connection refused;
Docker had reassigned the dynamic loopback port. The runner refreshes the actual
binding and opens a fresh client after restart; the passing rerun records both
ports. `docker-slow-01` returned three fault-phase 503 responses, then recovered
to 200. Atomic fault-file publication and an explicit container read-back were
added before the passing rerun. The original API log did not expose the predictor
exception, so the precise cause of that initial slow-request failure remains
unproven. Both failed runs stopped their owned containers and were preserved.

These observed local results use a synthetic predictor and a cached dependency
image with a source overlay. They supply no selected-model, native-amd64 release,
registry, notification-receiver or Virtech evidence. The small request samples
are procedure checks and cannot establish a capacity or latency SLO.

## Dependencies and remaining execution

These pending PR sources were integrated with their commit attribution. Their
human reviews and merge status are separate from implementation readiness.

| Task | Exact dependency | PR |
| --- | --- | --- |
| Deployment helpers #75 | `4fadb245b3e5542fec6027dcdffa1bc9620489c0` | [#94](https://github.com/mlops-2627q1-mds-upc/MLOps_Endesa/pull/94) |
| Publication #78 | `d6ded8cb954afc7df02af78ee997a883dea246c3` | [#98](https://github.com/mlops-2627q1-mds-upc/MLOps_Endesa/pull/98) |
| Rollout #79 | `b08fe64f1eddc07e761f614effb280ed62e97a2e` | [#99](https://github.com/mlops-2627q1-mds-upc/MLOps_Endesa/pull/99) |
| Service metrics #80 | `da39aa6acc4ebed5bdf6878c23f8f7ae4e2e88c9` | [#90](https://github.com/mlops-2627q1-mds-upc/MLOps_Endesa/pull/90) |
| Replay #81 | `067ff119530b0f8cf221778a6a2fc5ce8d5cdda9` | [#89](https://github.com/mlops-2627q1-mds-upc/MLOps_Endesa/pull/89) |
| Drift #82 | `c6938d0c394adb9064ee84c1fb276bdb431e1ffe` | [#91](https://github.com/mlops-2627q1-mds-upc/MLOps_Endesa/pull/91) |
| Quality #83 | `ae7a9151ee945292fff8d4a18343754f43ddeb12` | [#92](https://github.com/mlops-2627q1-mds-upc/MLOps_Endesa/pull/92) |
| Benchmark #76 | `920103eefbc2515fe682e200872d69be8b92c97c` | [#93](https://github.com/mlops-2627q1-mds-upc/MLOps_Endesa/pull/93) |

The user selected private GHCR at
`ghcr.io/mlops-2627q1-mds-upc/endesa-forecast`. Publication is unexecuted. A
verified publication-source update will need its own integration and refreshed
checks. New decision proposals remain only in ignored local `.txt` notes; the
already-open EDN PR #97 and formal EDN/index are unchanged.

After peer review and a separate concrete execution authorization, validate the
exact clean integrated source through [release validation](release-validation.md),
publish it through the reviewed private [publication procedure](image-publication.md),
and verify an independent native Linux amd64 selected-model startup. Obtain
target identity/capacity and a published baseline before any
[deployment/recovery](deployment-release.md) drill. If that work needs UPC/VPN
access, halt the dependent step and wait for the user; do not initiate a VPN,
SSH connection or tunnel.

On an authorized isolated target, capture the public #79 status immediately
before the failure, after the candidate failure and after restored forecasts:

```bash
python -m src.deployment.rollout --state-dir "$STATE" --lock-file "$STATE/transaction.lock" \
  --machine-id-sha256 "$MACHINE_SHA256" status > "$EVIDENCE/release-status-before.json"
```

Use the runbook's exact adopt/deploy/recover commands and expected-current fence.
Record `current`, `previous`, `pending`, generation, complete release/config/image
identities, attempt summaries and post-persistence commit receipts alongside
fresh monitoring timestamps. The status command exposes retained state and does
not measure target health. Re-observe HTTP 200/200/200/422, scrape health,
firing/clearing and an independent client after recovery. Keep one attempt,
one worker/two numerical threads, 90-second readiness, 30-second request bounds,
and a 30-minute maximum per scenario. The runner accepts only owned local
fixtures; it cannot stop a shared service or execute the target rollout.

Human peer review, selected-model/final-image/target failure and recovery,
notification delivery, and numerical latency/quality/observation-delay alert
policies remain acceptance gaps. Issue #84 remains open while those applicable
runtime and review prerequisites are unresolved.
