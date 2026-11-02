# Serving benchmark protocol

Issue [#76](https://github.com/mlops-2627q1-mds-upc/MLOps_Endesa/issues/76)
needs serving-machine evidence for [NFR-7](requirements.md). This harness measures
one isolated service through the HTTP contract in
[PR #86](https://github.com/mlops-2627q1-mds-upc/MLOps_Endesa/pull/86).
It imports neither the predictor nor training/evaluation code and requires only
Python 3.11's standard library on a POSIX host. Container source is supplied by
[PR #88](https://github.com/mlops-2627q1-mds-upc/MLOps_Endesa/pull/88).
These dependencies retain their own merge, deployment and human review gates.

The implementation and fixture checks do not supply Virtech measurements. No
target latency requirement, resource recommendation or deployment success is
established by the local fixture. New decision proposals remain in the ignored
issue-specific temporary note pending Dídac's review; this document is a proposed
experiment protocol, not a formal engineering decision.

## Predeclare the experiment

Retain a JSON plan before launch. The plan contains the complete request,
expected API identity/settings, actual hardware, exact lifecycle argv, resource
scope and measurement settings. The harness copies it to an exclusive output
directory, records its SHA-256, and never overwrites an earlier run.

| Item | Protocol |
| --- | --- |
| Input | State `TAS`; exactly 512 values, oldest first, `15 + 20*sin(2*pi*i/48)` for `i=0..511`, retained as JSON. Synthetic source scale, including negatives; no timestamp/unit inference. |
| Model | Selected `chronos-t5-small-fine-tuned`, weights SHA-256 `746f9c3a9b61d452754f461a64c1529b2c07d8327ab8c33f804daa98c94c8cd9`, training run `639eba4cc7e74de08ffab64beee715db`. Real runs must match tracked provenance. |
| Inference | 48 steps; 20 samples; seed 7; CPU float32; two model threads; temperature 1, top-k 50, top-p 1; linear quantiles and median point forecast. No timing-driven sampling changes. |
| Identity | Full service and benchmark commits, parameter-file SHA-256, immutable image ID for containers. Every 200 forecast must match identity/settings and contain finite, ordered 48-value p10/p50/p90 arrays. |
| Machine | Actual hostname (`platform.node()`), OS, CPU description/count and RAM bytes from #75's verified target receipt. Template screenshots are not hardware evidence. |
| Cold lifecycle | Start a stopped, dedicated service with an unused loopback port. Time launch to successful `/ready`, then the first forecast separately. Filesystem/model caches are uncontrolled; this is process/container cold start. |
| Warm-up | Predeclare a positive count; proposed target baseline: 5 requests, excluded from warm statistics. |
| Warm workload | Predeclare at least 20 requests at each of concurrency 1 and 2; proposed target baseline: 50 each. Synchronized batches drain before the next batch. At most two HTTP workers. |
| Overload probe | Predeclare a small bounded c2 phase, proposed baseline: 10 requests. Record queuing, statuses and timeouts; concurrency 2 may not overload the target. No higher load or shared-service saturation is authorized by this protocol. |
| Deadlines | Each discovery/confirmation lifecycle, including cleanup, has a maximum 900-second budget: at most 1,800 seconds for the pair. Proposed startup deadline 120 s, total HTTP request deadline 10 s, three consecutive timeout completions stop work. No automatic experiment retries. |
| Memory stop | Explicit byte threshold below measured host and container limits, leaving headroom. Stop on threshold, cgroup OOM event, or lost/unavailable target memory telemetry. HTTP 507 is a server resource error, not an OOM diagnosis. |
| Proposal margin | Predeclare a margin before measurement; proposed baseline `0.2`. This creates no numerical latency target until actual target discovery exists. |

The baseline values above are proposals for the isolated experiment. #75 supplies
the actual capacity and authorizes the exact declaration. A negative/stopped run
retains its evidence; do not silently reduce sample count, shorten warm-up, change
seeds, or retune a failed run. Do not train, load final-test targets, evaluate
forecast quality, download a replacement checkpoint or publish an image here.

## Target and isolation handoff from #75

Portal authentication or an allocated VM alone does not authorize a target run.
Wait for #75's verified guest access, selected-model health/readiness/forecast
check, and an explicit isolated benchmark receipt. #75 owns provisioning,
installation, image/model transfer and shared deployment changes. The benchmark
may start/stop only the dedicated lifecycle that its receipt authorizes.

The non-secret receipt is JSON with these fields:

| Field | Required contents |
| --- | --- |
| Verification flags | `target_identity_verified`, `readiness_verified`, `selected_forecast_verified`, `isolated_instance`, `authorized_for_benchmark`: all `true`, based on #75's actual checks. |
| `base_url` | `http://127.0.0.1:PORT` on the verified target; no credentials, public host, tunnel endpoint or redirects. |
| `source` | `service_commit`, `benchmark_commit` (full 40-character Git IDs), `params_sha256`, `image_id` (immutable `sha256:...` for container; `null` for native). |
| `expected` | API `model_id`, `model_revision`, `checkpoint` (`sha256`, `training_run_id`), `value_scale`, `context_length`, `horizon`, full executed `settings`. |
| `hardware` | `host_identity`, `os`, `cpu`, positive `cpu_count` and `memory_bytes`. |
| `startup` | Foreground `argv`, absolute `cwd`, `stop_argv` (empty for a directly owned native process; explicit bounded stop for a container). No shell expressions or credentials. |
| `resource` | `{"scope":"process"}` or `{"scope":"container","cgroup_path":"absolute isolated cgroup v2 directory"}`. |
| `measurement` | `budget_seconds`, `startup_seconds`, `request_timeout_seconds`, `warmup_requests`, `requests_per_level`, `concurrency:[1,2]`, `overload_requests`, `consecutive_timeout_limit`, `memory_limit_bytes`, `proposal_margin_fraction`. |
| `request_sha256` | `src.serving_benchmark.digest(request)` for the retained 512-value request. |

The receipt binds the plan's identity, hardware, lifecycle, request and workload;
it is an operator's evidence handoff, not independent attestation or human PR
approval. Real runs additionally check the actual clean committed benchmark
modules/HEAD and hostname. HTTP cannot prove the image/source commit: #75 must
establish them through the target/container inspection.

For a container, #75 can prepare a dedicated **stopped** container with known
immutable ID/cgroup path, mounts and loopback port. A foreground launch is
`["docker","start","--attach",NAME]`; cleanup is
`["docker","stop","--time","1",NAME]`. The cgroup must be absent before
launch. A live cgroup or occupied port refuses the run without invoking cleanup.
The harness does not create/remove containers or restart the serving deployment.
A native launcher must be a foreground single-worker process; its memory excludes
children. It must not fork/detach or share a listening socket.

The cloud manual's public NAT application mappings do not prove SSH access from
outside UPC. Use only #75's verified SSH/console route and actual VM mappings;
do not infer ports from the old PDF's examples or weaken VPN/firewall policy.

## Run and retain evidence

From the benchmark checkout on the verified target, first retain the request
and have #75 bind its hash in the receipt. All paths below are task-local output
paths; use a new directory for every attempted experiment.

```sh
mkdir -p reports/serving-benchmark
python3 -m src.benchmark_runtime request reports/serving-benchmark/request.json
python3 -m src.benchmark_runtime declare \
  reports/serving-benchmark/target-receipt.json \
  reports/serving-benchmark/request.json \
  reports/serving-benchmark/discovery-plan.json --purpose discovery
python3 -m src.benchmark_runtime run \
  reports/serving-benchmark/discovery-plan.json \
  reports/serving-benchmark/discovery \
  --receipt reports/serving-benchmark/target-receipt.json \
  --lock /tmp/m4-m6-compute.lock
```

On this Mac the default lock is the shared
`temporary/m4-m6-work-2026-10-23/m4-m6-compute.lock` in the main checkout.
For a target-local run use the target's shared advisory lock as above **and**
hold the mandated Mac lock across the entire invoking SSH subprocess. The
coordinator's wrapper uses `fcntl.flock(..., LOCK_EX | LOCK_NB)`, calls the
verified SSH argv with `subprocess.run`, and releases in `finally`. #75 supplies
the exact authorized connection argv; there is deliberately no guessed host or
credential in these commands. A busy lock fails before launch. Hold no lock
while waiting for access, review or another worker.

Each output contains `declaration.json`, `runtime-identity.json`, the target
receipt, `observations.jsonl`, `service.log`, and normally `report.json`. Raw
records distinguish dispatch, completed request, readiness probe, lifecycle PID,
and scoped memory. Each completed request retains its status/outcome, timing,
phase/concurrency/index and response-body checksum. The plan retains input values
and settings. Logs/output stay ignored under `reports/serving-benchmark/`; inspect
and sanitize evidence before any later publication.

Warm p50/p95 use linear sample quantiles of successful requests only, including
HTTP encoding/transfer/validation and queueing. Errors retain their own status
counts and are never hidden in latency statistics. Throughput divides successful
or attempted requests by the whole phase's wall time, not summed request times.
First request and warm-up are separate. This closed-loop batched load does not
estimate arrival-rate capacity or forecast accuracy.

Linux process memory uses the owned leader's `VmRSS`/`VmHWM`. Container memory
uses its cgroup v2 `memory.current`, `memory.peak` and OOM events, including cgroup
page cache. Sampling is 100 ms plus boundary samples; kernel high-water readings
capture peaks between polls. Host memory is explicitly unavailable. No Docker
client RSS or local client memory is substituted. Missing fixture memory (for
example `/proc` on macOS) remains `null`; real target work stops without telemetry.

Stop conditions prevent new batches and drain/account for dispatched work. The
external watchdog bounds blocked reads/startup as well; `watchdog.json` then
marks the forced stop and cleanup outcome, and raw dispatches expose incomplete
requests. A watchdog run is incomplete even if a partial `report.json` exists.
Nonzero exit means stopped, failed cleanup, or failed confirmation: inspect the
artifacts. The harness stops its owned service on success, failure and interruption.

## Derive and confirm an NFR-7 proposal

Only a complete **selected-target discovery** with memory evidence, verified
cleanup and an error-free warm level of at least 20 successes can create a
candidate. Select concurrency 1 or 2 using the measured latency, errors, peak
memory and overload observations; retain that rationale in the issue report.
The candidate latency ceiling is observed p95 times `(1 + declared margin)`.
The operating throughput floor is observed throughput divided by that factor.
These are proposed targets pending Dídac's review, never an accepted SLO.

```sh
python3 -m src.benchmark_runtime propose \
  reports/serving-benchmark/discovery/report.json \
  reports/serving-benchmark/nfr7-proposal.json --concurrency 1
python3 -m src.benchmark_runtime declare \
  reports/serving-benchmark/target-receipt.json \
  reports/serving-benchmark/request.json \
  reports/serving-benchmark/confirmation-plan.json --purpose confirmation
python3 -m src.benchmark_runtime run \
  reports/serving-benchmark/confirmation-plan.json \
  reports/serving-benchmark/confirmation \
  --receipt reports/serving-benchmark/target-receipt.json \
  --proposal reports/serving-benchmark/nfr7-proposal.json \
  --lock /tmp/m4-m6-compute.lock
```

Confirmation starts a separate isolated lifecycle/run ID and preserves the target,
source, model, request, settings, budget and method. Its `confirmation.json`
compares p95, throughput and errors against the retained discovery candidate.
A failed confirmation is evidence against the proposal; do not adjust the
threshold after seeing it. Further experiments need a new declared protocol and
budget. Formal EDN/requirements/evidence-register changes remain deferred to
reviewed actual evidence and Dídac's explicit decision-record review.

## Local harness verification

Fixture observations exercise HTTP/accounting/lifecycle behavior only. The
fixture deliberately reports all-zero checkpoint/run IDs; it loads no predictor.

```sh
uv sync --locked --dev
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  uv run pytest tests/test_serving_benchmark.py -q
python3 tests/fixtures/benchmark_http.py \
  --write-plan reports/serving-benchmark/fixture-plan.json
python3 -m src.benchmark_runtime run \
  reports/serving-benchmark/fixture-plan.json \
  reports/serving-benchmark/fixture-run
```

Use fresh paths on repetition. Deterministic fixtures cover quantile/throughput
arithmetic, all-request/error accounting, timeout resets, concurrent draining,
global/startup deadlines, HTTP failures/stalled bodies, memory/OOM/unavailable
scope, refusal of existing services, exclusive outputs, cleanup, lock exclusion,
watchdog ownership and separate confirmation. Applicable repository lint/tests
and human peer review remain separate from target performance evidence.
