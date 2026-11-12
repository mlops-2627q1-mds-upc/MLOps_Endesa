# Service monitoring

The API exposes Prometheus text metrics at `/metrics`. Each `create_app()` owns
an isolated registry. Keep the existing single Uvicorn worker: counters and
in-flight work describe that app, while `process_*` describes its Python process,
including native threads. Multiple workers need a separate aggregation design.

## Metric contract

| Metric | Meaning |
| --- | --- |
| `endesa_http_requests_total{method,route,status_class}` | Completed HTTP requests, including validation, unknown routes and controlled failures. |
| `endesa_http_request_duration_seconds{method,route}` | Histogram through response completion, including HTTP validation and inference; seconds. |
| `endesa_http_errors_total{status_class}` | All HTTP 4xx and 5xx responses, counted separately. |
| `endesa_model_ready` | Current predictor presence, matching `/ready`; 0 before startup and after shutdown. |
| `endesa_forecasts_in_flight` | Forecast handlers executing prediction or output validation; decremented on success and failure. This is not queue depth or an admission limit. |
| `process_cpu_seconds_total` | Cumulative user and system CPU seconds consumed by the API process. `rate(...[5m])` is consumed CPU cores, not percent of allocated capacity. |
| `process_resident_memory_bytes` | API process RSS, not host RAM, container working set, or a cgroup limit. |
| `endesa_process_metrics_available` | 1 only when the collector actually supplies both CPU and RSS samples. |

The Python client's [process collector](https://prometheus.github.io/client_python/collector/)
reads Linux `/proc`; native macOS scrapes report availability 0 and omit CPU/RSS.
Missing samples must stay missing. Other process families, such as open file
descriptors and start time, appear when the collector supports them. Host and
whole-container metrics require separate exporters and are outside this change.

Labels use fixed known routes, HTTP methods and status classes. Unknown routes
share `__unmatched__`; unknown methods share `OTHER`. No state values, URLs,
query strings, forecast IDs, histories, model hashes, exception messages or
event records enter labels. `/metrics` scrapes are excluded from HTTP counts
and duration. Forecast counters and histogram, and both error classes, start
at zero, so an idle app has observable zero activity. Other HTTP combinations
appear after their first request. Latency buckets span 5 ms to 120 s plus
`+Inf`; these are measurement ranges, not measured serving limits or SLOs.

The middleware measures until the final response body is sent, excluding any
background work after that response. It does not buffer requests, hold a predictor
lock or perform inference. Unhandled failures before response start count as 5xx. Once response
headers are sent, their HTTP status is retained even if transport later fails;
these metrics do not prove client receipt. Readiness means a predictor is loaded,
so an individual inference failure can coexist with readiness 1.

## Private stack

Configuration lives in [monitoring/service](../monitoring/service/compose.yml).
Prometheus and Grafana images use exact version tags and registry digests.
The API build uses the serving lock and explicit Docker allowlist, including
`src/api/metrics.py`. Its `linux/amd64` platform, one worker, two CPU threads,
two-CPU container ceiling, 2 GiB memory ceiling, read-only root filesystem and
64 MiB temporary filesystem follow the [container contract](container.md).
These ceilings are execution budgets, not measured capacity or alert limits.
The selected checkpoint is mounted read-only and still passes the production
loader's provenance checks. Services share a dedicated Compose bridge;
all published ports bind only `127.0.0.1`. Do not publish these unauthenticated API
or Prometheus endpoints on a public interface. The Grafana password is supplied
locally, with anonymous access and signup disabled. A required Compose project
name isolates each run's containers, network and volumes. Optional `FORECAST_PORT`,
`PROMETHEUS_PORT` and `GRAFANA_PORT` change only the loopback ports; the internal
API target remains `api:8000` and the Grafana data source remains
`http://prometheus:9090`.

The bridge permits outbound traffic. An `internal: true` bridge omitted actual
host port mappings on the tested Docker Engine 29.1.3, despite healthy containers
and valid Compose syntax. This stack uses bridge port publishing with explicit
loopback addresses; it does not claim egress isolation. See Docker's
[bridge networking](https://docs.docker.com/engine/network/drivers/bridge/) and
[loopback publishing](https://docs.docker.com/engine/network/port-publishing/)
contracts. The monitoring CI job checks a real published HTTP endpoint to catch
this integration failure.

When a working Docker engine and verified checkpoint are available, from the
repository root:

```sh
export GIT_SHA="$(git rev-parse HEAD)"
export COMPOSE_PROJECT_NAME="endesa-monitoring-issue80"
export FORECAST_CHECKPOINT_DIR="/absolute/path/to/chronos-t5-small-fine-tuned"
# Set GRAFANA_ADMIN_PASSWORD privately in this shell; do not commit its value.
docker compose -f monitoring/service/compose.yml config --quiet
docker compose -f monitoring/service/compose.yml up --build -d
curl --fail http://127.0.0.1:8000/ready
curl --fail http://127.0.0.1:8000/metrics
```

Choose an unused project name and ports before starting. Keep the same values
for subsequent inspection and stop commands; do not reuse another worktree's
project. The default API tag includes `GIT_SHA` rather than sharing a mutable
`monitoring-local` tag across checkouts. Check that the image revision label
matches the clean source used to build it.

For a separately authorized published release, set `FORECAST_IMAGE` to its
verified `registry/repository@sha256:digest`, use the clean source recorded by
the manifest, and start with `up --no-build --pull never -d` after pulling that
digest. Verify the full source/model/configuration bundle using #78 and the
deployment contract in #75 before doing so; Compose itself does not verify a
release manifest or generate deployment receipts. The build example above is
local packaging, not registry publication or a replacement for #79's deployment.

Open Grafana at `http://127.0.0.1:3000` (or `GRAFANA_PORT`), log in as `endesa`, and select
**Endesa / Endesa service monitoring** (`/d/endesa-service`). The provisioned
dashboard includes scrape reachability, readiness, in-flight work, process
collector availability, request/error rates, p50/p95 HTTP latency, process CPU,
RSS and scrape-result age. Its queries suppress old service values when scrapes
fail and do not replace absent data with zero. Aggregated rates and latency
filter each target by its own successful scrape before aggregation, so a healthy
target cannot retain a failed target's earlier samples. This uses PromQL
[label matching](https://prometheus.io/docs/prometheus/latest/querying/operators/).
Zero completed forecasts has no
latency estimate. Prometheus data is retained for seven days in a named volume;
Grafana state has its own volume. Stop only this project when finished, retaining
its containers and volumes for later inspection:

```sh
docker compose -f monitoring/service/compose.yml stop
```

## Alerts and checks

[Prometheus rules](../monitoring/service/rules/alerts.yml) cover failed scrapes,
an absent scrape job, reachable but unready models, missing core or expected
process telemetry, and unavailable process collection. Each annotation gives
an investigation step. They are evaluated by Prometheus, visible at
`http://127.0.0.1:9090/alerts`; no Alertmanager or external notification receiver
is configured. The one-minute hold is a provisional observation window to avoid
alerting on a single scrape, not a measured availability target. This stack
cannot alert externally on its own Prometheus failure: Grafana then shows a
data-source error, and an independent observer would be needed.

Latency, error-rate, CPU, RSS, saturation and timeout limits have no recorded
owner adoption in this monitoring change. The measurements in
[#76 / PR #93](https://github.com/mlops-2627q1-mds-upc/MLOps_Endesa/pull/93)
remain experimental evidence. A recorded owner decision and comparable final
source/model/target measurements are required before using them as thresholds.
No capacity alert is enabled here. The separate #75 runbook records historical
Virtech candidate evidence; this configuration has not been deployed there.

Without Docker, use the locked Python environment and a matching native
Prometheus 3.15.1 `promtool` binary from its
[official release](https://github.com/prometheus/prometheus/releases/tag/v3.15.1),
verified against the release's `sha256sums.txt`:

```sh
uv run --locked pytest tests/test_api_metrics.py tests/test_api.py tests/test_container.py
make lint
cd monitoring/service
promtool check config prometheus.yml
promtool check rules rules/alerts.yml
promtool test rules rules/tests.yml rules/dashboard-tests.yml
```

The separate `Monitoring / configuration` Actions job validates Compose, starts
only Prometheus to check its actual loopback HTTP publication, and runs the same
Prometheus checks in the pinned image. It is an additional
check; the required `CI / lint` and `CI / test` names are unchanged. To run the
Prometheus checks locally with Docker, from `monitoring/service`:

```sh
export PROMETHEUS_IMAGE="prom/prometheus:v3.15.1@sha256:fa3302279710be2f21a8301ec17a8b4f2301fd829bd113839b3b6b02b6f191a2"
docker run --rm --network none --workdir /etc/prometheus \
  --mount "type=bind,src=$PWD,dst=/etc/prometheus,readonly" \
  --entrypoint /bin/promtool "$PROMETHEUS_IMAGE" check config prometheus.yml
docker run --rm --network none --workdir /etc/prometheus \
  --mount "type=bind,src=$PWD,dst=/etc/prometheus,readonly" \
  --entrypoint /bin/promtool "$PROMETHEUS_IMAGE" check rules rules/alerts.yml
docker run --rm --network none --workdir /etc/prometheus \
  --mount "type=bind,src=$PWD,dst=/etc/prometheus,readonly" \
  --entrypoint /bin/promtool "$PROMETHEUS_IMAGE" test rules rules/tests.yml rules/dashboard-tests.yml
```

The nine alert scenarios cover healthy zero traffic, API failure and recovery, unready models,
missing readiness/error series, absent targets, macOS-style missing process
samples, and a missing RSS series despite claimed collector availability. API
tests use explicit synthetic predictors, never model weights. They check real
HTTP exposition, request/error counts and duration, input/output failures,
unknown-route/method cardinality, independent app registries, lifecycle readiness
and responsive probes during blocked inference, with in-flight cleanup on every
tested exit. The five dashboard scenarios use the configured aggregate expressions
to check idle zero rates without latency, active rates/quantiles, missing counters,
failed scrapes and a healthy target alongside a failed target. Floating point
comparison uses Prometheus's documented
[`fuzzy_compare`](https://prometheus.io/docs/prometheus/latest/configuration/unit_testing_rules/)
for the last mantissa bit. Linux `/proc` fixtures exercise the same process collector locally;
they do not replace a live Linux container check.

For a collection smoke check without weights, run the existing fixture API on
loopback and point a temporary copy of `prometheus.yml` at it:

```sh
uv run --locked uvicorn tests.container.fixture_app:create_app --factory \
  --host 127.0.0.1 --port 8000 --workers 1
```

Make `/health`, `/ready`, a valid synthetic `/forecast`, a 422 request and an
unknown-route request; check the corresponding series through Prometheus. Stop
the fixture process and confirm `up{job="forecast-api"}` becomes 0 and
`ForecastApiDown` fires after the hold. Restart it and check alert recovery.
This establishes synthetic collection only, not selected-model latency.

When Docker is available, additionally inspect the actual Grafana panels after
normal requests and a controlled `docker compose ... stop api`, then `start api`:
reachability must change to 0, service panels must lose current data, the outage
alert must fire, and all must recover after successful scrapes. Native Prometheus
checks and static Grafana provisioning checks do not establish Grafana rendering,
container collection, deployment or target capacity. Record those results
separately when run.

## Contract for #83 and #84

Consume `/metrics` through the `forecast-api` job on the private network. Keep
`job` and `instance` for matching; HTTP labels are only `method`, `route` and
`status_class`, plus the histogram's `le`. Model, configuration and event
identities belong in retained evidence, not metric labels. This service stack
does not supply delayed-observation quality, replay/drift metrics or an external
notification receiver.

Use the fixed alert names `ForecastApiDown`, `ForecastScrapeTargetMissing`,
`ForecastModelNotReady`, `ForecastTelemetryMissing` and
`ForecastProcessTelemetryUnavailable`. Their one-minute holds remain provisional.
With the 15-second scrape and evaluation intervals, outage/recovery checks must
wait for observed scrapes and evaluations rather than assuming an immediate
state change. Record Prometheus time, target error, `up`, alert state and the
source/image/configuration identity for each normal, stopped and recovered state.
Restarting the API resets its process-local counters; use counter-aware rates.

For a fixture-only collection check, override the API command with
`uvicorn fixture_app:create_app --factory --host 0.0.0.0 --port 8000 --workers 1`,
mount only `tests/container` at `/smoke:ro`, and set `PYTHONPATH=/app:/smoke`.
Remove the production model mount in that temporary override; do not add a
production fixture environment switch. The result must retain `fixture_only`
and the all-zero synthetic model identity. Actual selected-model execution,
registry publication, target deployment and rendering remain separate checks.
For visual evidence, inspect the Grafana dashboard in a browser during normal
traffic, outage and recovery, retain screenshots with state/source provenance,
and check both empty service panels and the visible reachability value. Query
responses alone do not establish rendering.
