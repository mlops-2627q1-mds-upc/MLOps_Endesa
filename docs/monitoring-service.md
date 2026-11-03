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

The middleware does not buffer requests, hold a predictor lock or perform
inference. Unhandled failures before response start count as 5xx. Once response
headers are sent, their HTTP status is retained even if transport later fails;
these metrics do not prove client receipt. Readiness means a predictor is loaded,
so an individual inference failure can coexist with readiness 1.

## Private stack

Configuration lives in [monitoring/service](../monitoring/service/compose.yml).
Prometheus and Grafana images use exact version tags and registry digests.
The API build uses the existing serving lock and explicit Docker allowlist.
The selected checkpoint is mounted read-only and still passes the production
loader's provenance checks. The internal Compose network has no external egress;
all published ports bind only `127.0.0.1`. Do not publish these unauthenticated API
or Prometheus endpoints on a public interface. The Grafana password is supplied
locally, with anonymous access and signup disabled.

When a working Docker engine and verified checkpoint are available, from the
repository root:

```sh
export GIT_SHA="$(git rev-parse HEAD)"
export FORECAST_CHECKPOINT_DIR="/absolute/path/to/chronos-t5-small-fine-tuned"
# Set GRAFANA_ADMIN_PASSWORD privately in this shell; do not commit its value.
docker compose -f monitoring/service/compose.yml config --quiet
docker compose -f monitoring/service/compose.yml up --build -d
curl --fail http://127.0.0.1:8000/ready
curl --fail http://127.0.0.1:8000/metrics
```

Open Grafana at `http://127.0.0.1:3000`, log in as `endesa`, and select
**Endesa / Endesa service monitoring** (`/d/endesa-service`). The provisioned
dashboard includes scrape reachability, readiness, in-flight work, process
collector availability, request/error rates, p50/p95 HTTP latency, process CPU,
RSS and scrape-result age. Its queries suppress old service values when scrapes
fail and do not replace absent data with zero. Zero completed forecasts has no
latency estimate. Prometheus data is retained for seven days in a named volume;
Grafana state has its own volume. Stop only this stack when finished:

```sh
docker compose -f monitoring/service/compose.yml down
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

Latency, error-rate, CPU, RSS, saturation and timeout limits remain pending
[#76](https://github.com/mlops-2627q1-mds-upc/MLOps_Endesa/issues/76). Measure the
selected checkpoint on the allocated target under comparable normal and
overload conditions before setting limits. No capacity alert is enabled from
invented measurements. Virtech access, allocation and deployment are unverified.

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
promtool test rules rules/tests.yml
```

Rule tests cover healthy zero traffic, API failure and recovery, unready models,
missing readiness/error series, absent targets, macOS-style missing process
samples, and a missing RSS series despite claimed collector availability. API
tests use explicit synthetic predictors, never model weights. They check real
HTTP exposition, request/error counts and duration, input/output failures,
unknown-route/method cardinality, independent app registries, lifecycle readiness
and responsive probes during blocked inference, with in-flight cleanup on every
tested exit. Linux `/proc` fixtures exercise the same process collector locally;
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
