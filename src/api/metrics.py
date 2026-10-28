"""Process-local service metrics with bounded labels and an isolated app registry."""

from collections.abc import Callable
from time import perf_counter

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, ProcessCollector
from prometheus_client.core import GaugeMetricFamily
from starlette.types import ASGIApp, Message, Receive, Scope, Send

# Never use a raw request URL, user-supplied method, or arbitrary route as a label.
ROUTES = frozenset(
    {"/health", "/ready", "/forecast", "/openapi.json", "/docs", "/docs/oauth2-redirect", "/redoc"}
)
METHODS = frozenset(
    {"GET", "HEAD", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "TRACE", "CONNECT"}
)
LATENCY_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60, 120)


class ProcessMetrics:  # pylint: disable=too-few-public-methods
    """Export the current process's /proc samples and their actual availability."""

    def __init__(self):
        self.collector = ProcessCollector(registry=None)

    def collect(self):
        """Leave unsupported CPU/RSS samples absent instead of synthesizing zeros."""
        metrics = list(self.collector.collect())
        yield from metrics
        observed = {sample.name for metric in metrics for sample in metric.samples}
        required = {"process_cpu_seconds_total", "process_resident_memory_bytes"}
        yield GaugeMetricFamily(
            "endesa_process_metrics_available",
            "1 when CPU and RSS samples for this API process are available; otherwise 0.",
            value=int(required <= observed),
        )


class ServiceMetrics:  # pylint: disable=too-few-public-methods
    """Own one app's registry; no mutation of the Python client's global registry."""

    def __init__(self, ready: Callable[[], bool]):
        self.registry = CollectorRegistry()
        self.requests = Counter(
            "endesa_http_requests_total",
            "Completed HTTP requests, excluding metrics scrapes; status is an HTTP class.",
            ("method", "route", "status_class"),
            registry=self.registry,
        )
        self.duration = Histogram(
            "endesa_http_request_duration_seconds",
            "HTTP time through response completion, including validation and inference.",
            ("method", "route"),
            buckets=LATENCY_BUCKETS,
            registry=self.registry,
        )
        self.errors = Counter(
            "endesa_http_errors_total",
            "Completed HTTP client (4xx) and server (5xx) errors, excluding metrics scrapes.",
            ("status_class",),
            registry=self.registry,
        )
        # The forecast workload has observable zero activity before its first request.
        for status_class in ("2xx", "4xx", "5xx"):
            self.requests.labels("POST", "/forecast", status_class)
        self.duration.labels("POST", "/forecast")
        for status_class in ("4xx", "5xx"):
            self.errors.labels(status_class)
        self.in_flight = Gauge(
            "endesa_forecasts_in_flight",
            "Forecast handlers currently executing model work or validating its output.",
            registry=self.registry,
        )
        Gauge(
            "endesa_model_ready",
            "1 when this app has a loaded predictor; 0 before startup and after shutdown.",
            registry=self.registry,
        ).set_function(lambda: int(ready()))
        self.registry.register(ProcessMetrics())


class MetricsMiddleware:  # pylint: disable=too-few-public-methods
    """Observe HTTP at the ASGI boundary without buffering requests or responses."""

    def __init__(self, app: ASGIApp, metrics: ServiceMetrics):
        self.app = app
        self.metrics = metrics

    async def __call__(self, scope: Scope, receive: Receive, send: Send):
        if scope["type"] != "http" or scope["path"] == "/metrics":
            await self.app(scope, receive, send)
            return
        started = perf_counter()
        status = 500  # An unhandled error before response start is a server failure.

        async def observe_send(message: Message):
            nonlocal status
            if message["type"] == "http.response.start":
                status = message["status"]
            await send(message)

        try:
            await self.app(scope, receive, observe_send)
        finally:
            route = getattr(scope.get("route"), "path", None)
            route = route if route in ROUTES else "__unmatched__"
            method = scope["method"] if scope["method"] in METHODS else "OTHER"
            status_class = f"{status // 100}xx" if 100 <= status < 600 else "other"
            self.metrics.requests.labels(method, route, status_class).inc()
            self.metrics.duration.labels(method, route).observe(perf_counter() - started)
            if status_class in ("4xx", "5xx"):
                self.metrics.errors.labels(status_class).inc()
