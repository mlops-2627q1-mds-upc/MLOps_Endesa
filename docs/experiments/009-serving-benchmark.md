# Selected-model serving benchmark, 2 November 2026

On the allocated FIB Virtech VM, two isolated selected-model lifecycles completed
94 forecast requests without HTTP errors. At concurrency 1, warm p95 was 4.791 s
in discovery and 5.028 s in the separate confirmation. Concurrency 2 increased
p95 to 9.189 s and 9.363 s while throughput stayed near 0.22 requests/s.

The [measurement manifest](../evidence/serving-benchmark-2026-11-02.json) retains
source/model/configuration identities, individual request timings, phase counts,
scoped memory, terminal supervision and hashes of the original journals.
The [protocol](../serving-benchmark.md) documents the client and reproduction.
Issue: [#76](https://github.com/mlops-2627q1-mds-upc/MLOps_Endesa/issues/76).
Human peer review and any requirement/decision-record changes remain pending.

| Observation | Discovery | Confirmation |
| --- | ---: | ---: |
| Launch to ready, process/container cold | 14.833 s | 15.544 s |
| First forecast | 4.657 s | 4.675 s |
| Warm c1 p50 / p95, 20 requests | 4.432 / 4.791 s | 4.469 / 5.028 s |
| Warm c1 throughput | 0.2227 requests/s | 0.2203 requests/s |
| Warm c2 p50 / p95, 20 requests | 6.646 / 9.189 s | 6.777 / 9.363 s |
| Warm c2 throughput | 0.2234 requests/s | 0.2238 requests/s |
| Concurrent probe c2 p95, 4 requests | 9.026 s | 9.145 s |
| Requests / HTTP errors, including first and warm-up | 47 / 0 | 47 / 0 |
| Container memory high-water | 1,359,327,232 bytes | 1,337,389,056 bytes |
| Lifecycle duration including cleanup | 227.021 s | 228.455 s |

The target had one logical Broadwell guest CPU, 4,127,555,584 RAM bytes,
Ubuntu 20.04.6 and kernel 5.4.0-66. The isolated container had one CPU quota and
a 2 GiB memory limit; its declared stop threshold was 1.5 GiB. Cgroup v1 usage,
high-water, limit, fail-count and OOM counters were read directly. Host memory
was not measured. The observed container maximum was about 1.266 GiB; recorded
OOM and limit-hit counters stayed zero.

Both runs used the retained 512-value TAS synthetic history, 48 forecast steps,
20 samples, seed 7, CPU float32, two model threads and unchanged sampling settings.
The selected weight hash was
`746f9c3a9b61d452754f461a64c1529b2c07d8327ab8c33f804daa98c94c8cd9`.
Benchmark code was pinned to `efdb0813a0a06599927dd6e49760f6f576500097`;
serving source was the unmerged candidate
`962471100c1f50838bd55be9ad4664e693082a0a`, with the immutable image ID in the
manifest. No source, input, seed or inference settings changed between runs.

The separately supervised confirmation passed its discovery-derived numerical
comparison using the margin declared before discovery. This records an
experimental comparison; it does not adopt an SLO or amend NFR-7. The complete
numerical artifact and raw outputs remain in the ignored task evidence and on
the target. Combined lifecycle execution was 455.476 seconds; the whole window,
including the gap between runs, was 722.408 seconds, below the 1,800-second budget.

Each stop used the exact declared one-second Docker stop command. Both container
states ended with exit 137, `OOMKilled=false`, zero recorded OOM/fail counters,
a refused loopback port and no running benchmark container. Graceful shutdown
was not demonstrated. This stop outcome is retained separately from the successful
HTTP requests and supervisor outcomes. The protected main API remained running,
Docker healthy, with `/health` and `/ready` returning 200 before and after each run.

The concurrent probe observed longer response times without rejection, timeouts
or higher throughput. It does not establish maximum capacity or admission control.
The results cover one retained input and a small sample on one VM. They exclude
external-network latency, cold filesystem caches, forecast accuracy and final-test
evaluation. Fixtures and CI checks remain separate evidence from these target runs.
No model, data, image, VM configuration, shared service or formal EDN was changed
by the benchmark.

## Release applicability checked on 12 November

PR #93 was refreshed with `main` at
`26f260e33766689bcdce4a6db6062a9e1e74475d`. The measured harness modules,
original manifest and retained raw journals were preserved. Integration tests
use the merged HTTP API with an injected predictor; they load no checkpoint.

An offline comparison with #79/PR #99 source
`b08fe64f1eddc07e761f614effb280ed62e97a2e` found matching API/inference and
runtime support source, model provenance, parameter bytes and dependency locks.
Its Dockerfile adds the repository source label. This comparison supplies no
published digest, final image inspection or executed-settings evidence for a
release. The [protocol's applicability checks](../serving-benchmark.md#applicability-to-release-and-monitoring-work)
describe the separate publication/deployment interfaces and evidence for #84/#85.

The retained discovery-derived c1 comparison used p95 at most 5.748611460 seconds
and throughput at least 0.185599796 requests/s; confirmation passed. These are the
original predeclared-margin comparison values, not adopted NFR-7 or alert limits.
The local operating proposal remains pending human assessment. Any final target
with different source/image, request, settings or resource inputs needs a separate
declared observation. No target benchmark, target connection, publication or model
execution ran during this refresh; graceful shutdown and human peer review
remain unverified for this experiment.
