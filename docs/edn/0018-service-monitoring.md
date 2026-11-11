# EDN-018: Bounded metrics with explicit missing samples

Date: 2026-11-11 (recorded).

Milestone: M6

Activity / Topic: Service monitoring (#80)

Decision Participants: Dídac Cayuela

Status: Draft for wording review before push.

## Decision

I chose a separate metrics registry per API instance, fixed request labels and explicit
unavailable resource samples. Prometheus evaluates the availability rules, with
provisioned Grafana views.

## Rationale

A shared registry could mix application instances. Raw request paths would create
unbounded labels, and zero-filled CPU/RSS values would hide missing measurements. I kept
the collector scoped to the Linux process and left resource/latency thresholds for a
separate operating decision.

## AI Involvement

Information seeking; Solution generation; Other. Codex implemented the instrumentation,
rules, dashboard and tests, and checked native and fixture-container behaviour.

## Response to AI

Other. I used implementation and measurement support.

## Assessment of the AI Contribution

I retained the scoped collector and single rule evaluator because they keep measurement
meaning and alert tests consistent. The checks support instrumentation behaviour;
host/container consumption needs separate measurement.

## Other Evidence

[PR #90](https://github.com/mlops-2627q1-mds-upc/MLOps_Endesa/pull/90). Browser
rendering and deployment of the monitoring stack on Virtech remain pending.
