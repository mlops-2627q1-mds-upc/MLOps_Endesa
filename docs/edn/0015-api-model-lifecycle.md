# EDN-015: One model per API lifecycle

Date: 2026-11-11 (recorded).

Milestone: M4

Activity / Topic: API lifecycle (#49)

Decision Participants: Dídac Cayuela

Status: Recorded; peer review pending.

## Decision

I chose an application factory that loads the selected model once at startup. Forecast
requests use the loaded instance, while health and readiness probes remain responsive
during inference.

## Rationale

Loading per request would repeat expensive work; an import-time singleton would make
lifecycle tests harder. I kept a strict 512-input/48-output contract and rejected
invalid model startup rather than leaving a permanently unavailable service.

## AI Involvement

Solution generation; Other. Codex implemented the factory, schemas and error handling,
wrote tests and ran the real-model HTTP smoke check.

## Response to AI

Other. I used implementation and testing support.

## Assessment of the AI Contribution

I retained the factory and injected test predictor because they let routine tests
exercise the production lifecycle without loading model weights. The blocked-predictor
test also checks probe responsiveness.

## Other Evidence

[PR #86](https://github.com/mlops-2627q1-mds-upc/MLOps_Endesa/pull/86). The HTTP smoke
checks interface behaviour, not target capacity.
