# Final delivery candidate and remaining evidence

**DRAFT, checked 12 November 2026.** Implementation readiness is available for
review; final delivery acceptance is incomplete. The report, talk and local
package remain drafts. No release tag, publication, deployment or course
submission is performed by preparing them.

## Pinned sources and dependencies

Merged `main` is `26f260e33766689bcdce4a6db6062a9e1e74475d`. It contains the
reviewed first delivery, inference PR #52 and FastAPI PR #86. The integrated
implementation candidate is `f9b5896af998331494175733c557e8dc5dd12cc7`,
[PR #100](https://github.com/mlops-2627q1-mds-upc/MLOps_Endesa/pull/100).
The documentation branch builds on that immutable candidate. Its focused PR is
stacked on #100 until dependency review/merges permit retargeting to `main`.
The build manifest records the report/presentation source commit separately.

| Issue / implementation | Pinned source | PR status at reconciliation |
| --- | --- | --- |
| #50 container | `6bd12575d7b80a1166b2370c0a98790ef7ad5075` | [#88](https://github.com/mlops-2627q1-mds-upc/MLOps_Endesa/pull/88), open |
| #75 target helpers | `4fadb245b3e5542fec6027dcdffa1bc9620489c0` | [#94](https://github.com/mlops-2627q1-mds-upc/MLOps_Endesa/pull/94), open |
| #76 benchmark | `920103eefbc2515fe682e200872d69be8b92c97c` | [#93](https://github.com/mlops-2627q1-mds-upc/MLOps_Endesa/pull/93), open |
| #77 release validation | `147e4500a086f4b5ceb7d58c0d121923c6b86915` | [#87](https://github.com/mlops-2627q1-mds-upc/MLOps_Endesa/pull/87), open |
| #78 private publication | `85452c98f2e1cf52b38156e6c1ecf1446e6a4b9b` | [#98](https://github.com/mlops-2627q1-mds-upc/MLOps_Endesa/pull/98), open |
| #79 release/recovery | `b08fe64f1eddc07e761f614effb280ed62e97a2e` | [#99](https://github.com/mlops-2627q1-mds-upc/MLOps_Endesa/pull/99), open |
| #80 service monitoring | `da39aa6acc4ebed5bdf6878c23f8f7ae4e2e88c9` | [#90](https://github.com/mlops-2627q1-mds-upc/MLOps_Endesa/pull/90), open |
| #81 chronological replay | `067ff119530b0f8cf221778a6a2fc5ce8d5cdda9` | [#89](https://github.com/mlops-2627q1-mds-upc/MLOps_Endesa/pull/89), open |
| #82 input drift | `c6938d0c394adb9064ee84c1fb276bdb431e1ffe` | [#91](https://github.com/mlops-2627q1-mds-upc/MLOps_Endesa/pull/91), open |
| #83 forecast quality | `ae7a9151ee945292fff8d4a18343754f43ddeb12` | [#92](https://github.com/mlops-2627q1-mds-upc/MLOps_Endesa/pull/92), open |
| #84 scenarios/integration | `f9b5896af998331494175733c557e8dc5dd12cc7` | [#100](https://github.com/mlops-2627q1-mds-upc/MLOps_Endesa/pull/100), open |

Each listed source is an ancestor of the implementation candidate. Integration
preserves commit attribution; it does not supply peer approval. EDN PR
[#97](https://github.com/mlops-2627q1-mds-upc/MLOps_Endesa/pull/97) remains open
at `e85b168c5bc282463cd6a6f8edf34f3c073ab6f5` and is not imported. New/revised
decision entries and their human response/assessment fields stay ignored local
`.txt` files for the user's review. Formal EDN/index changes are outside this PR.

## What the evidence establishes

The seven applicable #100 checks passed at its exact head. The
[required CI run](https://github.com/mlops-2627q1-mds-upc/MLOps_Endesa/actions/runs/43240322449)
reports 1,037 passed, 109 skipped and 82% rounded coverage. PR workflows checked
the generated merge `6604c5257f3dd91f437af40f7cd8b12c0758eccf`;
the PR head and generated merge are separate identities. The
[fresh synthetic Alibi run](https://github.com/mlops-2627q1-mds-upc/MLOps_Endesa/actions/runs/43240322406),
[drift tests](https://github.com/mlops-2627q1-mds-upc/MLOps_Endesa/actions/runs/43240322410),
[container fixture](https://github.com/mlops-2627q1-mds-upc/MLOps_Endesa/actions/runs/43240322457),
[service checks](https://github.com/mlops-2627q1-mds-upc/MLOps_Endesa/actions/runs/43240322420)
and [workflow semantics](https://github.com/mlops-2627q1-mds-upc/MLOps_Endesa/actions/runs/43240322493)
also passed. The 109 default-suite skips are not model passes.

| Evidence class | Observed scope | Acceptance gap |
| --- | --- | --- |
| Source and ordinary CI | Interfaces, failure paths, workflow semantics and synthetic/fixture tests | Human peer review and reviewed dependency merges |
| Native Linux amd64 container CI | CPU imports, non-root fixture HTTP and absent-weight refusal | Selected weights are absent; no real-model release startup |
| Fresh local #84 runtime | Synthetic production-handler HTTP, isolated Prometheus outage/recovery, fresh Alibi KS and delayed-truth export | Cached amd64 fixture image under Mac emulation; no final-source release image or target |
| Historical selected-model checks | October model behaviour/local HTTP; November local DVC validation and VM serving benchmark at their recorded sources | Different source/lock/image; no retrospective exact-release qualification |
| Historical model monitoring | 30 validation origins/state, 61.25-74.93% interval coverage; 8/30 historical drift control alarms and 56-112 nominal-day first detections/misses | No full-validation calibration, adopted threshold or live detection-delay claim |
| Registry and final target | Private GHCR destination selected in source | Package visibility/access, actual publication, independent pull/startup and final rollout/rollback unobserved |
| Notification | Local Prometheus alert firing/clearing observed | No external receiver configured or delivered notification observed |

The [#84 record](../monitoring/scenarios/local-evidence-2026-11-12.json) retains
both failed procedures as failures. A stale Docker port caused one recovery HTTP
failure; binding refresh/fresh-client handling was implemented and rerun. The
first slow-request attempt returned 503 before a passing atomic-publication rerun;
its precise predictor exception is not known. The historical benchmark's exit
137 with `OOMKilled=false` does not prove graceful shutdown or OOM.

## Reviewer access and local replication

Review the source at the candidate commit and the documentation PR's exact head.
The [replication guide](replication.md) separates fast checks, data/model retrieval,
synthetic scenarios and optional expensive execution. The
[artifact build guide](delivery/README.md) produces the report and slides offline
using installed TeX and standard-library Python; generated outputs remain under
ignored `temporary/issue-85/`.

Reviewers need their own GitHub access for source/Project and unexpired Actions
artifacts, DagsHub permission for DVC/MLflow, and private GHCR reader access for
the future published image. The container mounts the selected model separately;
image access alone is insufficient. Confirm those rights from each reviewer's
account without sharing credentials. No instructor access is inferred from the
author's account. Historical DVC retrieval is not a fresh independent-reader test.

The local delivery receipt records the clean offline setup, source checks,
small synthetic Actions artifact retrieval, PDF structure/page-count checks and
render inspection actually completed for this draft. Those checks do not retrieve
the selected model from a clean DVC cache or run it. Retain failures and skipped
checks alongside successful observations.

## Remaining delivery checklist

1. Obtain human peer review of the dependency implementation PRs and #85 docs;
   resolve review threads, perform reviewed merges in dependency order and check
   exact-head required CI again. Retarget the docs PR when `main` contains #100's
   implementation. No agent approval substitutes for a peer.
2. Have the team complete the final contribution consensus or CATME assessment.
   The earlier equal-contribution statement applies to the initial delivery only.
   Members must supply actual course names/connections and any missing attribution;
   do not invent them. Review proposed decision notes locally; the user controls
   EDN publication and the approved final EDN annex.
3. Resolve #74 serving/authentication choices and human decisions on numerical
   latency, interval/error, sample and observation-delay policies. Historical
   measurements are inputs, not adopted SLOs. Preserve the frozen test comparison.
4. Select the clean reviewed release SHA. With a separate execution instruction,
   run #77 in Actions, including fresh authenticated DVC retrieval and bounded
   selected-model checks. Retain run/attempt, source/tree, artifact ID/digest,
   complete reports and successful handoff. Revalidate if the commit changes.
5. With the reviewed prerequisites and a concrete publication instruction, run
   #78 for `ghcr.io/mlops-2627q1-mds-upc/endesa-forecast`. Check metadata and
   reader rights; default `first_publication=false`, explicit bootstrap only when
   intended. A 404 remains UNOBSERVED. Retain digest, manifest and actual private
   metadata. Do not change package visibility or external settings automatically.
6. Verify an independent native Linux amd64 selected-model start/pull at that
   digest and read-only artifact/config identities. Record actual health,
   readiness, forecast, invalid-input responses, executed settings and cleanup.
7. Halt UPC/VPN-dependent work and wait for the user. After authorized access,
   prove target host/machine/account/socket/capacity and publish/verify a baseline.
   Follow [#75 preparation](virtech-deployment.md) and
   [#79 transactions](deployment-release.md), one attempt/worker and two threads,
   90-second readiness and 30-second requests. Preserve the expected-current fence,
   current/previous/pending/generation and commit receipts. Prove recovery with
   restored 200/200/200/422 and independent-client observations; status alone is
   retained state. Keep each scenario within 30 minutes.
8. Verify target scrapes/dashboard and failure/recovery clocks. If notifications
   are required, configure a receiver only under separate authorization and retain
   actual receipt. Use no costly model run merely for a document demonstration.
9. Refresh the report/deck with accepted results and approved EDN/contribution
   material; rebuild, inspect every page/slide and rehearse the 15-minute talk.
   Reconfirm instructor access and the Atenea deadline. The supplied schedule
   states 14 December 2026 at 23:55 and presentation on 15 December; changes were
   not checked in Atenea. Tag, upload, publish or submit only after explicit
   authorization. Final replication stays incomplete until the listed observations
   are actually recorded.
