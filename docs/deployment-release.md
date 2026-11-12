# Release transactions and recovery

The manual release trigger is `python -m src.deployment.rollout` in an authorized
session on the native Linux amd64 target. It consumes two complete published
release slots, checks the candidate, and restores the accepted baseline if a
candidate check fails. Every deployment attempt holds the same exclusive lock
through its recovery forecast. A failed candidate returns exit 1 even after
successful recovery.

This implementation integrates #75/PR #94 at
`01ed11b93da2ea0f5ac050b8502359339bdf664d`, including #78/PR #98 at
`fad4e155554902834bfc09802ac78c834d14354d`, #77/PR #87 at
`147e4500a086f4b5ceb7d58c0d121923c6b86915` and #50/PR #88 at
`6bd12575d7b80a1166b2370c0a98790ef7ad5075`. Required human reviews, integrated
real-model validation, publication, independent startup and target execution
remain pending. Fixture tests establish transaction behavior using fabricated
release evidence and local helper packages. They execute no selected model.

## Prerequisites and access

Before any runtime execution, obtain the required peer reviews and resolve the
applicable #74 serving choices. Select an exact clean integrated source and
complete [release validation](release-validation.md) and
[publication](image-publication.md) at that source. Obtain each manifest and its
complete prepared validation directory from the trusted successful publication
run/attempt, recording the actual run URL, artifact ID/digest and conclusion.
Offline hashes establish agreement with that bundle; the operator still verifies
the producer through authenticated GitHub access.

Prepare the target through [the #75 runbook](virtech-deployment.md#release-contract-for-79).
Verify the portal VM allocation, console/SSH binding, SSH host key, actual guest
hostname, `/etc/machine-id`, local Docker daemon and measured capacity. Preserve
the exact owner account for the service state directory. The transaction checks
the machine-ID SHA-256, native Linux amd64, hostname, account ownership and Docker
socket; #75 checks container ownership, private port/mount/resource settings and
actual image identity.

Virtech's SSH route requires the supported UPC/VPN access path. Establish the
authorized private SSH session using its actual assigned endpoint and strict host
key checking, then run the commands below on the target. A public GitHub-hosted
runner has no established route to this endpoint. There is no automated Actions
deployment trigger in this change. A future runner must prove private reachability,
host binding, the same persistent lock/state directory and reviewed secret access
before automation can use this contract. Do not expose SSH or the API publicly to
make a workflow reachable.

Keep a dedicated controller checkout at the reviewed #79 source with its full
`uv sync --locked --dev` environment. Also retain each slot's own clean checkout
at its manifest's exact source/tree and its own locked `.venv`. The controller
calls the unchanged #75 helpers in that slot's Python process, which reuses
#78's `verify_manifest` through `src.deployment.release`. This permits recovery
to a different source/model/config without validating it against the candidate's
provenance. Never move a retained checkout or update it in place.

Pull each authorized immutable registry digest and provision its complete model
and params into separate read-only slots using #75's selection/provisioning
commands. These are preparation steps outside the transaction command. The
runner neither pulls images nor retrieves models. Both slots must use schema 2,
distinct owned container names, separate checkpoint/params paths, and the same
host/socket/loopback port. Read-only mount parents must be traversable by UID
10001; the private state directory is not a model mount. Preserve images, model
caches, weights, previous containers and all trusted bundles.

An historical schema-1 candidate cannot be adopted as a published baseline.
Prepare and verify a schema-2 baseline first, including the independent client
checks in #75. Adoption runs another target-local real forecast; it does not
replace the separate-client access check.

## Manual workflow

The variables below are explicit operator selections. `CONTROL` is the controller
checkout, `BASELINE_CHECKOUT` and `CANDIDATE_CHECKOUT` are the retained exact-source
checkouts, and each `*_CONFIG` is its selected schema-2 JSON. `STATE` is one
persistent absolute service directory without symlinks. Every operator and every
release uses this same directory and `STATE/transaction.lock`, including direct
#75 operations during maintenance. Do not wrap the transaction in another holder
of that same lock. Use a corresponding coordinator compute lock during separately
authorized preparation/transfer work, as described in #75.

Create `STATE` once, as the authorized service account, in an approved parent:

```sh
test ! -e "$STATE"
mkdir -m 700 "$STATE"
TARGET_MACHINE_ID_SHA256=$(sha256sum /etc/machine-id | cut -d ' ' -f 1)
PY="$CONTROL/.venv/bin/python"
cd "$CONTROL"
```

Compare the computed pin with the one independently obtained through the verified
console/SSH host binding before using it. Reuse the retained directory on later
attempts. Do not recreate it to bypass a busy lock or pending transaction.

Verify both trusted bundles offline in their respective source checkouts before
preparation. Run this once per slot, with that slot's paths and immutable digest:

```sh
cd "$SLOT_CHECKOUT"
.venv/bin/python -m src.release_manifest verify \
  --manifest "$MANIFEST" --validation-dir "$VALIDATION" \
  --image-reference "$IMAGE_REFERENCE" --architecture amd64
cd "$CONTROL"
```

Create explicit descriptors and a synthetic request in an approved evidence
directory. The descriptor contains exactly `checkout` and `config`; both paths
must be absolute. Set/export the path variables before this command:

```sh
export BASELINE_CHECKOUT BASELINE_CONFIG BASELINE_SLOT
export CANDIDATE_CHECKOUT CANDIDATE_CONFIG CANDIDATE_SLOT REQUEST
"$PY" - <<'PY'
import json, os
for name in ("BASELINE", "CANDIDATE"):
    with open(os.environ[name + "_SLOT"], "x", encoding="utf-8") as handle:
        json.dump({"checkout": os.environ[name + "_CHECKOUT"],
                   "config": os.environ[name + "_CONFIG"]}, handle)
with open(os.environ["REQUEST"], "x", encoding="utf-8") as handle:
    json.dump({"state": "TAS", "history": [-1.5] * 512}, handle)
PY
```

Register the already-running baseline once:

```sh
"$PY" -m src.deployment.rollout --state-dir "$STATE" \
  --lock-file "$STATE/transaction.lock" \
  --machine-id-sha256 "$TARGET_MACHINE_ID_SHA256" \
  adopt --slot "$BASELINE_SLOT" --request "$REQUEST"
```

Expected: exit 0, `adopted`, a complete current identity, and inspection plus
200/200/200/422 HTTP evidence. The forecast verifies 48 finite ordered quantiles,
selected weights/run, source scale and executed prediction settings through #75.
No container is started by adoption. Inspect status and record the accepted
`state.current.slot.config_sha256` as `EXPECTED_CURRENT`:

```sh
"$PY" -m src.deployment.rollout --state-dir "$STATE" \
  --lock-file "$STATE/transaction.lock" \
  --machine-id-sha256 "$TARGET_MACHINE_ID_SHA256" status
```

After the concrete release attempt has execution authorization:

```sh
"$PY" -m src.deployment.rollout --state-dir "$STATE" \
  --lock-file "$STATE/transaction.lock" \
  --machine-id-sha256 "$TARGET_MACHINE_ID_SHA256" \
  deploy --slot "$CANDIDATE_SLOT" --expected-current "$EXPECTED_CURRENT" \
  --request "$REQUEST"
```

The transaction checks the candidate bundle/image/artifacts and the baseline's
actual forecast before stopping the baseline. It fsyncs pending baseline/candidate
identities before the stop, requires a clean baseline exit, then starts and checks
the candidate. A successful forecast advances `current`, retains the complete
baseline as `previous`, clears `pending` and returns `deployed`/exit 0. Attempts
cannot replay a candidate slot already used for preflight; prepare a fresh slot.

Readiness waits at most 90 seconds and each HTTP request at most 30 seconds. Use
one attempt per authorized window with two CPU threads at most; there are no
automatic inference retry loops. A normal swap makes one baseline and one
candidate forecast request. Recovery adds one restored-baseline forecast. Runtime
resource limits come from the measured #75 config. Forecast checks establish
serving behavior, not accuracy, throughput or the adoption of performance targets.

## Controlled failure and interrupted recovery

After reviews, publication, independent startup and target/access prerequisites,
authorize one fresh drill slot with the same complete release contract. Add
`--drill-stop-before-forecast` to the deployment command:

```sh
"$PY" -m src.deployment.rollout --state-dir "$STATE" \
  --lock-file "$STATE/transaction.lock" \
  --machine-id-sha256 "$TARGET_MACHINE_ID_SHA256" \
  deploy --slot "$DRILL_SLOT" --expected-current "$EXPECTED_CURRENT" \
  --request "$REQUEST" --drill-stop-before-forecast
```

This records the requested drill, waits for real candidate readiness, stops only
the owned candidate and executes the actual HTTP verifier against its direct
target loopback port. Expect connection-refusal/failed-forecast evidence, followed
by restoration of the exact baseline image/model/config and an actual passing
200/200/200/422 forecast report. The command returns `failed_recovered`/exit 1;
status must retain the baseline identity and show `pending: null`. Retain the
candidate's clean-stop result separately. Exit 137 or OOM never proves clean stop.
Malformed manifest rejection happens before baseline stop and is a separate check.

Recovery quiesces the exact owned candidate, clears only a matching interrupted
#75 parser container, and restarts the baseline through its retained source's
helpers. It preserves service containers, images and checkpoint/params slots.
An unclean candidate exit remains recorded even when service recovery succeeds.
If candidate quiescence, baseline readiness, manifest/config verification or the
restored forecast fails, `pending` remains and the command fails. Inspect the
immutable records, restore missing exact dependencies/access if necessary and
authorize a separate recovery attempt:

```sh
"$PY" -m src.deployment.rollout --state-dir "$STATE" \
  --lock-file "$STATE/transaction.lock" \
  --machine-id-sha256 "$TARGET_MACHINE_ID_SHA256" \
  recover --expected-current "$EXPECTED_CURRENT" --request "$REQUEST"
```

Successful explicit recovery returns exit 0 with `failed_recovered`; the failed
deployment and its evidence retain their failure status. Failed recovery exits 1.

SIGINT/TERM requests wait for the active helper to settle before automatic
recovery. SIGKILL, host loss and storage failure can prevent that recovery;
the durable pending record blocks another deploy until explicit recovery.
A surviving helper inherits the same locked file description, so another
operator cannot enter while it finishes. Never delete or replace the lock file.
Restore a healthy writable evidence filesystem before retrying after an I/O
failure; the runner refuses operations whose intent cannot be recorded.

For a later interruption exercise, capture the exact owned controller PID and
pending attempt ID through the supervising session, issue a single authorized
TERM while that attempt is pending, and observe recovery. A SIGKILL exercise
requires a separately authorized outage window: confirm pending identity first,
terminate only that controller PID, confirm a competing deploy is refused, then
run the recovery command above after the inherited helper lock is released.
Do not terminate Docker, unrelated operators or the service host.

## Evidence interface for #84

`status` is read-only and works while a deployment holds the lock. Its schema-1
`release-transaction-status` output contains `state`, `releases` keyed by config
SHA-256, and `reports`. This is recorded transaction state; it makes no fresh
claim about target health. The public fields for scenarios are:

* `state.generation`, `identity`, `current`, `previous` and `pending`.
* Each accepted release's `slot.config_sha256`, `inspection` and `forecast`.
* `releases[hash]`, the complete image/model/config/manifest/source selection.
* `reports[*].status`, `attempt_id`, and recovery's `failed_attempt_id`.

Evidence references contain a path relative to `STATE` and its SHA-256. Per-attempt
`*.intent.json` precedes each helper call; matching operation JSON retains the
actual inspection/HTTP response or a sanitized error type and available owned
container exit/OOM state. Requests, deliberate-failure records and recovery
records remain in `attempts/<id>/`. Snapshot configs remain in `slots/`.

`summary.json` records `baseline_verified`, `candidate_verified`, `rejected` or
`recovery_failed`. Verified summaries precede the atomic state pointer. A separate
`commit.json` records `adopted`, `deployed` or `failed_recovered` only after state
persistence, so a crash between verification and commit cannot manufacture
deployment success. `state.last_result` references that decision summary; inspect
the pointer and commit receipt together. Hashes in historical commit receipts
refer to state bytes at that generation, before later state transitions.

Link final source/fixture CI independently from the trusted real validation and
publication runs, registry digest, native independent-startup evidence, actual
target/drill/recovery evidence and peer review. Keep #79 open until the runtime
acceptance and human review exist. Assess consequential proposals in the ignored
issue-specific notes; preserve the separate EDN PR #97 without inventing approval.
