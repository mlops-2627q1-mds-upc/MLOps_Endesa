# Virtech deployment and release consumer contract

This runbook operates an explicitly allocated course VM. It does not allocate VMs,
discover hosts, install SSH access, publish images or automate releases. Use the
course portal and its assigned template/quota first. Read the actual console NAT
banner; do not copy example IPs, ports, OS versions or capacities from a manual.
Keep existing VMs, services, credentials and artifacts intact.

The source integrates `main` at `26f260e33766689bcdce4a6db6062a9e1e74475d`
(including the API/inference merges) and publication PR #98 at
`fad4e155554902834bfc09802ac78c834d14354d`. The latter includes container PR #88
at `6bd12575d7b80a1166b2370c0a98790ef7ad5075` and release-validation PR #87 at
`147e4500a086f4b5ceb7d58c0d121923c6b86915`. These source dependencies still need
their required human reviews and merges. Source/fixture checks on this branch
do not establish a release validation, publication or deployment.

On 2 November 2026, VM 2245 ran candidate source
`962471100c1f50838bd55be9ad4664e693082a0a`, image
`sha256:780d8e3c4f059a6f89c5277bc272449190210a068d2177c2ba3aecbd86425cd9`.
Recorded host resources were Ubuntu 20.04.6, one CPU, 3936 MiB RAM and
20,629,221,376 filesystem bytes. Separate-client initial/restarted/recovered
checks returned 200/200/200/422; the missing-model probe exited 3, and a clean
stop exited 0 with direct target-local refusal. Those retained private receipts
are historical evidence for that candidate. This source integration did not
contact the VM, inspect its present state, open a tunnel or rerun inference.

## Target preparation

1. Verify the portal VM ID, instantiated template, guest hostname/IP, SSH host key
   and actual guest resources. A challenge written over SSH to that VM's console
   can bind the SSH connection to the authenticated portal console. Record
   `/etc/os-release`, `nproc`, `free -m`, `df -B1 /`, `ss -lntup`, running services
   and existing containers before setup. Keep this evidence outside Git.
2. Use the supported SSH endpoint shown by the VM. Preserve SSH/VPN restrictions.
   Application NAT reachability is not evidence of SSH access. Authenticate using
   the authorized course/guest account; keep passwords out of argv, logs and Git.
3. On a new, owned Ubuntu target, install Docker and Git only if absent:
   `sudo apt-get update` and
   `sudo apt-get install --no-install-recommends docker.io git ca-certificates`.
   Do not run a distribution upgrade or change the kernel/cgroup mode for this task.
   Inspect `sudo docker info` for the actual cgroup version and driver.
4. Use Python 3.11 for the deployment tools and independent client. Candidate
   schema 1 uses a minimal bootstrap; release schema 2 requires the full locked
   environment described in the #79 contract below. The deployment
   modules use its standard library; the existing `src` package bootstrap also
   imports the locked `python-dotenv` and `loguru` packages. Create an isolated
   environment with `uv venv --python 3.11.15 "$WORK/host-env"`, set
   `PY="$WORK/host-env/bin/python"`, and install only those two packages with
   `uv pip install --python "$PY" --no-deps python-dotenv==1.2.3 loguru==0.7.3`.
   Do not install into uv's managed base interpreter or use the training environment. The serving dependencies remain inside the
   [CPU image](container.md). An installed uv can provide Python with
   `uv python install 3.11.15`. The verified uv image reference used by the
   Dockerfile is also usable for a temporary `docker create` / `docker cp /uv`
   bootstrap; remove only that newly named bootstrap container afterward.
5. Create a new deployment directory and clone the organization repository there.
   Check out an explicit full candidate commit. Keep config, evidence and incoming
   weights in ignored `temporary/` or outside the checkout. Do not update a shared
   environment, overwrite a previous checkout or build from a dirty tree.

Serialize heavyweight work with an advisory `fcntl.flock(LOCK_EX)` for the entire
child-process lifetime. On the coordinating Mac use
`/Users/dcayuela/Desktop/UPC/MLOPS/MLOps_Endesa/temporary/m4-m6-work-2026-10-23/m4-m6-compute.lock`.
Hold it across SSH/scp/build/model operations, with a corresponding target lock if
there are target-local operators. The deployment/client CLIs require `--lock-file`
and fail immediately when another owner holds it. Do not nest the same lock.
Keep one API worker and at most two model/CPU threads. Fixture tests can run in
parallel with target work.

## Pin the image, checkpoint and configuration

In the clean candidate checkout, set `SOURCE` to its full commit, `PY` to the
verified Python 3.11 executable, and `WORK` to the new absolute deployment directory.
The following commands are target-local. Prefix Docker/deployment commands with
`sudo` when the authorized account needs it; do not add broad Docker group access.

```sh
git rev-parse HEAD
git status --short
docker --host unix:///var/run/docker.sock build \
  --cpu-period 100000 --cpu-quota 100000 \
  --build-arg GIT_SHA="$SOURCE" --tag endesa75-candidate:"$SOURCE" .
docker --host unix:///var/run/docker.sock image inspect endesa75-candidate:"$SOURCE"
```

Retain the resulting immutable `sha256:...` image ID as `IMAGE`; no registry push
is needed. Select the image ID, not a tag. The tooling checks its revision label,
normal Uvicorn command, non-root user and the current clean source tree. The image
label alone is not cryptographic source attestation; retain the actual build log.

Retrieve only `models/chronos-t5-small-fine-tuned` through existing authorized DVC
access (`uv run dvc pull models/chronos-t5-small-fine-tuned.dvc` in an isolated
prepared checkout), or transfer an already verified artifact over authenticated
SSH/scp to a **new** incoming directory. Do not copy credentials into the image.
The tracked DVC directory is `0377ee21f89c25da36b95db59841c3d4.dir`; selected weights
SHA-256 is `746f9c3a9b61d452754f461a64c1529b2c07d8327ab8c33f804daa98c94c8cd9`,
training run `639eba4cc7e74de08ffab64beee715db`. Keep all three checkpoint files
and tracked sibling provenance. No base-model/fixture fallback is accepted.

Create the explicit config from those local bytes. Supply the measured hostname,
memory budget and unused loopback port; the example uses shell variables rather
than assumed target values. `INCOMING`, `SELECTED` and `PARAMS` are absolute paths;
`SELECTED` must name a new slot for provisioning.

```sh
"$PY" -m src.deployment.select \
  --deployment-id "$DEPLOYMENT_ID" --target-hostname "$TARGET_HOSTNAME" \
  --source-commit "$SOURCE" --image-id "$IMAGE" \
  --from-checkpoint "$INCOMING" --checkpoint "$SELECTED" --params "$PARAMS" \
  --cpu-threads 2 --memory-mb "$MEMORY_MB" --port "$PORT" \
  --docker-socket /var/run/docker.sock --output "$WORK/deployment.json"
"$PY" -m src.deployment --config "$WORK/deployment.json" \
  --lock-file "$WORK/compute.lock" --output "$WORK/provision.json" \
  provision --from-checkpoint "$INCOMING"
"$PY" -m src.deployment --config "$WORK/deployment.json" \
  --lock-file "$WORK/compute.lock" --output "$WORK/preflight.json" preflight
```

Config schema 1 contains exactly: `schema_version`, `deployment_id` (prefix
`endesa75-`), `target_hostname`, `source_commit`, immutable `image_id`, absolute
`checkpoint`, selected `checkpoint_sha256`/`training_run_id`,
`checkpoint_tree_sha256`, absolute `params`, `params_sha256`, `cpu_threads` (1–2),
`memory_mb`, `port` and absolute `docker_socket`. No defaults select a real target.
The config hash includes every field. Missing/mismatched selection fails before
starting the API. Provisioning verifies **all** checkpoint files, copies into a
new read-only slot, and refuses overwrites. Preflight checks the files again and
reuses the image's serving configuration/provenance parser without loading Torch.

## Start and verify from a separate client

```sh
"$PY" -m src.deployment --config "$WORK/deployment.json" \
  --lock-file "$WORK/compute.lock" --output "$WORK/start.json" start
```

The named container has read-only model/params mounts and filesystem, a bounded
`/tmp`, no Linux capabilities, no privilege escalation, explicit CPU/RAM limits,
one worker and manual restart policy. It publishes **only** `127.0.0.1:PORT`.
Startup waits at most 90 seconds for readiness and stops its own failed instance.
The receipt separates actual image/container/host inspection from HTTP/model
verification. Container ownership labels must match the full config before any
stop, restart or removal. Never edit that config in place to operate another instance.

Copy the inspection receipt to the separate client via authenticated scp. Open
an SSH tunnel using the **actual** banner's endpoint, verified host key and unused
local `CLIENT_PORT`, forwarding to the selected target loopback port:

```sh
ssh -N -o ExitOnForwardFailure=yes -o StrictHostKeyChecking=yes \
  -L "127.0.0.1:$CLIENT_PORT:127.0.0.1:$PORT" -p "$SSH_PORT" "$SSH_USER@$SSH_HOST"
```

Generate an explicit synthetic smoke request on the client; it uses no dataset
split or evaluation target. The API keeps negative values in source scale.

```sh
python3.11 -c 'import json; print(json.dumps({"state":"TAS","history":[-1.5]*512}))' > temporary/request75.json
python3.11 -m src.deployment.verify --base-url "http://127.0.0.1:$CLIENT_PORT" \
  --receipt temporary/start75.json --request temporary/request75.json \
  --lock-file "$CLIENT_LOCK" --output temporary/http75.json
```

Expected actual statuses: `/health` 200 `ok`, `/ready` 200 `ready`, valid
`/forecast` 200 and a 511-value request 422. The client verifies all 48 finite,
ordered p10/p50/p90 values, source scale, state, selected checkpoint/run, revision
and every executed prediction setting against the receipt. It records response
bodies/hashes, statuses and request/receipt hashes. Redirects, ambiguous URLs,
unexpected settings and malformed responses fail. The receipt is supplied by the
operator: HTTP does not independently reveal source/image identity. Local fixture
tests only validate this tooling; retain real target/client evidence separately.

## Failure and lifecycle checks

Run `missing-model` with the same config/lock and a fresh output path. It starts
an isolated probe with no weights, ports or network and the normal image entry
point. Expected: model initialization diagnostic, exit 3, no OOM. The tooling
stops/removes only that probe, including on a failed check.

Run `restart`, retain its new receipt and repeat the Mac/client checks through
SSH. Run `stop`, then verify refusal with a **separate target-local client** using
the target's direct loopback URL (copy the same smoke request to the target first):

```sh
"$PY" -m src.deployment.verify --base-url "http://127.0.0.1:$PORT" \
  --receipt "$WORK/evidence/restarted.json" --request "$WORK/request75.json" \
  --lock-file "$WORK/compute.lock" --output "$WORK/evidence/http-stopped.json" \
  --expect-stopped
```

Both direct target probes must receive TCP connection refusal. An SSH forwarding
socket can stay open after the target service stops, so a forwarded disconnect is
not this refusal check. Initial, restarted and final live HTTP checks run from the
Mac through SSH; the stopped probe runs on the target. Run `start` again and repeat
the Mac/client verification so the final deployment remains healthy.
Use a new `--output` file for every operation; existing evidence is never replaced.
`stop` retains `exit_code`, `oom_killed`, `was_running` and `clean_shutdown`.
It exits 1 and writes a failed receipt for a nonzero exit or OOM; exit 137 with
`OOMKilled=false` cannot establish a clean stop. TCP refusal and exit status are
separate checks. `start` stops only its owned instance if final inspection fails.
`inspect` captures a fresh inspection. `remove` requires a matching stopped
container; it does not delete images, checkpoints, volumes or other services.

Reviewer access uses the verified course SSH endpoint/tunnel and private receipt,
not an unauthenticated public NAT port. Stop the owned client tunnel when QA ends;
leave the authorized deployment healthy. Share the guest password privately,
never in GitHub or this runbook. VM deletion/term-end expiry and the platform's
lack of backups require retaining source, model references and evidence elsewhere.

For #76, create a **separate** deployment ID/config and unused loopback port, using
the same immutable image/model/params/settings. Leave it stopped for handoff.
Supply actual Docker cgroup version/driver, container PID/cgroup and existing
v1 `memory.usage_in_bytes`/`memory.max_usage_in_bytes`/`memory.limit_in_bytes`/
`memory.failcnt`, or v2 `memory.current`/`memory.peak`/`memory.max`/`memory.events`
paths. Provide the actual Python 3.11 interpreter and measured memory budget.
Do not change kernel/boot configuration or substitute Docker CLI RSS for container
memory. Benchmark permission/isolation, raw results and performance conclusions
remain #76's evidence; a source test or this deployment smoke does not establish them.

## Release contract for #79

This section specifies the consumer interface for a later reviewed execution.
#75 supplies explicit provisioning, lifecycle operations and independent HTTP
checks; #79 owns the release trigger, stable deployment lock, candidate/previous
slot selection, controlled failure and rollback. No release automation runs as
part of this source-readiness change.

Obtain the complete publication bundle from the trusted successful Actions
run/attempt as described in [image publication](image-publication.md). Record its
run URL, actual conclusion and artifact identity. Hashes and JSON receipts alone
do not authenticate the producer. Use a separate clean checkout of the manifest's
full `source.git_commit` with `uv sync --locked --dev`; keep the model on CPU with
two threads and the declared sampling settings. Source or lock changes require
fresh validation and publication at the resulting integrated commit.

For each authorized release attempt, choose a fresh absolute `WORK` directory and
a separate clean absolute `CHECKOUT`. The suggested layout is an interface, not
a claim that these paths or resources exist on Virtech:

| Artifact | Exact path or field |
| --- | --- |
| Immutable source | `$CHECKOUT` at `manifest.source.git_commit`; tree equals `manifest.source.git_tree` |
| Publication manifest | `$WORK/publication/image-publication/manifest.json` |
| Prepared validation directory | `$WORK/publication/release-validation/$VALIDATOR_RUN_ID` (the validator's report run ID, not the numeric Actions ID) |
| Parameters mounted at `/app/params.yaml` | `$WORK/params.yaml`, byte-for-byte copy of `$CHECKOUT/params.yaml` |
| Incoming selected DVC model | `$CHECKOUT/models/chronos-t5-small-fine-tuned` after authorized retrieval and clean `dvc status` |
| Read-only checkpoint slot | `$WORK/checkpoint`, mounted at `/app/models/chronos-t5-small-fine-tuned` |
| Deployment config and receipts | `$WORK/deployment.json` and distinct `$WORK/evidence/*.json` |
| Shared deployment lock | Operator-supplied absolute `$DEPLOY_LOCK`, identical across release attempts and rollback slots |

The model remains directory MD5 `0377ee21f89c25da36b95db59841c3d4.dir`, weights
SHA-256 `746f9c3a9b61d452754f461a64c1529b2c07d8327ab8c33f804daa98c94c8cd9`,
training run `639eba4cc7e74de08ffab64beee715db`. Verify authorized DVC retrieval
against the tracked reference before selecting the slot; selection hashes every
checkpoint file and provisioning checks the copied bytes again. Preserve caches
and weights. The provenance stays in the image at the tracked sibling path.

Set `MANIFEST` and `VALIDATION` to the paths above and `IMAGE_REFERENCE` to the
trusted manifest's immutable `image.reference`. On both the target and separate
client, run the shared verifier before any runtime operation:

```sh
uv run --no-sync python -m src.release_manifest verify \
  --manifest "$MANIFEST" --validation-dir "$VALIDATION" \
  --image-reference "$IMAGE_REFERENCE" --architecture amd64
```

This command performs no network access or inference. The consumer calls this
same public verifier through `src/deployment/release.py`, without duplicating
the validation or manifest rules. Schema 2 binds the manifest SHA-256 and its
absolute path, the prepared validation directory and registry digest in the
config's ownership hash. Schema 1 retains its original fields/hash for historical
candidate operations and supplies no verified-publication claim.

For a published release, consume the existing digest; #78 owns building and
publication. The candidate build command above is available for a separately
authorized local candidate and does not establish registry identity. On the
native Linux amd64 target, pull the selected digest only after execution
authorization and any required registry login through a secret mechanism:

```sh
docker --host unix:///var/run/docker.sock pull --platform linux/amd64 "$IMAGE_REFERENCE"
docker --host unix:///var/run/docker.sock image inspect "$IMAGE_REFERENCE" \
  > "$WORK/evidence/image-inspect.json"
IMAGE=$(docker --host unix:///var/run/docker.sock image inspect \
  --format '{{.Id}}' "$IMAGE_REFERENCE")
SOURCE=$(git -C "$CHECKOUT" rev-parse HEAD)
cp "$CHECKOUT/params.yaml" "$WORK/params.yaml"
chmod 444 "$WORK/params.yaml"
```

Use `$CHECKOUT/.venv/bin/python` as `PY`, run from `$CHECKOUT`, and provide measured
`TARGET_HOSTNAME`, `MEMORY_MB` and an unused private `PORT`. `DEPLOYMENT_ID` must
be a fresh `endesa75-...` name for the attempt. Parent directories must permit UID
10001 to traverse to the read-only files; do not grant access to private credential
directories. Keep all credentials outside the source/build context and evidence.

```sh
"$PY" -m src.deployment.select \
  --deployment-id "$DEPLOYMENT_ID" --target-hostname "$TARGET_HOSTNAME" \
  --source-commit "$SOURCE" --image-id "$IMAGE" \
  --from-checkpoint "$CHECKOUT/models/chronos-t5-small-fine-tuned" \
  --checkpoint "$WORK/checkpoint" --params "$WORK/params.yaml" \
  --cpu-threads 2 --memory-mb "$MEMORY_MB" --port "$PORT" \
  --docker-socket /var/run/docker.sock --output "$WORK/deployment.json" \
  --release-manifest "$MANIFEST" --validation-dir "$VALIDATION" \
  --image-reference "$IMAGE_REFERENCE"
"$PY" -m src.deployment --config "$WORK/deployment.json" \
  --lock-file "$DEPLOY_LOCK" --output "$WORK/evidence/provision.json" \
  provision --from-checkpoint "$CHECKOUT/models/chronos-t5-small-fine-tuned"
"$PY" -m src.deployment --config "$WORK/deployment.json" \
  --lock-file "$DEPLOY_LOCK" --output "$WORK/evidence/preflight.json" preflight
"$PY" -m src.deployment --config "$WORK/deployment.json" \
  --lock-file "$DEPLOY_LOCK" --output "$WORK/evidence/start.json" start
```

`select` writes schema 2 only when all release arguments are present and valid.
It rejects mismatched source, weights, training run, parameter hash or thread
count before writing. Provision/preflight/start/inspect/restart recheck the
publication bundle. Preflight also inspects the actual image ID, Linux amd64
platform, source/revision labels, digest in `RepoDigests`, non-root user and normal
entry point; its no-network container parses the mounted configuration and
checkpoint without loading a model. It compares the executed settings with the
manifest. Start uses the inspected immutable image ID and the private runtime
spec described above. Image ID, registry digest and weights SHA-256 are distinct.

Copy the receipt and trusted bundle to the separate client, check out the same
source there, and use its full locked Python environment. The copied manifest
and validation paths may differ; their bytes and identities must agree with the
target config. With the authorized private tunnel and explicit synthetic request:

```sh
"$PY" -m src.deployment.verify --base-url "http://127.0.0.1:$CLIENT_PORT" \
  --receipt "$CLIENT_WORK/start.json" --request "$CLIENT_WORK/request.json" \
  --release-manifest "$CLIENT_MANIFEST" --validation-dir "$CLIENT_VALIDATION" \
  --lock-file "$CLIENT_LOCK" --output "$CLIENT_WORK/http-initial.json"
```

The client reuses #78 verification, compares the inspected release/settings and
then checks actual 200/200/200/422 responses and 48 ordered finite quantiles. Its
report binds the receipt and manifest hashes plus registry reference. HTTP cannot
independently attest the source or image; that evidence comes from the trusted
bundle and target inspection. Repeat after restart and after rollback. Check a
clean stop and direct target-local refusal separately before claiming shutdown.
Stop/remove remain ownership-fenced even if release evidence/model bytes break,
so an unusable bundle cannot prevent stopping its own instance.

Before #79 execution, retain a reviewed previous config/manifest/bundle/checkpoint
slot and prove its recovery path. Hold the stable lock across the complete
deployment/verification/rollback transaction; do not acquire that same lock again
inside a child CLI. An orchestrator can invoke `Deployment` methods while holding
`compute_lock`, or serialize the CLIs under its own distinct transaction lock.
An individual CLI lock does not serialize gaps between several commands. Registry
destination/reader access, final human reviews, EDN PR #97 assessment and actual
release validation/publication/independent startup remain pending prerequisites.
