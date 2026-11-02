# Virtech candidate deployment

This runbook operates an explicitly allocated course VM. It does not allocate VMs,
discover hosts, install SSH access, publish images or automate releases. Use the
course portal and its assigned template/quota first. Read the actual console NAT
banner; do not copy example IPs, ports, OS versions or capacities from a manual.
Keep existing VMs, services, credentials and artifacts intact.

The deployment branch carries unmerged dependencies #52 (`7c2a9a4`), #86
(`6f07154`) and #88 (`488bd4f`). #52/#86 had human approvals at those heads when
inspected on 2 November 2026; #88 did not. A running candidate does not establish
merge acceptance, #75 peer review, performance or final-test accuracy.

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
4. Use Python 3.11 for the deployment tools and independent client. The deployment
   modules use its standard library; the existing `src` package bootstrap also
   imports the locked `python-dotenv` and `loguru` packages. In the isolated
   interpreter install these with `uv pip install --python "$PY" --no-deps
   python-dotenv==1.2.3 loguru==0.7.3`. The serving dependencies remain inside the
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

Run `restart`, retain its new receipt and repeat the separate-client checks. Run
`stop`, then run the client with `--expect-stopped`; both probes must receive
connection refusal rather than a timeout or response from another service.
Run `start` again and repeat verification so the final deployment remains healthy.
Use a new `--output` file for every operation; existing evidence is never replaced.
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
