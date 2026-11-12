# Versioned image publication

The manual [image-publication workflow](../.github/workflows/image-publication.yml)
combines the [container contract](container.md) from #50 and the
[bounded validation handoff](release-validation.md#publication-handoff-for-78)
from #77. It retrieves an explicitly selected successful validation run/attempt
and artifact through authenticated GitHub API reads, checks the artifact ZIP's
SHA-256, and checks that evidence against the exact clean source before building
or pushing. It records the registry digest with the
selected checkpoint, prediction settings and hashes of the complete retained
validation reports. It calls #77's public `check_handoff` without inference;
validation runs separately before publication.

Implementation and fixture tests do not establish publication. Registry/access
agreement, human peer review, merged dependencies, an actual successful workflow
and an independent pull/start check remain release prerequisites. A failed push
or manifest step can leave a registry image without a completed release record;
retain that run's evidence and investigate before another dispatch. Never treat
an image alone as a completed release or delete it to hide a partial attempt.

## Registry and execution prerequisites

The concrete proposal is
`ghcr.io/mlops-2627q1-mds-upc/endesa-forecast`, initially private, associated with
this repository. The workflow would use its short-lived `GITHUB_TOKEN` with
`contents: read` and `packages: write`; independent authorized readers would need
package read access. This proposal has not been agreed or configured. The local
decision proposal is in ignored `temporary/issue-78/decision-proposal.txt`; EDN
PR #97 remains separate, with no invented assessment or approval.

The workflow requires an explicit `image_repository` and has no default registry.
GHCR uses the workflow token; another agreed registry requires narrowly scoped
`IMAGE_REGISTRY_USER` and `IMAGE_REGISTRY_PASSWORD` secrets. It never creates
these secrets or changes package visibility/access. A first package's absent tag
must produce an explicit `manifest unknown` or `no such manifest` response;
ambiguous authentication/network errors stop publication. Resolve registry access
as a separate authorized step if the registry cannot establish absence.

After review and integration, run the manual release validator on the intended
clean `main` source. Pin the actual successful run ID/attempt and the unexpired
`release-validation-RUN_ID-ATTEMPT` artifact ID/digest from GitHub's API. The
publication workflow requires the exact dispatch SHA, those identifiers and the
explicit destination. These commands are a later publication step:

```sh
source_sha=$(git rev-parse origin/main)
gh workflow run image-publication.yml \
  --repo mlops-2627q1-mds-upc/MLOps_Endesa --ref main \
  -f source_sha="$source_sha" \
  -f image_repository=AGREED_REGISTRY/AGREED_REPOSITORY \
  -f release_version=v0.1.0 \
  -f validation_run_id=RUN_ID -f validation_run_attempt=ATTEMPT \
  -f validation_artifact_id=ARTIFACT_ID \
  -f validation_artifact_digest=sha256:ARTIFACT_ZIP_SHA256
```

The API lookup requires `workflow_dispatch`, `completed`/`success`, this
repository's release-validation workflow on `main`, exact source/attempt, and an
unexpired artifact for that attempt. It compares all five recorded
`source.github` fields, including workflow ref; artifacts outside the attempt's
time interval fail. Missing artifacts, an unexpected event/workflow/source and
archive digest mismatches fail before a build. Downloads have hard command
timeouts; extraction rejects unsafe paths, symlinks, duplicate/unknown members,
more than 256 files, a compressed archive over 32 MiB or over 100 MiB of expanded
content. Original archive bytes and sanitized API receipts remain in the bundle.

If `main` advances before publication, a mismatched input fails; validate the
intended integrated source again. Each tag is `<release_version>-<full_source_sha>`; occupied
tags fail, and the workflow neither creates `latest` nor updates prior tags.
Consumers use the recorded `repository@sha256:...` digest. The workflow builds a
single `linux/amd64` image on the native runner. The manifest helper can verify
`amd64` or `arm64`, but this workflow does not qualify an arm64 release.

The job has a 35-minute cap, including 3 minutes for evidence download/ingestion,
12 minutes for the build and 6 minutes for push/manifest creation. It serializes
publication attempts without canceling a running attempt. The publication job
has `actions: read` for selected evidence; it receives no DVC credentials and
does not download weights or data. DVC credentials remain scoped to #77's
retrieval path. Registry credentials stay in the runner's private Docker
configuration and never enter build arguments, the allowlisted context, image
layers or evidence uploads. Weights/raw data also stay outside that context.

## Workflow semantic validation

[Workflow / semantics](../.github/workflows/workflow-semantics.yml) runs
checksum-pinned actionlint 1.7.12 against every Actions workflow on PRs and pushes
to `main`. It checks expression types, allowed contexts and workflow structure;
optional ShellCheck/Pyflakes integrations are disabled because this job has that
specific scope. It also requires rejection of the intentionally invalid fixture
in `tests/fixtures/workflows/invalid-runner-job-env.yml`, outside the directory
GitHub registers as workflows.

The original [failed run 43224118655](https://github.com/mlops-2627q1-mds-upc/MLOps_Endesa/actions/runs/43224118655)
at source `a7225cb47e3ef653d79aac7a6eed7310f5a19a53` created no jobs. Its inherited
publication file used `runner.temp` in job-level `env`, where GitHub disallows
the `runner` context. YAML/bash syntax and ordinary tests had passed; actionlint
reproduces the precise unavailable-context error. The workflow sets
`DOCKER_CONFIG` in a runner step through `$RUNNER_TEMP` and `$GITHUB_ENV`, keeping
registry credentials outside the checkout for later steps. GitHub documents this
restriction in its [context availability table](https://docs.github.com/en/actions/reference/workflows-and-actions/contexts#context-availability).

With the pinned actionlint binary available locally, reproduce both checks:

```sh
actionlint -shellcheck= -pyflakes= -oneline
if actionlint -shellcheck= -pyflakes= -oneline \
  tests/fixtures/workflows/invalid-runner-job-env.yml; then
  echo 'The unavailable-context regression was accepted' >&2
  exit 1
fi
```

The fixture must fail specifically because `runner` is unavailable in job `env`;
the CI check asserts that diagnostic. Retain the original failed run. Semantic
validation and ordinary CI do not establish that GitHub accepted or executed a
publication dispatch; record that separately during the authorized release step.

## Manifest v1 and read-only verification

`python -m src.release_manifest ingest` checks the API receipts against explicit
run/attempt/artifact/digest pins and unpacks the downloaded archive into a fresh
directory. It retains the original archive and sanitized `actions-provenance.json`.
The workflow obtains the API responses itself; accepting JSON files in a local
invocation does not authenticate their author.

`python -m src.release_manifest check` accepts that prepared run directory,
the exact Git SHA and explicit image naming/platform inputs. It rejects dirty
source, missing/mismatched source hashes, checkpoint provenance/DVC identities,
altered parameters, failed/missing steps, exceeded budgets or incomplete detailed
data/model reports. `create` additionally requires post-push `docker image
inspect` JSON, matching platform/revision/tag and exactly one destination digest;
its output must not already exist. `check --image-inspect PATH` checks platform,
source/revision labels and tag before the push. Every tracked source file's bytes
must also match its Git blob, including files hidden by `assume-unchanged`.

| JSON field | Bound identity |
| --- | --- |
| `schema_version` | Integer `1`; unknown fields/versions fail verification |
| `source` | Full `git_commit`, `git_tree`, `git_dirty: false` |
| `image` | Explicit `repository`, version `tag`, immutable `digest`, `reference`, `os`, `architecture`, source `revision` |
| `model` | Tracked `path`, directory `dvc_md5`, `weights_sha256`, `training_run_id`, `provenance_sha256` |
| `configuration` | Parameters, project/lock/executed-parameter hashes and actual CPU prediction context/horizon, sampling, seed and thread settings |
| `validation` | Passed status, validator `run_id`, `report_sha256`, `handoff_sha256`, `executed_params_sha256`, relative `evidence_sha256`, source/data hashes, all five GitHub identifiers and selected Actions artifact identity |

The evidence map covers the root summary, identities, executed parameters, every
step log, data-validation summary, minimum-functionality report, directional
plan/report, handoff/checksum and sanitized API receipt. `validation.artifact`
records repository/workflow, run ID/attempt, artifact ID/name/digest and run URL.
The original artifact ZIP is rehashed and every archived file is compared with
the extracted evidence during verification. The manifest contains no absolute checkpoint paths, command lines or arbitrary
environment fields. The digest refers to the registry image manifest, not a
local Docker image ID or model weights hash.

For #79, use a clean checkout of the recorded source with the locked environment:

```sh
uv run --no-sync python -m src.release_manifest verify \
  --manifest "$manifest" --validation-dir "$validation_dir" \
  --image-reference "$image_reference" --architecture amd64
```

Success prints the verified manifest and exits 0 without modifying source,
evidence, Git, Docker or any registry. Failure exits 1 with a controlled reason;
malformed inputs never echo their contents. The command checks retained evidence
and expected identities, and makes no network request. GitHub identifiers inside
a report are attribution, not authenticated proof that a remote workflow passed.
Obtain the bundle from the trusted Actions run and independently check its actual
conclusion, source SHA and attempt before using the manifest. JSON hashes detect
later changes; they do not sign or authenticate the original producer.

## Independent pull and real-model startup

Run this later in another authorized Linux amd64 environment with Docker and
enough memory for the selected model. This makes one bounded real forecast and
requires separate execution authorization. Fixture HTTP tests use synthetic
identities and cannot supply this evidence.

1. Inspect the actual publication run with
   `gh run view RUN_ID --json headSha,attempt,status,conclusion,url`. Require
   `completed`/`success`, the intended source and the selected attempt. Download
   `image-publication-RUN_ID-ATTEMPT` into an ignored release directory with
   `gh run download RUN_ID --name image-publication-RUN_ID-ATTEMPT --dir RELEASE_DIR`.
   Keep the Actions run URL and bundle beyond its 14-day retention window.
2. Use a separate clean checkout at the manifest's exact `source.git_commit`.
   Install `uv sync --locked --dev`, set the paths below to the downloaded files,
   read the expected digest from the trusted manifest, then run `verify` above.

```sh
manifest=/absolute/path/to/RELEASE_DIR/image-publication/manifest.json
validation_dir=/absolute/path/to/RELEASE_DIR/release-validation/VALIDATION_RUN_ID
mkdir -p temporary
verification_dir=$(mktemp -d "$(pwd)/temporary/release-check-XXXXXXXX")
container_name="endesa-${verification_dir##*/}"
image_reference=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["image"]["reference"])' "$manifest")
export RELEASE_MANIFEST="$manifest"
export RELEASE_VERIFICATION_DIR="$verification_dir"
```

3. Provision only the selected model through authorized DVC access. Supply
   `DVC_HTTP_USER`/`DVC_HTTP_PASSWORD` to this command through the environment's
   secret mechanism; the script passes them to DVC in memory and suppresses
   credential-bearing third-party errors. Omit both for public access. Reuse a
   verified model directory when available; do not remove caches or weights.

```sh
uv run --no-sync python - <<'PY'
import os
from dvc.repo import Repo
user, password = os.environ.get('DVC_HTTP_USER'), os.environ.get('DVC_HTTP_PASSWORD')
if bool(user) != bool(password):
    raise SystemExit('Supply both DVC credentials or neither')
config = {'auth': 'basic', 'user': user, 'password': password} if user else {}
try:
    with Repo('.', remote='origin', remote_config=config) as repo:
        repo.pull(targets=['models/chronos-t5-small-fine-tuned.dvc'], remote='origin', jobs=2)
except Exception:
    raise SystemExit('Model retrieval failed; check authorized DVC access') from None
PY
uv run --no-sync dvc status models/chronos-t5-small-fine-tuned.dvc --json \
  > "$verification_dir/model-status.json"
python3 - <<'PY'
import json, os
from pathlib import Path
assert json.loads((Path(os.environ['RELEASE_VERIFICATION_DIR']) / 'model-status.json').read_text()) == {}
PY
checkpoint_dir="$(pwd)/models/chronos-t5-small-fine-tuned"
```

4. Authenticate to the agreed registry using an authorized read token through
   `docker login --password-stdin` if needed. Pull by digest and check its actual
   Linux platform and revision before starting the normal production command:

```sh
docker pull --platform linux/amd64 "$image_reference"
docker image inspect "$image_reference" > "$verification_dir/image-inspect.json"
python3 - <<'PY'
import json, os
from pathlib import Path
m = json.loads(Path(os.environ['RELEASE_MANIFEST']).read_text())
i, = json.loads((Path(os.environ['RELEASE_VERIFICATION_DIR']) / 'image-inspect.json').read_text())
assert (i['Os'], i['Architecture']) == ('linux', m['image']['architecture'])
assert i['Config']['Labels']['org.opencontainers.image.revision'] == m['source']['git_commit']
assert m['image']['reference'] in i['RepoDigests']
PY
docker run --detach --name "$container_name" --cpus 2 --memory 2g \
  --read-only --tmpfs /tmp:rw,noexec,nosuid,size=64m \
  --publish 127.0.0.1:18000:8000 \
  --mount "type=bind,src=$checkpoint_dir,dst=/app/models/chronos-t5-small-fine-tuned,readonly" \
  "$image_reference"
```

5. Run the probes and one forecast. Startup verifies the mounted weights against
   tracked provenance; these assertions also compare the served identity/settings
   with the manifest and require finite ordered 48-step quantiles. The two GiB cap
   is an initial budget, not a capacity measurement.

```sh
python3 - <<'PY'
import json, math, os, platform, time
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen
m = json.loads(Path(os.environ['RELEASE_MANIFEST']).read_text())
def request(path, body=None, timeout=3):
    data = None if body is None else json.dumps(body).encode()
    r = Request('http://127.0.0.1:18000' + path, data=data,
                headers={'Content-Type': 'application/json'})
    with urlopen(r, timeout=timeout) as response:
        assert response.status == 200
        return json.load(response)
deadline = time.monotonic() + 120
while True:
    try:
        assert request('/health') == {'status': 'ok'}
        assert request('/ready') == {'status': 'ready'}
        break
    except (URLError, TimeoutError, ConnectionError):
        if time.monotonic() >= deadline:
            raise SystemExit('Real-model startup exceeded 120 seconds')
        time.sleep(1)
r = request('/forecast', {'state': 'TAS', 'history': [1000.0] * 512}, timeout=60)
assert r['checkpoint'] == {'sha256': m['model']['weights_sha256'],
                            'training_run_id': m['model']['training_run_id']}
assert r['model_revision'] == 'local-sha256:' + m['model']['weights_sha256']
assert (r['context_length'], r['horizon'], r['value_scale']) == (512, 48, 'source')
for key, value in m['configuration']['prediction'].items():
    if key not in ('context_length', 'horizon'):
        assert r['settings'][key] == value
assert all(len(r[key]) == 48 for key in ('p10', 'p50', 'p90'))
assert all(math.isfinite(v) for key in ('p10', 'p50', 'p90') for v in r[key])
assert all(a <= b <= c for a, b, c in zip(r['p10'], r['p50'], r['p90']))
receipt = {'fixture_only': False, 'health': 200, 'ready': 200, 'forecast': r,
           'image_reference': m['image']['reference'], 'source': m['source'],
           'environment': {'platform': platform.platform(), 'architecture': platform.machine()}}
with (Path(os.environ['RELEASE_VERIFICATION_DIR']) / 'startup.json').open('x', encoding='utf-8') as stream:
    json.dump(receipt, stream, indent=2, allow_nan=False)
print('Real-model health, readiness, identity and forecast checks passed')
PY
docker logs "$container_name" > "$verification_dir/startup.log" 2>&1
docker stop "$container_name"
```

If a check fails, retain logs/response and stop the owned container even when the
probe command exits nonzero. Record the environment/architecture and actual
command results together with the source, digest, Actions URL and verified bundle.
Only a successful independent execution closes the pull/start evidence gap; it
does not establish a Virtech deployment or throughput result.

References: [GitHub run attempts](https://docs.github.com/en/rest/actions/workflow-runs#get-a-workflow-run-attempt),
[Actions artifacts and SHA-256 digests](https://docs.github.com/en/rest/actions/artifacts),
[GHCR authentication and package access](https://docs.github.com/en/packages/working-with-a-github-packages-registry/working-with-the-container-registry),
[Docker push and digests](https://docs.docker.com/reference/cli/docker/image/push/),
[Docker registry manifest inspection](https://docs.docker.com/reference/cli/docker/buildx/imagetools/inspect/).
