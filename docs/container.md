# CPU forecast container

The image runs the [forecast API](api.md) as UID/GID `10001`, with one Uvicorn
factory worker and two CPU threads. Python and uv base images are pinned by
version and multi-architecture digest; Python packages come from `uv.lock` via
`uv sync --locked --only-group serving`. Linux uses the CPU Torch index.

The serving group retains the existing inference imports (including pandas,
evaluation and split helpers), while excluding notebook, DVC, MLflow and PEFT
tools. Only explicit runtime source files, parameters and the tracked selected
checkpoint provenance enter the image. Weights, datasets, environments,
credentials, caches, tests and temporary outputs are excluded from the build
context. The production image contains no fixture predictor.

## Build and provision

Run from a clean checkout using a working Docker engine. A full Git commit is
required for the image revision label; an absent or malformed value fails the
build. The commands below use a POSIX shell.

```sh
revision=$(git rev-parse HEAD)
docker build --build-arg GIT_SHA="$revision" --tag "endesa-forecast:$revision" .
docker image inspect "endesa-forecast:$revision" \
  --format 'image_id={{.Id}} revision={{index .Config.Labels "org.opencontainers.image.revision"}}'
```

Retrieve the selected artifact using existing authorized DVC access, following
the [inference setup](inference.md), or reuse an already retrieved copy:

```sh
uv run --locked dvc pull models/chronos-t5-small-fine-tuned.dvc
checkpoint_dir="$(pwd)/models/chronos-t5-small-fine-tuned"
```

The directory must contain the complete checkpoint and be readable by UID
`10001`. Mount that directory only, so the image's tracked provenance remains
available at `/app/models/chronos-t5-small-fine-tuned.provenance.json`. Startup
hashes `model.safetensors` and checks the selected SHA-256 and training run against
that provenance. Missing, unreadable or mismatched artifacts abort startup;
the service never downloads weights. A shared writable model cache is not needed.

## Run and check the selected model

```sh
docker run --detach --name endesa-forecast --cpus 2 --memory 2g \
  --read-only --tmpfs /tmp:rw,noexec,nosuid,size=64m \
  --publish 127.0.0.1:8000:8000 \
  --mount "type=bind,src=$checkpoint_dir,dst=/app/models/chronos-t5-small-fine-tuned,readonly" \
  "endesa-forecast:$revision"
docker logs endesa-forecast
curl --fail --silent --show-error http://127.0.0.1:8000/health
curl --fail --silent --show-error http://127.0.0.1:8000/ready
```

Wait for the startup log before probing. Docker's health check calls `/ready`;
it does not perform inference. The container listens on port 8000 and the host
mapping above exposes it only on loopback. The two-thread CPU setting is separate
from the container's `--cpus 2` scheduler limit. The 2 GiB memory cap is a starting
budget, not a measured capacity guarantee.

Generate the 512-value request using [the API example](api.md#requests-and-responses),
then make one bounded request and verify its artifact identity:

```sh
curl --fail-with-body --silent --show-error --max-time 60 \
  http://127.0.0.1:8000/forecast -H 'Content-Type: application/json' \
  --data-binary @reports/inference/api-request.json \
  -o reports/inference/container-response.json
python - <<'PY'
import json
from pathlib import Path

expected = json.loads(Path("models/chronos-t5-small-fine-tuned.provenance.json").read_text())
actual = json.loads(Path("reports/inference/container-response.json").read_text())
assert actual["checkpoint"]["sha256"] == expected["model_safetensors_sha256"]
assert actual["checkpoint"]["training_run_id"] == expected["training_run_id"]
assert actual["model_revision"] == "local-sha256:" + expected["model_safetensors_sha256"]
print(actual["model_revision"], actual["settings"])
PY
docker image inspect "endesa-forecast:$revision" \
  --format 'image_id={{.Id}} revision={{index .Config.Labels "org.opencontainers.image.revision"}}'
docker stop endesa-forecast
docker rm endesa-forecast
```

Retain the commit, image ID, probe results and full response together for a real
container smoke record. The API settings `FORECAST_CHECKPOINT`,
`FORECAST_PARAMS_PATH` and `FORECAST_CPU_THREADS` remain available; path overrides
must refer to files mounted inside the container. Use one worker and preserve
the selected provenance when relocating the weights. Keep artifacts unchanged
while the process is serving.

## PR smoke and local reproduction

[Container smoke](../.github/workflows/container-smoke.yml) is a separate workflow;
the existing `CI / lint` and `CI / test` names are unchanged. It builds the real
runtime, checks non-root execution and CPU imports, proves the normal production
command exits with status 3 when weights are absent, then overrides the command
to use a read-only mounted test factory. The HTTP check requires `/health` and
`/ready` to return 200, verifies a 48-step forecast with an explicitly synthetic
all-zero identity, and checks invalid-request rejection. No environment switch
can enable the fixture in the production factory.

To repeat the fixture check against the built image:

```sh
docker run --detach --name endesa-fixture --cpus 2 --memory 2g \
  --read-only --tmpfs /tmp:rw,noexec,nosuid,size=64m \
  --publish 127.0.0.1:8000:8000 \
  --mount "type=bind,src=$(pwd)/tests/container,dst=/smoke,readonly" \
  --env PYTHONPATH=/app:/smoke "endesa-forecast:$revision" \
  uvicorn fixture_app:create_app --factory --host 0.0.0.0 --port 8000 --workers 1
python3 tests/container/smoke.py http://127.0.0.1:8000
docker stop endesa-fixture
docker rm endesa-fixture
```

Without Docker, `uv run --locked pytest tests/test_container.py -q` checks the
context allowlist and stages the explicit source copies in an empty directory.
It exercises actual Uvicorn HTTP transport and the production missing-artifact
failure, using the local Python environment. This does not validate image layers,
Linux dependencies or real model execution. Fixture CI does not establish a
successful selected-model container smoke, publication or deployment.

References: [uv dependency groups](https://docs.astral.sh/uv/concepts/projects/sync/),
[Docker build context](https://docs.docker.com/build/concepts/context/).
