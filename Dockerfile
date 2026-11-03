# syntax=docker/dockerfile:1
FROM python:3.11.15-slim-bookworm@sha256:d29f48a31a8b408ed19272ca1e7b10ebae13b240a27e862d3d4217c528e2e0c3 AS dependencies
COPY --from=ghcr.io/astral-sh/uv:0.11.32@sha256:df4cae8f3a96d175e2e5f992e597550000edbe78fdc2594d5cd8de1a217f504c /uv /usr/local/bin/uv
ENV UV_PYTHON_DOWNLOADS=never UV_LINK_MODE=copy
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --only-group serving --no-cache

FROM python:3.11.15-slim-bookworm@sha256:d29f48a31a8b408ed19272ca1e7b10ebae13b240a27e862d3d4217c528e2e0c3 AS runtime
ARG GIT_SHA
LABEL org.opencontainers.image.title="Endesa forecast API" \
      org.opencontainers.image.revision="${GIT_SHA}"
RUN python -c 'import re, sys; sys.exit(not re.fullmatch(r"[0-9a-f]{40}", sys.argv[1]))' "${GIT_SHA}" && \
    groupadd --gid 10001 forecast && \
    useradd --uid 10001 --gid 10001 --no-create-home --shell /usr/sbin/nologin forecast
WORKDIR /app
ENV PATH="/app/.venv/bin:${PATH}" \
    PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    FORECAST_CHECKPOINT=/app/models/chronos-t5-small-fine-tuned \
    FORECAST_PARAMS_PATH=/app/params.yaml FORECAST_CPU_THREADS=2 \
    OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HOME=/tmp/huggingface
COPY --from=dependencies /app/.venv /app/.venv
COPY params.yaml ./
COPY src/__init__.py src/config.py src/evaluation.py src/splits.py ./src/
COPY src/api/__init__.py src/api/app.py src/api/schemas.py src/api/settings.py ./src/api/
COPY src/modeling/__init__.py src/modeling/predict.py src/modeling/checkpoints.py ./src/modeling/
COPY models/chronos-t5-small-fine-tuned.provenance.json ./models/
USER 10001:10001
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/ready', timeout=2).close()"
CMD ["uvicorn", "src.api.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
