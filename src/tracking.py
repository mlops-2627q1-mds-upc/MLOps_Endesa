"""Explicit MLflow runs for forecasting scripts and notebooks."""

from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
from importlib.metadata import distributions
import json
import logging
import math
import os
from pathlib import Path
import platform
import re
import subprocess
import tempfile
import threading
import time
from typing import Any, Literal
from urllib.parse import urlparse

from mlflow import MlflowClient
from mlflow.entities import Metric, Param
from mlflow.exceptions import MlflowException

from src.config import PROJ_ROOT

# Runs execute locally in Barcelona; the country sets CodeCarbon's grid carbon intensity.
ENERGY_COUNTRY_ISO_CODE = "ESP"
# Attribute power to this process by its CPU share, rather than to the whole machine.
ENERGY_TRACKING_MODE = "process"
# CodeCarbon reports how it obtained each power reading only in its log output.
_TRACKING_METHOD = re.compile(r"(CPU|GPU|RAM) Tracking Method: (.+)")
# CodeCarbon reads GPU energy only from NVML's energy counter; without it, it logs this
# warning and reports 0 kWh for the GPU.
_GPU_COUNTER_UNSUPPORTED = "Failed to retrieve gpu total energy consumption"
GPU_SAMPLES_METHOD = "nvml power samples (board power)"
GPU_UNMEASURED_METHOD = "not measured (NVML energy counter not supported)"
# NVML keeps about 120 power samples; under load they span only a few seconds.
_GPU_POLL_SECONDS = 0.5


def tracking_client(tracking_uri: str | None = None) -> MlflowClient:
    """Require a chosen store, rather than silently recording runs locally."""
    uri = (tracking_uri or os.environ.get("MLFLOW_TRACKING_URI", "")).strip()
    if not uri:
        raise ValueError("Set MLFLOW_TRACKING_URI or pass tracking_uri explicitly.")
    if urlparse(uri).hostname == "dagshub.com":
        missing = [
            name
            for name in ("MLFLOW_TRACKING_USERNAME", "MLFLOW_TRACKING_PASSWORD")
            if not os.environ.get(name, "").strip()
        ]
        if missing:
            raise ValueError(f"Set your DagsHub credentials: {', '.join(missing)}.")
    return MlflowClient(tracking_uri=uri)


def _experiment_id(client: MlflowClient, name: str) -> str:
    experiment = client.get_experiment_by_name(name)
    if experiment is None:
        try:
            return client.create_experiment(name)
        except MlflowException:
            experiment = client.get_experiment_by_name(name)
            if experiment is None:
                raise
    if experiment.lifecycle_stage != "active":
        raise ValueError(f"Experiment {name!r} is deleted; choose an active experiment.")
    return experiment.experiment_id


def _provenance(data_version: str, model_revision: str) -> dict[str, Any]:
    def git(*args: str) -> str:
        return subprocess.check_output(
            ["git", *args], cwd=PROJ_ROOT, text=True, stderr=subprocess.PIPE
        ).strip()

    return {
        "git_commit": git("rev-parse", "HEAD"),
        "git_dirty": bool(git("status", "--porcelain")),
        "data_version": data_version,
        "model_revision": model_revision,
        "uv_lock_sha256": hashlib.sha256((PROJ_ROOT / "uv.lock").read_bytes()).hexdigest(),
    }


def _emissions_tracker(output_dir: Path, country_iso_code: str):
    """Create an offline CodeCarbon tracker; tests replace this with a fake."""
    # Imported here so that importing this module does not load CodeCarbon.
    from codecarbon import OfflineEmissionsTracker
    from codecarbon.core.hardware_cache import clear_cache

    # CodeCarbon sets up hardware once per process and logs the tracking methods only
    # then; a fresh setup gives every run its own methods in its own log.
    clear_cache()
    return OfflineEmissionsTracker(
        country_iso_code=country_iso_code,
        output_dir=str(output_dir),
        tracking_mode=ENERGY_TRACKING_MODE,
        log_level="info",
    )


def _integrate_power_kwh(samples: list[tuple[int, float]], start_us: int, stop_us: int) -> float:
    """Energy of (timestamp in microseconds, watts) samples over [start_us, stop_us].

    Each sample's power holds until the next sample, as NVML reports averaged readings.
    """
    points = sorted(samples)
    joules = 0.0
    for (begin, watts), (end, _) in zip(points, [*points[1:], (stop_us, 0.0)]):
        overlap = min(end, stop_us) - max(begin, start_us)
        if overlap > 0:
            joules += watts * overlap / 1e6
    return joules / 3.6e6


class _GpuPowerSampler:
    """Integrate NVML's power samples for GPUs whose energy counter is not supported.

    The samples are the board power the driver reports, for the whole device.
    """

    def __init__(
        self,
        nvml: Any,
        handles: list[Any],
        poll_seconds: float = _GPU_POLL_SECONDS,
        clock: Callable[[], float] = time.time,
    ):
        self._nvml = nvml
        self._handles = handles
        self._poll_seconds = poll_seconds
        self._clock = clock
        self._samples: list[list[tuple[int, float]]] = [[] for _ in handles]
        self._capacity = [0] * len(handles)
        self._lost = False
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.kwh = 0.0

    def _read(self, index: int, since_us: int) -> list[tuple[int, float]]:
        try:
            _, samples = self._nvml.nvmlDeviceGetSamples(
                self._handles[index], self._nvml.NVML_TOTAL_POWER_SAMPLES, since_us
            )
        except self._nvml.NVMLError_NotFound:
            return []
        return [(sample.timeStamp, sample.sampleValue.uiVal / 1000) for sample in samples]

    def _poll(self) -> None:
        for index, samples in enumerate(self._samples):
            new = self._read(index, samples[-1][0])
            # A read as large as the whole buffer may have skipped overwritten samples.
            if len(new) >= self._capacity[index]:
                self._lost = True
            samples.extend(new)

    def _run(self) -> None:
        while not self._stop.wait(self._poll_seconds):
            self._poll()

    def start(self) -> None:
        self._start_us = int(self._clock() * 1e6)
        for index in range(len(self._handles)):
            # The full buffer gives its capacity and the reading in force at the start.
            buffer = self._read(index, 0)
            if not buffer:
                raise RuntimeError("NVML returned no GPU power samples.")
            self._capacity[index] = len(buffer)
            self._samples[index] = [buffer[-1]]
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        stop_us = int(self._clock() * 1e6)
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        self._poll()
        if self._lost:
            raise RuntimeError(
                "GPU power samples were overwritten before they were read; "
                "the GPU energy is incomplete."
            )
        self.kwh = sum(
            _integrate_power_kwh(samples, self._start_us, stop_us) for samples in self._samples
        )


def _gpu_power_sampler() -> _GpuPowerSampler | None:
    """Return a sampler only for NVIDIA GPUs that CodeCarbon cannot measure; tests fake this."""
    try:
        import pynvml
    except ImportError:
        return None
    try:
        pynvml.nvmlInit()
        handles = [
            pynvml.nvmlDeviceGetHandleByIndex(index)
            for index in range(pynvml.nvmlDeviceGetCount())
        ]
    except pynvml.NVMLError:
        # No NVIDIA driver or GPU: CodeCarbon reports the GPU itself.
        return None
    if not handles:
        return None
    for handle in handles:
        try:
            pynvml.nvmlDeviceGetTotalEnergyConsumption(handle)
        except pynvml.NVMLError:
            continue
        # A working energy counter on any GPU means CodeCarbon measures it as usual.
        return None
    try:
        for handle in handles:
            pynvml.nvmlDeviceGetSamples(handle, pynvml.NVML_TOTAL_POWER_SAMPLES, 0)
    except pynvml.NVMLError:
        return None
    return _GpuPowerSampler(pynvml, handles)


def _grid_emissions_kg(kwh: float, country_iso_code: str) -> float:
    """Emissions of kwh with CodeCarbon's own carbon intensity for the country."""
    from codecarbon.core.emissions import Emissions
    from codecarbon.core.units import Energy
    from codecarbon.external.geography import GeoMetadata
    from codecarbon.input import DataSource

    return Emissions(DataSource()).get_private_infra_emissions(
        Energy.from_energy(kWh=kwh), GeoMetadata(country_iso_code=country_iso_code)
    )


def _environment() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "system": platform.system(),
        "machine": platform.machine(),
        "packages": dict(
            sorted((package.metadata["Name"], package.version) for package in distributions())
        ),
    }


@dataclass(frozen=True)
class TrackedRun:
    """A run and its client; no global active-run state is changed."""

    client: MlflowClient
    run_id: str

    def log_metrics(self, values: Mapping[str, float], *, step: int = 0) -> None:
        timestamp = int(time.time() * 1000)
        metrics = []
        for name, value in values.items():
            value = float(value)
            if not math.isfinite(value):
                raise ValueError(f"Metric {name!r} must be finite.")
            metrics.append(Metric(name, value, timestamp, step))
        self.client.log_batch(self.run_id, metrics=metrics, synchronous=True)

    def log_artifact(self, path: Path, *, artifact_path: str | None = None) -> None:
        self.client.log_artifact(self.run_id, str(path), artifact_path=artifact_path)

    @contextmanager
    def measure_energy(self, *, country_iso_code: str = ENERGY_COUNTRY_ISO_CODE) -> Iterator[None]:
        """Measure the enclosed computation with CodeCarbon and log it to this run.

        Wrap only the work being compared, such as a forecast or training loop, and use it
        at most once per run. If the block fails, the error propagates and no energy is
        logged. CodeCarbon suppresses its own errors, so a missing result is raised here.
        GPUs without NVML's energy counter are measured from NVML's power samples instead.
        """
        codecarbon_logger = logging.getLogger("codecarbon")
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            log_path = output / "codecarbon.log"
            handler = logging.FileHandler(log_path, encoding="utf-8")
            codecarbon_logger.addHandler(handler)
            try:
                tracker = _emissions_tracker(output, country_iso_code)
                sampler = _gpu_power_sampler()
                tracker.start()
                try:
                    if sampler is not None:
                        sampler.start()
                    try:
                        yield
                    finally:
                        if sampler is not None:
                            sampler.stop()
                finally:
                    tracker.stop()
            finally:
                codecarbon_logger.removeHandler(handler)
                handler.close()
            data = getattr(tracker, "final_emissions_data", None)
            if data is None:
                raise RuntimeError("CodeCarbon reported no measurement; see its log output.")
            log = log_path.read_text(encoding="utf-8")
            methods = dict(_TRACKING_METHOD.findall(log))
            metrics = {
                "energy.kwh": data.energy_consumed,
                "energy.cpu_kwh": data.cpu_energy,
                "energy.gpu_kwh": data.gpu_energy,
                "energy.ram_kwh": data.ram_energy,
                "energy.emissions_kg": data.emissions,
                "energy.duration_seconds": data.duration,
            }
            if sampler is not None:
                # CodeCarbon counted 0 kWh for these GPUs; replace it with the samples.
                metrics["energy.gpu_kwh"] = sampler.kwh
                metrics["energy.kwh"] = data.energy_consumed - data.gpu_energy + sampler.kwh
                metrics["energy.emissions_kg"] = data.emissions + _grid_emissions_kg(
                    sampler.kwh, data.country_iso_code
                )
                methods["GPU"] = GPU_SAMPLES_METHOD
            elif _GPU_COUNTER_UNSUPPORTED in log:
                # Record the gap instead of a zero that reads as measured.
                del metrics["energy.gpu_kwh"]
                methods["GPU"] = GPU_UNMEASURED_METHOD
            self.log_metrics(metrics)
            params = {
                "energy.country_iso_code": data.country_iso_code,
                "energy.tracking_mode": data.tracking_mode,
                "energy.codecarbon_version": data.codecarbon_version,
                "energy.cpu_model": data.cpu_model,
                "energy.gpu_model": data.gpu_model or "none",
                **{
                    f"energy.{kind.lower()}_method": method.strip()
                    for kind, method in methods.items()
                },
            }
            self.client.log_batch(
                self.run_id,
                params=[Param(key, str(value)) for key, value in params.items()],
                synchronous=True,
            )
            for path in (output / "emissions.csv", log_path):
                if path.exists():
                    self.log_artifact(path, artifact_path="energy")


@contextmanager
def tracked_run(
    *,
    name: str,
    config: Mapping[str, Any],
    data_version: str,
    model_revision: str,
    experiment_name: str | None = None,
    tracking_uri: str | None = None,
    purpose: Literal["evaluation", "smoke"] = "evaluation",
) -> Iterator[TrackedRun]:
    """Record configuration and versions once; propagate failures to the caller.

    Config must contain only JSON-serializable experiment settings, never secrets.
    The caller supplies the version of the data/model it actually used; this helper
    does not download data, select splits, or load a model.
    """
    if not data_version.strip() or not model_revision.strip():
        raise ValueError("Provide explicit data_version and model_revision values.")
    config = dict(config)
    params = [
        Param(
            key,
            value
            if isinstance(value, str)
            else json.dumps(value, sort_keys=True, allow_nan=False),
        )
        for key, value in config.items()
    ]
    provenance = _provenance(data_version, model_revision)
    client = tracking_client(tracking_uri)
    experiment = experiment_name or os.environ.get("MLFLOW_EXPERIMENT_NAME", "MLOps_Endesa")
    run = client.create_run(
        _experiment_id(client, experiment),
        run_name=name,
        tags={
            "mlflow.source.git.commit": provenance["git_commit"],
            "git.dirty": str(provenance["git_dirty"]).lower(),
            "data.version": data_version,
            "model.revision": model_revision,
            "uv.lock.sha256": provenance["uv_lock_sha256"],
            "run.purpose": purpose,
        },
    )
    recorder = TrackedRun(client, run.info.run_id)
    try:
        client.log_batch(recorder.run_id, params=params, synchronous=True)
        client.log_dict(recorder.run_id, config, "config.json")
        client.log_dict(recorder.run_id, provenance, "provenance.json")
        client.log_dict(recorder.run_id, _environment(), "environment.json")
        recorder.log_artifact(PROJ_ROOT / "uv.lock")
        start = time.perf_counter()
        yield recorder
        recorder.log_metrics({"run_body_seconds": time.perf_counter() - start})
    except BaseException as error:
        try:
            client.set_terminated(recorder.run_id, status="FAILED")
        except MlflowException:
            error.add_note(f"Could not mark MLflow run {recorder.run_id} as FAILED.")
        raise
    else:
        client.set_terminated(recorder.run_id, status="FINISHED")
