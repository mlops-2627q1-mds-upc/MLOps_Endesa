"""Bounded sequential HTTP replay; inference stays in the forecast service."""

from contextlib import nullcontext
import json
import math
from pathlib import Path
import time

import httpx
import numpy as np

from src.api.schemas import ForecastResponse
from src.evaluation import seasonal_forecast
from src.monitoring.journal import ReplayJournal
from src.monitoring.source import ReplaySource
from src.monitoring.types import ModelIdentity, ReplayConfig, ReplayCounts, canonical_json, sha256

RETRYABLE_STATUS = {408, 425, 429, 500, 502, 503, 504}
MAX_RESPONSE_BYTES = 131072


class ReplayRequestError(RuntimeError):
    """An HTTP attempt failed without changing the forecast cursor."""


def _response(client, endpoint: str, payload: dict, timeout: float, deadline: float):
    with client.stream("POST", endpoint + "/forecast", json=payload, timeout=timeout) as response:
        if response.status_code != 200:
            return response.status_code, None
        body = bytearray()
        for chunk in response.iter_bytes():
            body.extend(chunk)
            if len(body) > MAX_RESPONSE_BYTES or time.monotonic() >= deadline:
                raise ReplayRequestError("Response exceeded the replay size/time budget")
        try:
            return 200, ForecastResponse.model_validate(json.loads(body))
        except (ValueError, TypeError) as error:
            raise ReplayRequestError("Invalid forecast response schema") from error


class Replay:
    """Own one writer session, with finite attempts shared across all restarts."""

    def __init__(
        self,
        journal: Path,
        source: ReplaySource,
        config: ReplayConfig,
        *,
        client: httpx.Client | None = None,
    ):
        self.journal = ReplayJournal(journal, source, config)
        self.source, self.config, self.client = source, config, client

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.close()

    def close(self) -> None:
        """Release the journal writer lock; an injected HTTP client belongs to its caller."""
        self.journal.close()

    def release(self, through_position: int) -> ReplayCounts:
        """Advance observation availability without skipping unfinished forecasts."""
        self.journal.advance_clock(through_position)
        return self.journal.counts()

    def _forecast(self, client, index: int, deadline: float) -> None:
        state, origin = self.config.request_at(index)
        self.journal.advance_clock(origin)
        history = self.source.history(state, origin)
        payload = {"state": state, "history": history}
        baseline = seasonal_forecast(np.asarray([history]), 48, 48)[0].tolist()
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise ReplayRequestError("Invocation time budget exhausted before HTTP")
            attempt = self.journal.start_attempt(index)
            try:
                status, response = _response(
                    client,
                    self.config.endpoint,
                    payload,
                    min(self.config.timeout_seconds, remaining),
                    deadline,
                )
            except httpx.RequestError:
                self.journal.fail_attempt(attempt, "transport_error")
                status, response = None, None
            except ReplayRequestError:
                self.journal.fail_attempt(attempt, "invalid_response")
                raise
            if status == 200:
                if (
                    response.state != state
                    or ModelIdentity.from_response(response) != self.config.model_identity
                ):
                    self.journal.fail_attempt(attempt, "identity_mismatch")
                    raise ReplayRequestError(
                        "Forecast state/model/settings differ from the declared run"
                    )
                self.journal.commit_forecast(
                    index, attempt, sha256(canonical_json(payload)), response, baseline
                )
                return
            if status is not None:
                self.journal.fail_attempt(attempt, f"http_{status}")
                if status not in RETRYABLE_STATUS:
                    raise ReplayRequestError(f"Nonretryable forecast HTTP status {status}")
            remaining = deadline - time.monotonic()
            if remaining <= self.config.retry_delay_seconds:
                raise ReplayRequestError("Invocation time budget exhausted after failed HTTP")
            time.sleep(self.config.retry_delay_seconds)

    def run(self, *, max_forecasts: int = 1, max_seconds: float = 60) -> ReplayCounts:
        """Commit at most max_forecasts requests, with finite per-call timeouts.

        The deadline is checked between requests and response chunks. HTTPX also
        limits connect/read/write/pool waits; this is not a hard process deadline.
        Final observations require an explicit release(), including after restart.
        """
        if type(max_forecasts) is not int or not 1 <= max_forecasts <= 1000:
            raise ValueError("max_forecasts must be 1..1000")
        if (
            isinstance(max_seconds, bool)
            or not math.isfinite(max_seconds)
            or not 0 < max_seconds <= 3600
        ):
            raise ValueError("max_seconds must be positive and at most 3600")
        deadline = time.monotonic() + max_seconds
        context = (
            nullcontext(self.client)
            if self.client is not None
            else httpx.Client(trust_env=False, follow_redirects=False)
        )
        with context as client:
            for _ in range(max_forecasts):
                index = self.journal.next_index()
                if index == self.config.expected_forecasts or time.monotonic() >= deadline:
                    break
                self._forecast(client, index, deadline)
        return self.journal.counts()
