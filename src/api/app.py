"""FastAPI lifecycle and thin HTTP handlers for the selected forecaster."""

from collections.abc import Callable
from contextlib import asynccontextmanager
import logging
from typing import Protocol

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from src.api.schemas import ForecastRequest, ForecastResponse
from src.api.settings import Settings
from src.modeling.predict import ChronosForecaster, ForecastConfig

LOGGER = logging.getLogger(__name__)


class Predictor(Protocol):
    """The single model operation needed by HTTP serving and test fixtures."""

    def predict(self, history: list[float]) -> dict:
        """Return quantiles and the actual loaded artifact's identity."""


def load_predictor(settings: Settings) -> Predictor:
    """Load the selected artifact once, using only local model files."""
    config = ForecastConfig.from_params(
        settings.params_path, checkpoint=settings.checkpoint, cpu_threads=settings.cpu_threads
    )
    return ChronosForecaster(config, local_files_only=True)


def create_app(
    settings: Settings | None = None,
    predictor_factory: Callable[[Settings], Predictor] = load_predictor,
) -> FastAPI:
    """Create an app without loading weights until its lifespan starts."""
    settings = settings if settings is not None else Settings.from_env()

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        try:
            application.state.predictor = predictor_factory(settings)
            if application.state.predictor is None:
                raise RuntimeError("Predictor factory returned no model")
        except Exception as error:  # pylint: disable=broad-exception-caught
            LOGGER.error("Forecast model initialization failed (%s)", type(error).__name__)
            raise RuntimeError("Forecast model initialization failed") from error
        try:
            yield
        finally:
            application.state.predictor = None

    application = FastAPI(title="Endesa forecast API", lifespan=lifespan)
    application.state.predictor = None

    @application.exception_handler(RequestValidationError)
    async def invalid_request(_request: Request, error: RequestValidationError):
        # Do not echo request values, including NaN/Infinity or arbitrary extra fields.
        details = [{key: item[key] for key in ("loc", "msg", "type")} for item in error.errors()]
        return JSONResponse(status_code=422, content={"detail": details})

    @application.get("/health")
    async def health():
        """Report process health without invoking or waiting for the model."""
        return {"status": "ok"}

    @application.get("/ready", responses={503: {"description": "Model unavailable"}})
    async def ready():
        """Report whether this lifecycle loaded a predictor successfully."""
        if application.state.predictor is None:
            raise HTTPException(status_code=503, detail="Model unavailable")
        return {"status": "ready"}

    @application.post(
        "/forecast",
        response_model=ForecastResponse,
        responses={
            503: {"description": "Model or inference unavailable"},
            500: {"description": "Invalid forecast output"},
        },
    )
    def forecast(payload: ForecastRequest):
        """Run model work in FastAPI's thread pool, leaving async probes responsive."""
        predictor = application.state.predictor
        if predictor is None:
            raise HTTPException(status_code=503, detail="Model unavailable")
        try:
            prediction = predictor.predict(payload.history)
        except Exception as error:  # pylint: disable=broad-exception-caught
            LOGGER.error("Forecast inference failed")
            raise HTTPException(status_code=503, detail="Forecast unavailable") from error
        try:
            return ForecastResponse.model_validate({**prediction, "state": payload.state})
        except (TypeError, ValueError) as error:
            LOGGER.error("Forecast output validation failed")
            raise HTTPException(status_code=500, detail="Invalid forecast output") from error

    return application
