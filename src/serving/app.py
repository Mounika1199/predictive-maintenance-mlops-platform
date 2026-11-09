"""The prediction web service.

Start it (with the MLflow server running: `docker compose up -d`):

    uvicorn src.serving.app:app --host 127.0.0.1 --port 8000

then open http://127.0.0.1:8000/docs for an interactive page where you can try
every address from the browser.

    GET  /health    "are you alive?" -- for monitoring tools
    GET  /model     which model version is answering, and how it prepares data
    POST /predict   an engine's raw readings in -> remaining useful life out
    GET  /metrics   request counts, response times, ... -- read by Prometheus

Requests are checked in layers (see src/serving/checks.py):
  shape    -- field names and types (schemas.py); refused with 422
  history  -- consecutive cycles, enough of them; refused with 422
  values   -- outside the validation contract; ANSWERED, with warnings

The champion is loaded ONCE, when the service starts -- not on every request.
A promotion in the registry is therefore picked up on the next restart: until
then, the service keeps answering with the version it loaded (and says which,
in every response).

Every answered prediction is also written to the diary
(src/monitoring/prediction_log.py), and every request is counted and timed
for Prometheus (src/monitoring/metrics.py).
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, HTTPException, Request, Response

from src.data.load_cmapss import PROJECT_ROOT, load_config
from src.models.tracking import check_server, resolve_tracking_uri
from src.monitoring.metrics import CONTENT_TYPE_LATEST, ServiceMetrics
from src.monitoring.prediction_log import PredictionLog, default_log_dir
from src.serving.checks import contract_warnings, load_contract
from src.serving.predictor import RulPredictor, load_champion
from src.serving.schemas import ModelInfo, PredictionRequest, PredictionResponse

TRAINING_CONFIG = PROJECT_ROOT / "configs" / "training.yaml"


def create_app(
    predictor: RulPredictor | None = None,
    prediction_log: PredictionLog | None = None,
) -> FastAPI:
    """Build the service.

    Normally the champion is fetched from MLflow at startup. Passing a
    ``predictor`` instead skips that -- which is how the tests run the service
    without MLflow or Docker. Likewise, the diary goes to the folder in
    configs/monitoring.yaml unless a ``prediction_log`` is passed (the tests
    pass a temporary one, so they never write into the real diary).
    """
    mlflow_config = load_config(TRAINING_CONFIG)["mlflow"]
    model_name = mlflow_config["registered_model_name"]
    metrics = ServiceMetrics()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Runs once at startup, before the first request is accepted.
        if predictor is None:
            tracking_uri = resolve_tracking_uri(mlflow_config["tracking_uri"])
            check_server(tracking_uri)  # a clear message if MLflow is down
            app.state.predictor = load_champion(tracking_uri, model_name)
        else:
            app.state.predictor = predictor
        # The validation contract, from the repository -- the same one the data
        # pipeline enforces. Built once, not per request.
        app.state.contract = load_contract()
        app.state.prediction_log = prediction_log or PredictionLog(default_log_dir())
        metrics.model.labels(model_name, app.state.predictor.model_version).set(1)
        yield

    app = FastAPI(
        title="Remaining Useful Life service",
        description="Predicts how many cycles a turbofan engine has left, "
                    "from its recent raw sensor readings (NASA C-MAPSS FD001).",
        version="1.0.0",
        lifespan=lifespan,
    )

    @app.middleware("http")
    async def count_and_time(request: Request, call_next):
        # Runs around EVERY request: start a stopwatch, let the request through,
        # then count it (by route and status code) and record how long it took.
        start = time.perf_counter()
        status = 500  # if the request crashes, it is counted as a server error
        try:
            response = await call_next(request)
            status = response.status_code
            return response
        finally:
            # FastAPI notes the matched route on the request; none means an
            # unknown address (404), all of which share one label.
            route = request.scope.get("route")
            path = route.path if route is not None else "unmatched"
            metrics.requests.labels(request.method, path, str(status)).inc()
            metrics.duration.labels(path).observe(time.perf_counter() - start)

    @app.get("/metrics", include_in_schema=False)
    def metrics_page() -> Response:
        return Response(metrics.page(), media_type=CONTENT_TYPE_LATEST)

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "model_version": app.state.predictor.model_version}

    @app.get("/model", response_model=ModelInfo)
    def model_info() -> ModelInfo:
        p: RulPredictor = app.state.predictor
        return ModelInfo(
            model_name=model_name,
            model_version=p.model_version,
            run_id=p.run_id,
            rolling_window=p.window,
            sensors=p.sensors,
            n_features=len(p.feature_columns),
        )

    @app.post("/predict", response_model=PredictionResponse)
    def predict(request: PredictionRequest) -> PredictionResponse:
        p: RulPredictor = app.state.predictor
        readings = pd.DataFrame([r.model_dump() for r in request.readings])
        try:
            rul = p.predict(readings)
        except ValueError as exc:  # broken history: gaps, repeats, too few cycles
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        warnings = contract_warnings(readings, app.state.contract)
        if warnings:
            metrics.warnings.inc()

        # The diary: the NEWEST cycle only (earlier ones were written by
        # earlier requests), and the answer given.
        written = app.state.prediction_log.write({
            "engine_id": request.engine_id,
            **request.readings[-1].model_dump(),
            "remaining_useful_life": rul,
            "model_name": model_name,
            "model_version": p.model_version,
            "warnings": warnings,
        })
        if not written:
            metrics.diary_failures.inc()

        return PredictionResponse(
            engine_id=request.engine_id,
            remaining_useful_life=rul,
            at_cycle=int(readings["time_in_cycles"].iloc[-1]),
            model_name=model_name,
            model_version=p.model_version,
            warnings=warnings,
        )

    return app


# What uvicorn runs. Nothing is loaded at import time -- only when the server
# actually starts (see `lifespan`).
app = create_app()
