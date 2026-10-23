"""The prediction web service.

Start it (with the MLflow server running: `docker compose up -d`):

    uvicorn src.serving.app:app --host 127.0.0.1 --port 8000

then open http://127.0.0.1:8000/docs for an interactive page where you can try
every address from the browser.

    GET  /health    "are you alive?" -- for monitoring tools
    GET  /model     which model version is answering, and how it prepares data
    POST /predict   an engine's raw readings in -> remaining useful life out

Requests are checked in layers (see src/serving/checks.py):
  shape    -- field names and types (schemas.py); refused with 422
  history  -- consecutive cycles, enough of them; refused with 422
  values   -- outside the validation contract; ANSWERED, with warnings

The champion is loaded ONCE, when the service starts -- not on every request.
A promotion in the registry is therefore picked up on the next restart: until
then, the service keeps answering with the version it loaded (and says which,
in every response).
"""

from __future__ import annotations

from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, HTTPException

from src.data.load_cmapss import PROJECT_ROOT, load_config
from src.models.tracking import check_server, resolve_tracking_uri
from src.serving.checks import contract_warnings, load_contract
from src.serving.predictor import RulPredictor, load_champion
from src.serving.schemas import ModelInfo, PredictionRequest, PredictionResponse

TRAINING_CONFIG = PROJECT_ROOT / "configs" / "training.yaml"


def create_app(predictor: RulPredictor | None = None) -> FastAPI:
    """Build the service.

    Normally the champion is fetched from MLflow at startup. Passing a
    ``predictor`` instead skips that -- which is how the tests run the service
    without MLflow or Docker.
    """
    mlflow_config = load_config(TRAINING_CONFIG)["mlflow"]
    model_name = mlflow_config["registered_model_name"]

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
        yield

    app = FastAPI(
        title="Remaining Useful Life service",
        description="Predicts how many cycles a turbofan engine has left, "
                    "from its recent raw sensor readings (NASA C-MAPSS FD001).",
        version="1.0.0",
        lifespan=lifespan,
    )

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

        return PredictionResponse(
            engine_id=request.engine_id,
            remaining_useful_life=rul,
            at_cycle=int(readings["time_in_cycles"].iloc[-1]),
            model_name=model_name,
            model_version=p.model_version,
            warnings=contract_warnings(readings, app.state.contract),
        )

    return app


# What uvicorn runs. Nothing is loaded at import time -- only when the server
# actually starts (see `lifespan`).
app = create_app()
