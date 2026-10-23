"""Turning one engine's raw sensor readings into a remaining-life prediction.

This is the ONLY place where serving prepares data, and it prepares it exactly
as training did -- by reusing the same code and the same numbers:

* the feature code from src/features/engineering.py (smoothing, the cycle copy,
  scaling) -- never a second copy of it;
* the champion's own feature_stats.json from MLflow -- the sensors, the rolling
  window and every scaling number it was trained with;
* predict_rul() from src/models/training.py -- including the clip at zero.

Any difference here would be TRAINING/SERVING SKEW: every prediction slightly
wrong, and nothing crashes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mlflow
import mlflow.artifacts
import mlflow.sklearn
import pandas as pd
from mlflow import MlflowClient

from src.features import engineering as fe
from src.models.tracking import CHAMPION
from src.models.training import predict_rul
from src.serving.checks import check_history


@dataclass(frozen=True)
class RulPredictor:
    """A model together with the exact numbers needed to prepare its inputs."""

    model: Any
    feature_stats: dict  # the champion's training-time feature_stats.json
    model_version: str
    run_id: str

    @property
    def sensors(self) -> list[str]:
        return self.feature_stats["sensors"]

    @property
    def window(self) -> int:
        return self.feature_stats["rolling_window"]

    @property
    def feature_columns(self) -> list[str]:
        return self.feature_stats["feature_columns"]

    def predict(self, readings: pd.DataFrame) -> float:
        """Remaining useful life, in cycles, at the LAST of the given cycles.

        ``readings`` holds ONE engine's raw cycles, oldest first:
        ``time_in_cycles`` plus the sensor columns, exactly as recorded.

        The same steps as build_features.py, using the champion's numbers:
        1. rolling mean and std over the last `window` cycles, per sensor;
        2. the scaled copy of the engine's age ("cycle");
        3. scaling with the champion's training-time means and spreads;
        then predict from the last row only -- "here is the engine now".

        Refuses (InvalidReadings) history that would make the features differ
        from training: gaps, repeats, or fewer than `window` cycles.
        """
        check_history(readings, self.window)

        frame = readings.copy()
        # The feature code groups rows by engine. A request is one engine, so
        # it gets a placeholder id; the number itself is never a model input.
        frame[fe.ID_COLUMN] = 0

        frame = fe.add_rolling_features(frame, self.sensors, self.window)
        frame = fe.add_cycle_feature(frame)
        frame = fe.apply_scaler(frame, self.feature_stats["scaling"])

        latest = frame[self.feature_columns].tail(1)
        return float(predict_rul(self.model, latest)[0])


def load_champion(tracking_uri: str, registered_name: str) -> RulPredictor:
    """Fetch the current champion and its feature_stats.json from MLflow.

    The alias is resolved ONCE, then the model and its numbers are both loaded
    by that version NUMBER. If someone promoted a new champion halfway through
    loading, we still get a matching pair -- never version 4's model with
    version 3's scaling numbers.
    """
    client = MlflowClient(tracking_uri)
    version = client.get_model_version_by_alias(registered_name, CHAMPION)

    mlflow.set_tracking_uri(tracking_uri)
    model = mlflow.sklearn.load_model(f"models:/{registered_name}/{version.version}")
    stats = mlflow.artifacts.load_dict(f"runs:/{version.run_id}/feature_stats.json")

    return RulPredictor(
        model=model,
        feature_stats=stats,
        model_version=str(version.version),
        run_id=version.run_id,
    )
