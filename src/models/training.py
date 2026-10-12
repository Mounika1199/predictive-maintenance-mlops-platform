"""Building models and making predictions.

This lives in src/ rather than in a script for one reason: the SAME prediction
code must be used everywhere -- when we score on validation, when we score on
the test set, and later when a real system asks for predictions. If each place
had its own copy, they would slowly drift apart.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

MODEL_NAMES = ("baseline", "linear", "random_forest")


def build_model(name: str, params: dict | None = None, random_state: int = 42):
    """Create an untrained model by name.

    * ``baseline``      -- always predicts the training average. The bar to beat.
    * ``linear``        -- a weighted sum of the features (a "recipe").
    * ``random_forest`` -- many yes/no flowcharts whose answers are averaged.

    ``random_state`` fixes the random choices the forest makes while it learns,
    so training twice on the same data gives exactly the same model.
    """
    params = dict(params or {})

    if name == "baseline":
        return DummyRegressor(strategy="mean")
    if name == "linear":
        return LinearRegression(**params)
    if name == "random_forest":
        return RandomForestRegressor(random_state=random_state, **params)

    raise ValueError(f"unknown model '{name}'; choose one of {MODEL_NAMES}")


def predict_rul(model, features) -> np.ndarray:
    """Predict remaining life, never below zero.

    Some models (linear regression in particular) can produce negative numbers
    for engines that look very worn. "Minus 12 cycles left" means nothing, so we
    clip to 0. This is part of the model's behaviour, which is why it lives here
    and not in whichever script happens to call it.
    """
    return np.clip(model.predict(features), 0, None)


def load_feature_columns(stats_path: Path | str) -> list[str]:
    """The exact, ordered list of feature columns, from feature_stats.json.

    Read from the file written by build_features.py rather than typed out again,
    so there is only ONE place that defines what the model sees.
    """
    with Path(stats_path).open("r", encoding="utf-8") as handle:
        return json.load(handle)["feature_columns"]
