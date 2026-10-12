"""Tests for building, saving and using models."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

from src.models.metrics import rmse
from src.models.training import build_model, load_feature_columns, predict_rul

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def tiny_data() -> tuple[pd.DataFrame, pd.Series]:
    """40 rows where the answer is roughly 3 x a."""
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"a": rng.uniform(0, 10, 40), "b": rng.uniform(0, 10, 40)})
    y = 3 * X["a"] + rng.normal(0, 0.5, 40)
    return X, y


# ---------------------------------------------------------------------------
# Building models
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name, expected_type",
    [
        ("baseline", DummyRegressor),
        ("linear", LinearRegression),
        ("random_forest", RandomForestRegressor),
    ],
)
def test_build_model_returns_the_right_kind(name: str, expected_type: type) -> None:
    assert isinstance(build_model(name), expected_type)


def test_unknown_model_name_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown model"):
        build_model("magic_model")


# ---------------------------------------------------------------------------
# Predicting
# ---------------------------------------------------------------------------


def test_predictions_are_never_negative() -> None:
    """'Minus 12 cycles left' means nothing, so predict_rul clips it to 0."""

    class SaysNegative:
        def predict(self, features):
            return np.array([-12.0, 5.0])

    assert predict_rul(SaysNegative(), None).tolist() == [0.0, 5.0]


# ---------------------------------------------------------------------------
# Reproducibility and saving
# ---------------------------------------------------------------------------


def test_same_random_state_gives_the_same_forest() -> None:
    X, y = tiny_data()
    params = {"n_estimators": 10}

    a = build_model("random_forest", params, random_state=42).fit(X, y)
    b = build_model("random_forest", params, random_state=42).fit(X, y)

    np.testing.assert_array_equal(a.predict(X), b.predict(X))


def test_saved_model_loads_back_identical(tmp_path: Path) -> None:
    """Train once, save, load somewhere else: the predictions must not change."""
    X, y = tiny_data()
    model = build_model("random_forest", {"n_estimators": 10}).fit(X, y)

    path = tmp_path / "model.joblib"
    joblib.dump(model, path)
    loaded = joblib.load(path)

    np.testing.assert_array_equal(model.predict(X), loaded.predict(X))


def test_feature_list_is_read_in_its_saved_order(tmp_path: Path) -> None:
    """The model needs its columns in the exact order it learned them."""
    path = tmp_path / "feature_stats.json"
    path.write_text(json.dumps({"feature_columns": ["b", "a", "c"]}), encoding="utf-8")

    assert load_feature_columns(path) == ["b", "a", "c"]


# ---------------------------------------------------------------------------
# On the real data
# ---------------------------------------------------------------------------


def test_real_models_clearly_beat_the_baseline(real) -> None:
    """The same safety gate train.py applies, as a permanent test.

    If a change ever broke the features or the training code, the models would
    fall back towards guessing the average -- and this would catch it.
    A small forest (30 trees) keeps the test quick.
    """
    train, val = real["scaled"]["train"], real["scaled"]["val"]
    X, y = train[real["columns"]], train["rul"]

    def val_rmse(model):
        return rmse(val["rul"], predict_rul(model, val[real["columns"]]))

    baseline = val_rmse(build_model("baseline").fit(X, y))
    linear = val_rmse(build_model("linear").fit(X, y))
    forest = val_rmse(
        build_model("random_forest", {"n_estimators": 30, "min_samples_leaf": 5, "n_jobs": -1}).fit(X, y)
    )

    assert linear < 0.7 * baseline
    assert forest < 0.7 * baseline


def test_training_script_never_reads_the_test_set() -> None:
    """A guard on the RULE, not just the code.

    Choosing a model must happen on validation only. This is a simple text
    check -- it looks for the test file's name or the split name "test" in
    train.py -- so it is a tripwire, not a proof. But it will catch the most
    likely way the rule gets broken: someone adding a test-set read "just to
    have a look".
    """
    source = (PROJECT_ROOT / "scripts" / "train.py").read_text(encoding="utf-8")

    assert "test_features" not in source
    assert '"test"' not in source
    assert "'test'" not in source
