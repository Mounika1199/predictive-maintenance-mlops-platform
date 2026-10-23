"""Tests for the prediction service.

They run the real web service in-process with FastAPI's TestClient -- no
running server, no network, and no MLflow or Docker: a small stand-in champion,
trained here from the real training data, is handed straight to create_app().

The most important test: for all 100 test engines, raw readings sent through
the WEB SERVICE give exactly the same answers as the offline pipeline that
evaluate.py uses. That is the permanent guard against training/serving skew.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.data.load_cmapss import load_cmapss, load_config
from src.features.labels import last_cycle_per_engine
from src.models.training import build_model, predict_rul
from src.serving.app import create_app
from src.serving.checks import InvalidReadings, check_history
from src.serving.predictor import RulPredictor

PROJECT_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Setup: a stand-in champion and the service around it
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def predictor(real) -> RulPredictor:
    """A small forest trained on the real training split, with the real
    scaling numbers -- the same pair the real champion is. One core
    (n_jobs=1), so predictions are exactly repeatable."""
    train = real["scaled"]["train"]
    model = build_model("random_forest", {"n_estimators": 30, "min_samples_leaf": 5, "n_jobs": 1})
    model.fit(train[real["columns"]], train["rul"])

    feature_stats = {
        "sensors": real["sensors"],
        "rolling_window": 5,
        "feature_columns": real["columns"],
        "scaling": real["stats"],
    }
    return RulPredictor(model=model, feature_stats=feature_stats, model_version="test", run_id="test-run")


@pytest.fixture(scope="module")
def client(predictor):
    # `with` runs the service's startup (lifespan): it gets our predictor
    # instead of contacting MLflow, and loads the validation contract.
    with TestClient(create_app(predictor)) as test_client:
        yield test_client


@pytest.fixture(scope="module")
def raw_test():
    """The test engines' readings exactly as recorded -- what a client sends."""
    return load_cmapss(load_config()).test


def readings_of(raw: pd.DataFrame, engine: int) -> pd.DataFrame:
    return raw[raw["unit_number"] == engine].drop(columns="unit_number")


def post(client, frame: pd.DataFrame, **extra):
    return client.post("/predict", json={"readings": frame.to_dict(orient="records"), **extra})


# ---------------------------------------------------------------------------
# THE test: the web route and the offline pipeline agree
# ---------------------------------------------------------------------------


def test_web_service_matches_the_offline_pipeline_for_every_test_engine(client, predictor, raw_test, real) -> None:
    """Route 1: raw readings -> HTTP -> service. Route 2: the pipeline's
    prepared features -> model (what evaluate.py does). Any difference would be
    training/serving skew: every prediction slightly wrong, nothing crashing."""
    offline = last_cycle_per_engine(real["scaled"]["test"])
    expected = predict_rul(predictor.model, offline[predictor.feature_columns])

    served = [
        post(client, readings_of(raw_test, engine)).json()["remaining_useful_life"]
        for engine in offline["unit_number"]
    ]

    assert len(served) == 100
    np.testing.assert_allclose(served, expected, rtol=0, atol=1e-9)


# ---------------------------------------------------------------------------
# The three addresses
# ---------------------------------------------------------------------------


def test_health(client) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model_version": "test"}


def test_model_info_describes_how_inputs_are_prepared(client) -> None:
    info = client.get("/model").json()
    assert info["model_version"] == "test"
    assert info["rolling_window"] == 5
    assert len(info["sensors"]) == 15
    assert info["n_features"] == 46


def test_a_prediction_names_the_engine_cycle_and_model(client, raw_test) -> None:
    response = post(client, readings_of(raw_test, 1), engine_id="engine-1")

    assert response.status_code == 200
    answer = response.json()
    assert answer["engine_id"] == "engine-1"  # echoed unchanged
    assert answer["at_cycle"] == 31  # the latest cycle sent
    assert answer["model_name"] == "rul-fd001"
    assert answer["model_version"] == "test"  # always says which model answered
    assert answer["remaining_useful_life"] >= 0
    assert answer["warnings"] == []


def test_the_last_window_of_cycles_gives_the_same_answer_as_the_full_history(client, raw_test) -> None:
    engine = readings_of(raw_test, 1)
    full = post(client, engine).json()["remaining_useful_life"]
    window = post(client, engine.tail(5)).json()["remaining_useful_life"]
    assert window == pytest.approx(full, abs=1e-9)


# ---------------------------------------------------------------------------
# Layer 1: the request's shape (refused by FastAPI before our code runs)
# ---------------------------------------------------------------------------


def test_a_misspelled_field_is_refused(client, raw_test) -> None:
    readings = readings_of(raw_test, 1).tail(5).to_dict(orient="records")
    readings[0]["sensor_22"] = readings[0].pop("sensor_21")
    assert client.post("/predict", json={"readings": readings}).status_code == 422


def test_text_instead_of_a_number_is_refused(client, raw_test) -> None:
    readings = readings_of(raw_test, 1).tail(5).to_dict(orient="records")
    readings[0]["sensor_4"] = "hot"
    assert client.post("/predict", json={"readings": readings}).status_code == 422


def test_an_empty_request_is_refused(client) -> None:
    assert client.post("/predict", json={"readings": []}).status_code == 422


# ---------------------------------------------------------------------------
# Layer 2: the history (refused -- the features would be silently wrong)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "make_broken, expected_message",
    [
        (lambda e: e.tail(3), "send at least the last 5"),  # too little history
        (lambda e: e.tail(6).drop(e.tail(6).index[3]), "followed by cycle"),  # a gap
        (lambda e: pd.concat([e.tail(5), e.tail(1)]), "followed by cycle"),  # a repeat
        (lambda e: e.tail(5).iloc[[1, 0, 2, 3, 4]], "followed by cycle"),  # out of order
    ],
    ids=["too-little-history", "gap", "repeat", "out-of-order"],
)
def test_broken_history_is_refused(client, raw_test, make_broken, expected_message) -> None:
    response = post(client, make_broken(readings_of(raw_test, 1)))

    assert response.status_code == 422
    assert expected_message in response.json()["detail"]


def test_a_young_engine_with_its_whole_history_is_accepted(client, raw_test) -> None:
    """Fewer than 5 cycles is fine if they start at cycle 1: the engine really
    is that young, and training saw exactly this at every engine's start."""
    response = post(client, readings_of(raw_test, 1).head(3))
    assert response.status_code == 200


# ---------------------------------------------------------------------------
# Layer 3: unusual values (answered, with warnings)
# ---------------------------------------------------------------------------


def test_a_reading_beyond_the_training_range_is_answered_with_a_warning(client, raw_test) -> None:
    readings = readings_of(raw_test, 1).tail(5).copy()
    readings.loc[readings.index[-1], "sensor_4"] = 1500.0  # training range ~1376-1447

    response = post(client, readings)

    assert response.status_code == 200  # still answered...
    warnings = response.json()["warnings"]
    assert len(warnings) == 1  # ...but flagged
    assert "sensor_4" in warnings[0]


def test_a_change_the_model_cannot_see_is_still_flagged(client, raw_test) -> None:
    """sensor_1 was dropped as constant, so the model cannot see it: the
    prediction does not move. The warning is the ONLY sign that something --
    a recalibration, a different data source -- has changed."""
    normal = readings_of(raw_test, 1).tail(5)
    recalibrated = normal.assign(sensor_1=520.0)

    before = post(client, normal).json()
    after = post(client, recalibrated).json()

    assert after["remaining_useful_life"] == before["remaining_useful_life"]
    assert any("sensor_1" in w for w in after["warnings"])


# ---------------------------------------------------------------------------
# check_history() on its own
# ---------------------------------------------------------------------------


def cycles(*numbers: int) -> pd.DataFrame:
    return pd.DataFrame({"time_in_cycles": list(numbers)})


@pytest.mark.parametrize(
    "numbers",
    [(10, 11, 12, 13, 14), (1, 2), (1,), (7, 8, 9, 10, 11, 12)],
    ids=["exactly-the-window", "young-engine", "first-cycle-only", "more-than-the-window"],
)
def test_valid_histories_pass(numbers) -> None:
    check_history(cycles(*numbers), window=5)  # must not raise


@pytest.mark.parametrize(
    "numbers",
    [(12, 13, 14), (10, 11, 13, 14, 15), (10, 11, 11, 12, 13), (11, 10, 12, 13, 14)],
    ids=["too-short", "gap", "repeat", "reordered"],
)
def test_invalid_histories_are_refused(numbers) -> None:
    with pytest.raises(InvalidReadings):
        check_history(cycles(*numbers), window=5)


# ---------------------------------------------------------------------------
# A guard on the rule: ONE copy of the feature logic
# ---------------------------------------------------------------------------


def test_serving_reuses_the_feature_code_instead_of_copying_it() -> None:
    """A tripwire. Serving must call src/features/engineering.py, never keep
    its own version of the smoothing or scaling -- two copies drift apart, and
    that drift IS training/serving skew. A simple text check, not a proof."""
    source = (PROJECT_ROOT / "src" / "serving" / "predictor.py").read_text(encoding="utf-8")

    assert "fe.add_rolling_features(" in source
    assert "fe.apply_scaler(" in source
    assert ".rolling(" not in source  # no hand-rolled smoothing
