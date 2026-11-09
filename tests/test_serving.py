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
from prometheus_client.parser import text_string_to_metric_families

from src.data.load_cmapss import load_cmapss, load_config
from src.features.labels import last_cycle_per_engine
from src.models.training import build_model, predict_rul
from src.monitoring.prediction_log import PredictionLog, read_prediction_log
from src.serving.app import create_app
from src.serving.checks import InvalidReadings, check_history
from src.serving.predictor import RulPredictor, use_one_core
from src.serving.schemas import MEASUREMENTS

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
def client(predictor, tmp_path_factory):
    # `with` runs the service's startup (lifespan): it gets our predictor
    # instead of contacting MLflow, and loads the validation contract. Its
    # diary goes to a temporary folder, never into the real logs/.
    diary = PredictionLog(tmp_path_factory.mktemp("diary"))
    with TestClient(create_app(predictor, diary)) as test_client:
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
# The diary: what the service writes down for monitoring
# ---------------------------------------------------------------------------


@pytest.fixture
def diary(predictor, tmp_path):
    """A fresh service with an empty diary of its own, so lines can be counted."""
    with TestClient(create_app(predictor, PredictionLog(tmp_path))) as test_client:
        yield test_client, tmp_path


def test_a_prediction_writes_one_line_with_the_newest_reading_and_the_answer(diary, raw_test) -> None:
    client, folder = diary
    engine = readings_of(raw_test, 1)  # 31 cycles
    answer = post(client, engine, engine_id="engine-1").json()

    written = read_prediction_log(folder)
    assert len(written) == 1  # one prediction, one line -- not 31
    line, newest = written.iloc[0], engine.iloc[-1]
    assert line["engine_id"] == "engine-1"
    assert line["time_in_cycles"] == 31
    assert line[MEASUREMENTS].tolist() == newest[MEASUREMENTS].tolist()  # exactly as sent
    assert line["remaining_useful_life"] == answer["remaining_useful_life"]
    assert line["model_name"] == "rul-fd001"
    assert line["model_version"] == "test"
    assert line["warnings"] == []


def test_an_engine_reporting_after_every_flight_writes_each_cycle_once(diary, raw_test) -> None:
    """A real engine would send its last 5 cycles after each flight: every
    request repeats 4 cycles already sent, but the diary records each once."""
    client, folder = diary
    engine = readings_of(raw_test, 1)
    for latest in range(5, 11):
        post(client, engine.iloc[:latest].tail(5))

    assert read_prediction_log(folder)["time_in_cycles"].tolist() == [5, 6, 7, 8, 9, 10]


def test_a_refused_request_is_not_written(diary, raw_test) -> None:
    client, folder = diary
    assert post(client, readings_of(raw_test, 1).tail(3)).status_code == 422
    assert read_prediction_log(folder).empty


def test_warnings_are_written_too(diary, raw_test) -> None:
    client, folder = diary
    readings = readings_of(raw_test, 1).tail(5).copy()
    readings.loc[readings.index[-1], "sensor_4"] = 1500.0

    post(client, readings)

    [warnings] = read_prediction_log(folder)["warnings"]
    assert len(warnings) == 1 and "sensor_4" in warnings[0]


def test_the_service_still_answers_when_the_diary_cannot_be_written(predictor, raw_test, tmp_path, caplog) -> None:
    """Monitoring must never take predictions down with it."""
    blocked = tmp_path / "a-file-not-a-folder"
    blocked.write_text("x", encoding="utf-8")

    with TestClient(create_app(predictor, PredictionLog(blocked))) as client:
        response = post(client, readings_of(raw_test, 1))

    assert response.status_code == 200
    assert "NOT written to the diary" in caplog.text


# ---------------------------------------------------------------------------
# /metrics: the numbers Prometheus collects
# ---------------------------------------------------------------------------


def metric(client, name: str, **labels) -> float:
    """One number from the /metrics page, exactly as Prometheus would read it."""
    for family in text_string_to_metric_families(client.get("/metrics").text):
        for sample in family.samples:
            if sample.name == name and all(sample.labels.get(k) == v for k, v in labels.items()):
                return sample.value
    return 0.0


def test_requests_are_counted_by_route_and_status(diary, raw_test) -> None:
    client, _ = diary
    post(client, readings_of(raw_test, 1))             # answered
    post(client, readings_of(raw_test, 1).tail(3))     # refused: too little history

    assert metric(client, "rul_http_requests_total", route="/predict", status="200") == 1
    assert metric(client, "rul_http_requests_total", route="/predict", status="422") == 1


def test_every_request_is_timed(diary, raw_test) -> None:
    client, _ = diary
    post(client, readings_of(raw_test, 1))

    assert metric(client, "rul_http_request_duration_seconds_count", route="/predict") == 1
    assert metric(client, "rul_http_request_duration_seconds_sum", route="/predict") > 0


def test_unknown_addresses_share_one_label(diary) -> None:
    """A thousand random URLs must not become a thousand counters."""
    client, _ = diary
    for address in ["/admin", "/wp-login.php", "/predict/../../etc"]:
        assert client.get(address).status_code == 404

    assert metric(client, "rul_http_requests_total", route="unmatched", status="404") == 3
    assert "wp-login" not in client.get("/metrics").text


def test_predictions_with_warnings_are_counted(diary, raw_test) -> None:
    client, _ = diary
    readings = readings_of(raw_test, 1).tail(5).copy()
    post(client, readings)                                      # normal
    readings.loc[readings.index[-1], "sensor_4"] = 1500.0
    post(client, readings)                                      # outside the training range

    assert metric(client, "rul_predictions_with_warnings_total") == 1


def test_diary_failures_are_counted(predictor, raw_test, tmp_path) -> None:
    """Monitoring the monitoring: a diary that stops saving shows up on a graph."""
    blocked = tmp_path / "a-file-not-a-folder"
    blocked.write_text("x", encoding="utf-8")

    with TestClient(create_app(predictor, PredictionLog(blocked))) as client:
        post(client, readings_of(raw_test, 1))
        assert metric(client, "rul_prediction_log_failures_total") == 1


def test_the_metrics_name_the_model_answering(diary) -> None:
    client, _ = diary
    assert metric(client, "rul_model_info", model_name="rul-fd001", model_version="test") == 1


# ---------------------------------------------------------------------------
# Speed: one processor core for single predictions
# ---------------------------------------------------------------------------


def test_one_core_gives_the_same_answers(real) -> None:
    """Trained on every core, like the champion; served on one. The core count
    changes how the work is split, not what is computed -- only rounding dust
    from adding the trees' answers in a different order."""
    train, test = real["scaled"]["train"], real["scaled"]["test"]
    model = build_model("random_forest", {"n_estimators": 30, "min_samples_leaf": 5, "n_jobs": -1})
    model.fit(train[real["columns"]], train["rul"])
    on_all_cores = model.predict(test[real["columns"]])

    served = use_one_core(model)

    assert served.get_params()["n_jobs"] == 1
    np.testing.assert_allclose(served.predict(test[real["columns"]]), on_all_cores, rtol=0, atol=1e-9)


def test_a_model_without_the_setting_is_left_alone() -> None:
    baseline = build_model("baseline", {})
    assert use_one_core(baseline) is baseline


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
