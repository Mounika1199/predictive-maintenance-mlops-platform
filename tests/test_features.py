"""Tests for feature engineering.

The most important tests here are about LEAKAGE: features secretly using
information they would not have in real life. Leakage never crashes. It produces
excellent scores and a model that fails in production -- so tests are the only
thing standing between us and it.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.data.load_cmapss import load_cmapss, load_config
from src.data.split import split_by_engine
from src.features import engineering as fe
from src.features.labels import add_test_rul, add_train_rul

CONSTANT_SENSORS = ["sensor_1", "sensor_5", "sensor_10", "sensor_16", "sensor_18", "sensor_19"]


def one_sensor(values_by_engine: dict[int, list[float]]) -> pd.DataFrame:
    """A tiny table with one sensor, e.g. {1: [1, 2, 3], 2: [50, 50]}."""
    rows = [
        {"unit_number": engine, "time_in_cycles": cycle, "sensor_a": value}
        for engine, values in values_by_engine.items()
        for cycle, value in enumerate(values, start=1)
    ]
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def real():
    """The real FD001 pipeline, run in memory: labels -> split -> features -> scale."""
    ds = load_cmapss(load_config())
    split = split_by_engine(add_train_rul(ds.train, cap=125), validation_fraction=0.2, seed=42)
    test = add_test_rul(ds.test, ds.rul_truth, cap=125)

    sensors = fe.select_sensor_columns(split.train)

    def build(frame):
        return fe.add_cycle_feature(fe.add_rolling_features(frame, sensors, window=5))

    built = {"train": build(split.train), "val": build(split.validation), "test": build(test)}
    columns = fe.feature_columns(sensors, include_cycle=True)
    stats = fe.fit_scaler(built["train"], columns)
    scaled = {name: fe.apply_scaler(frame, stats) for name, frame in built.items()}

    return {
        "split": split,
        "sensors": sensors,
        "built": built,
        "columns": columns,
        "stats": stats,
        "scaled": scaled,
    }


# ---------------------------------------------------------------------------
# Leakage 1: rolling features must never look into the future
# ---------------------------------------------------------------------------


def test_rolling_features_never_look_into_the_future(real) -> None:
    """THE most important test in this file.

    Take engine 1's readings and change ONE future cycle (cycle 150) to an
    absurd value. If the features at cycles 1-149 change in any way, they were
    peeking ahead -- using a reading that would not exist yet in real life.
    """
    sensors = real["sensors"]
    train = real["split"].train
    engine = train["unit_number"].iloc[0]

    before = fe.add_rolling_features(train, sensors, window=5)

    tampered = train.copy()
    future = (tampered["unit_number"] == engine) & (tampered["time_in_cycles"] == 150)
    tampered.loc[future, sensors] = 99_999.0
    after = fe.add_rolling_features(tampered, sensors, window=5)

    rolling = [c for c in before.columns if "_roll" in c]
    past = (train["unit_number"] == engine) & (train["time_in_cycles"] < 150)

    pd.testing.assert_frame_equal(before.loc[past, rolling], after.loc[past, rolling])

    # Make sure the tampering actually did something -- otherwise the check
    # above would pass even if we had tampered with nothing at all.
    assert not before.loc[future, rolling].equals(after.loc[future, rolling])


# ---------------------------------------------------------------------------
# Leakage 2: rolling windows must not cross from one engine into the next
# ---------------------------------------------------------------------------


def test_rolling_window_restarts_for_each_engine() -> None:
    """Engine 2's first row must not be averaged with engine 1's last rows."""
    frame = one_sensor({1: [10.0, 10.0, 10.0], 2: [50.0, 50.0, 50.0]})
    out = fe.add_rolling_features(frame, ["sensor_a"], window=5)

    engine_2_first = out[(out["unit_number"] == 2) & (out["time_in_cycles"] == 1)].iloc[0]
    assert engine_2_first["sensor_a_rollmean"] == 50.0  # not a mix of 10s and 50s
    assert engine_2_first["sensor_a_rollstd"] == 0.0


# ---------------------------------------------------------------------------
# Leakage 3: statistics come from the training split only
# ---------------------------------------------------------------------------


def test_constant_sensors_are_chosen_from_training_engines(real) -> None:
    all_sensors = [c for c in real["split"].train.columns if c.startswith("sensor_")]
    found = fe.find_constant_columns(real["split"].train, all_sensors)

    assert found == CONSTANT_SENSORS
    assert len(real["sensors"]) == 15


def test_scaling_uses_training_averages_only(real) -> None:
    """The proof from the demo, made permanent.

    Training was used to compute the averages, so its scaled mean is 0. If
    validation had been allowed to use its OWN averages, its scaled mean would
    be 0 too. It is close to 0 but clearly not 0 -- so it was not.
    """
    columns = real["columns"]
    train_means = real["scaled"]["train"][columns].mean()
    val_means = real["scaled"]["val"][columns].mean()

    assert train_means.abs().max() < 1e-9
    assert val_means.abs().max() > 0.01


def test_scaling_matches_the_formula_with_training_numbers(real) -> None:
    """(x - training mean) / training std, written out by hand."""
    stats = real["stats"]["sensor_2"]
    raw = real["built"]["val"]["sensor_2"]
    expected = (raw - stats["mean"]) / stats["std"]

    pd.testing.assert_series_equal(real["scaled"]["val"]["sensor_2"], expected)


# ---------------------------------------------------------------------------
# Correctness of the rolling numbers themselves
# ---------------------------------------------------------------------------


def test_rolling_mean_values_are_correct() -> None:
    """Window of 3 over 1,2,3,4,5,6: the first rows use what history exists."""
    out = fe.add_rolling_features(one_sensor({1: [1, 2, 3, 4, 5, 6]}), ["sensor_a"], window=3)
    assert out["sensor_a_rollmean"].tolist() == [1.0, 1.5, 2.0, 3.0, 4.0, 5.0]


def test_first_rolling_std_is_zero_not_missing() -> None:
    out = fe.add_rolling_features(one_sensor({1: [1, 2, 3]}), ["sensor_a"], window=3)
    assert out["sensor_a_rollstd"].iloc[0] == 0.0
    assert not out["sensor_a_rollstd"].isna().any()


def test_unsorted_rows_are_rejected() -> None:
    """'The last 5 rows' only means 'the last 5 cycles' if rows are in order."""
    backwards = one_sensor({1: [1, 2, 3]}).iloc[::-1]
    with pytest.raises(ValueError, match="sorted"):
        fe.add_rolling_features(backwards, ["sensor_a"], window=3)


def test_invalid_window_is_rejected() -> None:
    with pytest.raises(ValueError):
        fe.add_rolling_features(one_sensor({1: [1, 2]}), ["sensor_a"], window=0)


# ---------------------------------------------------------------------------
# What the model is (and is not) allowed to see
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("include_cycle", [True, False])
def test_feature_list_never_contains_ids_or_answers(include_cycle: bool) -> None:
    columns = fe.feature_columns(["sensor_2", "sensor_3"], include_cycle=include_cycle)

    for forbidden in ("unit_number", "time_in_cycles", "rul", "rul_uncapped"):
        assert forbidden not in columns


def test_feature_count(real) -> None:
    assert len(fe.feature_columns(real["sensors"], include_cycle=True)) == 46
    assert len(fe.feature_columns(real["sensors"], include_cycle=False)) == 45


def test_identifiers_and_answers_survive_scaling(real) -> None:
    """Scaling changes the features, never the 'who', 'when', or the answers."""
    before, after = real["built"]["test"], real["scaled"]["test"]
    for column in ("unit_number", "time_in_cycles", "rul", "rul_uncapped"):
        pd.testing.assert_series_equal(before[column], after[column])


def test_no_missing_values_anywhere(real) -> None:
    for name, frame in real["scaled"].items():
        assert not frame[real["columns"]].isna().any().any(), f"NaNs in {name}"


# ---------------------------------------------------------------------------
# The scaler refuses columns it cannot scale
# ---------------------------------------------------------------------------


def test_scaler_rejects_a_constant_column() -> None:
    frame = pd.DataFrame({"flat": [5.0, 5.0, 5.0]})
    with pytest.raises(ValueError, match="zero spread"):
        fe.fit_scaler(frame, ["flat"])


def test_scaler_rejects_real_sensor_5_despite_rounding_dust() -> None:
    """sensor_5 never changes, yet its computed std is 1.78e-15, not 0.0.

    A check for 'std == 0' would let it through, and dividing by 1.78e-15 would
    turn rounding noise into gigantic numbers. MIN_STD exists for exactly this.
    """
    train = load_cmapss(load_config()).train
    assert train["sensor_5"].std() > 0  # the dust really is there

    with pytest.raises(ValueError, match="zero spread"):
        fe.fit_scaler(train, ["sensor_5"])


def test_apply_scaler_rejects_missing_columns() -> None:
    with pytest.raises(ValueError, match="missing"):
        fe.apply_scaler(pd.DataFrame({"a": [1.0]}), {"b": {"mean": 0.0, "std": 1.0}})
