"""Shared test fixtures.

pytest reads this file automatically. Anything defined here can be used by any
test file just by naming it as an argument -- no import needed.
"""

from __future__ import annotations

import pytest

from src.data.load_cmapss import load_cmapss, load_config
from src.data.split import split_by_engine
from src.features import engineering as fe
from src.features.labels import add_test_rul, add_train_rul


@pytest.fixture(scope="session")
def real():
    """The real FD001 pipeline, run in memory: labels -> split -> features -> scale.

    Built once per test run (scope="session") and shared, because it takes a
    few seconds. Tests must treat it as read-only: copy before changing.
    """
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
