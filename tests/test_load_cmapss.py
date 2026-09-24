"""Tests for the C-MAPSS loader.

WHY TEST A LOADER AT ALL?
Loading looks like the least interesting part of an ML project, which is
exactly why it is a good place for silent bugs to hide. A phantom all-NaN
column, an off-by-one in the RUL alignment, or engine cycles that are not in
order will not raise an exception -- they will just make the model quietly
worse, months later, in a way that is very hard to trace back.

These tests run against the real FD001 files in data/raw/.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.data.load_cmapss import (
    build_column_names,
    load_cmapss,
    load_config,
)


@pytest.fixture(scope="module")
def config() -> dict:
    return load_config()


@pytest.fixture(scope="module")
def dataset(config: dict):
    return load_cmapss(config)


# ---------------------------------------------------------------------------
# Column naming
# ---------------------------------------------------------------------------


def test_build_column_names_has_expected_layout() -> None:
    columns = build_column_names(n_operational_settings=3, n_sensors=21)

    assert len(columns) == 26
    assert columns[0] == "unit_number"
    assert columns[1] == "time_in_cycles"
    assert columns[2:5] == ["op_setting_1", "op_setting_2", "op_setting_3"]
    assert columns[5] == "sensor_1"
    assert columns[-1] == "sensor_21"


def test_build_column_names_rejects_nonsense_input() -> None:
    with pytest.raises(ValueError):
        build_column_names(n_operational_settings=0, n_sensors=21)


# ---------------------------------------------------------------------------
# Shape and integrity of the loaded tables
# ---------------------------------------------------------------------------


def test_train_has_expected_shape(dataset, config: dict) -> None:
    expected = config["dataset"]["expected"]
    assert len(dataset.train) == expected["train_rows"]
    assert dataset.train.shape[1] == 26


def test_test_has_expected_shape(dataset, config: dict) -> None:
    expected = config["dataset"]["expected"]
    assert len(dataset.test) == expected["test_rows"]
    assert dataset.test.shape[1] == 26


def test_engine_counts(dataset, config: dict) -> None:
    expected = config["dataset"]["expected"]
    assert dataset.train["unit_number"].nunique() == expected["n_engines_train"]
    assert dataset.test["unit_number"].nunique() == expected["n_engines_test"]


@pytest.mark.parametrize("split", ["train", "test"])
def test_no_phantom_columns_from_trailing_whitespace(dataset, split: str) -> None:
    """The trap this loader exists to avoid.

    Parsing these files with sep=" " instead of sep=r"\\s+" appends two
    all-NaN columns, because every line ends in trailing spaces. If that ever
    regresses, this test fails immediately.
    """
    frame: pd.DataFrame = getattr(dataset, split)

    assert not frame.isna().any().any(), "Loaded table contains missing values"
    assert not any(str(column).startswith("Unnamed") for column in frame.columns)


@pytest.mark.parametrize("split", ["train", "test"])
def test_all_columns_are_numeric(dataset, split: str) -> None:
    frame: pd.DataFrame = getattr(dataset, split)
    non_numeric = [
        column
        for column in frame.columns
        if not pd.api.types.is_numeric_dtype(frame[column])
    ]
    assert non_numeric == [], f"Non-numeric columns found: {non_numeric}"


@pytest.mark.parametrize("split", ["train", "test"])
def test_ids_are_integers(dataset, split: str) -> None:
    frame: pd.DataFrame = getattr(dataset, split)
    assert frame["unit_number"].dtype == "int64"
    assert frame["time_in_cycles"].dtype == "int64"


# ---------------------------------------------------------------------------
# Time ordering -- everything we build later depends on this
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("split", ["train", "test"])
def test_cycles_are_consecutive_from_one_per_engine(dataset, split: str) -> None:
    """Each engine's cycles must be 1, 2, 3, ... with no gaps or duplicates.

    Rolling averages, lag features and any notion of "the last known state of
    this engine" are meaningless if the time axis has holes or is out of order.
    """
    frame: pd.DataFrame = getattr(dataset, split)

    for unit_number, group in frame.groupby("unit_number"):
        cycles = group["time_in_cycles"].to_numpy()
        expected = range(1, len(cycles) + 1)
        assert list(cycles) == list(expected), (
            f"Engine {unit_number} in {split} has non-consecutive cycles"
        )


# ---------------------------------------------------------------------------
# Ground-truth RUL table
# ---------------------------------------------------------------------------


def test_rul_truth_has_one_row_per_test_engine(dataset) -> None:
    rul_truth = dataset.rul_truth

    assert list(rul_truth.columns) == ["unit_number", "rul"]
    assert len(rul_truth) == dataset.test["unit_number"].nunique()

    # Every test engine must have exactly one ground-truth value, and there must
    # be no extra ids on either side -- a mismatch here would silently misalign
    # predictions with their targets during evaluation.
    assert set(rul_truth["unit_number"]) == set(dataset.test["unit_number"])
    assert rul_truth["unit_number"].is_unique


def test_rul_truth_values_are_plausible(dataset) -> None:
    rul = dataset.rul_truth["rul"]
    assert rul.dtype == "int64"
    assert (rul > 0).all(), "A test engine cannot have zero or negative RUL"
