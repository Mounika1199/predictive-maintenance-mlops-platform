"""Tests for the data validation stage.

TWO KINDS OF TEST HERE, AND BOTH MATTER

1. **Clean data must PASS.** This looks like a throwaway test. It is not -- it
   is the test that catches "validation is broken" as opposed to "the data is
   bad". We learned this the hard way: a pandera import problem made every
   single check raise an exception, which got reported as 26 column failures.
   The data was perfect. Only a happy-path test distinguishes those two cases.

2. **Broken data must FAIL.** A validation suite that has only ever seen good
   data has never been shown to catch anything. Each test below breaks the data
   in one specific, realistic way and asserts we notice.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.data.load_cmapss import load_cmapss, load_config
from src.data.validation import (
    build_reference_profile,
    build_schema,
    check_structure,
    load_reference_profile,
    load_validation_settings,
    validate_frame,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def config() -> dict:
    return load_config()


@pytest.fixture(scope="module")
def dataset(config: dict):
    return load_cmapss(config)


@pytest.fixture(scope="module")
def schema(config: dict):
    """The real schema, built from the committed reference profile."""
    profile_path, range_tolerance, constant_tolerance = load_validation_settings(config)
    profile = load_reference_profile(profile_path)
    return build_schema(profile, range_tolerance, constant_tolerance)


@pytest.fixture(scope="module")
def small(dataset) -> pd.DataFrame:
    """First three engines only -- enough to break, small enough to be fast.

    Engines 1..3 keep unit_number a contiguous 1..N block, so this slice is
    structurally valid on its own.
    """
    return dataset.train[dataset.train["unit_number"] <= 3].reset_index(drop=True)


def errors_of(frame: pd.DataFrame, schema, **kwargs) -> list[str]:
    return validate_frame(frame, schema, "case", **kwargs).errors


# ---------------------------------------------------------------------------
# 1. Clean data must pass
# ---------------------------------------------------------------------------


def test_clean_train_passes(dataset, schema) -> None:
    result = validate_frame(dataset.train, schema, "train")
    assert result.ok, f"clean training data failed validation: {result.errors}"


def test_clean_test_passes(dataset, schema) -> None:
    """The check that justifies our tolerance choice.

    The schema is built from TRAIN, yet 7 of 17 varying columns have test values
    outside the training range. With a zero tolerance this would fail. It
    passing is the evidence that our rules are not over-fitted to the training
    sample.
    """
    result = validate_frame(dataset.test, schema, "test")
    assert result.ok, f"clean test data failed validation: {result.errors}"


def test_small_slice_is_clean(small, schema) -> None:
    """Sanity check on the fixture itself, so later failures mean something."""
    assert errors_of(small, schema) == []


# ---------------------------------------------------------------------------
# 2. Column-level corruption must be caught
# ---------------------------------------------------------------------------


def test_constant_column_that_starts_moving_is_caught(small, schema) -> None:
    """sensor_1 is 518.67 in every row. A recalibration would show up here."""
    broken = small.copy()
    broken.loc[0, "sensor_1"] = 518.70  # 0.03 off -- tiny, but real

    errors = errors_of(broken, schema)
    assert any("sensor_1" in e for e in errors), errors


def test_floating_point_dust_does_not_trigger_a_false_alarm(small, schema) -> None:
    """The reason constant_tolerance exists.

    A difference in the 12th decimal place is rounding noise, not a sensor
    change. If this test fails, our alarm cries wolf and people stop trusting
    it.
    """
    noisy = small.copy()
    noisy["sensor_1"] = noisy["sensor_1"] + 1e-12

    assert errors_of(noisy, schema) == []


def test_value_far_outside_range_is_caught(small, schema) -> None:
    broken = small.copy()
    broken.loc[5, "sensor_9"] = 99_999.0

    errors = errors_of(broken, schema)
    assert any("sensor_9" in e for e in errors), errors


def test_missing_column_is_caught(small, schema) -> None:
    errors = errors_of(small.drop(columns=["sensor_4"]), schema)
    assert any("sensor_4" in e for e in errors), errors


def test_unexpected_extra_column_is_caught(small, schema) -> None:
    """strict=True. An extra column means the upstream format changed."""
    broken = small.copy()
    broken["sensor_22"] = 1.0

    errors = errors_of(broken, schema)
    assert errors, "an unexpected extra column should be rejected"


def test_reordered_columns_are_caught(small, schema) -> None:
    """ordered=True.

    The raw files have no header row, so a column's identity is purely its
    position. If an upstream export reorders two columns, our loader attaches
    the wrong names and every value still looks plausible.
    """
    columns = list(small.columns)
    columns[5], columns[6] = columns[6], columns[5]

    errors = errors_of(small[columns], schema)
    assert errors, "reordered columns should be rejected"


def test_swapped_values_between_columns_are_caught(small, schema) -> None:
    """The realistic version of the above, and the most dangerous case.

    Here the column *names* are right but the values behind two of them were
    exchanged upstream. Nothing crashes. It is caught only because sensor_2
    (~641) and sensor_3 (~1590) live in completely different ranges -- which is
    exactly what the range checks are for.
    """
    broken = small.copy()
    broken["sensor_2"], broken["sensor_3"] = small["sensor_3"], small["sensor_2"]

    errors = errors_of(broken, schema)
    assert any("sensor_2" in e for e in errors), errors
    assert any("sensor_3" in e for e in errors), errors


def test_missing_values_are_caught(small, schema) -> None:
    broken = small.copy()
    broken.loc[3, "sensor_7"] = None

    errors = errors_of(broken, schema)
    assert any("sensor_7" in e for e in errors), errors


# ---------------------------------------------------------------------------
# 3. Structural corruption must be caught
# ---------------------------------------------------------------------------


def test_gap_in_cycles_is_caught(small) -> None:
    """Delete one row from the middle of an engine's trajectory.

    Every time-aware feature we build later -- rolling means, lags, "the most
    recent state of this engine" -- silently produces nonsense across a gap.
    """
    broken = small.drop(index=10).reset_index(drop=True)

    errors = check_structure(broken)
    assert any("gap" in e for e in errors), errors


def test_duplicated_engine_cycle_is_caught(small) -> None:
    broken = pd.concat([small, small.iloc[[7]]], ignore_index=True)

    errors = check_structure(broken)
    assert any("duplicated" in e for e in errors), errors


def test_non_contiguous_engine_ids_are_caught(small) -> None:
    """Engine 2 vanishing usually means rows were dropped upstream."""
    broken = small[small["unit_number"] != 2].reset_index(drop=True)

    errors = check_structure(broken)
    assert any("contiguous" in e for e in errors), errors


def test_trajectory_not_starting_at_cycle_one_is_caught(small) -> None:
    broken = small[small["time_in_cycles"] != 1].reset_index(drop=True)

    errors = check_structure(broken)
    assert errors, "an engine whose trajectory does not start at cycle 1"


def test_empty_table_is_caught(small) -> None:
    assert check_structure(small.iloc[0:0]) == ["table is empty"]


def test_clean_structure_produces_no_errors(small) -> None:
    assert check_structure(small) == []


# ---------------------------------------------------------------------------
# 4. The schema builder itself
# ---------------------------------------------------------------------------


def test_build_schema_rejects_negative_tolerance(config: dict) -> None:
    profile_path, _, _ = load_validation_settings(config)
    profile = load_reference_profile(profile_path)

    with pytest.raises(ValueError):
        build_schema(profile, range_tolerance=-0.1, constant_tolerance=1e-6)


def test_profile_identifies_the_constant_columns(dataset) -> None:
    """Locks in a real property of FD001: exactly 7 columns never move."""
    profile = build_reference_profile(dataset.train, "FD001")
    constant = {n for n, s in profile["columns"].items() if s["constant"]}

    assert constant == {
        "op_setting_3",
        "sensor_1",
        "sensor_5",
        "sensor_10",
        "sensor_16",
        "sensor_18",
        "sensor_19",
    }


def test_profile_is_built_from_training_data_only(dataset) -> None:
    """Guards against the test set leaking into our definition of 'normal'."""
    profile = build_reference_profile(dataset.train, "FD001")

    assert profile["n_rows"] == len(dataset.train)
    assert profile["n_rows"] != len(dataset.test)
