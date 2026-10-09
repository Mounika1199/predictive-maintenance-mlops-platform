"""Tests for RUL labelling.

Most tests use tiny hand-made tables, where the right answer can be worked out in
your head. A few run on the real FD001 data to prove the arithmetic holds there.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.data.load_cmapss import load_cmapss, load_config
from src.features.labels import add_test_rul, add_train_rul, last_cycle_per_engine


def tiny_train() -> pd.DataFrame:
    """Engine 1 ran for 4 cycles, engine 2 for 3 cycles -- then both failed."""
    return pd.DataFrame(
        {
            "unit_number": [1, 1, 1, 1, 2, 2, 2],
            "time_in_cycles": [1, 2, 3, 4, 1, 2, 3],
        }
    )


@pytest.fixture(scope="module")
def fd001():
    return load_cmapss(load_config())


# ---------------------------------------------------------------------------
# Training labels: count backwards from failure
# ---------------------------------------------------------------------------


def test_train_rul_counts_down_to_zero() -> None:
    out = add_train_rul(tiny_train(), cap=125)
    assert out["rul_uncapped"].tolist() == [3, 2, 1, 0, 2, 1, 0]


def test_cap_flattens_only_values_above_it() -> None:
    out = add_train_rul(tiny_train(), cap=2)

    assert out["rul"].tolist() == [2, 2, 1, 0, 2, 1, 0]  # the 3 became 2
    assert out["rul_uncapped"].tolist() == [3, 2, 1, 0, 2, 1, 0]  # truth kept


def test_every_real_training_engine_ends_at_zero(fd001) -> None:
    """Training engines ran until they broke, so their last row is RUL 0.

    This is also why last_cycle_per_engine() must NOT be used to score
    validation engines without cutting them first: every answer would be 0.
    """
    last = last_cycle_per_engine(add_train_rul(fd001.train, cap=125))
    assert (last["rul_uncapped"] == 0).all()


def test_cap_is_never_exceeded_on_real_data(fd001) -> None:
    out = add_train_rul(fd001.train, cap=125)
    assert out["rul"].max() == 125
    assert out["rul_uncapped"].max() > 125  # the cap really had work to do


# ---------------------------------------------------------------------------
# Test labels: start from the truth file and count backwards from there
# ---------------------------------------------------------------------------


def test_test_rul_starts_from_the_truth_value() -> None:
    """Switched off after 4 cycles with 10 cycles of life still left."""
    test = pd.DataFrame({"unit_number": [1, 1, 1, 1], "time_in_cycles": [1, 2, 3, 4]})
    truth = pd.DataFrame({"unit_number": [1], "rul": [10]})

    out = add_test_rul(test, truth, cap=125)
    assert out["rul_uncapped"].tolist() == [13, 12, 11, 10]


def test_real_final_test_rul_matches_the_truth_file(fd001) -> None:
    """Two numbers from two different places must agree for all 100 engines.

    Our arithmetic produces one; NASA's RUL_FD001.txt provides the other. An
    off-by-one anywhere in add_test_rul would break this.
    """
    last = last_cycle_per_engine(add_test_rul(fd001.test, fd001.rul_truth, cap=125))
    merged = last.merge(fd001.rul_truth, on="unit_number", suffixes=("", "_truth"))

    assert len(merged) == 100
    assert (merged["rul_uncapped"] == merged["rul_truth"]).all()


def test_missing_truth_value_is_rejected() -> None:
    test = pd.DataFrame({"unit_number": [1, 2], "time_in_cycles": [1, 1]})
    truth = pd.DataFrame({"unit_number": [1], "rul": [10]})  # engine 2 missing

    with pytest.raises(ValueError, match="no ground-truth"):
        add_test_rul(test, truth, cap=125)


# ---------------------------------------------------------------------------
# Odds and ends
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cap", [0, -5])
def test_invalid_cap_is_rejected(cap: int) -> None:
    with pytest.raises(ValueError):
        add_train_rul(tiny_train(), cap=cap)


def test_last_cycle_per_engine_picks_each_final_row() -> None:
    out = last_cycle_per_engine(tiny_train())
    assert out["unit_number"].tolist() == [1, 2]
    assert out["time_in_cycles"].tolist() == [4, 3]
