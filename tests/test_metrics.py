"""Tests for the scoring metrics.

Every number we report comes out of these functions, so they are tested on
tiny examples where the right answer can be worked out by hand.
"""

from __future__ import annotations

import math

import pytest

from src.models.metrics import mae, nasa_score, rmse, score


def test_perfect_predictions_score_zero_everywhere() -> None:
    truth = [10, 50, 100]
    assert rmse(truth, truth) == 0.0
    assert mae(truth, truth) == 0.0
    assert nasa_score(truth, truth) == 0.0


def test_mae_is_the_average_size_of_a_miss() -> None:
    # misses of -10 and +30 -> sizes 10 and 30 -> average 20
    assert mae([50, 50], [40, 80]) == 20.0


def test_rmse_punishes_one_big_blunder_more_than_mae() -> None:
    """Same average miss (10), but one model has a single miss of 40."""
    truth = [50, 50, 50, 50]
    even = [60, 60, 60, 60]  # misses 10, 10, 10, 10
    blunder = [50, 50, 50, 90]  # misses  0,  0,  0, 40

    assert mae(truth, even) == mae(truth, blunder) == 10.0
    assert rmse(truth, even) == 10.0
    assert rmse(truth, blunder) == 20.0


def test_nasa_score_exact_values() -> None:
    assert nasa_score([50], [60]) == pytest.approx(math.exp(10 / 10) - 1)  # late by 10
    assert nasa_score([50], [40]) == pytest.approx(math.exp(10 / 13) - 1)  # early by 10


def test_nasa_score_punishes_late_more_than_early() -> None:
    """Saying MORE life is left than there really is, is the dangerous mistake."""
    for miss in (5, 20, 40):
        late = nasa_score([50], [50 + miss])
        early = nasa_score([50], [50 - miss])
        assert late > early


def test_nasa_score_adds_up_across_engines() -> None:
    one = nasa_score([50], [60])
    assert nasa_score([50, 50], [60, 60]) == pytest.approx(2 * one)


def test_score_counts_late_and_early_predictions() -> None:
    result = score([50, 50, 50, 50], [60, 70, 40, 50])  # late, late, early, exact

    assert result["n"] == 4
    assert result["n_late"] == 2
    assert result["n_early"] == 1


def test_mismatched_lengths_are_rejected() -> None:
    with pytest.raises(ValueError, match="shape"):
        rmse([1, 2, 3], [1, 2])


def test_empty_input_is_rejected() -> None:
    with pytest.raises(ValueError, match="zero"):
        mae([], [])
