"""Tests for the promotion rules (src/models/promotion.py).

The rules decide whether a challenger may replace the champion. They are pure
logic, so every case can be checked with hand-made candidates.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from src.models.promotion import Candidate, decide_promotion

MARGIN = 0.1
SAME_EXAM = {"split_seed": "42", "validation_fraction": "0.2", "validation_engines": "20", "rul_cap": "125"}


def candidate(version: str, val_rmse: float, uncommitted: bool | None = False, **exam_changes) -> Candidate:
    return Candidate(
        version=version,
        val_rmse=val_rmse,
        exam={**SAME_EXAM, **exam_changes},
        uncommitted=uncommitted,
    )


CHAMPION = candidate("1", 15.13)


# ---------------------------------------------------------------------------
# Promote
# ---------------------------------------------------------------------------


def test_the_first_model_becomes_champion() -> None:
    decision = decide_promotion(candidate("1", 15.13), None, MARGIN)
    assert decision.promote
    assert "no champion" in decision.reason


def test_a_clearly_better_challenger_is_promoted() -> None:
    decision = decide_promotion(candidate("2", 14.20), CHAMPION, MARGIN)
    assert decision.promote
    assert "better by 0.93" in decision.reason


def test_trying_different_features_is_allowed() -> None:
    """Feature settings are not part of the 'exam': changing them is exactly
    what a challenger is for. Only the engines and the answer key must match."""
    challenger = candidate("2", 14.20)  # e.g. trained with another rolling window
    assert decide_promotion(challenger, CHAMPION, MARGIN).promote


# ---------------------------------------------------------------------------
# Refuse
# ---------------------------------------------------------------------------


def test_a_worse_challenger_is_refused() -> None:
    decision = decide_promotion(candidate("2", 16.00), CHAMPION, MARGIN)
    assert not decision.promote
    assert "worse by 0.87" in decision.reason


def test_a_tiny_improvement_is_treated_as_a_tie() -> None:
    decision = decide_promotion(candidate("2", 15.10), CHAMPION, MARGIN)
    assert not decision.promote
    assert "less than the required" in decision.reason


def test_rounding_dust_never_wins() -> None:
    """A retrained copy of the champion can differ in the 14th decimal place,
    because the forest adds its trees' answers in a varying order."""
    dust = candidate("2", CHAMPION.val_rmse - 1e-14)
    assert not decide_promotion(dust, CHAMPION, MARGIN).promote


def test_the_champion_cannot_replace_itself() -> None:
    decision = decide_promotion(CHAMPION, CHAMPION, MARGIN)
    assert not decision.promote
    assert "already the champion" in decision.reason


@pytest.mark.parametrize(
    "setting, value",
    [
        ("split_seed", "7"),  # different validation engines
        ("validation_engines", "25"),  # a different number of them
        ("rul_cap", "130"),  # a different answer key
    ],
)
def test_scores_from_different_exams_are_not_compared(setting: str, value: str) -> None:
    """Even a much 'better' score means nothing if it came from another exam."""
    challenger = candidate("2", 10.00, **{setting: value})
    decision = decide_promotion(challenger, CHAMPION, MARGIN)

    assert not decision.promote
    assert "different exams" in decision.reason
    assert setting in decision.reason  # says WHICH setting differs


# ---------------------------------------------------------------------------
# Warnings about uncommitted code
# ---------------------------------------------------------------------------


def test_committed_code_gives_no_warning() -> None:
    assert decide_promotion(candidate("2", 14.20, uncommitted=False), CHAMPION, MARGIN).warnings == []


@pytest.mark.parametrize("uncommitted", [True, None])
def test_uncommitted_or_unknown_code_warns_but_does_not_block(uncommitted) -> None:
    decision = decide_promotion(candidate("2", 14.20, uncommitted=uncommitted), CHAMPION, MARGIN)

    assert decision.promote  # the warning does not block...
    assert len(decision.warnings) == 1  # ...but it is always said out loud


def test_warnings_appear_even_when_refused() -> None:
    decision = decide_promotion(candidate("2", 16.00, uncommitted=True), CHAMPION, MARGIN)
    assert not decision.promote
    assert decision.warnings


def test_the_candidate_is_never_modified() -> None:
    """Candidates are frozen: the rules can only read them."""
    challenger = candidate("2", 14.20)
    decide_promotion(challenger, CHAMPION, MARGIN)
    assert challenger == replace(challenger)
    with pytest.raises(Exception):
        challenger.val_rmse = 1.0  # type: ignore[misc]
