"""Tests for the engine-level split.

The split only depends on WHICH engine ids exist and on the seed, so a
hand-made table with engines 1..100 behaves exactly like the real FD001 data.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.data.split import split_by_engine


def engines(n_engines: int = 100, cycles: int = 3) -> pd.DataFrame:
    """n_engines engines, each with `cycles` rows."""
    return pd.DataFrame(
        {
            "unit_number": [e for e in range(1, n_engines + 1) for _ in range(cycles)],
            "time_in_cycles": [c for _ in range(n_engines) for c in range(1, cycles + 1)],
        }
    )


def test_no_engine_appears_on_both_sides() -> None:
    """The whole point of splitting by engine."""
    s = split_by_engine(engines(), validation_fraction=0.2, seed=42)

    assert set(s.train_engines).isdisjoint(s.validation_engines)
    assert set(s.train["unit_number"]).isdisjoint(s.validation["unit_number"])


def test_every_engine_lands_on_exactly_one_side() -> None:
    s = split_by_engine(engines(), validation_fraction=0.2, seed=42)

    assert len(s.validation_engines) == 20
    assert len(s.train_engines) == 80
    assert sorted(s.train_engines + s.validation_engines) == list(range(1, 101))


def test_all_rows_of_an_engine_stay_together() -> None:
    """No engine is cut in half: each keeps all of its rows on its own side."""
    data = engines(cycles=5)
    s = split_by_engine(data, validation_fraction=0.2, seed=42)

    assert len(s.train) + len(s.validation) == len(data)
    assert (s.train.groupby("unit_number").size() == 5).all()
    assert (s.validation.groupby("unit_number").size() == 5).all()


def test_same_seed_gives_the_same_split() -> None:
    a = split_by_engine(engines(), validation_fraction=0.2, seed=42)
    b = split_by_engine(engines(), validation_fraction=0.2, seed=42)
    assert a.validation_engines == b.validation_engines


def test_different_seed_gives_a_different_split() -> None:
    a = split_by_engine(engines(), validation_fraction=0.2, seed=42)
    b = split_by_engine(engines(), validation_fraction=0.2, seed=7)
    assert a.validation_engines != b.validation_engines


def test_row_order_does_not_change_the_split() -> None:
    """Shuffling the rows must not change which engines are picked."""
    data = engines()
    shuffled = data.sample(frac=1.0, random_state=0)

    a = split_by_engine(data, validation_fraction=0.2, seed=42)
    b = split_by_engine(shuffled, validation_fraction=0.2, seed=42)
    assert a.validation_engines == b.validation_engines


@pytest.mark.parametrize("fraction", [0.0, 1.0, -0.1, 1.5])
def test_invalid_fraction_is_rejected(fraction: float) -> None:
    with pytest.raises(ValueError):
        split_by_engine(engines(), validation_fraction=fraction, seed=42)


def test_fraction_too_small_to_pick_any_engine_is_rejected() -> None:
    """1% of 10 engines rounds to 0 -- an empty validation set is useless."""
    with pytest.raises(ValueError, match="empty"):
        split_by_engine(engines(n_engines=10), validation_fraction=0.01, seed=42)
