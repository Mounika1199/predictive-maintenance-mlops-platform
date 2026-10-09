"""Splitting training data into a training part and a validation part.

WHY WE NEED A VALIDATION SET
We cannot judge a model on the data it learned from -- it may simply have
memorised it. So we hide some data during training and test on it afterwards.

WHY WE SPLIT BY ENGINE, NOT BY ROW
Rows from the same engine are near-identical neighbours: cycle 100 looks almost
exactly like cycle 99 and cycle 101. If we split rows randomly, the validation
set is full of rows whose "twins" sit in the training set. The model can score
brilliantly by recognising the twin, without learning anything that works on a
brand-new engine.

Splitting by engine means every validation engine is a complete stranger to the
model -- which is exactly the situation it will face in real life.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class EngineSplit:
    """The result of a split: two tables, plus which engines went where."""

    train: pd.DataFrame
    validation: pd.DataFrame
    train_engines: list[int]
    validation_engines: list[int]


def split_by_engine(
    frame: pd.DataFrame,
    validation_fraction: float,
    seed: int,
) -> EngineSplit:
    """Put a random fraction of whole ENGINES into validation.

    Every row of a given engine goes to the same side. Never some rows here and
    some rows there.

    ``seed`` makes the "random" choice repeatable: the same seed always picks
    the same engines. Without that, every run would test on different engines,
    and you could not tell whether a better score came from a better model or a
    luckier split.
    """
    if not 0 < validation_fraction < 1:
        raise ValueError(
            f"validation_fraction must be between 0 and 1, got {validation_fraction}"
        )

    # Sort first, so the result depends only on WHICH engines exist -- not on
    # whatever order the rows happen to arrive in.
    engines = np.sort(frame["unit_number"].unique())

    n_validation = round(len(engines) * validation_fraction)
    if n_validation == 0 or n_validation == len(engines):
        raise ValueError(
            f"validation_fraction={validation_fraction} with {len(engines)} "
            "engines would leave one side empty"
        )

    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(engines)

    validation_engines = sorted(int(e) for e in shuffled[:n_validation])
    train_engines = sorted(int(e) for e in shuffled[n_validation:])

    in_validation = frame["unit_number"].isin(validation_engines)

    return EngineSplit(
        train=frame[~in_validation].reset_index(drop=True),
        validation=frame[in_validation].reset_index(drop=True),
        train_engines=train_engines,
        validation_engines=validation_engines,
    )
