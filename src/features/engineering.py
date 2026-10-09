"""Turning raw sensor readings into features a model can learn from.

Three jobs, in order:

1. CHOOSE which sensors to use -- drop the ones that never change.
2. SMOOTH each sensor using its recent history (rolling mean and rolling std).
3. SCALE everything to a common size, using numbers learned from TRAINING only.

THE ONE RULE THAT RUNS THROUGH ALL OF IT
Every decision and every statistic comes from the TRAINING split only. The
validation and test sets are never consulted -- not to pick columns, not to
compute averages, nothing. If they were, our scores would be measuring how well
we memorised the exam, not how well we predict new engines.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

ID_COLUMN = "unit_number"
CYCLE_COLUMN = "time_in_cycles"
SENSOR_PREFIX = "sensor_"

# The model's own copy of the cycle number (see add_cycle_feature).
CYCLE_FEATURE = "cycle"

# Below this, a column's standard deviation is treated as zero. Not exactly 0.0:
# remember sensor_5, which has ONE unique value yet a computed std of 1.78e-15
# because of floating-point rounding dust. Dividing by that would turn tiny
# rounding errors into enormous numbers.
MIN_STD = 1e-9


# ---------------------------------------------------------------------------
# 1. Choosing sensors
# ---------------------------------------------------------------------------


def find_constant_columns(train_frame: pd.DataFrame, columns: list[str]) -> list[str]:
    """Columns that have exactly one value throughout the TRAINING data.

    A column that never changes cannot help predict anything.
    """
    return [column for column in columns if train_frame[column].nunique() == 1]


def select_sensor_columns(train_frame: pd.DataFrame) -> list[str]:
    """All sensor columns except the constant ones.

    Operating settings are deliberately left out. FD001 has a single operating
    condition: the settings wobble by at most +/-0.0087, which is noise, not
    information. (FD002 and FD004 have six real operating conditions, and would
    need them.)
    """
    sensors = [c for c in train_frame.columns if c.startswith(SENSOR_PREFIX)]
    constant = set(find_constant_columns(train_frame, sensors))
    return [c for c in sensors if c not in constant]


# ---------------------------------------------------------------------------
# 2. Smoothing with recent history
# ---------------------------------------------------------------------------


def add_rolling_features(
    frame: pd.DataFrame,
    sensors: list[str],
    window: int,
) -> pd.DataFrame:
    """Add a rolling mean and rolling std for each sensor, per engine.

    For each row, look at that engine's last ``window`` readings (including the
    current one) and compute:

    * ``<sensor>_rollmean`` -- their average. Calms the cycle-to-cycle noise so
      the slow wear trend shows through.
    * ``<sensor>_rollstd``  -- how much they jump around. Instability often
      grows as a part approaches failure.

    THREE SAFETY RULES, each one deliberate:

    * BACKWARD ONLY. The window covers the current cycle and the ones BEFORE
      it, never after. A window that peeks ahead uses readings that do not
      exist yet at prediction time -- great test scores, useless in reality.
    * PER ENGINE. ``groupby(unit_number)`` restarts the window for every
      engine, so engine 2's first row is never averaged with engine 1's last.
    * ``min_periods=1``. At cycle 1 there is only one reading, so the "average
      of the last 5" is just that reading. Without this, the first 4 cycles of
      every engine would be empty (NaN) and we would have to throw them away --
      costly, since some test engines are only 31 cycles long.

    A spread needs at least two readings, so every engine's first rolling std is
    undefined; we fill it with 0, meaning "no spread observed yet".
    """
    if window < 1:
        raise ValueError(f"window must be at least 1, got {window}")

    _check_time_order(frame)

    grouped = frame.groupby(ID_COLUMN)[sensors]

    means = grouped.transform(lambda s: s.rolling(window, min_periods=1).mean())
    means.columns = [f"{c}_rollmean" for c in sensors]

    stds = grouped.transform(lambda s: s.rolling(window, min_periods=1).std())
    stds = stds.fillna(0.0)
    stds.columns = [f"{c}_rollstd" for c in sensors]

    return pd.concat([frame, means, stds], axis=1)


def _check_time_order(frame: pd.DataFrame) -> None:
    """Rolling windows mean "the previous rows". That only equals "the previous
    cycles" if each engine's rows are in time order. Check, rather than assume."""
    steps = frame.groupby(ID_COLUMN)[CYCLE_COLUMN].diff().dropna()
    if not (steps > 0).all():
        raise ValueError(
            "rows must be sorted by time_in_cycles within each engine before "
            "rolling features can be computed"
        )


def add_cycle_feature(frame: pd.DataFrame) -> pd.DataFrame:
    """Give the model its own copy of the cycle number.

    ``time_in_cycles`` is an IDENTIFIER: it says *when* a row happened, and later
    steps rely on it (finding each engine's last cycle, cutting validation
    engines at random points). Scaling would turn 31 into something like -1.2
    and break all of that.

    So the original stays untouched, and the model gets a separate ``cycle``
    column that can be scaled like any other feature.
    """
    out = frame.copy()
    out[CYCLE_FEATURE] = out[CYCLE_COLUMN].astype("float64")
    return out


def feature_columns(sensors: list[str], include_cycle: bool) -> list[str]:
    """The exact, ordered list of columns the model will see.

    Deliberately NEVER included:

    * ``unit_number`` -- engine 1 in the test file is a different physical
      engine from engine 1 in the training file. The number is just a label.
    * ``rul`` / ``rul_uncapped`` -- those are the answers.
    """
    columns = [
        *sensors,
        *(f"{s}_rollmean" for s in sensors),
        *(f"{s}_rollstd" for s in sensors),
    ]
    if include_cycle:
        columns.append(CYCLE_FEATURE)
    return columns


# ---------------------------------------------------------------------------
# 3. Scaling with statistics from training only
# ---------------------------------------------------------------------------


def fit_scaler(train_frame: pd.DataFrame, columns: list[str]) -> dict[str, dict]:
    """Learn each column's mean and standard deviation from TRAINING data.

    WHY SCALE AT ALL? Our columns live on wildly different scales -- sensor_9
    is around 9,000, sensor_15 around 8.4. Many models treat bigger numbers as
    more important just because they are bigger. Scaling puts every column on
    the same footing: 0 means "average", +1 means "one typical spread above
    average".

    These numbers are FITTED PARAMETERS. They are saved and reused, unchanged,
    for validation, test and -- later -- live data in production.
    """
    stats: dict[str, dict] = {}
    for column in columns:
        mean = float(train_frame[column].mean())
        std = float(train_frame[column].std())
        if not np.isfinite(std) or std < MIN_STD:
            raise ValueError(
                f"column '{column}' has (near-)zero spread in training "
                f"(std={std:g}); it cannot be scaled. A constant column has "
                "probably slipped through."
            )
        stats[column] = {"mean": mean, "std": std}
    return stats


def apply_scaler(frame: pd.DataFrame, stats: dict[str, dict]) -> pd.DataFrame:
    """Scale columns using previously learned statistics: (x - mean) / std.

    The SAME stored numbers are used for every dataset. Validation and test are
    never allowed to supply their own averages.
    """
    missing = [c for c in stats if c not in frame.columns]
    if missing:
        raise ValueError(f"columns missing from frame: {missing}")

    out = frame.copy()
    for column, s in stats.items():
        out[column] = (out[column] - s["mean"]) / s["std"]
    return out
