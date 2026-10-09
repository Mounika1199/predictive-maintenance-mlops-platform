"""Creating the RUL target -- the number we are trying to predict.

RUL = Remaining Useful Life = how many more cycles until this engine fails.

The two splits need different arithmetic, because they were recorded
differently:

* TRAIN engines were run until they broke. So the last row of an engine IS the
  failure, and we can count backwards from it.
* TEST engines were switched off early, before breaking. We cannot count
  backwards, because the failure is not in the data. Instead a separate file
  (RUL_FD001.txt) tells us how much life each test engine had left at the moment
  it was switched off.
"""

from __future__ import annotations

import pandas as pd


def add_train_rul(frame: pd.DataFrame, cap: int) -> pd.DataFrame:
    """Add the RUL target to a TRAINING table by counting backwards from failure.

    Worked example -- an engine that failed after 4 cycles::

        cycle   last_cycle   rul = last_cycle - cycle
          1          4                 3
          2          4                 2
          3          4                 1
          4          4                 0   <- the failure

    So RUL drops by exactly 1 each cycle and hits 0 on the final row.

    ``cap`` then flattens everything above it. With cap=125, an RUL of 300
    becomes 125, and an RUL of 40 stays 40. We keep the original in
    ``rul_uncapped`` so we can still analyse the true values later.
    """
    if cap <= 0:
        raise ValueError(f"cap must be positive, got {cap}")

    # transform("max") gives each ROW its own engine's final cycle, so the
    # subtraction below lines up row by row.
    last_cycle = frame.groupby("unit_number")["time_in_cycles"].transform("max")
    rul = last_cycle - frame["time_in_cycles"]

    out = frame.copy()
    out["rul_uncapped"] = rul
    out["rul"] = rul.clip(upper=cap)
    return out


def add_test_rul(
    frame: pd.DataFrame,
    rul_truth: pd.DataFrame,
    cap: int,
) -> pd.DataFrame:
    """Add the RUL target to a TEST table using the ground-truth file.

    The truth file gives RUL at the engine's LAST OBSERVED cycle only. To label
    earlier rows we walk backwards from there.

    Worked example -- an engine switched off after 4 cycles with 10 cycles of
    life still left::

        cycle   last_cycle   truth   rul = truth + (last_cycle - cycle)
          1          4         10          10 + 3 = 13
          2          4         10          10 + 2 = 12
          3          4         10          10 + 1 = 11
          4          4         10          10 + 0 = 10   <- matches the file

    Same rule as training (RUL falls by 1 per cycle); only the starting point
    differs, because this engine never actually failed.
    """
    if cap <= 0:
        raise ValueError(f"cap must be positive, got {cap}")

    missing = set(frame["unit_number"]) - set(rul_truth["unit_number"])
    if missing:
        raise ValueError(
            f"{len(missing)} test engine(s) have no ground-truth RUL "
            f"(e.g. {sorted(missing)[:5]})"
        )

    truth_by_engine = rul_truth.set_index("unit_number")["rul"]
    truth = frame["unit_number"].map(truth_by_engine)

    last_cycle = frame.groupby("unit_number")["time_in_cycles"].transform("max")
    rul = truth + (last_cycle - frame["time_in_cycles"])

    out = frame.copy()
    out["rul_uncapped"] = rul
    out["rul"] = rul.clip(upper=cap)
    return out


def last_cycle_per_engine(frame: pd.DataFrame) -> pd.DataFrame:
    """Keep only each engine's final row.

    This is the standard way C-MAPSS results are scored: one prediction per test
    engine, made at the latest moment we observed it. That mirrors the real
    question -- "given everything I know about this engine right now, how long
    has it got?" -- rather than re-scoring its whole history.

    WARNING -- only use this on engines that were stopped BEFORE failure.
    That is true of the real test set. It is NOT true of training engines, or
    of a validation set carved from them: those ran until they broke, so their
    last row is always the failure itself and every "answer" would be RUL 0. A
    model that always predicts 0 would then look perfect.

    To score validation engines this way, first cut each one at a random cycle
    (imitating how the test set was made), then call this function.
    """
    final_rows = frame.groupby("unit_number")["time_in_cycles"].idxmax()
    return frame.loc[final_rows].sort_values("unit_number").reset_index(drop=True)
