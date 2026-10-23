"""Checks on the readings a client sends, before and after predicting.

Two kinds of problem, treated differently on purpose:

1. BROKEN HISTORY -> refuse (InvalidReadings, HTTP 422).
   Gaps, repeated or reordered cycles, or too little history. The smoothed
   features are then guaranteed to differ from anything the model saw in
   training -- e.g. 124.46 instead of 123.96 for test engine 1, with no error.
   There is no honest answer to give.

2. UNUSUAL VALUES -> answer, with warnings.
   A reading outside the validation contract (configs/reference_profile.json,
   the same ranges scripts/validate_fd001.py checks). The features can still be
   computed, so the service answers -- but says plainly which readings are
   outside what the model learned from. An engine close to failure may be
   exactly such an engine, so refusing would mean staying silent when it
   matters most. These warnings are also a ready-made drift signal.
"""

from __future__ import annotations

import pandas as pd
import pandera.pandas as pa

from src.data.load_cmapss import load_config
from src.data.validation import (
    build_schema,
    load_reference_profile,
    load_validation_settings,
    validate_frame,
)

CYCLE_COLUMN = "time_in_cycles"


class InvalidReadings(ValueError):
    """The readings cannot give an honest prediction; the reason says why."""


def check_history(readings: pd.DataFrame, window: int) -> None:
    """Refuse readings whose history would make the features wrong.

    The rules, in plain words:
    * the cycles must be CONSECUTIVE: each one exactly 1 after the previous
      (no gaps, no repeats, no reordering);
    * there must be at least `window` of them -- unless they start at cycle 1,
      because an engine younger than the window genuinely has fewer, and
      training saw exactly that situation at the start of every engine.
    """
    cycles = readings[CYCLE_COLUMN].tolist()

    for previous, current in zip(cycles, cycles[1:]):
        if current != previous + 1:
            raise InvalidReadings(
                f"cycle {previous} is followed by cycle {current}: the readings must be "
                "consecutive cycles, oldest first (no gaps, repeats or reordering)"
            )

    if cycles[0] != 1 and len(cycles) < window:
        raise InvalidReadings(
            f"got {len(cycles)} cycle(s) ({cycles[0]} to {cycles[-1]}), but the model "
            f"smooths over the last {window} cycles: send at least the last {window}, or "
            "the engine's whole history from cycle 1. With fewer, the features would "
            "differ from training and the prediction would be silently wrong."
        )


def load_contract() -> pa.DataFrameSchema:
    """The validation contract, built exactly as scripts/validate_fd001.py does."""
    profile_path, range_tolerance, constant_tolerance = load_validation_settings(load_config())
    return build_schema(load_reference_profile(profile_path), range_tolerance, constant_tolerance)


def contract_warnings(readings: pd.DataFrame, contract: pa.DataFrameSchema) -> list[str]:
    """Readings outside the validation contract, as human-readable warnings.

    The contract describes whole tables, so the request gets the columns it
    expects, in its order: an engine id (a placeholder -- never a model input)
    first, then the readings. Only the column checks are applied; the
    table-level checks (engine ids, gaps) belong to check_history().
    """
    frame = readings.copy()
    frame.insert(0, "unit_number", 1)
    frame = frame[list(contract.columns)]

    result = validate_frame(frame, contract, "request", check_row_structure=False)
    return [f"outside the range seen in training -- {error}" for error in result.errors]
