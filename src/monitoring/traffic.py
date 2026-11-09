"""A pretend fleet: the test engines, replayed into the service flight by flight.

We have no real engines. This replays the 100 FD001 test engines the way a
fleet would use the service, so monitoring has traffic to watch -- traffic
whose answer we KNOW: normal, or with a broken sensor we put there ourselves.

    day 1:   every engine flies once and sends its cycle 1
    day 2:   every engine sends its cycles 1-2
    ...
    day t:   every engine still flying sends its last `window` cycles

An engine stops after its last recorded cycle. Every cycle of every engine is
the NEWEST cycle of exactly one request, so the diary gets one line per cycle.
"""

from __future__ import annotations

from collections.abc import Iterator

import pandas as pd

from src.features.engineering import CYCLE_COLUMN, ID_COLUMN


def engine_label(unit: int) -> str:
    return f"engine-{unit}"


def fleet_requests(raw: pd.DataFrame, window: int, n_engines: int | None = None) -> Iterator[dict]:
    """Bodies for POST /predict, in fleet order: day by day, engine by engine.

    ``raw`` holds the engines' readings as recorded (unit_number,
    time_in_cycles, op_setting_*, sensor_*). ``n_engines`` keeps only the
    first n engines, for a quick run.

    Day t is cycle t: validation guarantees every engine's cycles run 1, 2,
    3, ... without gaps, so the last `window` rows ARE the last `window` cycles.
    """
    # Each engine's cycles as a list of plain dicts, converted ONCE: slicing a
    # list is fast; converting a table 13,096 times took ~14 s.
    units = sorted(raw[ID_COLUMN].unique())[:n_engines]
    engines = {
        unit: raw[raw[ID_COLUMN] == unit].sort_values(CYCLE_COLUMN).drop(columns=ID_COLUMN).to_dict(orient="records")
        for unit in units
    }
    last_day = max(len(cycles) for cycles in engines.values())

    for day in range(1, last_day + 1):
        for unit, cycles in engines.items():
            if day <= len(cycles):
                yield {"engine_id": engine_label(unit), "readings": cycles[max(0, day - window):day]}


def break_sensor(raw: pd.DataFrame, sensor: str, offset: float) -> pd.DataFrame:
    """A copy of ``raw`` in which ``sensor`` reads ``offset`` too high in every
    cycle -- like a sensor recalibrated wrongly. ``raw`` itself is not changed."""
    broken = raw.copy()
    broken[sensor] = broken[sensor] + offset
    return broken
