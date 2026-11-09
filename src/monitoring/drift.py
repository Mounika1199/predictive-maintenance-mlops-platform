"""Has the data drifted? The service's diary compared with the training data.

Evidently does the comparing; the RULES are ours, and every one of them was
measured on our own data (see configs/monitoring.yaml):

* Like with like: both sides only at engine ages 1..max_cycle.
* PSI for every sensor the model uses; a sensor has drifted when its score
  is at least the threshold.
* DRIFT if ANY sensor drifted. Not Evidently's own "dataset drift" verdict,
  which needs half of the columns: one broken sensor out of 15 moved
  predictions by up to 20 cycles, and that verdict would have said "no".
* Too few readings: no verdict at all, rather than a false alarm.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from evidently import DataDefinition, Dataset, Report
from evidently.presets import DataDriftPreset

from src.data.load_cmapss import load_config
from src.features.engineering import CYCLE_COLUMN, ID_COLUMN
from src.monitoring.prediction_log import MONITORING_CONFIG


class NotEnoughReadings(ValueError):
    """Too little traffic to tell drift from chance."""


@dataclass(frozen=True)
class DriftResult:
    scores: dict[str, float]  # sensor -> PSI score
    threshold: float
    n_reference: int
    n_current: int

    @property
    def drifted(self) -> list[str]:
        return [sensor for sensor, score in self.scores.items() if score >= self.threshold]

    @property
    def drift(self) -> bool:
        return bool(self.drifted)


def load_drift_config() -> dict:
    return load_config(MONITORING_CONFIG)["drift"]


def reference_readings(train_raw: pd.DataFrame, train_engines: list[int], sensors: list[str],
                       max_cycle: int) -> pd.DataFrame:
    """What the model learned from: its training engines, at ages 1..max_cycle."""
    picked = train_raw[train_raw[ID_COLUMN].isin(train_engines) & (train_raw[CYCLE_COLUMN] <= max_cycle)]
    return picked[sensors]


def current_readings(diary: pd.DataFrame, sensors: list[str], max_cycle: int,
                     since: pd.Timestamp | None = None, until: pd.Timestamp | None = None) -> pd.DataFrame:
    """What the service saw: diary lines in [since, until), at ages 1..max_cycle."""
    if diary.empty:
        return pd.DataFrame(columns=sensors)
    if since is not None:
        diary = diary[diary["time"] >= since]
    if until is not None:
        diary = diary[diary["time"] < until]
    return diary[diary[CYCLE_COLUMN] <= max_cycle][sensors]


def check_drift(reference: pd.DataFrame, current: pd.DataFrame, sensors: list[str], method: str,
                threshold: float, min_readings: int):
    """Compare the two, sensor by sensor. Returns the result and Evidently's
    report (a "snapshot"), which can be saved as HTML."""
    if len(current) < min_readings:
        raise NotEnoughReadings(
            f"only {len(current):,} readings to judge; at least {min_readings:,} are needed "
            f"to tell drift from chance"
        )

    definition = DataDefinition(numerical_columns=sensors)
    report = Report([DataDriftPreset(columns=sensors, method=method, threshold=threshold)])
    snapshot = report.run(
        current_data=Dataset.from_pandas(current, data_definition=definition),
        reference_data=Dataset.from_pandas(reference, data_definition=definition),
    )

    # One "ValueDrift" result per sensor; the other result is Evidently's own
    # overall count, which we deliberately do not use (see the module notes).
    scores = {
        metric["config"]["column"]: float(metric["value"])
        for metric in snapshot.dict()["metrics"]
        if metric["config"]["type"] == "evidently:metric_v2:ValueDrift"
    }
    return DriftResult(scores, threshold, len(reference), len(current)), snapshot
