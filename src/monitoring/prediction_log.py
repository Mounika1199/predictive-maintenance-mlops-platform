"""The service's diary: one line per prediction.

Every answered /predict request adds ONE line of JSON ("JSON Lines") to the
file for that day (UTC), e.g. logs/predictions/predictions-2026-10-27.jsonl:

    {"time": "2026-10-27T09:15:02.123456+00:00", "engine_id": "17",
     "time_in_cycles": 31, "op_setting_1": -0.0006, ..., "sensor_21": 23.4,
     "remaining_useful_life": 112.4, "model_name": "rul-fd001",
     "model_version": "3", "warnings": []}

Only the NEWEST cycle of each request is written. A request repeats earlier
cycles because the rolling features need them; writing them all would record
the same cycle again with every request.

Refused requests (422) are not written: no prediction was made.

Monitoring reads these files: the readings, to check for drift; the
predictions, to compare with the true RUL once it is known.
"""

from __future__ import annotations

import json
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.data.load_cmapss import PROJECT_ROOT, load_config

MONITORING_CONFIG = PROJECT_ROOT / "configs" / "monitoring.yaml"

logger = logging.getLogger(__name__)


def default_log_dir() -> Path:
    """The diary folder from configs/monitoring.yaml (relative to the project)."""
    directory = Path(load_config(MONITORING_CONFIG)["prediction_log"]["directory"])
    return directory if directory.is_absolute() else PROJECT_ROOT / directory


class PredictionLog:
    """Adds predictions to one JSON Lines file per day."""

    def __init__(self, directory: Path) -> None:
        self.directory = Path(directory)
        # The service answers several requests at the same time, in threads.
        # The lock lets only one of them write at a time. Without it, lines
        # written at the same moment overwrite each other: in a test with 8
        # threads on Windows, 40 of 400 lines were silently lost.
        self._lock = threading.Lock()

    def path_for(self, moment: datetime) -> Path:
        return self.directory / f"predictions-{moment:%Y-%m-%d}.jsonl"

    def write(self, record: dict, now: datetime | None = None) -> bool:
        """Add one line. Returns False -- and never raises -- if it cannot.

        By the time we write, the prediction is made: a full disk must not
        stop it reaching the caller. The failure goes to the service's own log.
        """
        now = now or datetime.now(timezone.utc)
        line = json.dumps({"time": now.isoformat(), **record})
        try:
            with self._lock:
                self.directory.mkdir(parents=True, exist_ok=True)
                with self.path_for(now).open("a", encoding="utf-8") as file:
                    file.write(line + "\n")
        except OSError as exc:
            logger.warning("Prediction NOT written to the diary in %s: %s", self.directory, exc)
            return False
        return True


def read_prediction_log(directory: Path) -> pd.DataFrame:
    """Every diary line, oldest first, as one table (empty if there are none).

    Read with json.loads rather than pd.read_json: pandas would quietly turn
    an engine_id like "007" into the number 7.
    """
    records = []
    for path in sorted(Path(directory).glob("predictions-*.jsonl")):
        with path.open(encoding="utf-8") as file:
            for number, line in enumerate(file, start=1):
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path.name}, line {number}: not valid JSON ({exc})") from exc

    table = pd.DataFrame(records)
    if not table.empty:
        table["time"] = pd.to_datetime(table["time"], format="ISO8601")
    return table
