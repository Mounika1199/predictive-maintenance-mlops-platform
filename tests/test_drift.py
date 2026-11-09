"""Tests for the drift check (src/monitoring/drift.py, scripts/check_drift.py).

The test engines play a normal fleet -- exactly the readings the traffic
simulator sends -- and the same engines with sensor_11 reading half its usual
spread too high play a broken one. Each rule in configs/monitoring.yaml has a
test showing what goes wrong without it.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from src.data.load_cmapss import load_cmapss, load_config
from src.monitoring.drift import (
    NotEnoughReadings,
    check_drift,
    current_readings,
    load_drift_config,
    reference_readings,
)
from src.monitoring.traffic import break_sensor

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def setup() -> dict:
    stats = json.loads((PROJECT_ROOT / "data" / "processed" / "feature_stats.json").read_text())
    data = load_cmapss(load_config())
    return {
        "config": load_drift_config(),
        "sensors": stats["sensors"],
        "train_engines": stats["split"]["train_engines"],
        "offset": stats["scaling"]["sensor_11"]["std"] / 2,
        "train": data.train,
        "test": data.test,
    }


def run(setup: dict, current_raw: pd.DataFrame, *, max_cycle: int | None = None, method: str | None = None):
    """The drift check with the configured rules, optionally changing one."""
    config = setup["config"]
    max_cycle = max_cycle or config["max_cycle"]
    reference = reference_readings(setup["train"], setup["train_engines"], setup["sensors"], max_cycle)
    current = current_raw[current_raw["time_in_cycles"] <= max_cycle][setup["sensors"]]
    result, _ = check_drift(reference, current, setup["sensors"], method or config["method"],
                            config["threshold"], config["min_readings"])
    return result


# ---------------------------------------------------------------------------
# The verdicts
# ---------------------------------------------------------------------------


def test_a_normal_fleet_shows_no_drift(setup) -> None:
    result = run(setup, setup["test"])
    assert not result.drift, f"false alarm: {result.drifted}"


def test_a_broken_sensor_is_caught_and_only_that_sensor(setup) -> None:
    broken = break_sensor(setup["test"], "sensor_11", setup["offset"])
    result = run(setup, broken)
    assert result.drifted == ["sensor_11"]
    assert result.drift  # ONE sensor of 15 is enough (Evidently's own verdict needs half)


# ---------------------------------------------------------------------------
# Why each rule exists: break it, and a normal fleet raises a false alarm
# ---------------------------------------------------------------------------


def test_comparing_all_ages_would_raise_a_false_alarm(setup) -> None:
    """The test fleet is younger than the training engines; comparing all
    ages measures that, not a fault."""
    result = run(setup, setup["test"], max_cycle=400)
    assert result.drift


def test_evidently_default_method_would_raise_a_false_alarm(setup) -> None:
    """Wasserstein's line is stricter than the natural differences between engines."""
    result = run(setup, setup["test"], method="wasserstein")
    assert len(result.drifted) > 10


def test_too_few_readings_get_no_verdict(setup) -> None:
    """With a few hundred readings, chance alone crosses the line."""
    config = setup["config"]
    reference = reference_readings(setup["train"], setup["train_engines"], setup["sensors"], config["max_cycle"])
    current = setup["test"][setup["sensors"]].head(config["min_readings"] - 1)
    with pytest.raises(NotEnoughReadings, match="at least 1,000"):
        check_drift(reference, current, setup["sensors"], config["method"], config["threshold"], config["min_readings"])


# ---------------------------------------------------------------------------
# Choosing what to compare
# ---------------------------------------------------------------------------


def test_reference_is_the_training_engines_at_the_chosen_ages(setup) -> None:
    reference = reference_readings(setup["train"], setup["train_engines"], setup["sensors"], 100)
    assert len(reference) == len(setup["train_engines"]) * 100  # every training engine lives past 100
    assert list(reference.columns) == setup["sensors"]


def test_current_readings_keep_the_time_window_and_the_ages() -> None:
    diary = pd.DataFrame({
        "time": pd.to_datetime(["2026-11-06T10:00Z", "2026-11-06T11:00Z", "2026-11-06T12:00Z", "2026-11-06T13:00Z"]),
        "time_in_cycles": [5, 50, 150, 60],
        "sensor_2": [1.0, 2.0, 3.0, 4.0],
    })
    picked = current_readings(diary, ["sensor_2"], max_cycle=100,
                              since=pd.Timestamp("2026-11-06T11:00Z"), until=pd.Timestamp("2026-11-06T13:00Z"))
    assert picked["sensor_2"].tolist() == [2.0]  # 10:00 too early, cycle 150 too old, 13:00 is the end


def test_an_empty_diary_gives_no_readings() -> None:
    assert current_readings(pd.DataFrame(), ["sensor_2"], max_cycle=100).empty


# ---------------------------------------------------------------------------
# The script: exit codes a scheduler can react to
# ---------------------------------------------------------------------------


def write_diary(folder: Path, raw: pd.DataFrame) -> None:
    folder.mkdir()
    lines = raw.drop(columns="unit_number").assign(time="2026-11-06T12:00:00+00:00").to_dict(orient="records")
    with (folder / "predictions-2026-11-06.jsonl").open("w", encoding="utf-8") as file:
        file.writelines(json.dumps(line) + "\n" for line in lines)


def check_drift_script(diary: Path, reports: Path) -> int:
    return subprocess.run(
        [sys.executable, "scripts/check_drift.py", "--diary", str(diary), "--report-dir", str(reports)],
        cwd=PROJECT_ROOT, capture_output=True, text=True,
    ).returncode


def test_the_script_exits_1_on_drift_and_saves_a_report(setup, tmp_path: Path) -> None:
    write_diary(tmp_path / "diary", break_sensor(setup["test"], "sensor_11", setup["offset"]))
    assert check_drift_script(tmp_path / "diary", tmp_path / "reports") == 1
    assert len(list((tmp_path / "reports").glob("drift-*.html"))) == 1


def test_the_script_exits_2_when_there_is_too_little_to_judge(tmp_path: Path) -> None:
    (tmp_path / "diary").mkdir()
    assert check_drift_script(tmp_path / "diary", tmp_path / "reports") == 2
