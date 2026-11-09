"""Tests for the diary itself: files, reading back, and writing safely.

What the SERVICE writes into it (newest cycle, the answer, warnings) is tested
in test_serving.py.
"""

from __future__ import annotations

import subprocess
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import pytest

from src.monitoring.prediction_log import PredictionLog, default_log_dir, read_prediction_log

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def at(*parts: int) -> datetime:
    return datetime(*parts, tzinfo=timezone.utc)


def test_one_file_per_day_read_back_oldest_first(tmp_path: Path) -> None:
    log = PredictionLog(tmp_path)
    log.write({"engine_id": "late"}, now=at(2026, 10, 28, 0, 0, 1))
    log.write({"engine_id": "early"}, now=at(2026, 10, 27, 23, 59, 59))

    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "predictions-2026-10-27.jsonl",
        "predictions-2026-10-28.jsonl",
    ]
    table = read_prediction_log(tmp_path)
    assert table["engine_id"].tolist() == ["early", "late"]
    assert table["time"].dt.tz is not None  # times are kept in UTC


def test_engine_ids_stay_text(tmp_path: Path) -> None:
    """pd.read_json would turn "007" into the number 7."""
    PredictionLog(tmp_path).write({"engine_id": "007"})
    assert read_prediction_log(tmp_path)["engine_id"].tolist() == ["007"]


def test_an_empty_or_missing_diary_reads_as_an_empty_table(tmp_path: Path) -> None:
    assert read_prediction_log(tmp_path).empty
    assert read_prediction_log(tmp_path / "never-created").empty


def test_a_damaged_line_is_reported_with_its_place(tmp_path: Path) -> None:
    (tmp_path / "predictions-2026-10-27.jsonl").write_text(
        '{"engine_id": "a"}\n{"engine_id": \n', encoding="utf-8"
    )
    with pytest.raises(ValueError, match="predictions-2026-10-27.jsonl, line 2"):
        read_prediction_log(tmp_path)


def test_a_failed_write_is_reported_not_raised(tmp_path: Path) -> None:
    blocked = tmp_path / "a-file-not-a-folder"
    blocked.write_text("x", encoding="utf-8")
    assert PredictionLog(blocked).write({"engine_id": "a"}) is False


def test_lines_stay_whole_when_many_requests_write_at_once(tmp_path: Path) -> None:
    """The service answers requests in parallel threads. Without the lock,
    lines written at the same moment overwrite each other (tried: 360 of 400
    lines survived, in 5 runs out of 5)."""
    log = PredictionLog(tmp_path)
    record = {"engine_id": "x", "padding": "y" * 20_000}

    def write_many(_: int) -> None:
        for _ in range(50):
            log.write(record)

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(write_many, range(8)))

    table = read_prediction_log(tmp_path)  # raises if any line were broken
    assert len(table) == 8 * 50


def test_the_diary_lives_in_the_project_logs_folder() -> None:
    assert default_log_dir() == PROJECT_ROOT / "logs" / "predictions"


def test_the_diary_is_hidden_from_git() -> None:
    """Like data, the diary is an output of the running service, not source code."""
    ignored = subprocess.run(
        ["git", "check-ignore", "-q", "logs/predictions/predictions-2026-10-27.jsonl"], cwd=PROJECT_ROOT
    )
    assert ignored.returncode == 0
