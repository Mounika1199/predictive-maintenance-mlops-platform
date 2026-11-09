"""Tests for the pretend fleet (src/monitoring/traffic.py).

The simulator is only useful if its traffic is exactly what we think it is:
every cycle sent once as the newest, in fleet order, and never refused.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.load_cmapss import load_cmapss, load_config
from src.monitoring.traffic import break_sensor, fleet_requests
from src.serving.checks import check_history
from src.serving.schemas import MEASUREMENTS


@pytest.fixture(scope="module")
def raw_test() -> pd.DataFrame:
    return load_cmapss(load_config()).test


@pytest.fixture(scope="module")
def requests(raw_test) -> list[dict]:
    return list(fleet_requests(raw_test, window=5))


def newest_cycle(body: dict) -> int:
    return body["readings"][-1]["time_in_cycles"]


def test_every_cycle_of_every_engine_is_the_newest_of_exactly_one_request(requests, raw_test) -> None:
    """So the diary gets one line per cycle: 13,096 for the 100 test engines."""
    newest = [(body["engine_id"], newest_cycle(body)) for body in requests]
    assert len(newest) == len(set(newest)) == len(raw_test)


def test_the_fleet_flies_day_by_day(requests) -> None:
    days = [newest_cycle(body) for body in requests]
    assert days == sorted(days)

    day_1 = requests[:100]
    assert [body["engine_id"] for body in day_1] == [f"engine-{unit}" for unit in range(1, 101)]
    assert all(len(body["readings"]) == 1 for body in day_1)


def test_each_request_carries_the_last_5_cycles_or_all_there_are(requests) -> None:
    for body in requests:
        assert len(body["readings"]) == min(newest_cycle(body), 5)


def test_every_request_is_one_the_service_accepts(requests) -> None:
    """Simulated traffic must never be refused: exactly the fields the service
    expects (no unit_number -- it forbids unknown fields), and a history that
    passes the service's own check."""
    for body in requests[::7]:  # every 7th request: ~1,900, all days and engines
        assert list(body["readings"][0]) == ["time_in_cycles", *MEASUREMENTS]
        check_history(pd.DataFrame(body["readings"]), window=5)


def test_a_quick_run_keeps_the_first_engines(raw_test) -> None:
    ids = {body["engine_id"] for body in fleet_requests(raw_test, window=5, n_engines=3)}
    assert ids == {"engine-1", "engine-2", "engine-3"}


def test_a_broken_sensor_reads_too_high_everywhere_and_nothing_else_changes(raw_test) -> None:
    original = raw_test.copy()
    broken = break_sensor(raw_test, "sensor_11", 0.134)

    np.testing.assert_allclose(broken["sensor_11"] - raw_test["sensor_11"], 0.134)
    pd.testing.assert_frame_equal(broken.drop(columns="sensor_11"), raw_test.drop(columns="sensor_11"))
    pd.testing.assert_frame_equal(raw_test, original)  # the input itself is untouched
