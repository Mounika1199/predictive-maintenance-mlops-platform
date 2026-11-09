"""Has the data the service sees drifted away from the training data?

    python scripts/check_drift.py                                      # the whole diary
    python scripts/check_drift.py --since 2026-11-06T19:13:43+00:00    # only lines from then on
    python scripts/check_drift.py --since ... --until ...              # a time window (UTC)

Reads the service's diary (logs/predictions/), compares its readings with the
champion's training readings using Evidently (rules in src/monitoring/drift.py
and configs/monitoring.yaml), prints a verdict and saves an HTML report with a
chart per sensor in reports/.

EXIT CODES -- so a scheduler or the retraining stage can react:
    0  no drift
    1  drift: at least one sensor has drifted
    2  no verdict: too few readings in the window to tell drift from chance
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.load_cmapss import load_cmapss, load_config  # noqa: E402
from src.monitoring.drift import (  # noqa: E402
    NotEnoughReadings,
    check_drift,
    current_readings,
    load_drift_config,
    reference_readings,
)
from src.monitoring.prediction_log import default_log_dir, read_prediction_log  # noqa: E402

FEATURES_CONFIG = PROJECT_ROOT / "configs" / "features.yaml"


def utc(text: str) -> pd.Timestamp:
    """A timestamp from the command line; without a time zone, UTC is assumed."""
    moment = pd.Timestamp(text)
    return moment.tz_localize("UTC") if moment.tzinfo is None else moment


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check the service's diary for data drift.")
    parser.add_argument("--since", type=utc, help="only diary lines from this time on (UTC)")
    parser.add_argument("--until", type=utc, help="only diary lines before this time (UTC)")
    parser.add_argument("--diary", type=Path, default=default_log_dir(), help="diary folder (default: %(default)s)")
    parser.add_argument("--report-dir", type=Path, help="where the HTML report goes (default: from the config)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_drift_config()
    report_dir = args.report_dir or PROJECT_ROOT / config["report_dir"]

    # The reference: the readings the model learned from.
    stats_file = PROJECT_ROOT / load_config(FEATURES_CONFIG)["output"]["stats_file"]
    stats = json.loads(stats_file.read_text())
    sensors = stats["sensors"]
    reference = reference_readings(
        load_cmapss(load_config()).train, stats["split"]["train_engines"], sensors, config["max_cycle"]
    )

    # The current data: what the service saw, in the chosen window.
    diary = read_prediction_log(args.diary)
    current = current_readings(diary, sensors, config["max_cycle"], args.since, args.until)
    window = f"{args.since or 'the beginning'} to {args.until or 'now'}"
    print(f"Diary {args.diary}, {window}")
    print(f"Comparing {len(current):,} readings (engine ages 1-{config['max_cycle']}) "
          f"with {len(reference):,} training readings, using {config['method'].upper()}.\n")

    try:
        result, snapshot = check_drift(
            reference, current, sensors, config["method"], config["threshold"], config["min_readings"]
        )
    except NotEnoughReadings as exc:
        print(f"NO VERDICT: {exc}.")
        return 2

    for sensor, score in sorted(result.scores.items(), key=lambda item: -item[1]):
        mark = "  <-- DRIFTED" if score >= result.threshold else ""
        print(f"  {sensor:10s} {score:6.3f}{mark}")

    report_dir.mkdir(parents=True, exist_ok=True)
    report = report_dir / f"drift-{datetime.now(timezone.utc):%Y-%m-%dT%H%M%SZ}.html"
    snapshot.save_html(str(report))

    print(f"\nA sensor has drifted at a score of {result.threshold} or more.")
    if result.drift:
        print(f"DRIFT: {', '.join(result.drifted)}")
    else:
        print("NO DRIFT")
    print(f"Report: {report}")
    return 1 if result.drift else 0


if __name__ == "__main__":
    raise SystemExit(main())
