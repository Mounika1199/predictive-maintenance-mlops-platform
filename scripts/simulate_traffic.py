"""Pretend to be a fleet of engines using the prediction service.

    python scripts/simulate_traffic.py                              # all 100 test engines (~20 min)
    python scripts/simulate_traffic.py --engines 10                 # a quick run (~2 min)
    python scripts/simulate_traffic.py --broken-sensor sensor_11    # every reading of sensor_11 is off

Needs the service running (`docker compose up -d`). Every answered request is
written to the service's diary (logs/predictions/), which monitoring reads.
The run's start and end times are printed, so its lines can be picked out of
the diary later.

HOW THE FLEET FLIES (see src/monitoring/traffic.py)
Day by day: every engine flies once a day and, after each flight, sends its
last few cycles -- as many as the service says it needs (GET /model).

THE BROKEN SENSOR
--broken-sensor adds HALF of that sensor's usual spread in training (its
standard deviation, from data/processed/feature_stats.json) to every reading.
For sensor_11: +0.134, on readings that normally range from 46.85 to 48.53.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.load_cmapss import load_cmapss, load_config  # noqa: E402
from src.monitoring.traffic import break_sensor, fleet_requests  # noqa: E402

FEATURES_CONFIG = PROJECT_ROOT / "configs" / "features.yaml"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay the test engines into the prediction service.")
    parser.add_argument("--engines", type=int, help="only the first N engines (default: all 100)")
    parser.add_argument("--broken-sensor", metavar="SENSOR",
                        help="e.g. sensor_11: reads half its usual spread too high, in every reading")
    parser.add_argument("--url", default="http://127.0.0.1:8000", help="the service (default: %(default)s)")
    return parser.parse_args(argv)


def call(url: str, body: dict | None = None) -> dict:
    """GET (no body) or POST a JSON body; returns the service's JSON answer."""
    data = None if body is None else json.dumps(body).encode()
    request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    # Ask the service what it needs, like any well-behaved client.
    try:
        model = call(f"{args.url}/model")
    except urllib.error.URLError as exc:
        print(f"Cannot reach the service at {args.url} ({exc.reason}).")
        print("Start it with: docker compose up -d")
        return 1

    raw = load_cmapss(load_config()).test
    if args.broken_sensor:
        if args.broken_sensor not in model["sensors"]:
            print(f"--broken-sensor must be one of the model's sensors: {', '.join(model['sensors'])}")
            return 1
        stats_file = PROJECT_ROOT / load_config(FEATURES_CONFIG)["output"]["stats_file"]
        spread = json.loads(stats_file.read_text())["scaling"][args.broken_sensor]["std"]
        offset = spread / 2
        raw = break_sensor(raw, args.broken_sensor, offset)
        print(f"BROKEN SENSOR: {args.broken_sensor} reads {offset:.3f} too high in every reading.")

    requests = list(fleet_requests(raw, window=model["rolling_window"], n_engines=args.engines))
    n_engines = len({body["engine_id"] for body in requests})
    print(f"Sending {len(requests):,} requests ({n_engines} engines, day by day) to {args.url}, "
          f"model {model['model_name']} v{model['model_version']}")

    started = now()
    clock = time.perf_counter()
    answered = refused = with_warnings = 0
    for number, body in enumerate(requests, start=1):
        try:
            answer = call(f"{args.url}/predict", body)
        except urllib.error.HTTPError as exc:  # the service refused this request (e.g. 422)
            refused += 1
            if refused == 1:
                print(f"  refused ({exc.code}): {exc.read().decode()[:200]}")
            continue
        except urllib.error.URLError as exc:  # the service went away mid-run
            print(f"Lost the service after {number - 1:,} requests ({exc.reason}).")
            return 1
        answered += 1
        with_warnings += bool(answer["warnings"])
        if number % max(1, len(requests) // 10) == 0:
            print(f"  {number / len(requests):4.0%}  {number:,} sent  ({time.perf_counter() - clock:,.0f} s)")

    print(f"\nRun from {started} to {now()} (UTC), {time.perf_counter() - clock:,.0f} s")
    print(f"  answered     : {answered:,}  (each one is a line in the diary)")
    print(f"  with warnings: {with_warnings:,}  (a reading outside the training range)")
    print(f"  refused      : {refused:,}")
    return 0 if refused == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
