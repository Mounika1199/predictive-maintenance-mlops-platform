"""Build model-ready features: interim Parquet -> processed Parquet.

    python scripts/build_features.py

Run AFTER scripts/ingest_fd001.py, which writes the data/interim/ files this
script reads. Together they form a chain:

    raw .txt  --ingest-->  data/interim/  --build_features-->  data/processed/

This script contains almost no logic of its own. It calls the pieces in
src/ in the right order. The ORDER is the important part:

    1. labels      add the RUL answer column
    2. split       hold back 20% of engines for validation
    3. features    choose sensors (from TRAIN only), add rolling features
    4. scale       learn mean/std (from TRAIN only), apply to everything
    5. save

Splitting happens BEFORE anything is learned from the data. If we chose
sensors or computed averages first, the validation engines would already have
influenced those choices, and the validation score would no longer be honest.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.load_cmapss import load_config  # noqa: E402
from src.data.split import split_by_engine  # noqa: E402
from src.features import engineering as fe  # noqa: E402
from src.features.labels import add_test_rul, add_train_rul  # noqa: E402

FEATURES_CONFIG = PROJECT_ROOT / "configs" / "features.yaml"

# Columns kept alongside the features: who, when, and the answers.
ID_COLUMNS = ["unit_number", "time_in_cycles"]
LABEL_COLUMNS = ["rul", "rul_uncapped"]


def read_interim_tables(data_config: dict) -> dict[str, pd.DataFrame]:
    """Load the Parquet files written by the ingestion step."""
    dataset = data_config["dataset"]
    interim_dir = PROJECT_ROOT / dataset["interim_dir"]
    name = dataset["name"]

    paths = {
        key: interim_dir / f"{key}_{name}.parquet"
        for key in ("train", "test", "rul_truth")
    }
    missing = [str(p.relative_to(PROJECT_ROOT)) for p in paths.values() if not p.is_file()]
    if missing:
        raise FileNotFoundError(
            f"Missing ingestion output: {missing}\n"
            "Run this first: python scripts/ingest_fd001.py"
        )
    return {key: pd.read_parquet(path) for key, path in paths.items()}


def main() -> int:
    data_config = load_config()
    feature_config = load_config(FEATURES_CONFIG)

    dataset_name = data_config["dataset"]["name"]
    cap = feature_config["label"]["cap"]
    validation_fraction = feature_config["split"]["validation_fraction"]
    seed = feature_config["split"]["seed"]
    window = feature_config["features"]["rolling_window"]
    include_cycle = feature_config["features"]["include_cycle_feature"]

    tables = read_interim_tables(data_config)

    # 1. Labels -- the answer column, worked out differently for train and test.
    train = add_train_rul(tables["train"], cap)
    test = add_test_rul(tables["test"], tables["rul_truth"], cap)

    # 2. Split -- whole engines, BEFORE anything is learned from the data.
    split = split_by_engine(train, validation_fraction, seed)

    # 3. Features -- which sensors to use is decided from the TRAINING split.
    sensors = fe.select_sensor_columns(split.train)

    def build(frame: pd.DataFrame) -> pd.DataFrame:
        return fe.add_cycle_feature(fe.add_rolling_features(frame, sensors, window))

    built = {
        "train": build(split.train),
        "val": build(split.validation),
        "test": build(test),
    }

    # 4. Scale -- averages learned from the TRAINING split, applied to all three.
    columns = fe.feature_columns(sensors, include_cycle)
    stats = fe.fit_scaler(built["train"], columns)

    keep = [*ID_COLUMNS, *columns, *LABEL_COLUMNS]
    outputs = {name: fe.apply_scaler(frame, stats)[keep] for name, frame in built.items()}

    # Fail loudly rather than save broken data.
    for name, frame in outputs.items():
        n_missing = int(frame.isna().sum().sum())
        if n_missing:
            raise ValueError(f"{name}: {n_missing} missing values after feature building")

    # 5. Save.
    processed_dir = PROJECT_ROOT / feature_config["output"]["processed_dir"]
    processed_dir.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    for name, frame in outputs.items():
        path = processed_dir / f"{name}_features.parquet"
        frame.to_parquet(path, index=False)
        written.append(path)

    # Everything needed to rebuild these features EXACTLY, and later to apply
    # the identical transformation to live data.
    stats_path = PROJECT_ROOT / feature_config["output"]["stats_file"]
    stats_document = {
        "dataset": dataset_name,
        "rul_cap": cap,
        "rolling_window": window,
        "split": {
            "seed": seed,
            "validation_fraction": validation_fraction,
            "train_engines": split.train_engines,
            "validation_engines": split.validation_engines,
        },
        "sensors": sensors,
        "feature_columns": columns,
        "scaling": stats,
    }
    with stats_path.open("w", encoding="utf-8") as handle:
        json.dump(stats_document, handle, indent=2)
        handle.write("\n")
    written.append(stats_path)

    # Summary.
    print(f"Built features for {dataset_name}")
    print(f"  RUL cap        : {cap}")
    print(f"  rolling window : {window} cycles")
    print(f"  sensors used   : {len(sensors)}")
    print(f"  feature count  : {len(columns)}")
    print()
    print(f"  {'split':<6}{'engines':>9}{'rows':>9}")
    for name, frame in outputs.items():
        print(f"  {name:<6}{frame['unit_number'].nunique():>9}{len(frame):>9,}")
    print(f"\n  validation engines: {split.validation_engines}")

    print("\nWrote:")
    for path in written:
        print(f"  {path.relative_to(PROJECT_ROOT)}  ({path.stat().st_size / 1024:,.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
