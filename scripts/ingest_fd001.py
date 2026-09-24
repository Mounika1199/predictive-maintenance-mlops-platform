"""Ingestion entry point: raw C-MAPSS .txt files -> typed Parquet tables.

Run it from the repository root:

    python scripts/ingest_fd001.py

WHAT "INGESTION" MEANS HERE
It is the first stage of the pipeline and its job is narrow on purpose: read the
raw files faithfully, attach real column names and types, and save the result in
a good format. It does **not** compute RUL labels, engineer features, or split
the data. Keeping ingestion lossless means that when a later stage produces a
suspicious result, we can always come back to a known-good starting point.

WHY PARQUET INSTEAD OF CSV
A CSV is just text: every load has to re-guess whether a column is an int, a
float or a string, and it is large and slow. Parquet stores the schema
alongside the data in a compressed columnar layout -- so a reload gives back
exactly the dtypes we saved, in a fraction of the size and time. It is the
default artifact format in production data pipelines, and it is what DVC will
version for us in a later task.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

# Make `src` importable when this file is run directly as a script.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.load_cmapss import CmapssDataset, load_cmapss, load_config  # noqa: E402


def summarize_table(name: str, frame: pd.DataFrame) -> None:
    """Print a short human-readable description of one table."""
    print(f"\n  {name}")
    print(f"    rows x columns : {frame.shape[0]:,} x {frame.shape[1]}")

    if "unit_number" in frame.columns:
        n_engines = frame["unit_number"].nunique()
        print(f"    engines        : {n_engines}")

    if "time_in_cycles" in frame.columns:
        cycles_per_engine = frame.groupby("unit_number")["time_in_cycles"].max()
        print(
            f"    cycles/engine  : min {cycles_per_engine.min()}, "
            f"median {int(cycles_per_engine.median())}, "
            f"max {cycles_per_engine.max()}"
        )

    n_missing = int(frame.isna().sum().sum())
    print(f"    missing values : {n_missing}")


def write_parquet(dataset: CmapssDataset, interim_dir: Path) -> list[Path]:
    """Write each table to Parquet and return the paths written."""
    interim_dir.mkdir(parents=True, exist_ok=True)

    tables = {
        f"train_{dataset.name}": dataset.train,
        f"test_{dataset.name}": dataset.test,
        f"rul_truth_{dataset.name}": dataset.rul_truth,
    }

    written: list[Path] = []
    for stem, frame in tables.items():
        path = interim_dir / f"{stem}.parquet"
        frame.to_parquet(path, index=False)
        written.append(path)
    return written


def main() -> int:
    config = load_config()
    dataset_cfg = config["dataset"]

    print(f"Ingesting C-MAPSS subset: {dataset_cfg['name']}")
    dataset = load_cmapss(config)

    summarize_table("train", dataset.train)
    summarize_table("test", dataset.test)
    summarize_table("rul_truth", dataset.rul_truth)

    # Loud sanity check against the shapes recorded in the config. A silent
    # change in the input data is one of the nastiest failure modes in ML:
    # nothing crashes, the model just quietly gets worse.
    expected = dataset_cfg["expected"]
    problems: list[str] = []
    if len(dataset.train) != expected["train_rows"]:
        problems.append(
            f"train rows: expected {expected['train_rows']}, got {len(dataset.train)}"
        )
    if len(dataset.test) != expected["test_rows"]:
        problems.append(
            f"test rows: expected {expected['test_rows']}, got {len(dataset.test)}"
        )
    if problems:
        print("\nWARNING: input data does not match the expected shape:")
        for problem in problems:
            print(f"  - {problem}")
        return 1

    interim_dir = PROJECT_ROOT / dataset_cfg["interim_dir"]
    written = write_parquet(dataset, interim_dir)

    print("\nWrote:")
    for path in written:
        size_kb = path.stat().st_size / 1024
        print(f"  {path.relative_to(PROJECT_ROOT)}  ({size_kb:,.0f} KB)")

    print("\nIngestion complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
