"""Loading logic for the NASA C-MAPSS turbofan degradation dataset.

This module is deliberately *pure*: it reads files and returns DataFrames. It
does not write anything, does not print, and does not know about the command
line. All of that lives in ``scripts/ingest_fd001.py``.

WHY SPLIT IT THAT WAY?
Functions that only transform inputs into outputs are trivial to test and to
reuse. The moment a function also writes files or prints, testing it means
dealing with the filesystem. We will reuse this same split for training and for
serving, where "pure transformation" vs "side effects" matters a great deal
more.

ABOUT THE DATASET
Each row is one operational cycle (roughly, one flight) of one engine:

    unit_number  time_in_cycles  op_setting_1..3  sensor_1..21

* ``train_FD001``: 100 engines run to failure. The last row of an engine is the
  cycle at which it failed.
* ``test_FD001``: 100 *different* engines whose trajectories are cut off at some
  random point *before* failure.
* ``RUL_FD001``: the true Remaining Useful Life (in cycles) of each test engine
  at the point where its trajectory was cut off -- one value per test engine.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import yaml

# Repository root, derived from this file's location (src/data/load_cmapss.py).
# This makes the code work no matter which directory you run it from.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "data.yaml"

# Column names that are always present, before the sensor columns.
ID_COLUMNS = ["unit_number", "time_in_cycles"]


@dataclass(frozen=True)
class CmapssDataset:
    """The three raw tables of a single C-MAPSS subset (e.g. FD001).

    Grouping them in one object means callers cannot accidentally pair the
    train table of FD001 with the RUL file of FD002. ``frozen=True`` makes it
    read-only, which prevents a whole class of "something mutated my data
    halfway through the pipeline" bugs.
    """

    name: str
    train: pd.DataFrame
    test: pd.DataFrame
    rul_truth: pd.DataFrame


def load_config(path: Path | str = DEFAULT_CONFIG_PATH) -> dict:
    """Read the YAML data configuration into a plain dictionary."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def build_column_names(n_operational_settings: int, n_sensors: int) -> list[str]:
    """Build the column names for a raw C-MAPSS table.

    The raw .txt files contain no header row, so we supply the names ourselves,
    in the exact order the dataset documentation specifies.

    >>> build_column_names(3, 21)[:4]
    ['unit_number', 'time_in_cycles', 'op_setting_1', 'op_setting_2']
    """
    if n_operational_settings < 1 or n_sensors < 1:
        raise ValueError(
            "n_operational_settings and n_sensors must both be >= 1, got "
            f"{n_operational_settings} and {n_sensors}"
        )

    op_columns = [f"op_setting_{i}" for i in range(1, n_operational_settings + 1)]
    sensor_columns = [f"sensor_{i}" for i in range(1, n_sensors + 1)]
    return [*ID_COLUMNS, *op_columns, *sensor_columns]


def load_raw_table(path: Path | str, columns: list[str]) -> pd.DataFrame:
    """Read one raw C-MAPSS .txt file into a typed DataFrame.

    THE TRAILING-WHITESPACE TRAP
    Every line in these files ends with trailing spaces. Parsing with
    ``sep=" "`` makes pandas treat each run of spaces as a separator and invent
    two extra, entirely empty columns at the end. Those phantom columns are
    all-NaN and silently poison anything downstream that iterates over columns.
    ``sep=r"\\s+"`` treats *any run of whitespace* as a single separator, which
    is what we actually want. The test suite locks this behaviour down.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Raw data file not found: {path}\n"
            "Expected the C-MAPSS .txt files to live in data/raw/."
        )

    frame = pd.read_csv(path, sep=r"\s+", header=None, engine="python")

    if frame.shape[1] != len(columns):
        raise ValueError(
            f"{path.name}: expected {len(columns)} columns but found "
            f"{frame.shape[1]}. The file format may have changed."
        )

    frame.columns = columns

    # Set dtypes by MEANING, not by whatever pandas happened to infer.
    #
    # unit_number and time_in_cycles are counters -> integers.
    #
    # Everything else is a continuous physical measurement -> float64. This
    # matters more than it looks: sensor_17 and sensor_18 infer as int64 purely
    # because the raw text prints them as "392" and "2388" with no decimal
    # point. If a future export wrote "392.0" instead, the dtype would silently
    # flip and any strict schema check would reject perfectly valid data. Dtype
    # should describe what a column *is*, not how it was formatted.
    dtypes = {column: "float64" for column in columns}
    dtypes["unit_number"] = "int64"
    dtypes["time_in_cycles"] = "int64"

    return frame.astype(dtypes)


def load_rul_truth(path: Path | str) -> pd.DataFrame:
    """Read the ground-truth RUL file into a keyed two-column table.

    The raw file is just 100 bare numbers, one per line, with no engine id. The
    id is implicit: line *i* is the RUL of test engine *i*. We make that
    explicit and return a proper ``unit_number`` / ``rul`` table, because a
    bare vector whose meaning depends on row order is an invitation to
    misalignment bugs later when we join it to predictions.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"RUL truth file not found: {path}")

    frame = pd.read_csv(path, sep=r"\s+", header=None, engine="python")

    if frame.shape[1] != 1:
        raise ValueError(
            f"{path.name}: expected a single column of RUL values, found "
            f"{frame.shape[1]} columns."
        )

    frame.columns = ["rul"]
    frame = frame.astype({"rul": "int64"})

    # Engine ids are 1-based, matching unit_number in the test table.
    frame.insert(0, "unit_number", range(1, len(frame) + 1))
    return frame


def load_cmapss(config: dict, project_root: Path = PROJECT_ROOT) -> CmapssDataset:
    """Load the train, test and RUL-truth tables described by ``config``."""
    dataset_cfg = config["dataset"]
    raw_dir = project_root / dataset_cfg["raw_dir"]
    files = dataset_cfg["files"]

    columns = build_column_names(
        n_operational_settings=dataset_cfg["n_operational_settings"],
        n_sensors=dataset_cfg["n_sensors"],
    )

    return CmapssDataset(
        name=dataset_cfg["name"],
        train=load_raw_table(raw_dir / files["train"], columns),
        test=load_raw_table(raw_dir / files["test"], columns),
        rul_truth=load_rul_truth(raw_dir / files["rul_truth"]),
    )
