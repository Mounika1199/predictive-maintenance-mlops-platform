# Predictive Maintenance MLOps Platform

End-to-end MLOps platform for predicting the **Remaining Useful Life (RUL)** of
turbofan engines, built on the NASA C-MAPSS Turbofan Engine Degradation dataset
(starting with the **FD001** subset).

This repository is built incrementally, one stage at a time.

## Current status

| Stage | Status |
|---|---|
| Data ingestion | Done |
| Data validation | Not started |
| Data versioning (DVC) | Not started |
| Feature engineering | Not started |
| Model training | Not started |
| Model evaluation | Not started |
| Experiment tracking (MLflow) | Not started |
| Model registry | Not started |
| Model serving | Not started |
| Monitoring & drift detection | Not started |
| Retraining | Not started |

## Setup

```powershell
python -m venv mlops_env
.\mlops_env\Scripts\Activate.ps1
pip install -r requirements.txt
```

The dataset is **not** in Git (see *Data* below). Place the three FD001 files in
`data/raw/`:

```
data/raw/train_FD001.txt
data/raw/test_FD001.txt
data/raw/RUL_FD001.txt
```

## Usage

```powershell
python scripts/ingest_fd001.py   # raw .txt -> typed Parquet in data/interim/
pytest                            # run the test suite
```

## Project layout

```
configs/data.yaml            Paths, column layout, expected shapes
data/raw/                    Original .txt files            (git-ignored)
data/interim/                Ingested Parquet tables        (git-ignored)
src/data/load_cmapss.py      Pure loading functions (importable, testable)
scripts/ingest_fd001.py      Entry point: reads config, loads, writes Parquet
tests/                       Pytest suite
```

The split between `src/` and `scripts/` is deliberate: `src/` holds **pure
logic** (inputs in, values out -- no printing, no file writing), `scripts/`
holds the **side effects**. Pure functions are easy to test and to reuse from a
training job or an API later; entry points are not.

## About the data

FD001 contains 100 engines in training and 100 *different* engines in test.
Each row is one operational cycle of one engine:

```
unit_number  time_in_cycles  op_setting_1..3  sensor_1..21
```

* **Train** trajectories run until the engine fails, so the last cycle of an
  engine is its failure point.
* **Test** trajectories are truncated at a random point *before* failure.
* `RUL_FD001.txt` gives the true remaining cycles for each test engine at its
  truncation point -- one value per engine, in engine-id order.

| Table | Rows | Engines | Cycles per engine (min / median / max) |
|---|---|---|---|
| train | 20,631 | 100 | 128 / 199 / 362 |
| test | 13,096 | 100 | 31 / 133 / 303 |
| rul_truth | 100 | 100 | — |

### Data is not stored in Git

`data/` is git-ignored. Git stores a complete copy of every version of every
file; putting datasets in it makes the repository permanently large and slow.
Data versioning is handled by DVC in a later stage, which keeps the data
elsewhere and only a small pointer file in Git.

### Parsing gotcha

Every line of the raw `.txt` files ends with **trailing spaces**. Reading them
with `sep=" "` makes pandas invent two extra all-NaN columns (28 instead of 26).
The loader uses `sep=r"\s+"`, and a test asserts the column count and the
absence of missing values so this cannot silently regress.

## Notes for later stages

* **Splitting must be done by engine (`unit_number`), never by row.** Rows from
  one engine are consecutive snapshots of the same degradation process; putting
  some in train and some in validation leaks the answer and produces
  validation scores that will not survive contact with reality.
* **Ingestion is intentionally lossless.** RUL labels and features are *not*
  computed here, so there is always a known-good starting point to fall back to
  when a later stage looks wrong.
