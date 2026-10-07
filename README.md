# Predictive Maintenance MLOps Platform

End-to-end MLOps platform for predicting the **Remaining Useful Life (RUL)** of
turbofan engines, built on the NASA C-MAPSS Turbofan Engine Degradation dataset
(starting with the **FD001** subset).

This repository is built incrementally, one stage at a time.

## Current status

| Stage | Status |
|---|---|
| Data ingestion | Done |
| Data validation | Done |
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
python scripts/ingest_fd001.py              # raw .txt -> typed Parquet in data/interim/
python scripts/build_reference_profile.py   # regenerate the validation contract (rarely)
python scripts/validate_fd001.py            # check the data against that contract
pytest                                       # run the test suite
```

`validate_fd001.py` exits non-zero when validation fails, so it can become an
automatic gate in a pipeline or in CI later.

## Project layout

```
configs/data.yaml                  Paths, column layout, expected shapes, tolerances
configs/reference_profile.json     Generated contract: per-column dtype/min/max
data/raw/                          Original .txt files             (git-ignored)
data/interim/                      Ingested Parquet tables         (git-ignored)
src/data/load_cmapss.py            Pure loading functions (importable, testable)
src/data/validation.py             Schema building + structural checks
scripts/ingest_fd001.py            Entry point: loads raw, writes Parquet
scripts/build_reference_profile.py Entry point: writes the reference profile
scripts/validate_fd001.py          Entry point: validates, exits non-zero on failure
tests/                             Pytest suite
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

## Data validation

Validation declares a contract the data must satisfy, and fails loudly when it
does not. The failures that matter in ML are silent ones -- a recalibrated
sensor, a reordered column, a burst of NaNs -- which produce a quietly worse
model rather than a crash.

Rules are **not** hardcoded. `build_reference_profile.py` measures the training
data once into `configs/reference_profile.json` (per column: dtype, min, max,
whether it is constant), and validation checks against that committed file. At
serving time the training set is not in memory -- a stored contract is what you
actually have. The same file will later be the baseline for drift detection.

Two tolerances, solving unrelated problems:

| | `constant_tolerance` | `range_tolerance` |
|---|---|---|
| Fixes | floating-point rounding noise | training data being a limited sample |
| Value | `1e-6` | `0.10` (10%) |
| Kind | absolute | relative, as a fraction of the column's span |

The second one is not optional padding. **7 of 17 varying columns have test
values outside the training range** -- a zero-tolerance rule would reject our own
valid held-out data, and an alarm that cries wolf gets ignored.

The profile also records that **7 of 26 columns are perfectly constant**
(`op_setting_3`, `sensor_1`, `sensor_5`, `sensor_10`, `sensor_16`, `sensor_18`,
`sensor_19`). They carry no predictive information, but they make excellent
tripwires: if one ever moves, something upstream changed.

### Import note

`src/data/validation.py` imports `pandera.pandas`, not bare `pandera`. Pandera
dispatches checks to per-library implementations, and importing the pandas
namespace is what registers them. A bare `import pandera` left the registry
empty and made every check fail with
`KeyError("<class 'pandas.core.series.Series'>")` -- an error that looks like a
data problem but is purely an import problem.

## Notes for later stages

* **Splitting must be done by engine (`unit_number`), never by row.** Rows from
  one engine are consecutive snapshots of the same degradation process; putting
  some in train and some in validation leaks the answer and produces
  validation scores that will not survive contact with reality.
* **Ingestion is intentionally lossless.** RUL labels and features are *not*
  computed here, so there is always a known-good starting point to fall back to
  when a later stage looks wrong.
