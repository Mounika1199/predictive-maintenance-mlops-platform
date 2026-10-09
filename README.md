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
| RUL labelling & feature engineering | Done |
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
python scripts/build_features.py            # labels, split, features -> data/processed/
pytest                                       # run the test suite
```

`build_features.py` reads the files written by `ingest_fd001.py`, so run
ingestion first. If the loading code changes, re-run ingestion too -- the files
in `data/interim/` do not update themselves. (Tracking which outputs are out of
date is one of the jobs DVC will take over in a later stage.)

`validate_fd001.py` exits non-zero when validation fails, so it can become an
automatic gate in a pipeline or in CI later.

## Project layout

```
configs/data.yaml                  Paths, column layout, expected shapes, tolerances
configs/features.yaml              Modelling choices: RUL cap, split, rolling window
configs/reference_profile.json     Generated contract: per-column dtype/min/max
data/raw/                          Original .txt files             (git-ignored)
data/interim/                      Ingested Parquet tables         (git-ignored)
data/processed/                    Model-ready features + stats    (git-ignored)
src/data/load_cmapss.py            Pure loading functions (importable, testable)
src/data/validation.py             Schema building + structural checks
src/data/split.py                  Engine-level train/validation split
src/features/labels.py             RUL target for train and test
src/features/engineering.py        Sensor choice, rolling features, scaling
scripts/ingest_fd001.py            Entry point: loads raw, writes Parquet
scripts/build_reference_profile.py Entry point: writes the reference profile
scripts/validate_fd001.py          Entry point: validates, exits non-zero on failure
scripts/build_features.py          Entry point: interim -> processed features
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

## RUL labels and features

`build_features.py` turns validated data into a supervised learning problem, in
this order: **labels → split → features → scaling**. The split comes before
anything is learned from the data, so validation engines cannot influence any
decision.

### The target: Remaining Useful Life

RUL = how many more cycles (roughly, flights) until the engine fails. Train and
test need different arithmetic because they were recorded differently:

* **Train** engines ran until they broke, so we count backwards from the last
  row: `rul = last_cycle - time_in_cycles`.
* **Test** engines were switched off early. `RUL_FD001.txt` gives their
  remaining life at the last observed cycle, and we count back from that:
  `rul = truth + (last_cycle - time_in_cycles)`.

### Capping RUL at 125

Average sensor readings at different amounts of life remaining (training data):

| life left | sensor_4 | sensor_11 |
|---|---|---|
| 300 | 1402.48 | 47.24 |
| 200 | 1401.64 | 47.34 |
| 100 | 1406.73 | 47.46 |
| 5 | 1427.52 | 48.10 |

Between 300 and 200 cycles left, the sensors barely move -- `sensor_4` even
drifts the wrong way. Between 100 and 5 they move up to 25x more. An engine with
300 cycles left and one with 200 look the same, so asking a model to tell them
apart is asking the impossible.

So the training target is capped: anything above 125 becomes 125. The true value
is kept in `rul_uncapped`. The cost is small: only 11 of 100 test engines truly
exceed 125, by at most 20 cycles.

**Capping is a training aid, not the truth.** Final scores should be computed
against `rul_uncapped`.

### Split

80 engines for training, 20 for validation, chosen by shuffling **engine ids**
with a fixed seed (42). Every row of an engine stays on the same side. A row-level
split would put near-identical neighbouring cycles on both sides and produce a
flattering, meaningless validation score.

### Features (46)

* **15 sensors** -- the 6 sensors that never change in training are dropped.
  Operating settings are dropped too: FD001 has a single operating condition, so
  they are noise.
* **Rolling mean and rolling std** of each sensor over the last 5 cycles, per
  engine. Smooths noise and captures growing instability.
* **`cycle`** -- a scaled copy of the engine's age. Optional
  (`include_cycle_feature`); `time_in_cycles` itself is never modified, because
  later steps rely on it as a timestamp.

Rolling windows look **backward only** and **restart for every engine**. A window
that peeks ahead would use readings that do not exist yet at prediction time. A
test changes a future reading to 99,999 and checks that no earlier feature moves.

### Scaling

Every feature is scaled with `(x - mean) / std`, using means and standard
deviations computed from the **training split only** and saved to
`data/processed/feature_stats.json`. Validation, test and (later) live data are
all translated with the same numbers.

`feature_stats.json` is git-ignored, unlike `configs/reference_profile.json`.
The reference profile is a **contract** that rarely changes; scaling statistics
are **fitted parameters** that change with every retrain and belong with the
model.

## Notes for later stages

* **Ingestion is intentionally lossless.** RUL labels and features are computed
  in a separate stage, so there is always a known-good starting point to fall
  back to when a later stage looks wrong.
* **Do not score uncut validation engines on their last cycle.** Validation
  engines come from the training file, so they ran to failure and every
  last-cycle answer is RUL 0. Cut each engine at a random cycle first
  (imitating how the test set was made), then use `last_cycle_per_engine()`.
* **Score against `rul_uncapped`**, not the capped `rul`.
* **Test the `cycle` feature both ways.** Engine age is a fair but rough clue: at
  cycle 100, training engines have anywhere from 28 to 262 cycles left. Train
  with and without it and compare.
