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
| Data versioning (DVC) | Done (pointer files, local storage, data pipeline) |
| RUL labelling & feature engineering | Done |
| Model training | Done (first models) |
| Model evaluation | Done (first version) |
| Experiment tracking (MLflow) | Done (server in Docker) |
| Model registry | Done (champion/challenger, gated promotion) |
| Model serving | Done (FastAPI, run locally; Docker next) |
| Monitoring & drift detection | Not started |
| Retraining | Not started |

## Setup

```powershell
python -m venv mlops_env
.\mlops_env\Scripts\Activate.ps1
pip install -r requirements.txt
dvc pull                      # fetch the data from DVC storage (see below)
docker compose up -d          # start the MLflow server (needs Docker Desktop)
setx MLFLOW_DISABLE_TELEMETRY true   # once per machine; see the telemetry note
```

The MLflow server runs in a Docker container; see *Experiment tracking* below.

The dataset is **not** in Git. Git holds only the pointer file `data/raw.dvc`;
`dvc pull` fetches the files themselves from DVC storage (see *Data versioning*).
The storage location is a per-machine setting, kept in `.dvc/config.local`
(never committed, so no local paths are published). On a new machine, tell DVC
where your copy of the storage is, once:

```powershell
dvc remote add --local -d localstore <folder>   # e.g. a folder outside the project
```

Or place the three FD001 files in `data/raw/` by hand.

**Telemetry.** DVC and MLflow both send anonymous usage statistics by default.
Both are switched off: DVC in `.dvc/config`, the MLflow server in
`compose.yaml`, and the MLflow client by the `MLFLOW_DISABLE_TELEMETRY`
environment variable above (it takes effect in newly opened terminals).

## Usage

```powershell
dvc repro                                   # validate -> ingest -> build_features (only what's out of date)
python scripts/build_reference_profile.py   # regenerate the validation contract (rarely)
docker compose up -d                        # MLflow server must be running for train.py
python scripts/train.py                     # train, compare on validation, record in MLflow
python scripts/promote_model.py             # put the challenger into use, if it is better
python scripts/evaluate.py                  # score the champion on the test set
python scripts/experiment_cycle_feature.py  # 5-fold experiment: does `cycle` help?
uvicorn src.serving.app:app --port 8000     # the prediction service; docs at http://127.0.0.1:8000/docs
pytest                                       # run the test suite (no Docker needed)
```

`dvc repro` runs the data scripts (`validate_fd001.py`, `ingest_fd001.py`,
`build_features.py`) in order, and only those whose inputs changed. It calls
`python`, so activate `mlops_env` first. `train.py` refuses to run if the
pipeline is out of date.

## Project layout

```
compose.yaml                       The MLflow server, run as a Docker container
dvc.yaml                           The data pipeline: validate -> ingest -> build_features
dvc.lock                           The pipeline's receipt: fingerprints of every input/output
.dvc/config                        DVC settings: analytics off (storage path: config.local)
configs/data.yaml                  Paths, column layout, expected shapes, tolerances
configs/features.yaml              Modelling choices: RUL cap, split, rolling window
configs/training.yaml              Which models to train, and their settings
configs/reference_profile.json     Generated contract: per-column dtype/min/max
data/raw.dvc                       Pointer file: the raw data's fingerprint (in Git)
data/raw/                          Original .txt files             (DVC-tracked)
data/validation/report.json        Validation verdict: the pipeline's gate (DVC)
data/interim/                      Ingested Parquet tables         (DVC pipeline)
data/processed/                    Model-ready features + stats    (DVC pipeline)
src/data/load_cmapss.py            Pure loading functions (importable, testable)
src/data/validation.py             Schema building + structural checks
src/data/split.py                  Engine-level train/validation split
src/features/labels.py             RUL target for train and test
src/features/engineering.py        Sensor choice, rolling features, scaling
src/models/metrics.py              RMSE, MAE, NASA score
src/models/training.py             Building models, predicting (clipped at 0)
src/models/tracking.py             MLflow helpers: server check, what to record, registry
src/models/promotion.py            The rules for promoting a model to champion
src/serving/predictor.py           Raw readings -> features -> prediction (reuses src/features)
src/serving/checks.py              Request checks: history (refuse), contract (warn)
src/serving/schemas.py             Request/response shapes, checked by FastAPI
src/serving/app.py                 The web service: /health, /model, /predict
scripts/ingest_fd001.py            Entry point: loads raw, writes Parquet
scripts/build_reference_profile.py Entry point: writes the reference profile
scripts/validate_fd001.py          Entry point: validates, writes the gate report, exit 1 on failure
scripts/build_features.py          Entry point: interim -> processed features
scripts/train.py                   Entry point: train, select on validation, register
scripts/promote_model.py           Entry point: move the champion tag, after checks
scripts/evaluate.py                Entry point: score the champion on test
scripts/experiment_cycle_feature.py Experiment: cycle feature, 5-fold CV by engine
tests/                             Pytest suite (shared setup in conftest.py)
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

Git stores a complete copy of every version of every file; putting datasets in
it makes the repository permanently large and slow. So the data lives in DVC
storage, and Git holds only small pointer files -- see *Data versioning and the
pipeline (DVC)*.

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

## Models and evaluation

`train.py` trains three models on the 80 training engines, scores them on the 20
validation engines, picks the lowest validation RMSE, records everything in
MLflow and registers the winner as a challenger (see *Model registry*). It
**never reads the test set**. `evaluate.py` loads the registry's champion --
training nothing -- and scores it on the test set.

### Results

| Model | Validation RMSE | Validation avg miss | Test RMSE | Test avg miss | Test NASA score |
|---|---|---|---|---|---|
| Always guess the average (87.5) | 42.0 | 36.8 | 43.2 | 35.9 | 35,723 |
| Linear regression | 19.1 | 15.4 | -- | -- | -- |
| **Random forest (selected)** | **15.1** | **9.8** | **18.8** | **13.7** | **694** |

Only the selected model is scored on test (the baseline appears as a fixed
reference point, not a candidate).

### How each set is scored

| | Purpose | Rows | Answer key |
|---|---|---|---|
| Validation | choose between models | every row of the 20 held-back engines | capped `rul` |
| Test | the honest, real-world number | last observed cycle of each test engine | true `rul_uncapped` |

Validation uses the capped key because it only *compares* candidates, all graded
the same way; an uncapped key would let the unknowable far-future rows drown out
real differences. The number reported as real-world accuracy comes from test,
against the truth. (Against the capped key, test RMSE would have looked better:
17.7.)

**The test-set rule:** once the test score has been seen, the model is not
changed because of it. Improvements are made on validation; test is revisited
rarely, at the end of a round.

### Metrics

All start from **error = predicted - true**. Positive = *late* (more life
predicted than there really was) -- the dangerous direction.

* **MAE** -- the average size of a miss, in cycles.
* **RMSE** -- like MAE, but large misses count extra. The standard headline
  metric for C-MAPSS.
* **NASA score** -- from the original C-MAPSS challenge. Penalises each miss
  exponentially, late misses harder than early ones (late by 40 costs 53.6, early
  by 40 costs 20.7). Summed over engines, so only comparable on the same engines.

### What we learned

* **Bending matters.** Wear is slow for most of an engine's life, then speeds up.
  Linear regression (a straight line) was too optimistic in the last 50 cycles by
  +16.7 cycles on average; the random forest by +6.4.
* **The forest partly memorises.** Average miss 4.4 on its own training engines,
  9.8 on unseen ones. The held-back engines are what reveal the honest number.
* **One sensor dominates.** Smoothed `sensor_4` drives about half of the
  forest's decisions; smoothed readings beat raw ones throughout.
* **Test is harder than validation, as expected.** Validation includes many easy
  early-life rows; test asks once per engine, mostly mid-life. The 11 test
  engines with more than 125 cycles left are missed by 26.5 on average, all in
  the safe (early) direction -- the cost of the cap.
* **Averages hide the dangerous misses.** Test engine 37 truly had 21 cycles left;
  the model said 70. That one engine is 19% of the whole NASA score. A real
  system would add a safety margin or report how confident the model is.

## Experiment tracking (MLflow)

MLflow is a logbook for experiments. Every time `train.py` runs, each model it
trains becomes a **run** in the `rul-fd001` **experiment**, holding:

| Kind | What is recorded |
|---|---|
| Parameters (settings chosen beforehand) | the model's settings, and how the data was built: `rul_cap`, `rolling_window`, `include_cycle_feature`, `n_features`, split seed, engine counts |
| Metrics (scores) | `val_rmse`, `val_mae` |
| Tags | `selected` (true for its session's winner), the script, the Git commit, and `git.uncommitted_changes` |
| Files | the trained model with its input signature (46 columns), plus `feature_stats.json` -- the scaling numbers it needs |

Data settings are read from `feature_stats.json`, which records how the processed
data was *actually* built, not from `configs/features.yaml`, which may have been
edited since. Two runs with identical model settings but different data are
different experiments and look different in MLflow.

### The server runs in Docker

```
YOUR PROJECT (Windows)                         DOCKER CONTAINER
scripts/train.py  -- sends results -->   MLflow server (127.0.0.1:5000)
                     over HTTP             +- settings + scores -> SQLite database
                                           +- trained models    -> files
your browser  -- http://localhost:5000 --> the MLflow web page
```

* `compose.yaml` runs the official image `ghcr.io/mlflow/mlflow:v3.17.1`.
* The project installs only the **client**, `mlflow-skinny` (33 packages instead of
  70 for full MLflow). Its version must match the server's; a test checks this.
* Data lives in the Docker **volume** `mlflow_data`, so history survives restarts
  and `docker compose down`. Only `docker compose down -v` deletes it.
* The port is bound to `127.0.0.1`: only this computer can reach the server,
  which has no login.
* `train.py` checks the server is reachable *before* training, and stops with
  "start it with `docker compose up -d`" if not.

```powershell
docker compose ps          # is it running?
docker compose logs -f     # what is it doing?
docker compose down        # stop it (history kept)
```

### Safe model files

MLflow saves scikit-learn models with **skops** instead of pickle: opening a
pickle file can run any code hidden inside it. skops refuses object types it
cannot prove safe. A random forest's flowcharts (`sklearn.tree._tree.Tree`)
are one such type, so exactly that type is trusted, in
`src/models/tracking.py` -- acceptable because we create these files ourselves,
on a server only this computer can reach. A test fails if more types are added
without updating it deliberately.

### Windows note

MLflow prints messages containing emoji, which the default Windows terminal
character set cannot show; Python crashed mid-run and left a run stuck as
RUNNING. `train.py` switches its own output to UTF-8 to prevent this.

## Experiment: does engine age (`cycle`) help?

**Question.** `cycle` -- the engine's age -- was linear regression's biggest clue
and the forest's third. Is it genuinely useful, or does it make the models lean
on age instead of reading the sensors? Only that one setting was changed; models,
settings and engines stayed the same. Decided on validation; the test set stayed
closed.

**Step 1: one split** (our usual 20 validation engines). Removing `cycle` made
the random forest worse by **3.00** RMSE and linear regression by **1.78**.

**Step 2: five splits** (`scripts/experiment_cycle_feature.py`). 5-fold
cross-validation by engine (`engine_folds()` in `src/data/split.py`): every
engine is a validation engine exactly once. Recorded in MLflow as the parent run
`cycle-feature-experiment` with 20 child runs.

| Validation RMSE cost of removing `cycle` | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Average | Folds where it helps |
|---|---|---|---|---|---|---|---|
| Random forest | +3.00 | +2.52 | -1.53 | +1.81 | +2.10 | **+1.58** | 4 of 5 |
| Linear regression | +1.78 | +1.76 | -4.02 | +2.35 | +1.42 | **+0.66** | 4 of 5 |

**Decision: keep `cycle`.** It helps on most groups of engines, especially for
the forest. No change was needed -- it was already the default.

**What we learned**

* **One split exaggerated the effect about 2x.** Fold 1 is our usual split, and it
  turned out to be the *most* favourable of the five for `cycle` (+3.00 against an
  average of +1.58). The direction was right; the size was not. On fold 3 alone we
  would have concluded the opposite.
* **Age misleads on unusual engines.** Fold 3 holds 7 engines that lived over 250
  cycles (other folds: 1-4), including the longest-lived of all (362), and its
  engines are on average 48 cycles away from the typical lifespan of 199 (other
  folds: 23-36). On that fold both models did better *without* age. Age is a rule
  of thumb about the typical engine.

## Model registry

A catalogue of numbered model versions, with movable name tags (aliases):

```
rul-fd001
 +- version 1
 +- version 2
 +- version 3   <- champion    (the version in use)
 +- version 4   <- challenger  (the newest candidate)
```

Anything that needs "the model in use" asks for `models:/rul-fd001@champion` --
no version numbers, run IDs or file paths. Upgrading means moving the tag; a
rollback means moving it back. Versions are never overwritten or deleted, so an
old champion stays loadable by number (`models:/rul-fd001/1`).

Each version traces back to its run: settings, scores, Git commit, and whether
the code had uncommitted changes (**lineage**).

### Training a model and putting it into use are separate decisions

1. `train.py` registers each session's winner as a **new version** and moves the
   **`challenger`** tag to it. It never touches `champion`.
2. `promote_model.py` is the **only** way `champion` moves. Before moving it,
   it checks (rules in `src/models/promotion.py`):
   * both versions were scored on the **same exam** -- same validation engines,
     same answer key (`rul_cap`) -- otherwise their scores mean nothing side by
     side;
   * the challenger is better by at least `min_improvement_rmse` (0.1). Anything
     smaller is a tie: a retrained copy of the champion can differ by ~1e-14 just
     from the order the forest adds its trees.
   * It warns if the challenger was trained from uncommitted code.
3. `--force --reason "..."` overrides the checks (a rollback is the classic case).
   A reason is required.

Aliases keep no history, so every promotion writes its own record onto the
promoted version: `promoted_at`, `promoted_from_version`, `promotion_reason`,
`promotion_forced`. Exit codes: 0 promoted, 1 refused.

```powershell
python scripts/promote_model.py                              # consider the challenger
python scripts/promote_model.py --version 1 --force --reason "roll back: v3 misbehaves"
```

### The registry is the only source

There are no local model files. `evaluate.py` loads `rul-fd001@champion` and,
before scoring, compares the current `feature_stats.json` -- every column and
every scaling number -- with the copy stored on the champion's run. Data
prepared differently from the model's training data (**training/serving skew**)
never crashes; it just makes every prediction slightly wrong. A change of 0.01 to
a single average is enough to stop the evaluation. Test scores are written back
to MLflow: `test_*` metrics on the champion's run, a `test_rmse` tag on its
version.

## Data versioning and the pipeline (DVC)

DVC does for data what Git does for code. The real files live in DVC
**storage**; Git holds only a small **pointer file** with each dataset's
fingerprint (a hash that changes if a single byte changes). Every commit
therefore pins an exact version of the data.

```
Git (the catalogue)                  DVC storage (the bookshelf)
data/raw.dvc  --fingerprint-->       a storage folder outside the project
  a095ca28....dir                      each file stored under its own fingerprint
```

```powershell
dvc pull       # get the data for the current commit from storage
dvc push       # send new data versions to storage
dvc status     # what is out of date, and why
dvc repro      # rebuild what is out of date -- and nothing else
```

Tested end to end: with `data/raw` **and** DVC's local cache deleted, `dvc pull`
restored every file byte for byte.

### The pipeline

```
data/raw --> validate --> data/validation/report.json --+
    |                                                    v
    +--------------------------------------------> ingest --> data/interim --> build_features --> data/processed
```

`dvc.yaml` lists each step's command, what it reads (including its own code) and
what it writes; `dvc.lock` is the receipt -- the fingerprints of all of it from
the last run. `dvc repro` compares the two and re-runs only what changed:

| Change | What re-runs |
|---|---|
| nothing | nothing (about 1 second) |
| `rolling_window` in `features.yaml` | only `build_features` |
| a setting changed back to one already built | nothing -- restored from DVC's run cache |

**Validation is a real gate.** `ingest` depends on validation's report, so if
validation fails (exit code 1) nothing downstream runs. The report contains no
timestamp: an unchanged result gives an identical file, so re-validating does
not trigger rebuilds.

**Training is not a pipeline step**, on purpose: each training run registers a
new challenger, so it stays a deliberate command.

### Lineage now reaches the data

`train.py` refuses to train if `dvc status` reports anything out of date (the
stale `data/interim/` of Task 3 is now impossible), and records two parameters
on every run:

* `data_raw_md5` -- the raw files' fingerprint, from `data/raw.dvc`
* `data_processed_md5` -- the model-ready tables' fingerprint, from `dvc.lock`

So any registered version traces back through its run to the exact data:
`rul-fd001 vN -> run -> data_processed_md5 -> the commit whose dvc.lock holds
it -> git checkout + dvc checkout`.

### A .gitignore trap

`data/` contents are hidden from Git with `/data/?*`, and pointer files shown
with `!/data/*.dvc`. The obvious `/data/*` is wrong: `*` can match nothing, so
it also matches the folder `data/` itself. DVC asks "is `data/` ignored?" when
searching for pointer files, got "yes", and `dvc push` reported "Everything is
up to date" while pushing nothing. A test now asks DVC directly.

## Model serving

A small web service (FastAPI, run by uvicorn) answers "here are this engine's
readings -- how long has it got?" for other systems.

```powershell
docker compose up -d                          # MLflow must be running
uvicorn src.serving.app:app --port 8000       # then open http://127.0.0.1:8000/docs
```

| Address | Purpose |
|---|---|
| `GET /health` | "are you alive?" -- for monitoring tools |
| `GET /model` | which version is answering, and how it prepares data (sensors, window, features) |
| `POST /predict` | an engine's **raw** readings in -> remaining useful life out, plus the model version and any warnings |

At startup the service fetches `rul-fd001@champion` from MLflow -- the model and
its `feature_stats.json` as a matched pair, by version number. It loads once;
a promotion is picked up on the next restart, and every response names the
version that answered.

### No training/serving skew

The model needs 46 prepared features, not raw readings. The service prepares
them with the **same code** (`src/features/engineering.py`) and the **champion's
own numbers** -- there is no second copy of the feature logic. A test sends all
100 test engines' raw readings through the web service and requires the same
answers as the offline pipeline `evaluate.py` uses. Deliberately breaking one
number -- a smoothing window of 4 instead of 5 -- changes every one of the 100
answers, by up to 21 cycles, and the test fails.

### Requests are checked in layers

| Layer | Problem | Response |
|---|---|---|
| shape | missing or misspelled field, text instead of a number | refused, 422 |
| history | a gap, a repeat, wrong order, or fewer than the last 5 cycles (unless starting at cycle 1) | refused, 422 -- the features would be silently wrong |
| values | a reading outside the validation contract | **answered, with a warning** |

Too little history matters: with only the last 3 cycles, test engine 1 gets
124.46 instead of 123.96, with no error. Unusual values are answered rather
than refused because an engine near failure may be exactly such an engine; the
warning says plainly that the reading is outside what the model learned from.
A changed `sensor_1` does not move the prediction at all -- the model never
sees that sensor -- so the warning is the only sign that something changed.

## Notes for later stages

* **Ingestion is intentionally lossless.** RUL labels and features are computed
  in a separate stage, so there is always a known-good starting point to fall
  back to when a later stage looks wrong.
* **Scoring every validation row already covers every possible cut.** Because
  features look backward only, cutting an engine at cycle *t* and rebuilding its
  features gives exactly the same values as row *t* of the full engine (verified:
  difference 0.0). No separate random-cutting step is needed. Never score uncut
  validation engines on their *last* cycle only, though: they ran to failure, so
  every such answer is RUL 0.
* **Watch engine lifespans in monitoring.** The `cycle` feature helps on typical
  engines but hurt on a group of unusually long-lived ones (see the experiment
  above). If engines in service start living longer than the training engines
  did, this feature becomes a liability -- a concrete drift signal to monitor.
* **Compare options on several folds, not one split.** A single split got the
  direction of the `cycle` effect right but its size wrong by about 2x.
  `engine_folds()` exists for this.
* **The forest is still slightly optimistic near failure** (+6.4 cycles on
  validation, +5.4 on test). The direction to improve.
* **The service runs on this machine only.** Putting it in a Docker container
  next to MLflow (its own Dockerfile; inside Docker it reaches MLflow at a
  different address) is the next step.
* **A promotion needs a service restart.** The service loads the champion once.
  A reload endpoint, or a periodic check of the alias, would pick up promotions
  without one.
* **Warnings are returned, not yet collected.** They are a ready-made drift
  signal; monitoring should record them over time.
* **About 0.1-0.2 s per prediction**, mostly the forest spreading 200 trees
  over every CPU core (`n_jobs: -1`) for a single engine. Fine for occasional
  requests; tune if throughput ever matters.
* **Commit before training runs you want to reproduce.** Runs now record
  `git.uncommitted_changes`, and promotion warns about it, but a run trained from
  uncommitted code still cannot be rebuilt from its commit.
* **Promotion's "same exam" could include the raw data fingerprint.** Two
  versions validated on different raw data are not comparable either; today
  only the split seed, engine counts and RUL cap are checked. Versions before
  DVC have no fingerprint, so adding it would need one deliberate forced
  promotion.
* **Only `train.py` checks for a stale pipeline.** `evaluate.py` and the
  experiment script read DVC outputs too; `evaluate.py`'s skew check catches
  mismatched features, but neither refuses stale data outright.
* **DVC storage is a local folder.** Its path lives in `.dvc/config.local`, so
  it is per machine and nothing is shared. Shared or cloud storage (S3, MinIO)
  would be one `dvc remote add` in `.dvc/config`.
* **Old versions accumulate.** Nothing deletes them, which is what rollbacks and
  audits need. A clean-up rule (e.g. archive after a year) can come later.
* **Compare parallel predictions within a tolerance.** With `n_jobs: -1` the
  forest adds its trees' answers in a varying order, so repeated predictions
  differ by ~1e-14. Exact comparisons need one core.
