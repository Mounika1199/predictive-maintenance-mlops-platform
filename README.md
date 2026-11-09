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
| Model serving | Done (FastAPI, in Docker next to MLflow) |
| Monitoring & drift detection | Done (prediction diary, drift check with Evidently, Prometheus + Grafana); accuracy against late-arriving truth not yet |
| Retraining | Not started |

## Setup

```powershell
python -m venv mlops_env
.\mlops_env\Scripts\Activate.ps1
pip install -r requirements.txt
dvc pull                      # fetch the data from DVC storage (see below)
# once: a random Grafana admin password into .env (git-ignored, never committed)
python -c "import secrets, pathlib; pathlib.Path('.env').write_text('GRAFANA_ADMIN_PASSWORD=' + secrets.token_urlsafe(24) + '\n')"
docker compose up -d --build  # MLflow, the prediction service, Prometheus, Grafana (needs Docker Desktop)
setx MLFLOW_DISABLE_TELEMETRY true   # once per machine; see the telemetry note
```

MLflow, the prediction service, Prometheus and Grafana run in Docker
containers; see *Experiment tracking*, *Model serving* and *Monitoring* below.
The `.env` line is written by Python rather than PowerShell's `>`, which would
save it in an encoding Docker Compose cannot read; Compose refuses to start
Grafana without that password.

The dataset is **not** in Git. Git holds only the pointer file `data/raw.dvc`;
`dvc pull` fetches the files themselves from DVC storage (see *Data versioning*).
The storage location is a per-machine setting, kept in `.dvc/config.local`
(never committed, so no local paths are published). On a new machine, tell DVC
where your copy of the storage is, once:

```powershell
dvc remote add --local -d localstore <folder>   # e.g. a folder outside the project
```

Or place the three FD001 files in `data/raw/` by hand.

**Telemetry.** DVC, MLflow and Grafana all send anonymous usage statistics by
default. All are switched off: DVC in `.dvc/config`, the MLflow server and
Grafana in `compose.yaml` (Grafana's update checks, news feed and external
snapshot sharing too -- confirmed in its live settings), and the MLflow client
by the `MLFLOW_DISABLE_TELEMETRY` environment variable above (it takes effect
in newly opened terminals). Prometheus sends nothing.

## Usage

```powershell
dvc repro                                   # validate -> ingest -> build_features (only what's out of date)
python scripts/build_reference_profile.py   # regenerate the validation contract (rarely)
docker compose up -d --build                # MLflow (needed by train.py), the service, Prometheus, Grafana
python scripts/train.py                     # train, compare on validation, record in MLflow
python scripts/promote_model.py             # put the challenger into use, if it is better
docker compose restart rul-service          # the service picks up the new champion
python scripts/evaluate.py                  # score the champion on the test set
python scripts/experiment_cycle_feature.py  # 5-fold experiment: does `cycle` help?
python scripts/simulate_traffic.py --engines 10   # pretend fleet -> the service (all 100: drop --engines)
python scripts/check_drift.py --since <UTC time>  # has the data drifted? exit 0 / 1 drift / 2 too little
pytest                                       # run the test suite (no Docker needed)
```

Dashboards: Grafana at http://127.0.0.1:3000 (user `admin`, password in
`.env`), Prometheus at http://127.0.0.1:9090. See *Monitoring*.

`dvc repro` runs the data scripts (`validate_fd001.py`, `ingest_fd001.py`,
`build_features.py`) in order, and only those whose inputs changed. It calls
`python`, so activate `mlops_env` first. `train.py` refuses to run if the
pipeline is out of date.

## Project layout

```
compose.yaml                       The four containers: MLflow, prediction service, Prometheus, Grafana
Dockerfile                         How the prediction service's image is built
.dockerignore                      What the image may contain (an allow-list)
requirements-serving.txt           The image's packages: a subset of requirements.txt
.env                               Local secrets (Grafana password)  (git-ignored, never committed)
configs/monitoring.yaml            Diary folder; drift rules, each with its measured reason
monitoring/prometheus/prometheus.yml   What Prometheus collects, how often
monitoring/grafana/provisioning/   Grafana set-up as code: data source, dashboard loader
monitoring/grafana/dashboards/     The dashboard, as JSON
logs/predictions/                  The service's diary, one file per day  (git-ignored)
reports/                           Drift reports, HTML                    (git-ignored)
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
src/serving/app.py                 The web service: /health, /model, /predict, /metrics
src/monitoring/prediction_log.py   The diary: write one line per prediction, read it back
src/monitoring/traffic.py          The pretend fleet: which request comes next, a broken sensor
src/monitoring/drift.py            The drift rules: what to compare, Evidently, the verdict
src/monitoring/metrics.py          The service's numbers for Prometheus
scripts/ingest_fd001.py            Entry point: loads raw, writes Parquet
scripts/build_reference_profile.py Entry point: writes the reference profile
scripts/validate_fd001.py          Entry point: validates, writes the gate report, exit 1 on failure
scripts/build_features.py          Entry point: interim -> processed features
scripts/train.py                   Entry point: train, select on validation, register
scripts/promote_model.py           Entry point: move the champion tag, after checks
scripts/evaluate.py                Entry point: score the champion on test
scripts/experiment_cycle_feature.py Experiment: cycle feature, 5-fold CV by engine
scripts/simulate_traffic.py        Entry point: replay the test engines into the service
scripts/check_drift.py             Entry point: diary vs training data, HTML report, exit code
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
docker compose up -d --build    # build the service's image; start MLflow, then the service (and monitoring)
docker compose ps               # every container should say "healthy"
```

Then open http://127.0.0.1:8000/docs to try it. To run the service straight
from your code instead (while developing), free its port first:

```powershell
docker compose stop rul-service
uvicorn src.serving.app:app --port 8000
```

| Address | Purpose |
|---|---|
| `GET /health` | "are you alive?" -- for monitoring tools |
| `GET /model` | which version is answering, and how it prepares data (sensors, window, features) |
| `POST /predict` | an engine's **raw** readings in -> remaining useful life out, plus the model version and any warnings |
| `GET /metrics` | request counts, response times and more, read by Prometheus (see *Monitoring*) |

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

### The service runs in Docker

```
WINDOWS                                 DOCKER (a private network)
browser, other systems
   -- http://127.0.0.1:8000 -->   rul-service  -- http://mlflow:5000 -->  mlflow
your scripts (train.py, ...)
   -- http://localhost:5000 ---------------------------------------->  mlflow
```

* **The image** (`rul-service:1.0.0`) is built by the `Dockerfile` on an exact
  Python (3.11.17, slim). Packages are installed *before* the code is copied in,
  so editing code rebuilds in seconds (Docker reuses unchanged steps). It runs
  as an ordinary user, not root, and has a health check.
* **Only what the service needs goes in.** `.dockerignore` is an allow-list:
  `src/`, `configs/` and `requirements-serving.txt`, nothing else (85 KB sent to
  Docker). `mlops_env/`, `data/`, `.git/` and `.dvc/config.local` -- which holds a
  local path with the Windows username -- never reach the image.
  `requirements-serving.txt` leaves out DVC and the test tools; its versions must
  equal `requirements.txt`'s, so the service runs the libraries the model was
  trained with.
* **Inside Docker, `localhost` means the container itself.** The service finds
  MLflow by its service name, `http://mlflow:5000`, set by the
  `MLFLOW_TRACKING_URI` variable in `compose.yaml` (it wins over
  `configs/training.yaml`).
* **Startup order:** the service waits until MLflow is *healthy*, not just
  started -- it fetches the champion at startup.
* **Both ports are bound to `127.0.0.1`:** neither service has a login.

Checked: all 100 test engines' raw readings sent to the container (Linux) give
the same answers as the offline pipeline on Windows -- largest difference
4.3e-14 cycles. `tests/test_container.py` reads the Docker files (no Docker
needed) and fails if any decision above is undone; each check was tested by
breaking that decision on purpose.

### Two traps we hit

**A passing health check is not proof.** MLflow answered the service's `/health`
check with OK, but refused every real request with 403 *"Invalid Host header --
possible DNS rebinding attack detected"*. MLflow only answers requests addressed
to names it trusts (by default localhost and private IP addresses). This stops
**DNS rebinding**: a web page tricking your browser into talking to servers on
your own computer. The service addresses it as `mlflow:5000`, so exactly that
name was added to `--allowed-hosts` -- never `*`, which would switch the
protection off. Setting the list replaces the defaults, so localhost is listed
too.

**"Port is not available" on Windows.** A service started earlier with `uvicorn`
had been stopped, yet port 8000 was still taken. The venv's `python.exe` is a
small launcher that starts the real Python as a *separate* process; stopping the
launcher can leave the real one running. Find it, then stop that process:

```powershell
Get-CimInstance Win32_Process | Where-Object CommandLine -like "*uvicorn src.serving.app:app*" |
    Select-Object ProcessId, CommandLine
```

## Monitoring

A model in production never says "I don't know": if the world changes, it keeps
answering, confidently and wrongly, and nothing crashes. Monitoring watches two
things:

| Watches | Question | How |
|---|---|---|
| the **data** | Do incoming readings still look like the training data? | the diary + a drift check (Evidently) |
| the **service** | Is it up? How busy, how fast, how often does it refuse? | Prometheus collects numbers, Grafana draws them |

```
requests ──> rul-service ──> answers
               │      │
  counts,      │      │ every answered prediction
  timings      ▼      ▼
         Prometheus   diary (logs/predictions/) ──> scripts/check_drift.py ──> verdict + HTML report
               │
               ▼
            Grafana (dashboard)
```

### The diary (prediction log)

Every answered `/predict` adds one line of JSON to `logs/predictions/
predictions-<day>.jsonl`: the time, the engine, the **newest** cycle's readings,
the answer, the model version and any warnings.

* **Newest cycle only.** A request carries the last 5 cycles (the rolling
  features need them); writing all of them would record each cycle five times.
* **Refused requests are not written** -- no prediction was made.
* **A failed write never fails the prediction**: the service still answers and
  logs a warning (and counts it, see Prometheus below).
* **A lock around each write.** Requests are answered in parallel threads;
  without it, 40 of 400 lines written at once were silently lost on Windows, in
  5 runs out of 5.
* `logs/` is a **bind mount**: the container's `/app/logs` is the project's
  `logs/` folder, so the files open in VS Code. Git-ignored, never in the image.
* Read with `json.loads`, not `pd.read_json`, which turns an engine id `"007"`
  into the number 7.

### A pretend fleet: the traffic simulator

We have no real engines, so `scripts/simulate_traffic.py` replays the 100 test
engines into the service the way a fleet would: day by day, every engine flies
once and sends its last 5 cycles -- 13,096 requests, one per recorded cycle,
sent one at a time. `--engines 10` makes a quick run (1,088 requests).

`--broken-sensor sensor_11` adds half that sensor's usual spread (+0.134) to
every reading, like a badly recalibrated sensor. Every faulty reading stayed
inside the training range, so the service's warnings stayed silent (0 of
13,096) -- yet 75% of the answers went down, by up to 19.7 cycles. Only a drift
check can find this kind of fault.

### Data drift: `scripts/check_drift.py`

Compares the diary's readings with the champion's training readings, sensor by
sensor, using Evidently, and saves an HTML report with a chart per sensor in
`reports/`. Exit code 0 = no drift, 1 = drift, 2 = too few readings to judge --
so a scheduler or a retraining job can react.

Evidently does the comparing; **the rules are ours**, and each was measured on
our own data (normal fleet = the test engines; fault = the broken sensor):

| Rule | Why (measured) |
|---|---|
| Compare only engine ages **1-100**, on both sides | Readings change as engines wear -- that is what the model reads. The test fleet is younger than the training engines, so comparing all ages flagged 7 of 15 sensors on a normal fleet, and *hid* the broken sensor (its score went down). Beyond cycle 100, old training engines are near failure while old fleet engines are long-lived survivors, so even same-age comparison fails there |
| **PSI**, not Evidently's default (Wasserstein) | Wasserstein flagged 13 of 15 sensors on the normal fleet -- even engines of exactly the same ages differ by more than its strict line. PSI: normal fleet highest 0.043, broken sensor 0.446 |
| A sensor drifts at **score >= 0.1** | Evidently's PSI default; a wide margin on both sides |
| Drift if **any one** sensor drifts | Evidently's own verdict needs half the columns; one broken sensor of 15 would have been "no drift" |
| At least **1,000 readings** | With 100 or 300, chance alone raised a false alarm in 10 of 10 samples; with 1,000, in 0 of 10 |

On the full runs: normal fleet -> no drift (exit 0); broken sensor -> drift on
`sensor_11`, and only that sensor (exit 1); 47 seconds of traffic -> no verdict
(exit 2). (Evidently's PSI cuts the range into equal-width bins -- about 15 here
-- not quantiles. Its optional dashboard server, litestar, warns about its own
code; `pyproject.toml` hides those warnings, and only those.)

### Service health: Prometheus

The service offers `GET /metrics` (`src/monitoring/metrics.py`); Prometheus
reads it every 5 seconds and keeps the history (http://127.0.0.1:9090).

| Metric | Kind | Answers |
|---|---|---|
| `rul_http_requests_total` (method, route, status) | counter | requests per second; how many refused (422) |
| `rul_http_request_duration_seconds` (route) | histogram | how fast: median, 95% |
| `rul_predictions_with_warnings_total` | counter | are unusual readings becoming common? |
| `rul_prediction_log_failures_total` | counter | is the diary failing to save? |
| `rul_model_info` (model_name, model_version) | gauge | which model is answering |

* A **middleware** times and counts every request; counting happens in
  `finally`, so even a crash is counted (as 500).
* Requests are labelled by **route, never the raw address**: a thousand random
  URLs share one `unmatched` label instead of becoming a thousand counters.
* Each app has its **own registry**, so the tests -- which build many apps --
  each start from zero.
* Rates and increases are **estimates** (Prometheus only has a reading every
  5 s); a counter itself is exact. A run of 1,088 requests showed as 1,082 over
  its window, while the counter said 1,088.

Example (Prometheus's Graph tab): `rate(rul_http_requests_total{route="/predict"}[1m])`
-- requests per second, averaged over the minute before each moment.

### The dashboard: Grafana

http://127.0.0.1:3000/d/rul-service -- user `admin`, password in `.env`. Six
panels: service reachable, model answering, requests per second, response time
(median and 95%), refused requests (%), warnings and diary failures.

* **Set up as code.** The data source and the dashboard are files in
  `monitoring/grafana/`, loaded at start-up (`provisioning`): in Git, and
  rebuilt identically anywhere. The file is the source of truth: changes made by
  clicking cannot be saved over it ("Save as" makes a personal copy).
* **No password in Git.** `compose.yaml` reads it from `.env` with
  `${GRAFANA_ADMIN_PASSWORD:?...}` -- Compose refuses to start Grafana without
  one, instead of falling back to `admin`/`admin`. Grafana uses it only on its
  first start, when it creates the admin user.
* **Nothing leaves the machine**: usage reporting, update checks, the news feed
  and external snapshot sharing are off -- confirmed in Grafana's live settings
  (`/api/admin/settings`), not just in the file.

### Monitoring found a real problem: the slow model

The response-time panel showed ~90 ms per prediction. The forest kept its
training setting -- use every core (`n_jobs: -1`) -- and for ONE row, starting
and coordinating 12 workers costs more than asking 200 trees. The service now
switches its loaded copy to one core (`use_one_core()` in
`src/serving/predictor.py`); training still uses every core, and the
registered model is untouched. The same traffic before and after:

| | All 12 cores | One core |
|---|---|---|
| average per prediction | 91 ms | **55 ms** |
| 95% answered within | 206 ms | **74 ms** |
| requests per second | 10.1 | **15.0** |
| answers | | identical within 4.3e-14 cycles |

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
* **The service has no login.** That is why it is reachable from this computer
  only. Before it is opened to other machines, it needs authentication and HTTPS.
* **The image tag (`1.0.0`) is set by hand.** An automated build (CI) would tag
  each image with its Git commit, so a running container traces back to its code
  the way a model version traces back to its run.
* **A promotion needs a service restart** (`docker compose restart rul-service`).
  The service loads the champion once. A reload endpoint, or a periodic check of
  the alias, would pick up promotions without one.
* **Accuracy against late-arriving truth (not built yet).** An engine's true
  remaining life is known only once it fails; then every prediction made for it
  can be graded. Measured on the diary, using NASA's truth file as the
  "maintenance record": grading each engine's last cycle reproduces the offline
  test RMSE (18.8) exactly; over the last 125 cycles of life -- where
  maintenance decisions are made -- RMSE 19.8, bias +5.1 cycles (optimistic).
  Grading every cycle against the uncapped truth (57.3) is misleading: the model
  is capped at 125 by design. Lesson: the broken sensor barely moved accuracy
  (19.8 -> 19.6, bias even "improved" to +2.0) -- accuracy alone would have
  missed it; drift is the early warning, truth the late verdict.
* **CPU and memory of the service (planned).** prometheus_client's process
  collector, added to `ServiceMetrics`, would report the service's CPU (as
  "cores in use", up to 12) and memory; two dashboard panels. It reads `/proc`,
  so it reports nothing on Windows -- test that it is registered, not its values.
  Every container's CPU (cAdvisor) needs access to Docker's internals.
* **Grafana logs two harmless errors**: it looks for optional
  `provisioning/plugins` and `provisioning/alerting` folders we do not use.
  Adding them would keep the log clean, so real errors stand out.
* **The drift check covers engine ages 1-100 only.** Older engines are watched
  through their predictions and, eventually, the truth.
* **The simulated fleet starts on the same day**, so a short time window of the
  diary is also a narrow band of engine ages. Check whole runs, or simulate
  engines entering service at different times.
* **Nothing cleans up yet.** The diary grows by one line per prediction (about
  18 MB for two full runs) and each drift report is about 4.6 MB.
* **Metrics assume one service process.** With several uvicorn workers, each
  would count separately; prometheus_client then needs its multiprocess mode.
* **Warnings are now collected** (diary and Prometheus counter); a rise could
  trigger a drift check automatically.
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
