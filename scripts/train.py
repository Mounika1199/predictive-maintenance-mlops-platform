"""Train every model, choose the best one on validation, and record everything.

    python scripts/train.py

Run AFTER `dvc repro` (which builds data/processed/), with the MLflow server
running (`docker compose up -d`). Everything it produces goes to MLflow --
nothing is saved locally.

WHAT IT DOES
0. Refuse to train if the DVC data pipeline is out of date ("run dvc repro").
1. Check the MLflow server is reachable -- BEFORE spending time training.
2. Train each model listed in configs/training.yaml on the 80 training engines.
   Each model becomes one MLflow RUN, holding its settings (parameters), its
   validation scores (metrics) and the trained model itself (an artifact).
3. Pick the winner: the lowest validation RMSE. Tag its run selected=true.
4. Register the winner in the model registry as a new version, tagged
   'challenger'. Putting it into use ('champion') is a separate, deliberate
   step: scripts/promote_model.py.

THIS SCRIPT NEVER READS THE TEST SET. Choosing a model is a decision, and
decisions are made on validation data only. The test set is opened once,
afterwards, by scripts/evaluate.py -- to measure the choice, not to make it.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import mlflow
import pandas as pd
from mlflow import MlflowClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.load_cmapss import load_config  # noqa: E402
from src.models.metrics import score  # noqa: E402
from src.models.tracking import (  # noqa: E402
    CHAMPION,
    check_server,
    data_versions,
    has_uncommitted_changes,
    log_trained_model,
    register_challenger,
    resolve_tracking_uri,
    run_params,
    stale_pipeline_stages,
)
from src.models.training import build_model, predict_rul  # noqa: E402

FEATURES_CONFIG = PROJECT_ROOT / "configs" / "features.yaml"
TRAINING_CONFIG = PROJECT_ROOT / "configs" / "training.yaml"

# The column the models learn to predict: RUL capped at 125 (see Task 3).
TARGET = "rul"


def read_split(processed_dir: Path, name: str) -> pd.DataFrame:
    path = processed_dir / f"{name}_features.parquet"
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing {path.relative_to(PROJECT_ROOT)}\n"
            "Run this first: python scripts/build_features.py"
        )
    return pd.read_parquet(path)


def main() -> int:
    # MLflow prints messages containing emoji ("View run at: ..."). Windows
    # terminals default to an old character set with no emoji, which made
    # Python crash mid-run and left the run stuck as RUNNING. Printing in UTF-8
    # (and replacing anything still unprintable) makes that impossible.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    data_config = load_config()
    feature_config = load_config(FEATURES_CONFIG)
    training_config = load_config(TRAINING_CONFIG)

    # 0. Never train on stale data. DVC compares today's fingerprints of every
    # pipeline input (raw files, settings, code) with its receipt, dvc.lock. If
    # anything changed since the data was built, the features may not match.
    stale = stale_pipeline_stages(PROJECT_ROOT)
    if stale:
        print(f"ERROR: the data pipeline is out of date ({', '.join(stale)}).")
        print("Rebuild the data first:  dvc repro")
        return 1

    # The exact data this training uses -- recorded on every run below.
    data_fingerprints = data_versions(
        PROJECT_ROOT,
        raw_dir=data_config["dataset"]["raw_dir"],
        processed_dir=feature_config["output"]["processed_dir"],
    )

    stats_path = PROJECT_ROOT / feature_config["output"]["stats_file"]
    feature_stats = json.loads(stats_path.read_text(encoding="utf-8"))
    features = feature_stats["feature_columns"]

    processed_dir = PROJECT_ROOT / feature_config["output"]["processed_dir"]
    train = read_split(processed_dir, "train")
    val = read_split(processed_dir, "val")

    model_settings = training_config["models"]
    metric = training_config["selection_metric"]
    random_state = training_config["random_state"]
    if "baseline" not in model_settings:
        raise ValueError("configs/training.yaml must list 'baseline': it is the bar to beat")

    # 1. Connect to MLflow first: better to fail in one second than after training.
    tracking_uri = resolve_tracking_uri(training_config["mlflow"]["tracking_uri"])
    experiment_name = training_config["mlflow"]["experiment_name"]
    check_server(tracking_uri)
    mlflow.set_tracking_uri(tracking_uri)
    experiment = mlflow.set_experiment(experiment_name)

    # 2. Train each model on TRAIN, score it on VALIDATION -- one MLflow run each.
    # Can these runs be reproduced from their recorded commit alone?
    uncommitted = has_uncommitted_changes(PROJECT_ROOT)
    if uncommitted:
        print("Note: the code has uncommitted changes -- recorded on every run.\n")

    results = {}
    run_ids = {}
    model_uris = {}
    for name, params in model_settings.items():
        with mlflow.start_run(run_name=name) as run:
            mlflow.log_params(run_params(name, params, random_state, feature_stats))
            mlflow.log_params(data_fingerprints)  # which exact data (DVC)
            mlflow.set_tag("selected", "false")  # the winner is re-tagged below
            mlflow.set_tag("git.uncommitted_changes", str(uncommitted))

            model = build_model(name, params, random_state)
            model.fit(train[features], train[TARGET])
            predictions = predict_rul(model, val[features])
            scores = score(val[TARGET], predictions)

            mlflow.log_metrics({"val_rmse": scores["rmse"], "val_mae": scores["mae"]})

            # The model itself, its expected inputs, and its scaling numbers.
            info = log_trained_model(model, val[features], predictions, stats_path)

        results[name] = {"rmse": scores["rmse"], "mae": scores["mae"]}
        run_ids[name] = run.info.run_id
        model_uris[name] = info.model_uri

    print(f"\nValidation scores ({val['unit_number'].nunique()} engines never seen in training)\n")
    print(f"  {'model':<16}{'RMSE':>8}{'average miss':>15}")
    for name, r in results.items():
        print(f"  {name:<16}{r['rmse']:8.1f}{r['mae']:15.1f}")

    # 3. Pick the winner among the real candidates.
    candidates = [name for name in results if name != "baseline"]
    best = min(candidates, key=lambda name: results[name][metric])

    # A safety gate: if nothing beats guessing the average, something is
    # broken (features, data, code). Refuse to save rather than hand over a
    # useless model that looks like a real one. (The runs stay in MLflow, all
    # tagged selected=false -- a failed attempt is still worth a record.)
    if results[best][metric] >= results["baseline"][metric]:
        print(f"\nERROR: best model '{best}' does not beat the baseline. Nothing saved.")
        return 1

    client = MlflowClient(tracking_uri)
    client.set_tag(run_ids[best], "selected", "true")
    print(f"\nSelected: {best}  (lowest validation {metric.upper()})")

    # Add the winner to the model registry as a new version, tagged
    # "challenger". This does NOT put it into use: the "champion" tag stays
    # where it is until someone runs scripts/promote_model.py on purpose.
    registered_name = training_config["mlflow"]["registered_model_name"]
    commit = client.get_run(run_ids[best]).data.tags.get("mlflow.source.git.commit", "unknown")[:7]
    if uncommitted:
        commit += " PLUS UNCOMMITTED CHANGES (not reproducible from the commit alone)"
    version = register_challenger(
        client,
        model_uris[best],
        registered_name,
        description=(f"{best} trained by scripts/train.py at commit {commit}; "
                     f"validation RMSE {results[best]['rmse']:.2f}."),
    )
    print(f"\nRegistered: {registered_name} version {version.version}, tagged 'challenger'")
    try:
        champion = client.get_model_version_by_alias(registered_name, CHAMPION)
        print(f"            The champion (in use) is still version {champion.version}.")
    except mlflow.exceptions.MlflowException:
        print("            There is no champion yet.")
    print("            To put the challenger into use: python scripts/promote_model.py")

    print(f"\nCompare the runs at: {tracking_uri}/#/experiments/{experiment.experiment_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
