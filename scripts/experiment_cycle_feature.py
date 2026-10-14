"""Experiment: does the engine's age (the `cycle` feature) really help?

    python scripts/experiment_cycle_feature.py

Run AFTER scripts/ingest_fd001.py, with the MLflow server running
(`docker compose up -d`). Takes a few minutes. Writes nothing to disk; every
result goes to MLflow.

WHY THIS SCRIPT EXISTS
On our usual 80/20 split, removing `cycle` made the random forest worse by 3.0
RMSE. But that was ONE set of 20 validation engines -- one draw of luck. Here
the same comparison is repeated on 5 different sets of validation engines
(5-fold cross-validation by engine). If "with cycle" wins on all five, the
difference is real; if it wins three and loses two, it is probably noise.

WHAT CHANGES BETWEEN THE TWO SETTINGS: only whether the model may see `cycle`.
Same engines, same models, same settings -- otherwise we could not tell which
change caused a difference.

LEAKAGE RULES, as in build_features.py: inside each fold, which sensors to use
and the scaling numbers are learned from that fold's TRAINING engines only.

NO MODEL FILES are saved: 20 trainings x a 35 MB forest would fill the server
for nothing. This is a question about settings; only the scores matter.

THE TEST SET IS NEVER READ. Decisions are made on validation data.
"""

from __future__ import annotations

import sys
from pathlib import Path

import mlflow
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.load_cmapss import load_config  # noqa: E402
from src.data.split import engine_folds  # noqa: E402
from src.features import engineering as fe  # noqa: E402
from src.features.labels import add_train_rul  # noqa: E402
from src.models.metrics import score  # noqa: E402
from src.models.tracking import check_server, resolve_tracking_uri  # noqa: E402
from src.models.training import build_model, predict_rul  # noqa: E402

FEATURES_CONFIG = PROJECT_ROOT / "configs" / "features.yaml"
TRAINING_CONFIG = PROJECT_ROOT / "configs" / "training.yaml"

N_FOLDS = 5
# The baseline (always guess the average) ignores features entirely, so it
# cannot be affected by this experiment. Only the real models are compared.
MODELS = ("linear", "random_forest")
QUESTION = "Does the cycle feature (engine age) improve validation RMSE?"


def read_training_engines(data_config: dict) -> pd.DataFrame:
    """The ingested TRAINING file only. The test file is not even opened."""
    dataset = data_config["dataset"]
    path = PROJECT_ROOT / dataset["interim_dir"] / f"train_{dataset['name']}.parquet"
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}. Run: python scripts/ingest_fd001.py")
    return pd.read_parquet(path)


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # see train.py

    data_config = load_config()
    feature_config = load_config(FEATURES_CONFIG)
    training_config = load_config(TRAINING_CONFIG)

    cap = feature_config["label"]["cap"]
    window = feature_config["features"]["rolling_window"]
    seed = feature_config["split"]["seed"]
    random_state = training_config["random_state"]
    model_settings = training_config["models"]

    tracking_uri = resolve_tracking_uri(training_config["mlflow"]["tracking_uri"])
    check_server(tracking_uri)
    mlflow.set_tracking_uri(tracking_uri)
    experiment = mlflow.set_experiment(training_config["mlflow"]["experiment_name"])

    train = add_train_rul(read_training_engines(data_config), cap)
    folds = engine_folds(train, N_FOLDS, seed)

    rows = []
    with mlflow.start_run(run_name="cycle-feature-experiment") as parent:
        mlflow.set_tags({
            "run_type": "experiment",
            "question": QUESTION,
            # Shown as the run's description on the MLflow page.
            "mlflow.note.content": f"{QUESTION} {N_FOLDS}-fold cross-validation by "
                                   "engine; one child run per fold, setting and model.",
        })
        mlflow.log_params({
            "n_folds": N_FOLDS,
            "fold_seed": seed,
            "rul_cap": cap,
            "rolling_window": window,
            "random_state": random_state,
            "models": ", ".join(MODELS),
        })

        for fold_number, fold in enumerate(folds, start=1):
            # Learned from this fold's TRAINING engines only.
            sensors = fe.select_sensor_columns(fold.train)

            def build(frame: pd.DataFrame) -> pd.DataFrame:
                return fe.add_cycle_feature(fe.add_rolling_features(frame, sensors, window))

            fold_train, fold_val = build(fold.train), build(fold.validation)

            for include_cycle in (True, False):
                columns = fe.feature_columns(sensors, include_cycle)
                stats = fe.fit_scaler(fold_train, columns)  # TRAINING engines only
                train_scaled = fe.apply_scaler(fold_train, stats)
                val_scaled = fe.apply_scaler(fold_val, stats)

                for name in MODELS:
                    model = build_model(name, model_settings[name], random_state)
                    model.fit(train_scaled[columns], train_scaled["rul"])
                    s = score(val_scaled["rul"], predict_rul(model, val_scaled[columns]))

                    setting = "with" if include_cycle else "without"
                    with mlflow.start_run(run_name=f"fold{fold_number}-{setting}-cycle-{name}", nested=True):
                        mlflow.set_tag("run_type", "cv_fold")
                        mlflow.log_params({
                            "model": name,
                            "fold": fold_number,
                            "include_cycle_feature": include_cycle,
                            "n_features": len(columns),
                            "validation_engines": len(fold.validation_engines),
                        })
                        mlflow.log_metrics({"val_rmse": s["rmse"], "val_mae": s["mae"]})

                    rows.append({"fold": fold_number, "model": name,
                                 "cycle": setting, "rmse": s["rmse"]})
                    print(f"  fold {fold_number}  {setting:>7} cycle  {name:<14} "
                          f"RMSE {s['rmse']:5.1f}", flush=True)

        # Summary: per model, per fold, how much does removing `cycle` cost?
        results = pd.DataFrame(rows)
        table = results.pivot_table(index=["model", "fold"], columns="cycle", values="rmse")
        table["cost of removing cycle"] = table["without"] - table["with"]

        print(f"\nValidation RMSE on each fold (20 different engines each time)\n")
        print(table[["with", "without", "cost of removing cycle"]].round(2).to_string())

        print("\nSummary")
        summary_lines = []
        for name in MODELS:
            t = table.loc[name]
            cost = t["cost of removing cycle"]
            wins = int((cost > 0).sum())
            line = (f"  {name:<14} with cycle better on {wins} of {N_FOLDS} folds;  "
                    f"average cost of removing it {cost.mean():+.2f} RMSE "
                    f"(smallest {cost.min():+.2f}, largest {cost.max():+.2f})")
            print(line)
            summary_lines.append(line)
            mlflow.log_metrics({
                f"{name}_rmse_with_cycle_mean": float(t["with"].mean()),
                f"{name}_rmse_without_cycle_mean": float(t["without"].mean()),
                f"{name}_folds_won_by_cycle": wins,
            })

        mlflow.log_text(
            table.round(3).to_string() + "\n\n" + "\n".join(summary_lines) + "\n",
            "summary.txt",
        )

    print(f"\nRecorded in MLflow: {tracking_uri}/#/experiments/"
          f"{experiment.experiment_id}/runs/{parent.info.run_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
