"""Score the CHAMPION model on the TEST set -- the honest, final number.

    python scripts/evaluate.py

Needs the MLflow server running (`docker compose up -d`) and the processed
features from scripts/build_features.py. Trains nothing.

WHICH MODEL? Whichever version currently carries the 'champion' tag in the
model registry: models:/rul-fd001@champion. This script never needs to know a
version number, a run ID or a file path.

THE RULE THAT COMES WITH THIS SCRIPT
Once you have seen these numbers, do not go back and change the model because
of them. If we tweaked the model until the test score looked good, the test set
would quietly become a second validation set, and we would have nothing left
that tells us honestly how the model does on engines it has never seen.

HOW THE TEST SET IS SCORED
* One prediction per test engine, at the last cycle we observed it -- the real
  question: "here is this engine right now, how long has it got?"
* Against the TRUE remaining life (rul_uncapped), not the capped value the
  models were trained on. Capping helped the model learn; it is not reality.

The scores are written back to MLflow: as test_* metrics on the champion's run,
and as tags on the registered version.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import mlflow
import mlflow.artifacts
import pandas as pd
from mlflow import MlflowClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.load_cmapss import load_config  # noqa: E402
from src.features.labels import last_cycle_per_engine  # noqa: E402
from src.models.metrics import score  # noqa: E402
from src.models.tracking import CHAMPION, check_server, resolve_tracking_uri  # noqa: E402
from src.models.training import build_model, predict_rul  # noqa: E402

FEATURES_CONFIG = PROJECT_ROOT / "configs" / "features.yaml"
TRAINING_CONFIG = PROJECT_ROOT / "configs" / "training.yaml"

TRUTH = "rul_uncapped"


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # see train.py

    feature_config = load_config(FEATURES_CONFIG)
    training_config = load_config(TRAINING_CONFIG)

    tracking_uri = resolve_tracking_uri(training_config["mlflow"]["tracking_uri"])
    check_server(tracking_uri)
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri)

    # Which version is in use, and which run produced it?
    name = training_config["mlflow"]["registered_model_name"]
    champion = client.get_model_version_by_alias(name, CHAMPION)
    source_run = client.get_run(champion.run_id)
    model_label = f"{name} v{champion.version} ({source_run.data.params['model']})"

    # Safety check: the test data must be prepared EXACTLY as the champion's
    # training data was -- same columns, same order, and the same scaling
    # numbers. If the features were rebuilt differently since (another split,
    # another window...), the column names could still match while every value
    # is translated with a different "dictionary". The model would then read
    # subtly wrong numbers without any error: training/serving skew. The
    # champion's run carries its own feature_stats.json, so compare them all.
    stats_path = PROJECT_ROOT / feature_config["output"]["stats_file"]
    current = json.loads(stats_path.read_text(encoding="utf-8"))
    trained_with = mlflow.artifacts.load_dict(f"runs:/{champion.run_id}/feature_stats.json")
    for key in ("feature_columns", "scaling"):
        if current[key] != trained_with[key]:
            print(f"ERROR: the processed features differ from the ones {model_label} was "
                  f"trained with ('{key}' does not match). Rebuild the features with the "
                  "champion's settings, or train and promote a new model.")
            return 1
    features = trained_with["feature_columns"]

    # The champion, fetched from the registry by name.
    model = mlflow.sklearn.load_model(f"models:/{name}@{CHAMPION}")

    # The bar to beat, for context: always guess the training average.
    processed_dir = PROJECT_ROOT / feature_config["output"]["processed_dir"]
    train = pd.read_parquet(processed_dir / "train_features.parquet")
    baseline = build_model("baseline").fit(train[features], train["rul"])

    test = last_cycle_per_engine(pd.read_parquet(processed_dir / "test_features.parquet"))
    truth = test[TRUTH]

    predictions = {
        model_label: predict_rul(model, test[features]),
        "baseline": predict_rul(baseline, test[features]),
    }
    results = {label: score(truth, p) for label, p in predictions.items()}

    commit = source_run.data.tags.get("mlflow.source.git.commit", "unknown")[:7]
    print(f"Champion: {model_label}, from run {champion.run_id[:8]} at commit {commit}\n")
    print(f"TEST SET: {len(test)} engines, scored at their last observed cycle, "
          f"against the TRUE remaining life\n")
    print(f"  {'model':<34}{'RMSE':>7}{'avg miss':>10}{'NASA score':>12}{'late':>7}{'early':>7}")
    for label, r in results.items():
        print(f"  {label:<34}{r['rmse']:7.1f}{r['mae']:10.1f}{r['nasa_score']:12.1f}"
              f"{r['n_late']:7d}{r['n_early']:7d}")

    # A few engines side by side, spread from "about to fail" to "lots of life".
    table = pd.DataFrame({
        "engine": test["unit_number"],
        "true": truth,
        "predicted": predictions[model_label].round(1),
    })
    table["miss"] = (table["predicted"] - table["true"]).round(1)
    sample = table.sort_values("true").iloc[::12]

    print("\nA few engines (miss > 0 means LATE, the dangerous direction):\n")
    print(sample.to_string(index=False))

    worst = table.loc[table["miss"].idxmax()]
    print(f"\nWorst LATE miss: engine {int(worst['engine'])} -- truly {int(worst['true'])} "
          f"cycles left, predicted {worst['predicted']:.0f} (late by {worst['miss']:.0f})")

    # Record the result in MLflow, next to everything else about this model.
    champion_scores = results[model_label]
    for metric in ("rmse", "mae", "nasa_score", "n_late", "n_early"):
        client.log_metric(champion.run_id, f"test_{metric}", float(champion_scores[metric]))
    client.set_model_version_tag(name, champion.version, "test_rmse", f"{champion_scores['rmse']:.2f}")
    client.set_model_version_tag(
        name, champion.version, "evaluated_at",
        datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )
    print(f"\nRecorded test_* metrics on run {champion.run_id[:8]} "
          f"and a test_rmse tag on {name} v{champion.version}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
