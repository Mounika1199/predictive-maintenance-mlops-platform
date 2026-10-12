"""Train every model, choose the best one on validation, and save the results.

    python scripts/train.py

Run AFTER scripts/build_features.py. Reads data/processed/, writes models/.

WHAT IT DOES
1. Train each model listed in configs/training.yaml on the 80 training engines.
2. Score each one on the 20 validation engines it has never seen.
3. Pick the winner: the lowest validation RMSE.
4. Save every trained model, plus a small record (models/selected.json) of
   which one won, its settings, and its scores.

THIS SCRIPT NEVER READS THE TEST SET. Choosing a model is a decision, and
decisions are made on validation data only. The test set is opened once,
afterwards, by scripts/evaluate.py -- to measure the choice, not to make it.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
import sklearn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.load_cmapss import load_config  # noqa: E402
from src.models.metrics import score  # noqa: E402
from src.models.training import build_model, load_feature_columns, predict_rul  # noqa: E402

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
    feature_config = load_config(FEATURES_CONFIG)
    training_config = load_config(TRAINING_CONFIG)

    processed_dir = PROJECT_ROOT / feature_config["output"]["processed_dir"]
    features = load_feature_columns(PROJECT_ROOT / feature_config["output"]["stats_file"])
    train = read_split(processed_dir, "train")
    val = read_split(processed_dir, "val")

    model_settings = training_config["models"]
    metric = training_config["selection_metric"]
    if "baseline" not in model_settings:
        raise ValueError("configs/training.yaml must list 'baseline': it is the bar to beat")

    # 1 + 2. Train each model on TRAIN, score it on VALIDATION.
    fitted = {}
    results = {}
    for name, params in model_settings.items():
        model = build_model(name, params, training_config["random_state"])
        model.fit(train[features], train[TARGET])
        scores = score(val[TARGET], predict_rul(model, val[features]))
        fitted[name] = model
        results[name] = {"rmse": scores["rmse"], "mae": scores["mae"]}

    print(f"Validation scores ({val['unit_number'].nunique()} engines never seen in training)\n")
    print(f"  {'model':<16}{'RMSE':>8}{'average miss':>15}")
    for name, r in results.items():
        print(f"  {name:<16}{r['rmse']:8.1f}{r['mae']:15.1f}")

    # 3. Pick the winner among the real candidates.
    candidates = [name for name in results if name != "baseline"]
    best = min(candidates, key=lambda name: results[name][metric])

    # A safety gate: if nothing beats guessing the average, something is
    # broken (features, data, code). Refuse to save rather than hand over a
    # useless model that looks like a real one.
    if results[best][metric] >= results["baseline"][metric]:
        print(f"\nERROR: best model '{best}' does not beat the baseline. Nothing saved.")
        return 1

    print(f"\nSelected: {best}  (lowest validation {metric.upper()})")

    # 4. Save every model, and a record of what happened.
    model_dir = PROJECT_ROOT / training_config["output"]["model_dir"]
    model_dir.mkdir(parents=True, exist_ok=True)

    print("\nWrote:")
    for name, model in fitted.items():
        path = model_dir / f"{name}.joblib"
        joblib.dump(model, path)
        print(f"  {path.relative_to(PROJECT_ROOT)}  ({path.stat().st_size / 1024 / 1024:,.1f} MB)")

    record = {
        "selected_model": best,
        "params": model_settings[best],
        "selection_metric": metric,
        "target": TARGET,
        "feature_columns": features,
        "validation": results,
        "training_data": {"rows": len(train), "engines": int(train["unit_number"].nunique())},
        "validation_data": {"rows": len(val), "engines": int(val["unit_number"].nunique())},
        "scikit_learn_version": sklearn.__version__,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    record_path = model_dir / "selected.json"
    with record_path.open("w", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2)
        handle.write("\n")
    print(f"  {record_path.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
