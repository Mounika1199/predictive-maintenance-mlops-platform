"""Score the selected model on the TEST set -- the honest, final number.

    python scripts/evaluate.py

Run AFTER scripts/train.py. Loads the saved model; trains nothing.

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
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.load_cmapss import load_config  # noqa: E402
from src.features.labels import last_cycle_per_engine  # noqa: E402
from src.models.metrics import score  # noqa: E402
from src.models.training import load_feature_columns, predict_rul  # noqa: E402

FEATURES_CONFIG = PROJECT_ROOT / "configs" / "features.yaml"
TRAINING_CONFIG = PROJECT_ROOT / "configs" / "training.yaml"

TRUTH = "rul_uncapped"


def main() -> int:
    feature_config = load_config(FEATURES_CONFIG)
    training_config = load_config(TRAINING_CONFIG)
    model_dir = PROJECT_ROOT / training_config["output"]["model_dir"]

    record_path = model_dir / "selected.json"
    if not record_path.is_file():
        raise FileNotFoundError("No trained model found. Run: python scripts/train.py")
    record = json.loads(record_path.read_text(encoding="utf-8"))

    # Safety check: the model must be fed exactly the features it learned from.
    # If the features were rebuilt differently after training (a different
    # window, a sensor dropped...), the columns would no longer line up and the
    # model would silently read the wrong numbers.
    current = load_feature_columns(PROJECT_ROOT / feature_config["output"]["stats_file"])
    if current != record["feature_columns"]:
        print("ERROR: features have changed since this model was trained. Re-run train.py.")
        return 1
    features = record["feature_columns"]

    selected_name = record["selected_model"]
    models = {
        selected_name: joblib.load(model_dir / f"{selected_name}.joblib"),
        "baseline": joblib.load(model_dir / "baseline.joblib"),
    }

    processed_dir = PROJECT_ROOT / feature_config["output"]["processed_dir"]
    test = last_cycle_per_engine(pd.read_parquet(processed_dir / "test_features.parquet"))
    truth = test[TRUTH]

    predictions = {name: predict_rul(m, test[features]) for name, m in models.items()}
    results = {name: score(truth, p) for name, p in predictions.items()}

    print(f"TEST SET: {len(test)} engines, scored at their last observed cycle, "
          f"against the TRUE remaining life\n")
    print(f"  {'model':<16}{'RMSE':>7}{'avg miss':>10}{'NASA score':>12}{'late':>7}{'early':>7}")
    for name, r in results.items():
        print(f"  {name:<16}{r['rmse']:7.1f}{r['mae']:10.1f}{r['nasa_score']:12.1f}"
              f"{r['n_late']:7d}{r['n_early']:7d}")

    # A few engines side by side, spread from "about to fail" to "lots of life".
    table = pd.DataFrame({
        "engine": test["unit_number"],
        "true": truth,
        "predicted": predictions[selected_name].round(1),
    })
    table["miss"] = (table["predicted"] - table["true"]).round(1)
    sample = table.sort_values("true").iloc[::12]

    print(f"\nA few engines ({selected_name}; miss > 0 means LATE, the dangerous direction):\n")
    print(sample.to_string(index=False))

    worst = table.loc[table["miss"].idxmax()]
    print(f"\nWorst LATE miss: engine {int(worst['engine'])} -- truly {int(worst['true'])} "
          f"cycles left, predicted {worst['predicted']:.0f} (late by {worst['miss']:.0f})")

    report = {
        "selected_model": selected_name,
        "scored_against": TRUTH,
        "n_engines": len(test),
        "results": results,
        "evaluated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    report_path = model_dir / "test_results.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"\nWrote {report_path.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
