"""Generate the reference profile that validation checks against.

    python scripts/build_reference_profile.py

WHY THIS IS A SEPARATE, MANUAL STEP
The profile defines what "normal data" means. If it regenerated automatically on
every run, then any corruption that slipped into the training data would quietly
become the new definition of normal -- the alarm would rewrite itself to agree
with whatever it was handed. So it is generated deliberately, committed to Git,
and changes to it show up in code review like any other change to a contract.

The profile is built from TRAINING DATA ONLY. The test set must stay untouched:
the moment we let it influence our definition of normal, it stops being an
honest estimate of performance on unseen data.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.load_cmapss import load_cmapss, load_config  # noqa: E402
from src.data.validation import (  # noqa: E402
    build_reference_profile,
    load_validation_settings,
    save_reference_profile,
)


def main() -> int:
    config = load_config()
    profile_path, _, _ = load_validation_settings(config)

    dataset = load_cmapss(config)
    profile = build_reference_profile(dataset.train, dataset.name)

    n_constant = sum(1 for spec in profile["columns"].values() if spec["constant"])
    n_varying = len(profile["columns"]) - n_constant

    print(f"Reference profile for {profile['dataset']}, built from TRAIN only")
    print(f"  rows            : {profile['n_rows']:,}")
    print(f"  engines         : {profile['n_engines']}")
    print(f"  columns         : {len(profile['columns'])}")
    print(f"    constant      : {n_constant}")
    print(f"    varying       : {n_varying}")

    constant_names = [n for n, s in profile["columns"].items() if s["constant"]]
    print(f"\n  constant columns: {', '.join(constant_names)}")

    written = save_reference_profile(profile, profile_path)
    print(f"\nWrote {written.relative_to(PROJECT_ROOT)}")
    print("Commit this file -- it is the contract validation enforces.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
