"""Validate the FD001 tables against the committed reference profile.

    python scripts/validate_fd001.py

Exits 0 if everything passes, 1 if anything fails. That exit code is the point:
it is what lets this become an automatic gate in a pipeline or in CI later, so
bad data stops rather than flowing on to a model.

Note that we validate the TEST split against a profile built from TRAIN. That is
not a mistake and it is not leakage -- nothing about test influenced the rules.
It is a genuine rehearsal of production, where live data arrives and gets judged
against a contract derived from historical data. If our rules were too tight,
the test split is exactly where we would find out.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.load_cmapss import load_cmapss, load_config  # noqa: E402
from src.data.validation import (  # noqa: E402
    ValidationResult,
    build_schema,
    load_reference_profile,
    load_validation_settings,
    validate_frame,
)


def report(result: ValidationResult) -> None:
    status = "PASS" if result.ok else "FAIL"
    print(f"\n  [{status}] {result.name}  ({result.n_rows:,} rows)")
    for error in result.errors:
        print(f"         - {error}")


def main() -> int:
    config = load_config()
    profile_path, range_tolerance, constant_tolerance = load_validation_settings(config)

    profile = load_reference_profile(profile_path)
    schema = build_schema(profile, range_tolerance, constant_tolerance)
    dataset = load_cmapss(config)

    print(f"Validating {dataset.name} against {profile_path.name}")
    print(f"  range tolerance    : {range_tolerance:.0%} of each column's span")
    print(f"  constant tolerance : {constant_tolerance:g} (absolute)")

    results = [
        validate_frame(dataset.train, schema, "train"),
        validate_frame(dataset.test, schema, "test"),
    ]

    for result in results:
        report(result)

    # The RUL truth table has a different shape (unit_number, rul), so the
    # sensor schema does not apply. Its invariants were already covered by the
    # ingestion tests; we only re-check the one that would silently corrupt
    # evaluation: one row per engine, no duplicates.
    rul = dataset.rul_truth
    rul_errors: list[str] = []
    if not rul["unit_number"].is_unique:
        rul_errors.append("duplicate unit_number in RUL truth table")
    if set(rul["unit_number"]) != set(dataset.test["unit_number"]):
        rul_errors.append("RUL truth engines do not match the test engines")
    if (rul["rul"] <= 0).any():
        rul_errors.append("RUL truth contains non-positive values")
    report(ValidationResult("rul_truth", len(rul), rul_errors))

    failed = [r for r in results if not r.ok] + ([1] if rul_errors else [])
    if failed:
        print("\nVALIDATION FAILED")
        return 1

    print("\nAll checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
