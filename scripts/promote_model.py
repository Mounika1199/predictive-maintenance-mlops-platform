"""Put a registered model version into use, by moving the 'champion' tag.

    python scripts/promote_model.py                     # consider the current challenger
    python scripts/promote_model.py --version 3         # consider a specific version
    python scripts/promote_model.py --version 1 --force --reason "roll back: v3 misbehaves"

This is the ONLY place that moves 'champion'. Training a model (train.py) and
putting it into use (this script) are separate, deliberate decisions.

BEFORE MOVING THE TAG it checks, using the rules in src/models/promotion.py:
  * the two versions were scored on the same exam (same validation engines,
    same answer key) -- otherwise their scores mean nothing side by side;
  * the challenger is better by a real margin (configs/training.yaml);
and it warns if the challenger was trained from uncommitted code.

--force skips the checks (a rollback is the classic case) but needs --reason.

Aliases keep no history, so every promotion writes its own record onto the
promoted version: when, which version it replaced, why, and whether it was
forced. Exit code: 0 = promoted (or already champion), 1 = refused.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import mlflow
from mlflow import MlflowClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.load_cmapss import load_config  # noqa: E402
from src.models.promotion import EXAM_SETTINGS, Candidate, decide_promotion  # noqa: E402
from src.models.tracking import (  # noqa: E402
    CHALLENGER,
    CHAMPION,
    check_server,
    resolve_tracking_uri,
)

TRAINING_CONFIG = PROJECT_ROOT / "configs" / "training.yaml"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Move the 'champion' tag, after checks.")
    parser.add_argument("--version", help="version to consider (default: the current challenger)")
    parser.add_argument("--force", action="store_true",
                        help="promote even if the checks refuse (e.g. a rollback); needs --reason")
    parser.add_argument("--reason", help="why -- recorded on the promoted version")
    args = parser.parse_args(argv)
    if args.force and not args.reason:
        parser.error("--force needs --reason: the record must say why the checks were overridden")
    return args


def version_with_alias(client: MlflowClient, name: str, alias: str) -> str | None:
    try:
        return str(client.get_model_version_by_alias(name, alias).version)
    except mlflow.exceptions.MlflowException:
        return None


def load_candidate(client: MlflowClient, name: str, version: str) -> Candidate:
    """Everything the rules need, read from the version's source run."""
    model_version = client.get_model_version(name, version)
    run = client.get_run(model_version.run_id)
    flag = run.data.tags.get("git.uncommitted_changes")
    return Candidate(
        version=str(model_version.version),
        val_rmse=run.data.metrics["val_rmse"],
        exam={key: run.data.params.get(key) for key in EXAM_SETTINGS},
        uncommitted={"True": True, "False": False}.get(flag),  # anything else: unknown
    )


def show(champion: Candidate | None, challenger: Candidate) -> None:
    def describe(c: Candidate | None) -> list[str]:
        if c is None:
            return ["(none)", "", "", ""]
        exam = f"seed {c.exam['split_seed']}, {c.exam['validation_engines']} engines, cap {c.exam['rul_cap']}"
        code = {True: "UNCOMMITTED changes", False: "committed", None: "unknown"}[c.uncommitted]
        return [f"version {c.version}", f"{c.val_rmse:.2f}", exam, code]

    labels = ["", "validation RMSE", "scored on", "code"]
    left, right = describe(champion), describe(challenger)
    print(f"  {'':<17}{'CHAMPION (in use)':<40}{'CANDIDATE':<40}")
    for label, a, b in zip(labels, left, right):
        print(f"  {label:<17}{a:<40}{b:<40}")


def main(argv: list[str] | None = None) -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # see train.py
    args = parse_args(argv)

    config = load_config(TRAINING_CONFIG)
    tracking_uri = resolve_tracking_uri(config["mlflow"]["tracking_uri"])
    check_server(tracking_uri)
    client = MlflowClient(tracking_uri)

    name = config["mlflow"]["registered_model_name"]
    min_improvement = float(config["promotion"]["min_improvement_rmse"])

    candidate_version = args.version or version_with_alias(client, name, CHALLENGER)
    if candidate_version is None:
        print(f"ERROR: '{name}' has no challenger. Train one with: python scripts/train.py")
        return 1
    champion_version = version_with_alias(client, name, CHAMPION)

    candidate = load_candidate(client, name, candidate_version)
    champion = load_candidate(client, name, champion_version) if champion_version else None

    print(f"Promotion check for '{name}'\n")
    show(champion, candidate)

    if champion is not None and candidate.version == champion.version:
        print(f"\nNothing to do: version {candidate.version} is already the champion.")
        return 0

    decision = decide_promotion(candidate, champion, min_improvement)
    for warning in decision.warnings:
        print(f"\nWARNING: {warning}")
    print(f"\nCheck: {'PASS' if decision.promote else 'FAIL'} -- {decision.reason}")

    if not decision.promote and not args.force:
        print(f"\nNOT promoted. The champion is still version {champion.version}.")
        return 1

    forced = not decision.promote
    if forced:
        print(f"\nFORCED past the check. Reason given: {args.reason}")

    client.set_registered_model_alias(name, CHAMPION, candidate.version)
    record = {
        "promoted_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "promoted_from_version": champion.version if champion else "none",
        "promotion_reason": args.reason or decision.reason,
        "promotion_forced": str(forced),
    }
    for key, value in record.items():
        client.set_model_version_tag(name, candidate.version, key, value)

    previous = f"version {champion.version}" if champion else "no champion"
    print(f"\nPROMOTED: {name}@champion -> version {candidate.version} (was {previous}).")
    print(f"Anything loading models:/{name}@champion now gets version {candidate.version}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
