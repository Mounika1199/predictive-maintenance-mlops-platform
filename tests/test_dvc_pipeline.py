"""Tests for the DVC setup: the pipeline's shape, the validation gate, the data
fingerprints recorded on training runs, and the .gitignore rules DVC relies on.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from src.models.tracking import data_versions

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PIPELINE = yaml.safe_load((PROJECT_ROOT / "dvc.yaml").read_text(encoding="utf-8"))["stages"]


def deps(stage: str) -> list[str]:
    return PIPELINE[stage]["deps"]


def outs(stage: str) -> list[str]:
    return PIPELINE[stage]["outs"]


# ---------------------------------------------------------------------------
# The pipeline's shape: validate -> ingest -> build_features
# ---------------------------------------------------------------------------


def test_the_pipeline_has_its_three_steps() -> None:
    assert set(PIPELINE) == {"validate", "ingest", "build_features"}


def test_ingest_is_gated_by_validation() -> None:
    """The gate: ingest depends on validation's report, so a failed validation
    (exit code 1) stops everything downstream. If someone removed this
    dependency, bad data could reach the model again -- silently."""
    report = "data/validation/report.json"
    assert report in outs("validate")
    assert report in deps("ingest")


def test_features_are_built_from_ingested_data() -> None:
    assert "data/interim" in outs("ingest")
    assert "data/interim" in deps("build_features")
    assert "data/processed" in outs("build_features")


@pytest.mark.parametrize("stage", ["validate", "ingest", "build_features"])
def test_each_step_depends_on_its_own_script(stage: str) -> None:
    """A code change must re-run the step, or its output silently goes stale."""
    script = PIPELINE[stage]["cmd"].split()[1]  # "python scripts/x.py ..." -> "scripts/x.py"
    assert script in deps(stage)


@pytest.mark.parametrize("stage", ["validate", "ingest", "build_features"])
def test_every_listed_dependency_exists(stage: str) -> None:
    """A typo in dvc.yaml would make DVC fail -- catch it here first."""
    for path in deps(stage):
        assert (PROJECT_ROOT / path).exists(), f"{stage}: missing dependency {path}"


# ---------------------------------------------------------------------------
# The validation report (the file the gate is made of)
# ---------------------------------------------------------------------------


def run_validation(report: Path) -> int:
    return subprocess.run(
        [sys.executable, "scripts/validate_fd001.py", "--report", str(report)],
        cwd=PROJECT_ROOT, capture_output=True, text=True,
    ).returncode


def test_validation_writes_a_passing_report(tmp_path: Path) -> None:
    report = tmp_path / "report.json"

    assert run_validation(report) == 0
    document = json.loads(report.read_text(encoding="utf-8"))
    assert document["passed"] is True
    assert set(document["tables"]) == {"train", "test", "rul_truth"}


def test_the_report_is_identical_every_time(tmp_path: Path) -> None:
    """No timestamps: an unchanged result must give an identical file, or DVC
    would think something changed and re-run everything downstream."""
    first, second = tmp_path / "a.json", tmp_path / "b.json"
    run_validation(first)
    run_validation(second)
    assert first.read_bytes() == second.read_bytes()


# ---------------------------------------------------------------------------
# Data fingerprints recorded on every training run
# ---------------------------------------------------------------------------


def test_data_versions_reads_pointer_and_receipt(tmp_path: Path) -> None:
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "raw.dvc").write_text(
        "outs:\n- md5: aaa111.dir\n  path: raw\n", encoding="utf-8"
    )
    (tmp_path / "dvc.lock").write_text(
        "schema: '2.0'\n"
        "stages:\n"
        "  ingest:\n"
        "    outs:\n"
        "    - path: data/interim\n"
        "      md5: bbb222.dir\n"
        "  build_features:\n"
        "    outs:\n"
        "    - path: data/processed\n"
        "      md5: ccc333.dir\n",
        encoding="utf-8",
    )

    assert data_versions(tmp_path) == {
        "data_raw_md5": "aaa111.dir",
        "data_processed_md5": "ccc333.dir",  # the processed data, not interim
    }


def test_data_versions_says_unknown_rather_than_crashing(tmp_path: Path) -> None:
    assert data_versions(tmp_path) == {
        "data_raw_md5": "unknown",
        "data_processed_md5": "unknown",
    }


def test_the_real_fingerprints_match_dvcs_own_files() -> None:
    versions = data_versions(PROJECT_ROOT)
    assert versions["data_raw_md5"] in (PROJECT_ROOT / "data" / "raw.dvc").read_text(encoding="utf-8")
    assert versions["data_processed_md5"] in (PROJECT_ROOT / "dvc.lock").read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# .gitignore: data hidden from Git, pointer files visible to Git AND to DVC
# ---------------------------------------------------------------------------


def is_ignored_by_git(path: str) -> bool:
    return subprocess.run(["git", "check-ignore", "-q", path], cwd=PROJECT_ROOT).returncode == 0


@pytest.mark.parametrize("path", ["data/raw/train_FD001.txt", "data/processed/feature_stats.json"])
def test_data_files_are_hidden_from_git(path: str) -> None:
    assert is_ignored_by_git(path)


def test_pointer_files_are_visible_to_git() -> None:
    assert not is_ignored_by_git("data/raw.dvc")


def test_dvc_can_find_the_pointer_file() -> None:
    """The bug this guards against failed SILENTLY. With the rule "/data/*",
    DVC asked "is data/ ignored?" (with the slash), got yes, never looked
    inside -- and `dvc push` reported "Everything is up to date" while pushing
    nothing. Git itself looked fine. Only asking DVC reveals it."""
    from dvc.repo import Repo

    with Repo(str(PROJECT_ROOT)) as repo:
        assert not repo.scm.is_ignored(os.path.join(repo.root_dir, "data") + os.sep)
        found = {Path(stage.relpath).as_posix() for stage in repo.index.stages}
    assert "data/raw.dvc" in found
