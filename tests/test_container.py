"""Tests for the Docker setup of the prediction service.

They read the files (Dockerfile, .dockerignore, compose.yaml, the two
requirements lists) rather than running Docker, so they need no Docker. Each
locks in a decision that would be easy to undo by accident.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
COMPOSE = yaml.safe_load((PROJECT_ROOT / "compose.yaml").read_text(encoding="utf-8"))["services"]


def pins(filename: str) -> dict[str, str]:
    """{package: version} for every 'name==version' line in a requirements file."""
    found = {}
    for line in (PROJECT_ROOT / filename).read_text(encoding="utf-8").splitlines():
        match = re.match(r"^([A-Za-z0-9_.\-]+)==([^\s#]+)", line.strip())
        if match:
            found[match.group(1).lower()] = match.group(2)
    return found


# ---------------------------------------------------------------------------
# The service image runs the same library versions the model was trained with
# ---------------------------------------------------------------------------


def test_serving_packages_match_the_main_requirements() -> None:
    """The image installs requirements-serving.txt; training and tests use
    requirements.txt. If the two disagreed, the service could run a different
    scikit-learn or pandas from the one the model was trained and tested with."""
    serving, main = pins("requirements-serving.txt"), pins("requirements.txt")

    assert serving, "requirements-serving.txt has no pinned packages"
    for package, version in serving.items():
        assert package in main, f"{package} is in the image but not in requirements.txt"
        assert version == main[package], (
            f"{package}: image has {version}, requirements.txt has {main[package]}"
        )


def test_the_image_leaves_out_tools_the_service_never_uses() -> None:
    serving = pins("requirements-serving.txt")
    for unused in ("dvc", "pytest", "httpx2"):
        assert unused not in serving


# ---------------------------------------------------------------------------
# Nothing private or bulky reaches the image
# ---------------------------------------------------------------------------


def dockerignore_rules() -> list[str]:
    text = (PROJECT_ROOT / ".dockerignore").read_text(encoding="utf-8")
    return [line.strip() for line in text.splitlines() if line.strip() and not line.startswith("#")]


def test_dockerignore_is_an_allow_list() -> None:
    """Ignore everything first, then let back in only what the service needs:
    anything added to the project later stays out by default."""
    rules = dockerignore_rules()
    assert rules[0] == "*"


@pytest.mark.parametrize("private", ["data", ".dvc", ".git", "mlops_env", "tests", "models"])
def test_private_or_bulky_folders_are_never_let_back_in(private: str) -> None:
    """.dvc/config.local holds a local path with the Windows username; data/
    and mlops_env/ are large; none of it belongs in the service."""
    let_back_in = [rule[1:].strip("/") for rule in dockerignore_rules() if rule.startswith("!")]
    assert not any(rule.split("/")[0] == private for rule in let_back_in)


def test_the_image_runs_as_an_ordinary_user() -> None:
    dockerfile = (PROJECT_ROOT / "Dockerfile").read_text(encoding="utf-8")
    users = re.findall(r"^USER\s+(\S+)", dockerfile, flags=re.MULTILINE)
    assert users and users[-1] not in ("root", "0")


# ---------------------------------------------------------------------------
# compose.yaml: the two containers find each other, and only this computer
# can reach either of them
# ---------------------------------------------------------------------------


def test_the_service_finds_mlflow_by_its_service_name() -> None:
    """Inside Docker, 'localhost' is the container itself. The service must
    use MLflow's service name on the compose network."""
    uri = COMPOSE["rul-service"]["environment"]["MLFLOW_TRACKING_URI"]
    assert uri == "http://mlflow:5000"
    assert "mlflow" in COMPOSE


def test_mlflow_trusts_exactly_the_names_it_needs() -> None:
    """MLflow refused requests addressed to 'mlflow:5000' (403, DNS-rebinding
    protection) -- while /health still said OK. The fix adds that one name;
    switching the protection off with '*' would not be a fix."""
    command = COMPOSE["mlflow"]["command"]
    allowed = re.search(r"--allowed-hosts\s+(\S+)", command).group(1).split(",")

    assert "mlflow:5000" in allowed
    assert "*" not in allowed
    assert any(host.startswith("localhost") for host in allowed)  # our Windows scripts


def test_the_service_waits_until_mlflow_is_healthy() -> None:
    """It fetches the champion at startup, so MLflow must be ready, not just started."""
    assert COMPOSE["rul-service"]["depends_on"]["mlflow"]["condition"] == "service_healthy"
    assert "healthcheck" in COMPOSE["mlflow"]


@pytest.mark.parametrize("service", ["mlflow", "rul-service"])
def test_only_this_computer_can_reach_each_service(service: str) -> None:
    """Neither has a login, so neither may be published to the whole network."""
    for mapping in COMPOSE[service]["ports"]:
        assert mapping.startswith("127.0.0.1:"), f"{service} is exposed beyond this computer: {mapping}"
