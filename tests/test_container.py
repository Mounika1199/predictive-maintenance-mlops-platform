"""Tests for the Docker setup: the prediction service, Prometheus and Grafana.

They read the files (Dockerfile, .dockerignore, compose.yaml, the two
requirements lists, the monitoring settings) rather than running Docker, so
they need no Docker. Each locks in a decision that would be easy to undo by
accident.
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

import pytest
import yaml
from prometheus_client.parser import text_string_to_metric_families

from src.monitoring.metrics import ServiceMetrics

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


@pytest.mark.parametrize("private", ["data", ".dvc", ".git", "mlops_env", "tests", "models", "logs", "reports", ".env"])
def test_private_or_bulky_folders_are_never_let_back_in(private: str) -> None:
    """.dvc/config.local holds a local path with the Windows username; data/
    and mlops_env/ are large; logs/ is the diary, shared at run time (a bind
    mount), never baked into the image. None of it belongs in the service."""
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


@pytest.mark.parametrize("service", ["mlflow", "rul-service", "prometheus", "grafana"])
def test_only_this_computer_can_reach_each_service(service: str) -> None:
    """Neither has a login, so neither may be published to the whole network."""
    for mapping in COMPOSE[service]["ports"]:
        assert mapping.startswith("127.0.0.1:"), f"{service} is exposed beyond this computer: {mapping}"


# ---------------------------------------------------------------------------
# Prometheus: pinned, and pointed at the service it is meant to watch
# ---------------------------------------------------------------------------


def test_prometheus_image_is_pinned_to_an_exact_version() -> None:
    image = COMPOSE["prometheus"]["image"]
    assert re.fullmatch(r"prom/prometheus:v\d+\.\d+\.\d+", image), image


def test_prometheus_collects_from_the_service() -> None:
    """The target must be the service's compose NAME and port. A typo here
    fails silently: Prometheus just shows the target as down."""
    config = yaml.safe_load((PROJECT_ROOT / "monitoring" / "prometheus" / "prometheus.yml").read_text(encoding="utf-8"))
    targets = [t for job in config["scrape_configs"] for c in job["static_configs"] for t in c["targets"]]
    service_port = COMPOSE["rul-service"]["ports"][0].rsplit(":", 1)[1]

    assert targets == [f"rul-service:{service_port}"]


def test_prometheus_settings_are_mounted_read_only() -> None:
    assert "./monitoring/prometheus/prometheus.yml:/etc/prometheus/prometheus.yml:ro" in COMPOSE["prometheus"]["volumes"]


# ---------------------------------------------------------------------------
# Grafana: no secrets in Git, nothing sent out, and wired to what exists
# ---------------------------------------------------------------------------

GRAFANA = COMPOSE["grafana"]
GRAFANA_DIR = PROJECT_ROOT / "monitoring" / "grafana"
DATASOURCE = yaml.safe_load((GRAFANA_DIR / "provisioning" / "datasources" / "prometheus.yml").read_text(encoding="utf-8"))["datasources"][0]
DASHBOARD = json.loads((GRAFANA_DIR / "dashboards" / "rul-service.json").read_text(encoding="utf-8"))


def test_grafana_image_is_pinned_to_an_exact_version() -> None:
    assert re.fullmatch(r"grafana/grafana:\d+\.\d+\.\d+", GRAFANA["image"]), GRAFANA["image"]


def test_no_password_is_written_in_compose() -> None:
    """This repository is public. The password must come from the local .env
    file, and ':?' makes compose refuse to start if it is missing."""
    assert GRAFANA["environment"]["GF_SECURITY_ADMIN_PASSWORD"].startswith("${GRAFANA_ADMIN_PASSWORD:?")


def test_the_env_file_is_hidden_from_git() -> None:
    ignored = subprocess.run(["git", "check-ignore", "-q", ".env"], cwd=PROJECT_ROOT)
    assert ignored.returncode == 0


@pytest.mark.parametrize("setting", [
    "GF_ANALYTICS_REPORTING_ENABLED",         # usage counters to stats.grafana.org
    "GF_ANALYTICS_CHECK_FOR_UPDATES",
    "GF_ANALYTICS_CHECK_FOR_PLUGIN_UPDATES",
    "GF_NEWS_NEWS_FEED_ENABLED",
    "GF_SNAPSHOTS_EXTERNAL_ENABLED",
])
def test_grafana_sends_nothing_out(setting: str) -> None:
    assert GRAFANA["environment"][setting] == "false"


def test_grafana_data_source_is_the_prometheus_service() -> None:
    host, port = DATASOURCE["url"].removeprefix("http://").split(":")
    assert host == "prometheus"
    assert COMPOSE[host]["ports"][0].endswith(f":{port}")


def test_grafana_set_up_files_are_mounted_read_only() -> None:
    assert "./monitoring/grafana/provisioning:/etc/grafana/provisioning:ro" in GRAFANA["volumes"]
    assert "./monitoring/grafana/dashboards:/etc/grafana/dashboards:ro" in GRAFANA["volumes"]


def test_grafana_looks_for_dashboards_where_compose_puts_them() -> None:
    """A mismatch fails silently: Grafana starts fine, with no dashboards."""
    provider = yaml.safe_load((GRAFANA_DIR / "provisioning" / "dashboards" / "dashboards.yml").read_text(encoding="utf-8"))
    path = provider["providers"][0]["options"]["path"]
    assert f"./monitoring/grafana/dashboards:{path}:ro" in GRAFANA["volumes"]


def test_every_panel_uses_the_provisioned_data_source() -> None:
    for panel in DASHBOARD["panels"]:
        assert panel["datasource"]["uid"] == DATASOURCE["uid"], panel["title"]
        for target in panel["targets"]:
            assert target["datasource"]["uid"] == DATASOURCE["uid"], panel["title"]


def test_every_panel_asks_only_for_metrics_the_service_produces() -> None:
    """Renaming a metric in src/monitoring/metrics.py without updating the
    dashboard would otherwise just leave a panel quietly empty."""
    metrics = ServiceMetrics()
    metrics.requests.labels("POST", "/predict", "200").inc()
    metrics.duration.labels("/predict").observe(0.1)
    metrics.model.labels("rul-fd001", "1").set(1)
    produced = {s.name for f in text_string_to_metric_families(metrics.page().decode()) for s in f.samples}

    for panel in DASHBOARD["panels"]:
        for target in panel["targets"]:
            for name in re.findall(r"\brul_\w+", target["expr"]):
                assert name in produced, f"panel '{panel['title']}' asks for {name}, which the service never produces"
