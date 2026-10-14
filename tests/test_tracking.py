"""Tests for the MLflow tracking helpers.

None of these need Docker or the real MLflow server:

* the server check is tested against a tiny fake web server started by the test;
* saving a model is tested against a temporary folder (see `temporary_mlflow`).
"""

from __future__ import annotations

import re
import socket
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import pytest
from mlflow import MlflowClient

from src.models import tracking
from src.models.training import build_model

PROJECT_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Which server to use
# ---------------------------------------------------------------------------


def test_environment_variable_wins_over_config(monkeypatch) -> None:
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://somewhere-else:5000")
    assert tracking.resolve_tracking_uri("http://localhost:5000") == "http://somewhere-else:5000"


def test_config_is_used_when_no_environment_variable(monkeypatch) -> None:
    # Cleared explicitly: mlflow.set_tracking_uri() silently sets this variable,
    # so another test could otherwise leak its address into this one.
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    assert tracking.resolve_tracking_uri("http://localhost:5000") == "http://localhost:5000"


# ---------------------------------------------------------------------------
# Checking the server is up before training
# ---------------------------------------------------------------------------


@contextmanager
def fake_server(status: int):
    """A tiny web server whose /health page answers with the given status."""

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(status if self.path == "/health" else 404)
            self.end_headers()
            self.wfile.write(b"OK")

        def log_message(self, *args):  # keep test output quiet
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)  # port 0 = any free port
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()


def unused_port() -> int:
    """A port number that nothing is listening on."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_healthy_server_passes_the_check() -> None:
    with fake_server(200) as url:
        tracking.check_server(url)  # must not raise


def test_unhealthy_server_is_reported() -> None:
    with fake_server(503) as url:
        with pytest.raises(ConnectionError, match="503"):
            tracking.check_server(url)


def test_stopped_server_gives_a_helpful_message() -> None:
    """The message must tell a beginner exactly what to do next."""
    with pytest.raises(ConnectionError, match="docker compose up -d"):
        tracking.check_server(f"http://127.0.0.1:{unused_port()}", timeout=2)


def test_local_storage_is_not_checked() -> None:
    """A local folder or database has no server to ask -- nothing to check."""
    tracking.check_server("file:///C:/somewhere/mlruns")  # must not raise


# ---------------------------------------------------------------------------
# What gets recorded about each run
# ---------------------------------------------------------------------------


def feature_stats(include_cycle: bool) -> dict:
    """A hand-made, cut-down feature_stats.json."""
    columns = ["sensor_2", "sensor_2_rollmean", "sensor_2_rollstd"]
    if include_cycle:
        columns.append("cycle")
    return {
        "feature_columns": columns,
        "sensors": ["sensor_2"],
        "rul_cap": 125,
        "rolling_window": 5,
        "split": {
            "seed": 42,
            "validation_fraction": 0.2,
            "train_engines": [1, 2, 3, 4],
            "validation_engines": [5],
        },
    }


def test_run_params_record_both_model_and_data_settings() -> None:
    params = tracking.run_params(
        "random_forest", {"n_estimators": 200}, random_state=42, feature_stats=feature_stats(True)
    )

    assert params["model"] == "random_forest"
    assert params["model.n_estimators"] == 200  # model settings get a "model." prefix
    assert params["random_state"] == 42
    assert params["rul_cap"] == 125
    assert params["rolling_window"] == 5
    assert params["n_features"] == 4
    assert params["train_engines"] == 4
    assert params["validation_engines"] == 1


@pytest.mark.parametrize("include_cycle", [True, False])
def test_run_params_show_whether_the_cycle_feature_was_used(include_cycle: bool) -> None:
    """The setting the next experiment will change -- it must be visible in MLflow."""
    params = tracking.run_params("linear", {}, 42, feature_stats(include_cycle))
    assert params["include_cycle_feature"] is include_cycle


# ---------------------------------------------------------------------------
# Saving a model end to end
# ---------------------------------------------------------------------------


@pytest.fixture
def temporary_mlflow(tmp_path, monkeypatch):
    """Point MLflow at a throwaway folder for one test, then undo it.

    MLflow has retired folder-based storage for real use ("maintenance mode")
    and only allows it if you opt in. It is fine as a stand-in here: these tests
    check OUR saving code, not MLflow's storage. The real server is untouched.
    """
    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")
    mlflow.set_tracking_uri((tmp_path / "mlruns").as_uri())
    mlflow.set_experiment("test")
    yield
    mlflow.set_tracking_uri(None)  # also clears the MLFLOW_TRACKING_URI it set


def test_logged_forest_loads_back_and_predicts_identically(temporary_mlflow, tmp_path) -> None:
    """Would have caught both Step 3 surprises: missing skops, and the untrusted
    tree type. A random forest is the hard case -- linear models only hold plain
    numbers."""
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"a": rng.uniform(0, 10, 40), "b": rng.uniform(0, 10, 40)})
    y = 3 * X["a"]
    # Default n_jobs (one core), so predictions are exactly repeatable.
    model = build_model("random_forest", {"n_estimators": 5}).fit(X, y)

    stats_path = tmp_path / "feature_stats.json"
    stats_path.write_text('{"feature_columns": ["a", "b"]}', encoding="utf-8")

    with mlflow.start_run() as run:
        info = tracking.log_trained_model(model, X, model.predict(X), stats_path)

    loaded = mlflow.sklearn.load_model(info.model_uri)
    np.testing.assert_array_equal(loaded.predict(X), model.predict(X))

    # The model records what it expects as input...
    assert info.signature.inputs.input_names() == ["a", "b"]
    # ...and its scaling numbers travel with it.
    artifacts = [a.path for a in MlflowClient().list_artifacts(run.info.run_id)]
    assert "feature_stats.json" in artifacts


def test_only_one_type_is_trusted() -> None:
    """A tripwire. If loading ever fails with 'untrusted types', the tempting fix
    is to trust whatever the error lists. Each extra type should be a deliberate
    decision -- if you add one, update this test on purpose."""
    assert tracking.SKOPS_TRUSTED_TYPES == ["sklearn.tree._tree.Tree"]


# ---------------------------------------------------------------------------
# Client and server must speak the same version
# ---------------------------------------------------------------------------


def test_client_and_server_versions_match() -> None:
    """requirements.txt (client) and compose.yaml (server) are separate files.
    Upgrading one and forgetting the other is an easy mistake; this catches it."""
    requirements = (PROJECT_ROOT / "requirements.txt").read_text(encoding="utf-8")
    compose = (PROJECT_ROOT / "compose.yaml").read_text(encoding="utf-8")

    client = re.search(r"^mlflow-skinny==([\d.]+)", requirements, re.MULTILINE).group(1)
    server = re.search(r"ghcr\.io/mlflow/mlflow:v([\d.]+)", compose).group(1)

    assert client == server


# ---------------------------------------------------------------------------
# The model registry: new versions are challengers, never champions
# ---------------------------------------------------------------------------


def test_registering_a_challenger_never_moves_the_champion(temporary_mlflow) -> None:
    """Training a model and putting it into use are separate decisions."""
    client = MlflowClient()
    X = pd.DataFrame({"a": [1.0, 2.0, 3.0]})

    def log_a_model() -> str:
        with mlflow.start_run():
            model = build_model("baseline").fit(X, [1.0, 2.0, 3.0])
            return mlflow.sklearn.log_model(model, name="model").model_uri

    first = tracking.register_challenger(client, log_a_model(), "demo-model", "first")
    client.set_registered_model_alias("demo-model", tracking.CHAMPION, first.version)

    second = tracking.register_challenger(client, log_a_model(), "demo-model", "second")

    aliases = {name: str(v) for name, v in client.get_registered_model("demo-model").aliases.items()}
    assert aliases[tracking.CHAMPION] == str(first.version)  # untouched
    assert aliases[tracking.CHALLENGER] == str(second.version)  # moved to the newest
    assert client.get_model_version("demo-model", second.version).description == "second"


# ---------------------------------------------------------------------------
# Detecting uncommitted code
# ---------------------------------------------------------------------------


@pytest.fixture
def tiny_repo(tmp_path):
    """A one-file Git repository with a single commit."""
    import git

    repo = git.Repo.init(tmp_path)
    repo.git.config("user.name", "Test")
    repo.git.config("user.email", "test@example.com")
    # write_bytes keeps Windows from changing the line endings, which Git
    # would otherwise report as a modification.
    (tmp_path / "code.py").write_bytes(b"x = 1\n")
    repo.git.add("code.py")
    repo.git.commit("-m", "first")
    return tmp_path


def test_committed_code_is_clean(tiny_repo) -> None:
    assert tracking.has_uncommitted_changes(tiny_repo) is False


def test_an_edited_file_counts_as_uncommitted(tiny_repo) -> None:
    (tiny_repo / "code.py").write_bytes(b"x = 2\n")
    assert tracking.has_uncommitted_changes(tiny_repo) is True


def test_a_new_file_counts_as_uncommitted(tiny_repo) -> None:
    """A brand-new script is code too, even before Git knows about it."""
    (tiny_repo / "new_script.py").write_bytes(b"print('hi')\n")
    assert tracking.has_uncommitted_changes(tiny_repo) is True


def test_a_folder_outside_git_is_unknown(tmp_path) -> None:
    assert tracking.has_uncommitted_changes(tmp_path) is None
