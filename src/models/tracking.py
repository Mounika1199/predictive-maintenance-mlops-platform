"""Small helpers for recording training runs in MLflow.

MLflow vocabulary, in plain words:

* EXPERIMENT -- a folder of related attempts ("RUL prediction for FD001").
* RUN        -- one attempt: one model, trained once.
* PARAMETER  -- a setting chosen BEFORE the run (n_estimators = 200).
* METRIC     -- a score measured DURING or AFTER the run (val_rmse = 15.1).
* ARTIFACT   -- a file kept with the run (the trained model itself).
"""

from __future__ import annotations

import os
import urllib.request
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow import MlflowClient
from mlflow.entities.model_registry import ModelVersion
from mlflow.models import infer_signature
from mlflow.models.model import ModelInfo

from src.features.engineering import CYCLE_FEATURE

# Registry aliases: movable name tags on model versions.
#   champion   -- the version in use. Moved ONLY by scripts/promote_model.py.
#   challenger -- the newest candidate. Moved by every training session.
CHAMPION = "champion"
CHALLENGER = "challenger"

# MLflow saves scikit-learn models with skops rather than pickle, because
# opening a pickle file can run any code hidden inside it. skops refuses to open
# object types it cannot prove are safe. A random forest's flowcharts are stored
# as sklearn.tree._tree.Tree, whose "go to node N" numbers scikit-learn follows
# without checking -- a tampered file could crash the process or read memory it
# should not.
#
# We trust exactly this one type, and only because these files are created by
# our own training code, on a server only this computer can reach (compose.yaml
# binds it to 127.0.0.1). Never add types just to make an unknown file load.
SKOPS_TRUSTED_TYPES = ["sklearn.tree._tree.Tree"]


def has_uncommitted_changes(repo_path: Path | str) -> bool | None:
    """Does the code differ from its last Git commit (edited or new files)?

    MLflow records the commit a run came from -- but if files had been edited
    since that commit, the commit alone cannot reproduce the run. So we record
    this too, rather than relying on everyone to remember to commit first.

    Returns None if the folder is not a Git repository. Ignored files (data/,
    models/) do not count -- they are not code.
    """
    try:
        import git

        repo = git.Repo(repo_path, search_parent_directories=True)
    except Exception:  # not a repository, or git not available
        return None
    return repo.is_dirty(untracked_files=True)


def resolve_tracking_uri(configured: str) -> str:
    """Where to send runs: MLFLOW_TRACKING_URI if set, else the configured address.

    The environment variable is MLflow's own standard name, so this behaves the
    way anyone who knows MLflow would expect.
    """
    return os.environ.get("MLFLOW_TRACKING_URI") or configured


def check_server(uri: str, timeout: float = 3.0) -> None:
    """Fail fast, with a clear message, if the tracking server is not reachable.

    Without this, a stopped server only shows up AFTER training -- as a slow
    series of retries and then a confusing network error. Checking first costs
    a fraction of a second.

    Only network addresses (http/https) are checked. A local database such as
    sqlite:///mlflow.db has no server to check.
    """
    if not uri.startswith(("http://", "https://")):
        return

    health_url = f"{uri.rstrip('/')}/health"
    try:
        with urllib.request.urlopen(health_url, timeout=timeout) as response:
            if response.status == 200:
                return
            problem = f"it answered with HTTP {response.status}"
    except OSError as exc:  # connection refused, timeout, bad address...
        problem = str(exc)

    raise ConnectionError(
        f"Cannot reach the MLflow server at {uri} ({problem}).\n"
        "Is Docker running? Start the server with:  docker compose up -d"
    )


def run_params(
    model_name: str,
    model_params: dict,
    random_state: int,
    feature_stats: dict,
) -> dict[str, object]:
    """Everything worth recording about HOW one model was trained.

    Not just the model's own settings, but also how the DATA was prepared --
    because two runs with identical model settings but different data (say,
    with and without the cycle feature) are different experiments, and should
    look different in MLflow.

    The data settings come from feature_stats.json, which records how the
    processed data was ACTUALLY built. configs/features.yaml might have been
    edited since without the features being rebuilt; the stats file cannot lie
    about that.
    """
    features = feature_stats["feature_columns"]
    split = feature_stats["split"]

    return {
        "model": model_name,
        **{f"model.{key}": value for key, value in model_params.items()},
        "random_state": random_state,
        "rul_cap": feature_stats["rul_cap"],
        "rolling_window": feature_stats["rolling_window"],
        "n_sensors": len(feature_stats["sensors"]),
        "n_features": len(features),
        "include_cycle_feature": CYCLE_FEATURE in features,
        "split_seed": split["seed"],
        "validation_fraction": split["validation_fraction"],
        "train_engines": len(split["train_engines"]),
        "validation_engines": len(split["validation_engines"]),
    }


def log_trained_model(
    model,
    example_features: pd.DataFrame,
    predictions,
    stats_path: Path | str,
) -> ModelInfo:
    """Attach a trained model, and the numbers needed to use it, to the ACTIVE run.

    Call this inside ``with mlflow.start_run():``. It stores:

    * the model itself, saved with skops (see SKOPS_TRUSTED_TYPES);
    * its SIGNATURE -- the exact input columns it expects and the shape of what
      it returns -- so anyone loading it later knows what to feed it without
      reading our code;
    * feature_stats.json -- the scaling numbers that turn raw sensor readings
      into the inputs this model expects. Model and preprocessing travel
      together, so they can never be mismatched.
    """
    signature = infer_signature(example_features.head(100), predictions[:100])
    info = mlflow.sklearn.log_model(
        model,
        name="model",
        signature=signature,
        input_example=example_features.head(3),
        skops_trusted_types=SKOPS_TRUSTED_TYPES,
    )
    mlflow.log_artifact(str(stats_path))
    return info


def register_challenger(
    client: MlflowClient,
    model_uri: str,
    registered_name: str,
    description: str,
) -> ModelVersion:
    """Add a trained model to the registry as a new version tagged 'challenger'.

    A new version number is created every time (versions are never
    overwritten), and the 'challenger' tag moves to it. The 'champion' tag --
    the version in use -- is deliberately NOT touched here: training a model
    and putting it into use are separate decisions.
    """
    version = mlflow.register_model(model_uri, registered_name)
    client.update_model_version(registered_name, version.version, description=description)
    client.set_registered_model_alias(registered_name, CHALLENGER, version.version)
    return version
