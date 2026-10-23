"""The shapes of the service's requests and responses.

FastAPI uses these to check every request automatically -- right field names,
numbers where numbers belong -- and to write the interactive documentation
page (/docs). A request that does not fit is turned away with a message saying
exactly which field is wrong, before any of our code runs.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, create_model

from src.data.load_cmapss import ID_COLUMNS, build_column_names, load_config

# The measurement columns of one cycle (op_setting_1..3, sensor_1..21), taken
# from the SAME list the data loader uses -- never typed out a second time, so
# the service and the loader cannot disagree about column names.
_dataset = load_config()["dataset"]
MEASUREMENTS = [
    column
    for column in build_column_names(_dataset["n_operational_settings"], _dataset["n_sensors"])
    if column not in ID_COLUMNS
]

# One cycle of one engine, exactly as recorded. Built from the list above.
# extra="forbid": an unknown field (a typo like "sensor_22") is an error, not
# something silently ignored.
Reading = create_model(
    "Reading",
    __config__=ConfigDict(extra="forbid"),
    time_in_cycles=(int, Field(ge=1, description="Cycle number, counting from 1.")),
    **{name: (float, ...) for name in MEASUREMENTS},
)


class PredictionRequest(BaseModel):
    """One engine's recent cycles, oldest first."""

    model_config = ConfigDict(extra="forbid")

    engine_id: str | None = Field(
        None, description="Your own label for the engine. Returned unchanged; never used to predict."
    )
    readings: list[Reading] = Field(  # type: ignore[valid-type]
        min_length=1, description="The engine's cycles, oldest first, as recorded."
    )


class PredictionResponse(BaseModel):
    engine_id: str | None
    remaining_useful_life: float = Field(description="Predicted cycles left, at `at_cycle`.")
    at_cycle: int = Field(description="The latest cycle in the request: 'the engine now'.")
    model_name: str
    model_version: str
    warnings: list[str] = Field(
        default_factory=list,
        description="Readings outside the range seen in training. Empty when all is normal; "
                    "otherwise treat the prediction with caution.",
    )


class ModelInfo(BaseModel):
    """Which model is answering, and how it prepares its inputs."""

    model_name: str
    model_version: str
    run_id: str
    rolling_window: int
    sensors: list[str]
    n_features: int
