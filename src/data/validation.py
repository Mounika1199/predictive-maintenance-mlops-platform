"""Data validation for the C-MAPSS tables.

WHAT A "CHECK" IS
A yes/no question asked of every value in a column -- exactly like a web form
saying "age must be a number between 0 and 120". If any value fails, we report
which column, which rule, and how many rows.

TWO KINDS OF RULE
1. Column rules (Pandera): right columns, right order, right dtypes, no missing
   values, values inside a plausible range.
2. Structural rules (written by hand below): invariants that span *rows* rather
   than living inside one column -- engine ids contiguous, no repeated
   (engine, cycle) pair, each engine's cycles running 1, 2, 3, ... with no gaps.
   Pandera is built around columns, so these are clearer written out.

WHY BOTHER
The dangerous data problems in ML are the silent ones. A sensor recalibrated
into different units, a reordered column, a batch of NaNs: none of these raise
an exception. They produce a model that is quietly worse, and you find out weeks
later from a business metric. Validation turns silent corruption into an
immediate, loud failure.

THE REFERENCE PROFILE
Bounds are not hard-coded. We measure them once from the training data into
``configs/reference_profile.json`` and validate against that file. This matters
because at serving time the training set is not sitting in memory -- what a
deployed service has is a stored contract. It is also the exact baseline that
drift detection will compare against later.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

# Import the PANDAS namespace explicitly, not bare ``pandera``.
#
# Pandera supports several dataframe libraries (pandas, polars, pyspark) and
# dispatches each check to a per-library implementation. Importing
# ``pandera.pandas`` is what guarantees the pandas implementations are
# registered. A bare ``import pandera`` left that registry empty for us and made
# every single check blow up with
# ``KeyError("<class 'pandas.core.series.Series'>")`` -- an error that looked
# like a data problem but was purely an import problem.
import pandera.pandas as pa

from src.data.load_cmapss import PROJECT_ROOT

# Columns that are counters (discrete, 1-based), not physical measurements.
COUNTER_COLUMNS = ("unit_number", "time_in_cycles")


@dataclass(frozen=True)
class ValidationResult:
    """The outcome of validating one table."""

    name: str
    n_rows: int
    errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


# ---------------------------------------------------------------------------
# Building the reference profile
# ---------------------------------------------------------------------------


def build_reference_profile(frame: pd.DataFrame, dataset_name: str) -> dict:
    """Summarise a (training) table into a JSON-serialisable contract.

    We record only coarse, stable facts: dtype, min, max, and whether the column
    is constant.

    Note what is deliberately absent -- mean and standard deviation. Those
    *will* shift as different engines degrade in different ways, and that shift
    is normal, not an error. Watching distributions move is drift detection: a
    separate job, with a separate response (investigate, maybe retrain) from
    validation's response (stop the pipeline now).
    """
    columns: dict[str, dict] = {}

    for name in frame.columns:
        series = frame[name]
        n_unique = int(series.nunique())
        columns[name] = {
            "dtype": str(series.dtype),
            "min": _to_python_number(series.min()),
            "max": _to_python_number(series.max()),
            "n_unique": n_unique,
            "constant": n_unique == 1,
        }

    return {
        "dataset": dataset_name,
        "n_rows": int(len(frame)),
        "n_engines": int(frame["unit_number"].nunique()),
        "columns": columns,
    }


def _to_python_number(value) -> float | int:
    """Convert a numpy scalar to a plain Python number so json can write it."""
    return value.item() if hasattr(value, "item") else value


def save_reference_profile(profile: dict, path: Path | str) -> Path:
    """Write the profile as indented JSON (indented so Git diffs are readable)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(profile, handle, indent=2)
        handle.write("\n")
    return path


def load_reference_profile(path: Path | str) -> dict:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Reference profile not found: {path}\n"
            "Generate it with: python scripts/build_reference_profile.py"
        )
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


# ---------------------------------------------------------------------------
# Building the Pandera schema from the profile
# ---------------------------------------------------------------------------


def build_schema(
    profile: dict,
    range_tolerance: float,
    constant_tolerance: float,
) -> pa.DataFrameSchema:
    """Turn a reference profile into an executable schema.

    Three schema-level flags, all deliberate:

    * ``strict=True``  -- reject unexpected *extra* columns.
    * ``ordered=True`` -- reject *reordered* columns. The raw files have no
      header row, so a column's identity comes purely from its position. If an
      upstream export ever swaps two sensors, every value still looks perfectly
      plausible and nothing crashes -- the model just silently learns from
      mislabelled inputs. This catches that.
    * ``coerce=False`` (the default) -- report a dtype mismatch instead of
      quietly repairing it. A surprise dtype is information, not a nuisance.
    """
    if range_tolerance < 0:
        raise ValueError(f"range_tolerance must be >= 0, got {range_tolerance}")
    if constant_tolerance < 0:
        raise ValueError(f"constant_tolerance must be >= 0, got {constant_tolerance}")

    columns = {
        name: pa.Column(
            dtype=spec["dtype"],
            checks=_build_checks(name, spec, range_tolerance, constant_tolerance),
            nullable=False,
            required=True,
        )
        for name, spec in profile["columns"].items()
    }

    return pa.DataFrameSchema(columns, strict=True, ordered=True)


def _build_checks(
    name: str,
    spec: dict,
    range_tolerance: float,
    constant_tolerance: float,
) -> list[pa.Check]:
    """Pick the right value check for one column.

    Our columns come in three genuinely different kinds, and each wants a
    different question asked of it:

    =========  =========================  ==================================
    kind       example                    question
    =========  =========================  ==================================
    counter    unit_number                "is it at least 1?"
    constant   sensor_1 (always 518.67)   "is it STILL 518.67?"
    varying    sensor_2 (641.21..644.53)  "is it in the usual range?"
    =========  =========================  ==================================

    That table is this whole function.
    """
    # --- Counters -----------------------------------------------------------
    # Lower bound only. Engine and cycle numbering is 1-based, so anything <= 0
    # is meaningless. We deliberately set NO upper bound: a dataset with 150
    # engines, or an engine that survives 400 cycles, is perfectly legitimate.
    # Capping counters at whatever training happened to contain would reject
    # valid data.
    if name in COUNTER_COLUMNS:
        return [pa.Check.ge(1, name=f"{name}_is_positive")]

    # --- Constant columns ---------------------------------------------------
    # Seven columns never move at all (sensor_1 is 518.67 in every one of the
    # 20,631 training rows). If one ever does move, something upstream changed
    # -- a recalibration, a units switch, a different export -- and that is
    # exactly what we want to hear about.
    #
    # Why a tolerance rather than ``== 518.67``? Computers cannot store most
    # decimals exactly. Our own profile showed sensor_5 with n_unique == 1 but
    # std == 1.78e-15: the standard deviation of identical values is
    # mathematically zero, and that residue is pure floating-point rounding
    # dust. An exact-equality check would raise an alarm about nothing.
    #
    # 1e-6 sits in the wide, safe gap between the dust (~1e-13 and below) and
    # any real sensor change (518.67 -> 518.70 is 0.03).
    if spec["constant"]:
        value = spec["min"]
        return [
            pa.Check.in_range(
                value - constant_tolerance,
                value + constant_tolerance,
                name=f"{name}_is_constant",
            )
        ]

    # --- Varying columns ----------------------------------------------------
    # The training range, widened by a margin.
    #
    # Why widen at all? Training data is only a SAMPLE. The lowest sensor_2 we
    # saw (641.21) is not a law of physics, just the smallest value that turned
    # up in 100 engines. We measured this directly: 7 of 17 varying columns have
    # test values outside the training range. A zero-tolerance rule would reject
    # our own valid held-out data.
    #
    # Why a PERCENTAGE rather than a fixed pad? Column widths differ by roughly
    # 185,000x (op_setting_2 spans 0.0012; sensor_9 spans 222.86). Any fixed pad
    # is absurdly large for one and invisible for the other. A fraction of the
    # column's own span adapts itself:
    #
    #     sensor_2:     span 3.32    -> margin 0.332     -> [640.878, 644.862]
    #     op_setting_2: span 0.0012  -> margin 0.00012   -> [-0.00072, 0.00072]
    low, high = spec["min"], spec["max"]
    margin = (high - low) * range_tolerance
    return [
        pa.Check.in_range(
            low - margin,
            high + margin,
            name=f"{name}_in_expected_range",
        )
    ]


# ---------------------------------------------------------------------------
# Structural rules (row-spanning invariants)
# ---------------------------------------------------------------------------


def check_structure(frame: pd.DataFrame) -> list[str]:
    """Check invariants that span rows rather than living inside one column.

    Everything we build later -- rolling means, lag features, "the most recent
    state of this engine" -- silently produces nonsense if the time axis has
    gaps or repeats. Better to fail here, loudly.

    Returns a list of human-readable problems; empty means all good.
    """
    errors: list[str] = []

    if frame.empty:
        return ["table is empty"]

    # 1. Engine ids should form a contiguous 1..N block. A gap usually means
    #    rows were dropped somewhere upstream.
    ids = frame["unit_number"]
    present = set(ids.unique())
    expected = set(range(1, ids.nunique() + 1))
    if present != expected:
        missing = sorted(expected - present)[:5]
        errors.append(
            f"unit_number is not a contiguous 1..N block "
            f"(n={ids.nunique()}, first missing: {missing})"
        )

    # 2. Exactly one row per (engine, cycle). A duplicate would double-count a
    #    single point in time.
    n_duplicates = int(frame.duplicated(["unit_number", "time_in_cycles"]).sum())
    if n_duplicates:
        errors.append(
            f"{n_duplicates} duplicated (unit_number, time_in_cycles) row(s)"
        )

    # 3. Each engine's cycles must run 1, 2, 3, ... with no gaps.
    #
    #    Because rule 2 already guarantees no duplicates, "consecutive from 1"
    #    is exactly equivalent to "the minimum is 1 and the maximum equals the
    #    number of rows". That turns a slow per-engine loop into one groupby.
    grouped = frame.groupby("unit_number")["time_in_cycles"]
    lengths, minima, maxima = grouped.size(), grouped.min(), grouped.max()

    bad_start = minima[minima != 1]
    if len(bad_start):
        errors.append(
            f"{len(bad_start)} engine(s) do not start at cycle 1 "
            f"(e.g. engine {bad_start.index[0]} starts at {bad_start.iloc[0]})"
        )

    has_gaps = maxima[maxima != lengths]
    if len(has_gaps):
        engine = has_gaps.index[0]
        errors.append(
            f"{len(has_gaps)} engine(s) have gaps in time_in_cycles "
            f"(e.g. engine {engine}: {lengths[engine]} rows but highest cycle "
            f"is {maxima[engine]})"
        )

    return errors


# ---------------------------------------------------------------------------
# Putting it together
# ---------------------------------------------------------------------------


def validate_frame(
    frame: pd.DataFrame,
    schema: pa.DataFrameSchema,
    name: str,
    check_row_structure: bool = True,
) -> ValidationResult:
    """Validate one table and collect *all* problems found.

    ``lazy=True`` tells Pandera to gather every failure instead of stopping at
    the first one. When data is wrong you want the complete list, not to fix one
    thing, rerun, and discover the next.
    """
    errors: list[str] = []

    try:
        schema.validate(frame, lazy=True)
    except pa.errors.SchemaErrors as exc:
        errors.extend(_summarize_schema_errors(exc))

    if check_row_structure:
        errors.extend(check_structure(frame))

    return ValidationResult(name=name, n_rows=len(frame), errors=errors)


def _summarize_schema_errors(exc: pa.errors.SchemaErrors) -> list[str]:
    """Condense Pandera's failure table into one readable line per problem.

    Pandera reports one row per failing *value*, which for a badly broken column
    could be thousands of lines. We group by (column, check) so a human sees
    "sensor_2 out of range for 431 values" rather than 431 separate messages.
    """
    cases = exc.failure_cases

    grouped = (
        cases.groupby(["column", "check"], dropna=False)
        .agg(n=("failure_case", "size"), example=("failure_case", "first"))
        .reset_index()
    )

    messages: list[str] = []
    for row in grouped.itertuples(index=False):
        column = row.column if pd.notna(row.column) else "<table>"
        messages.append(
            f"column '{column}': check '{row.check}' failed for "
            f"{row.n} value(s) (e.g. {row.example})"
        )
    return messages


def load_validation_settings(config: dict) -> tuple[Path, float, float]:
    """Pull validation settings out of the config into plain values."""
    settings = config["validation"]
    return (
        PROJECT_ROOT / settings["reference_profile"],
        float(settings["range_tolerance"]),
        float(settings["constant_tolerance"]),
    )
