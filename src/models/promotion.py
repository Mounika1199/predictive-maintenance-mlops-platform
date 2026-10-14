"""The rules for putting a model into use ("promoting" it to champion).

Pure logic: two candidates in, a decision out. Nothing here talks to MLflow,
which keeps the rules easy to read and easy to test. scripts/promote_model.py
fetches the candidates from the registry and acts on the decision.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# What defines the "exam" a model was scored on. Scores are only comparable if
# these all match:
#   split_seed, validation_fraction, validation_engines -> the SAME engines
#   rul_cap                                             -> the SAME answer key
# (Feature settings such as rolling_window are deliberately NOT here: trying
# different features is exactly what a challenger is allowed to do.)
EXAM_SETTINGS = ("split_seed", "validation_fraction", "validation_engines", "rul_cap")


@dataclass(frozen=True)
class Candidate:
    """What the rules need to know about one registered model version."""

    version: str
    val_rmse: float
    exam: dict[str, str]  # the EXAM_SETTINGS values it was scored with
    uncommitted: bool | None  # trained from uncommitted code? None = unknown


@dataclass(frozen=True)
class Decision:
    promote: bool
    reason: str
    warnings: list[str] = field(default_factory=list)


def decide_promotion(
    challenger: Candidate,
    champion: Candidate | None,
    min_improvement: float,
) -> Decision:
    """Should `challenger` replace `champion`? Checks run in this order:

    1. no champion yet               -> promote (someone has to be first)
    2. it IS the champion            -> nothing to do
    3. scored on a different exam    -> refuse: the scores are not comparable
    4. worse, or better by less than min_improvement
                                     -> refuse: no real reason to switch
    5. otherwise                     -> promote

    Uncommitted code never blocks promotion, but always produces a warning.
    """
    warnings = []
    if challenger.uncommitted is True:
        warnings.append(
            f"version {challenger.version} was trained from code with uncommitted "
            "changes: it cannot be reproduced from its recorded commit"
        )
    elif challenger.uncommitted is None:
        warnings.append(
            f"it is unknown whether version {challenger.version}'s code was committed"
        )

    if champion is None:
        return Decision(True, "there is no champion yet", warnings)

    if challenger.version == champion.version:
        return Decision(False, f"version {challenger.version} is already the champion", warnings)

    differences = [
        f"{key}: {champion.exam.get(key)} vs {challenger.exam.get(key)}"
        for key in EXAM_SETTINGS
        if champion.exam.get(key) != challenger.exam.get(key)
    ]
    if differences:
        return Decision(
            False,
            "the two versions were scored on different exams, so their scores "
            "cannot be compared (" + "; ".join(differences) + ")",
            warnings,
        )

    improvement = champion.val_rmse - challenger.val_rmse
    scores = f"(champion {champion.val_rmse:.2f}, challenger {challenger.val_rmse:.2f})"
    if improvement < 0:
        return Decision(False, f"worse by {-improvement:.2f} RMSE {scores}", warnings)
    if improvement < min_improvement:
        return Decision(
            False,
            f"better by only {improvement:.2f} RMSE, less than the required "
            f"{min_improvement} {scores}; too small a difference to be a reason to switch",
            warnings,
        )

    return Decision(
        True,
        f"better by {improvement:.2f} RMSE "
        f"(champion {champion.val_rmse:.2f}, challenger {challenger.val_rmse:.2f})",
        warnings,
    )
