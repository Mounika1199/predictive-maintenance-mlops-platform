"""How we measure a model's mistakes.

Every metric here starts from the same quantity, the ERROR of one prediction:

    error = predicted RUL - true RUL

* error > 0  -> LATE:  we said the engine had MORE life left than it really did.
* error < 0  -> EARLY: we said it had LESS life left than it really did.

Late is the dangerous direction. An early prediction means servicing an engine
sooner than necessary -- it costs money. A late one means trusting an engine
that is closer to failure than we think.
"""

from __future__ import annotations

import numpy as np


def _errors(y_true, y_pred) -> np.ndarray:
    """predicted - true, as a plain array, after basic sanity checks."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"y_true and y_pred differ in shape: {y_true.shape} vs {y_pred.shape}"
        )
    if y_true.size == 0:
        raise ValueError("cannot score zero predictions")

    return y_pred - y_true


def mae(y_true, y_pred) -> float:
    """Mean Absolute Error: the average size of a miss, in cycles.

    The easiest metric to explain: "on average, we are off by N cycles".
    """
    return float(np.mean(np.abs(_errors(y_true, y_pred))))


def rmse(y_true, y_pred) -> float:
    """Root Mean Squared Error: like MAE, but big misses count extra.

    Square each miss, average them, take the square root. Still in cycles, but
    squaring makes large errors weigh more:

        misses 10, 10, 10, 10  ->  MAE 10, RMSE 10
        misses  0,  0,  0, 40  ->  MAE 10, RMSE 20

    Same average miss, but the second model has one bad blunder -- and RMSE
    notices. It is the standard headline metric for C-MAPSS.
    """
    return float(np.sqrt(np.mean(_errors(y_true, y_pred) ** 2)))


def nasa_score(y_true, y_pred) -> float:
    """The scoring function from the original C-MAPSS challenge (lower is better).

    Each prediction is penalised exponentially, and LATE errors more steeply
    than early ones:

        late  by d cycles:  exp(d / 10) - 1
        early by d cycles:  exp(d / 13) - 1

    For example:

        early by 20 ->  3.7        late by 20 ->   6.4
        early by 40 -> 20.7        late by 40 ->  53.6

    The penalties are SUMMED, not averaged, so the score grows with the number
    of predictions. Only compare scores computed on the same set of engines.
    A perfect model scores 0.
    """
    d = _errors(y_true, y_pred)
    penalties = np.where(d >= 0, np.exp(d / 10.0) - 1, np.exp(-d / 13.0) - 1)
    return float(np.sum(penalties))


def score(y_true, y_pred) -> dict[str, float]:
    """All metrics at once, plus how many predictions were late or early."""
    d = _errors(y_true, y_pred)
    return {
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "nasa_score": nasa_score(y_true, y_pred),
        "n": int(d.size),
        "n_late": int(np.sum(d > 0)),
        "n_early": int(np.sum(d < 0)),
    }
