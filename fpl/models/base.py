"""The contract every expected-points model implements.

A predictor answers exactly one question: *how many points will each player
score in gameweek N*. It does not know about budgets, squad rules, or which
players you already own -- that is the optimiser's job in Phase 4. Keeping the
two apart is what makes either of them testable.

The crucial term in the contract is ``history``: it contains **only** rows from
gameweeks strictly before the one being predicted. The harness guarantees that,
and a predictor must not go looking for data anywhere else. Everything a model
knows has to be something it could have known at the deadline.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import pandas as pd

PREDICTION_COLUMNS = ["element", "expected_points"]


@runtime_checkable
class Predictor(Protocol):
    """Predicts points for a single upcoming gameweek."""

    name: str

    def predict(self, history: pd.DataFrame, gameweek: int) -> pd.DataFrame:
        """Return ``element`` and ``expected_points``, one row per player.

        ``history`` holds completed gameweeks only, strictly earlier than
        ``gameweek``. Players absent from the returned frame are treated as
        having no prediction rather than a prediction of zero.
        """
        ...


def empty_predictions() -> pd.DataFrame:
    """A well-formed, empty prediction frame.

    Returned when there is nothing to go on -- the first gameweek of a season,
    say. An empty frame with the right columns lets callers concatenate and
    merge without special-casing.
    """
    return pd.DataFrame(
        {"element": pd.Series(dtype="int64"), "expected_points": pd.Series(dtype="float64")}
    )
