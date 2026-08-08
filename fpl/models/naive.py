"""The baseline every real model has to beat.

``NaiveFormPredictor`` predicts that a player will score whatever they have
been averaging lately. That is all. It knows nothing about fixtures, minutes,
opposition or position.

It is here because a model is only worth having if it beats the obvious thing,
and "recent form continues" is the obvious thing -- it is roughly what a human
does when they glance at the form column. A sophisticated model that cannot
beat this one is not sophisticated, it is just expensive.

``SeasonMeanPredictor`` is the even blunter floor beneath it: a player's
average over everything played so far, with no recency weighting at all. If
form does not beat the season mean, recency is not carrying information.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from fpl.models.base import empty_predictions

DEFAULT_WINDOW = 5


@dataclass
class NaiveFormPredictor:
    """Predicts the mean points over the last ``window`` gameweeks played."""

    window: int = DEFAULT_WINDOW

    @property
    def name(self) -> str:
        return f"NaiveForm({self.window})"

    def predict(
        self,
        history: pd.DataFrame,
        gameweek: int,
        fixtures: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        if history.empty:
            return empty_predictions()

        recent = (
            history.sort_values("gameweek").groupby("element", as_index=False).tail(self.window)
        )
        predictions = recent.groupby("element", as_index=False)["total_points"].mean()
        return predictions.rename(columns={"total_points": "expected_points"})


@dataclass
class SeasonMeanPredictor:
    """Predicts a player's mean points across every gameweek so far."""

    @property
    def name(self) -> str:
        return "SeasonMean"

    def predict(
        self,
        history: pd.DataFrame,
        gameweek: int,
        fixtures: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        if history.empty:
            return empty_predictions()

        predictions = history.groupby("element", as_index=False)["total_points"].mean()
        return predictions.rename(columns={"total_points": "expected_points"})


@dataclass
class ZeroPredictor:
    """Predicts zero for everyone.

    Not a serious model -- it exists so the metrics have a known-worst anchor.
    A metric that makes this look respectable is a metric that is not measuring
    what you think.
    """

    @property
    def name(self) -> str:
        return "Zero"

    def predict(
        self,
        history: pd.DataFrame,
        gameweek: int,
        fixtures: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        if history.empty:
            return empty_predictions()

        elements = history["element"].drop_duplicates()
        return pd.DataFrame({"element": elements, "expected_points": 0.0})
