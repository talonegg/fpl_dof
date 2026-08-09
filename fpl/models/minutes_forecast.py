"""Forecasting minutes, as a question in its own right.

Not playing is the largest single cause of a zero score, and minutes are the
one input where the data is complete. Every other component of an FPL
prediction runs into something unpublished — BPS into Opta events, penalties
into who took them — but minutes are recorded exactly, every week, for every
player. There is no ceiling here imposed by missing data, only by modelling.

So minutes deserve to be predicted and scored as their own quantity rather
than being a term buried inside a points model. A points predictor that is
wrong can be wrong for many reasons; a minutes forecast that is wrong is
wrong about one thing, and can be fixed.

Two quantities matter and they are not the same:

**Expected minutes** drives the per-90 rates — a player expected to last 30
minutes earns a third of what their rate implies.

**Probability of a full appearance** drives the discrete payouts. Appearance
points step at 60 minutes, and clean sheet points pay nothing below it, so a
player who plays 59 minutes every week scores very differently from one who
plays 61.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import pandas as pd

MINUTES_PER_MATCH = 90
FULL_APPEARANCE_MINUTES = 60
DEFAULT_WINDOW = 5

FORECAST_COLUMNS = ["element", "expected_minutes", "start_probability"]


@runtime_checkable
class MinutesForecaster(Protocol):
    """Predicts how long each player will be on the pitch."""

    name: str

    def forecast(self, history: pd.DataFrame, gameweek: int) -> pd.DataFrame:
        """Return ``element``, ``expected_minutes`` and ``start_probability``.

        ``history`` holds only gameweeks earlier than ``gameweek``, exactly as
        for a points predictor.
        """
        ...


def empty_forecast() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "element": pd.Series(dtype="int64"),
            "expected_minutes": pd.Series(dtype="float64"),
            "start_probability": pd.Series(dtype="float64"),
        }
    )


@dataclass
class RecentMinutes:
    """The mean of the last ``window`` gameweeks. The obvious baseline.

    Averaging across a rotation pattern gives a number the player never
    actually plays — someone alternating 90 and 0 is forecast at 45, which is
    wrong every week while being right on average. That is the weakness the
    other forecasters exist to address.
    """

    window: int = DEFAULT_WINDOW

    @property
    def name(self) -> str:
        return f"RecentMinutes({self.window})"

    def forecast(self, history: pd.DataFrame, gameweek: int) -> pd.DataFrame:
        if history.empty:
            return empty_forecast()

        recent = (
            history.sort_values("gameweek").groupby("element", as_index=False).tail(self.window)
        )
        forecast = recent.groupby("element", as_index=False).agg(
            expected_minutes=("minutes", "mean"),
            start_probability=(
                "minutes",
                lambda values: float((values >= FULL_APPEARANCE_MINUTES).mean()),
            ),
        )
        return forecast[FORECAST_COLUMNS]


@dataclass
class StartWeightedMinutes:
    """Separate whether they start from how long they last once started.

    A starter who is usually substituted on the hour and a substitute who
    usually gets half an hour can average the same minutes and are not the
    same player. Modelling the two states separately keeps them apart:

        expected = P(start) × minutes when starting
                 + P(cameo) × minutes when appearing off the bench

    ``recency`` weights later gameweeks more heavily, because a change in role
    is the thing being detected and the most recent evidence is the strongest.
    """

    window: int = DEFAULT_WINDOW
    recency: float = 1.5

    @property
    def name(self) -> str:
        return f"StartWeighted({self.window})"

    def forecast(self, history: pd.DataFrame, gameweek: int) -> pd.DataFrame:
        if history.empty:
            return empty_forecast()

        recent = (
            history.sort_values("gameweek").groupby("element", as_index=False).tail(self.window)
        ).copy()

        # Geometric recency weights: the most recent gameweek counts most.
        order = recent.groupby("element").cumcount(ascending=False)
        recent["weight"] = self.recency ** (-order.astype(float))
        recent["started"] = recent["minutes"] >= FULL_APPEARANCE_MINUTES
        recent["played"] = recent["minutes"] > 0

        rows = []
        for element, group in recent.groupby("element"):
            weight = group["weight"]
            total_weight = weight.sum()
            if total_weight <= 0:
                continue

            start_probability = float((weight * group["started"]).sum() / total_weight)
            cameo_probability = float(
                (weight * (group["played"] & ~group["started"])).sum() / total_weight
            )

            starts = group[group["started"]]
            cameos = group[group["played"] & ~group["started"]]
            minutes_when_starting = (
                float((starts["weight"] * starts["minutes"]).sum() / starts["weight"].sum())
                if not starts.empty
                else float(MINUTES_PER_MATCH)
            )
            minutes_when_cameo = (
                float((cameos["weight"] * cameos["minutes"]).sum() / cameos["weight"].sum())
                if not cameos.empty
                else 0.0
            )

            rows.append(
                {
                    "element": element,
                    "expected_minutes": start_probability * minutes_when_starting
                    + cameo_probability * minutes_when_cameo,
                    "start_probability": start_probability,
                }
            )

        if not rows:
            return empty_forecast()
        return pd.DataFrame(rows)[FORECAST_COLUMNS]


@dataclass
class SeasonMinutes:
    """A player's average across the whole season so far.

    The floor, and the same idea that beat every form window at predicting
    points: stability over recency. Worth including because if it wins here
    too, that is a fact about the problem rather than about one metric.
    """

    @property
    def name(self) -> str:
        return "SeasonMinutes"

    def forecast(self, history: pd.DataFrame, gameweek: int) -> pd.DataFrame:
        if history.empty:
            return empty_forecast()

        forecast = history.groupby("element", as_index=False).agg(
            expected_minutes=("minutes", "mean"),
            start_probability=(
                "minutes",
                lambda values: float((values >= FULL_APPEARANCE_MINUTES).mean()),
            ),
        )
        return forecast[FORECAST_COLUMNS]


def forecasters() -> list[MinutesForecaster]:
    """The standard comparison set."""
    return [
        SeasonMinutes(),
        RecentMinutes(window=3),
        RecentMinutes(window=5),
        StartWeightedMinutes(window=5),
        StartWeightedMinutes(window=8),
    ]


@dataclass
class PreseasonMinutes:
    """Minutes forecast for a season that has not started.

    The other forecasters read recent gameweeks. Before gameweek 1 there are
    none, so this works from career aggregates instead: how much of the
    available time a player has historically been on the pitch for.

    The measured stakes are unusually clear. A squad picked from per-90 rates
    with a *constant* minutes assumption played 1,121 minutes across an opening
    run where the naive heuristic's squad played 10,234, and scored below a
    randomly chosen legal squad. Minutes are not a refinement here; they are
    most of the answer.

    Two separate quantities, as elsewhere: how often a player features at all,
    and how long they last when they do.
    """

    gameweeks_per_season: int = 38
    # A player at a new club, or returning from a long absence, regresses
    # towards the population. Full weight on a thin history is how the naive
    # version bought players who never play.
    reliable_appearances: int = 30

    @property
    def name(self) -> str:
        return "PreseasonMinutes"

    def forecast(self, history: pd.DataFrame, gameweek: int = 1) -> pd.DataFrame:
        """Forecast from career totals rather than recent gameweeks.

        ``history`` is expected to carry ``career_minutes``,
        ``career_appearances`` and ``seasons_seen`` — the output of
        :func:`fpl.features.career.blend_career_rates` — rather than raw
        per-gameweek rows.
        """
        if history.empty:
            return empty_forecast()

        required = {"career_minutes", "career_appearances", "seasons_seen"}
        if not required <= set(history.columns):
            return empty_forecast()

        seasons = history["seasons_seen"].clip(lower=1)
        available = seasons * self.gameweeks_per_season

        # How often they featured at all.
        appearance_rate = (history["career_appearances"] / available).clip(0, 1)
        # How long they lasted when they did.
        minutes_when_playing = (
            history["career_minutes"] / history["career_appearances"].replace(0, pd.NA)
        ).clip(0, MINUTES_PER_MATCH)

        expected = (appearance_rate * minutes_when_playing.fillna(0)).clip(0, MINUTES_PER_MATCH)

        # A player who usually lasts an hour is a starter; one averaging twenty
        # minutes off the bench is not, even at the same appearance rate.
        start_rate = (appearance_rate * (minutes_when_playing.fillna(0) / MINUTES_PER_MATCH)).clip(
            0, 1
        )

        # Regress thin histories towards the population, in proportion to doubt.
        weight = (history["career_appearances"] / self.reliable_appearances).clip(0, 1)
        population_minutes = float(expected.mean()) if len(expected) else 0.0
        population_start = float(start_rate.mean()) if len(start_rate) else 0.0

        return pd.DataFrame(
            {
                "element": history.get("element", pd.Series(range(len(history)))),
                "match_key": history.get("match_key", pd.Series([None] * len(history))),
                "expected_minutes": weight * expected + (1 - weight) * population_minutes,
                "start_probability": weight * start_rate + (1 - weight) * population_start,
            }
        )
