"""Models that separate "will they play" from "how good are they".

The baselines all predict points directly from past points, which conflates
two very different things. A player scoring 2 points a week because they play
20 minutes and a player scoring 2 points a week because they play 90 minutes
badly look identical to ``NaiveForm`` -- but the first is one injury to a
teammate away from being excellent, and the second is not.

Splitting them also exploits what the baseline comparison showed: season-long
rates are more stable than recent points. So take the *rate* from the whole
season, where there is enough data for it to mean something, and take
*minutes* from recent weeks, where the rotation news actually lives.

The FPL-specific reason this matters more than it would elsewhere: a player
who does not play scores almost exactly zero. Minutes are not a modifier on
the prediction, they very nearly are the prediction.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from fpl.models.base import empty_predictions

MINUTES_PER_MATCH = 90
DEFAULT_MINUTES_WINDOW = 4

# Below this many minutes, a per-90 rate is one lucky cameo away from nonsense,
# so fall back to the player's raw average instead of extrapolating.
MIN_MINUTES_FOR_RATE = 180


@dataclass
class MinutesAdjustedPredictor:
    """Expected minutes, times points per 90.

    Expected minutes come from the last ``minutes_window`` gameweeks, because
    that is where rotation and injury information shows up. The scoring rate
    comes from the whole season so far, because it needs the sample size.
    """

    minutes_window: int = DEFAULT_MINUTES_WINDOW

    @property
    def name(self) -> str:
        return f"MinutesAdjusted({self.minutes_window})"

    def predict(
        self,
        history: pd.DataFrame,
        gameweek: int,
        fixtures: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        if history.empty or "minutes" not in history.columns:
            return empty_predictions()

        season = history.groupby("element", as_index=False).agg(
            total_points=("total_points", "sum"),
            season_minutes=("minutes", "sum"),
            appearances=("minutes", "size"),
        )

        recent = (
            history.sort_values("gameweek")
            .groupby("element", as_index=False)
            .tail(self.minutes_window)
            .groupby("element", as_index=False)
            .agg(expected_minutes=("minutes", "mean"))
        )

        combined = season.merge(recent, on="element", how="left")

        points_per_90 = combined["total_points"] * MINUTES_PER_MATCH / combined["season_minutes"]
        # A player with barely any minutes has no meaningful rate; use their
        # plain average rather than extrapolating from a handful of minutes.
        fallback = combined["total_points"] / combined["appearances"]
        thin = combined["season_minutes"] < MIN_MINUTES_FOR_RATE

        expected = (combined["expected_minutes"].fillna(0) / MINUTES_PER_MATCH) * points_per_90
        combined["expected_points"] = expected.where(~thin, fallback).fillna(0.0)

        return combined[["element", "expected_points"]]


@dataclass
class OpponentAdjustedPredictor:
    """Minutes-adjusted points, scaled by how generous the opponent has been.

    Opponent strength is derived from the history the harness supplies, never
    from a fixed table: how many points have players actually scored against
    this team so far, relative to the league average. That keeps it strictly
    point-in-time -- early in a season it barely moves the numbers, which is
    correct, because early in a season nobody knows who is good yet.
    """

    minutes_window: int = DEFAULT_MINUTES_WINDOW
    # How far the adjustment is allowed to swing a prediction. Opponent
    # generosity is noisy, so an unclamped ratio produces silly multipliers
    # off a handful of matches.
    max_adjustment: float = 0.35

    @property
    def name(self) -> str:
        return f"OpponentAdjusted({self.minutes_window})"

    def _generosity(self, history: pd.DataFrame) -> pd.Series:
        """Mean points conceded to opposing players, per team, indexed by team."""
        league_mean = history["total_points"].mean()
        if not league_mean:
            return pd.Series(dtype="float64")

        by_opponent = history.groupby("opponent_team")["total_points"].mean()
        ratio = by_opponent / league_mean
        return ratio.clip(1 - self.max_adjustment, 1 + self.max_adjustment)

    def predict(
        self,
        history: pd.DataFrame,
        gameweek: int,
        fixtures: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        base = MinutesAdjustedPredictor(self.minutes_window).predict(history, gameweek, fixtures)
        if base.empty:
            return base

        if (
            fixtures is None
            or fixtures.empty
            or "opponent_team" not in fixtures.columns
            or "opponent_team" not in history.columns
        ):
            # Nothing to adjust with. Degrade to the unadjusted model rather
            # than guessing.
            return base

        generosity = self._generosity(history)
        if generosity.empty:
            return base

        adjusted = base.merge(
            fixtures[["element", "opponent_team"]].drop_duplicates("element"),
            on="element",
            how="left",
        )
        multiplier = adjusted["opponent_team"].map(generosity).fillna(1.0)
        adjusted["expected_points"] = adjusted["expected_points"] * multiplier

        return adjusted[["element", "expected_points"]]
