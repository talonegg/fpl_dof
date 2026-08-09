"""Evaluating minutes forecasts.

Minutes are scored differently from points, because the useful errors are
different. Points prediction is a ranking problem — you only need to know who
will out-score whom. Minutes prediction is a *calibration* problem: "this
player has a 70% chance of starting" is only useful if, across all such
players, roughly 70% start.

So the metrics here are error and calibration, not rank correlation:

``minutes_mae``      average miss in minutes; interpretable directly
``start_brier``      Brier score of the start probability, lower is better
``start_accuracy``   share of appearances where the more likely outcome happened
``zero_recall``      of players who did not play, how many were forecast under 30

The last one matters more than its weight suggests. A player who does not play
scores almost exactly nothing, so failing to see a zero coming is the single
most expensive error available.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from fpl.backtest.harness import DEFAULT_FIRST_GAMEWEEK, prepare_season
from fpl.models.minutes_forecast import FULL_APPEARANCE_MINUTES, MinutesForecaster

# Below this a player is treated as forecast not to feature.
BENCHED_MINUTES = 30


@dataclass
class MinutesResult:
    """Forecasts joined to the minutes actually played."""

    forecaster: str
    forecasts: pd.DataFrame

    @property
    def gameweeks(self) -> list[int]:
        if self.forecasts.empty:
            return []
        return sorted(self.forecasts["gameweek"].unique().tolist())


def replay_minutes(
    season: pd.DataFrame,
    forecaster: MinutesForecaster,
    first_gameweek: int = DEFAULT_FIRST_GAMEWEEK,
    last_gameweek: int | None = None,
) -> MinutesResult:
    """Replay a season, scoring each minutes forecast against what happened.

    Same point-in-time rule as everywhere else: the forecaster sees only
    gameweeks strictly earlier than the one it is forecasting.
    """
    season = prepare_season(season)
    if season.empty:
        return MinutesResult(forecaster.name, pd.DataFrame())

    available = sorted(season["gameweek"].unique())
    last = last_gameweek if last_gameweek is not None else max(available)
    targets = [gw for gw in available if first_gameweek <= gw <= last]

    frames = []
    for gameweek in targets:
        history = season[season["gameweek"] < gameweek]
        if history.empty:
            continue

        forecast = forecaster.forecast(history, gameweek)
        if forecast.empty:
            continue

        actual = season[season["gameweek"] == gameweek][["element", "minutes"]]
        joined = forecast.merge(actual, on="element", how="inner")
        joined["gameweek"] = gameweek
        frames.append(joined)

    if not frames:
        return MinutesResult(forecaster.name, pd.DataFrame())

    return MinutesResult(forecaster.name, pd.concat(frames, ignore_index=True))


def minutes_mae(joined: pd.DataFrame) -> float:
    """Average miss, in minutes."""
    if joined.empty:
        return float("nan")
    return float((joined["expected_minutes"] - joined["minutes"]).abs().mean())


def start_brier(joined: pd.DataFrame) -> float:
    """Brier score of the start probability. Lower is better; 0.25 is a coin toss.

    Chosen over accuracy because it rewards being *calibrated* rather than
    merely being on the right side of a half. A forecaster that says 0.9 and is
    right beats one that says 0.51 and is right.
    """
    if joined.empty:
        return float("nan")
    started = (joined["minutes"] >= FULL_APPEARANCE_MINUTES).astype(float)
    return float(((joined["start_probability"] - started) ** 2).mean())


def start_accuracy(joined: pd.DataFrame) -> float:
    """Share of appearances where the more likely outcome is what happened."""
    if joined.empty:
        return float("nan")
    started = joined["minutes"] >= FULL_APPEARANCE_MINUTES
    predicted = joined["start_probability"] >= 0.5
    return float((started == predicted).mean())


def zero_recall(joined: pd.DataFrame) -> float:
    """Of players who did not play, how many were forecast under 30 minutes.

    The most expensive error in FPL is fielding someone who does not play, so
    this is the metric to read when two forecasters look similar elsewhere.
    """
    if joined.empty:
        return float("nan")
    absent = joined[joined["minutes"] == 0]
    if absent.empty:
        return float("nan")
    return float((absent["expected_minutes"] < BENCHED_MINUTES).mean())


def evaluate_minutes(joined: pd.DataFrame) -> dict[str, float]:
    """Every minutes metric for one set of forecasts."""
    return {
        "rows": float(len(joined)),
        "minutes_mae": minutes_mae(joined),
        "start_brier": start_brier(joined),
        "start_accuracy": start_accuracy(joined),
        "zero_recall": zero_recall(joined),
    }


def compare_forecasters(
    season: pd.DataFrame,
    candidates: list[MinutesForecaster],
    first_gameweek: int = DEFAULT_FIRST_GAMEWEEK,
) -> pd.DataFrame:
    """Every forecaster on the same season, best Brier score first."""
    rows = []
    for forecaster in candidates:
        result = replay_minutes(season, forecaster, first_gameweek=first_gameweek)
        if result.forecasts.empty:
            continue
        summary = evaluate_minutes(result.forecasts)
        summary["forecaster"] = result.forecaster
        rows.append(summary)

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).set_index("forecaster").sort_values("start_brier")
