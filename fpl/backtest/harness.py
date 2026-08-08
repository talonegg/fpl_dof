"""Replaying a season, one gameweek at a time.

The entire value of this module is one guarantee: **a predictor is only ever
handed data from gameweeks strictly before the one it is predicting.**

Lookahead is the easiest bug to write in this whole codebase and the hardest to
notice. It does not crash, it does not look wrong, it just makes a model appear
excellent in testing and useless in August. So the slicing happens here, once,
rather than being each model's responsibility -- a predictor is never given the
opportunity to cheat, instead of being trusted not to.

``replay`` returns predictions joined to what actually happened, which is the
input to everything in :mod:`fpl.backtest.metrics`.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from fpl.domain.history import collapse_to_gameweeks
from fpl.models.base import Predictor

# Predicting gameweek 1 from nothing is not a test of anything, and early
# gameweeks are mostly noise, so a backtest starts once there is some history.
DEFAULT_FIRST_GAMEWEEK = 6

REQUIRED_COLUMNS = ("element", "gameweek", "total_points")


@dataclass
class BacktestResult:
    """Predictions joined to actuals, plus what produced them."""

    model: str
    predictions: pd.DataFrame
    first_gameweek: int
    last_gameweek: int

    @property
    def gameweeks(self) -> list[int]:
        if self.predictions.empty:
            return []
        return sorted(self.predictions["gameweek"].unique().tolist())


def prepare_season(season: pd.DataFrame) -> pd.DataFrame:
    """Normalise a season frame into what the harness expects.

    Collapses double gameweeks to one row per player per gameweek. Without
    this a player appears twice in the same gameweek and is scored twice.
    """
    missing = [column for column in REQUIRED_COLUMNS if column not in season.columns]
    if missing:
        raise ValueError(f"season data is missing required columns: {missing}")

    collapsed = collapse_to_gameweeks(season, ["element", "gameweek"])
    return collapsed.sort_values(["gameweek", "element"]).reset_index(drop=True)


def replay(
    season: pd.DataFrame,
    predictor: Predictor,
    first_gameweek: int = DEFAULT_FIRST_GAMEWEEK,
    last_gameweek: int | None = None,
) -> BacktestResult:
    """Run ``predictor`` across a season, gameweek by gameweek.

    For each gameweek the predictor sees only earlier gameweeks, predicts, and
    is scored against what happened. Predictions for players who did not
    feature in that gameweek are dropped -- there is no actual to compare
    against, and keeping them would quietly reward predicting for absentees.
    """
    season = prepare_season(season)
    if season.empty:
        return BacktestResult(predictor.name, pd.DataFrame(), first_gameweek, 0)

    available = sorted(season["gameweek"].unique())
    last = last_gameweek if last_gameweek is not None else max(available)
    targets = [gw for gw in available if first_gameweek <= gw <= last]

    frames = []
    for gameweek in targets:
        # The one line that matters: strictly earlier, never <=.
        history = season[season["gameweek"] < gameweek]
        if history.empty:
            continue

        predictions = predictor.predict(history, gameweek)
        if predictions.empty:
            continue

        actual = season[season["gameweek"] == gameweek][["element", "total_points"]]
        joined = predictions.merge(actual, on="element", how="inner")
        joined["gameweek"] = gameweek
        frames.append(joined)

    if not frames:
        return BacktestResult(predictor.name, pd.DataFrame(), first_gameweek, last)

    return BacktestResult(
        model=predictor.name,
        predictions=pd.concat(frames, ignore_index=True),
        first_gameweek=first_gameweek,
        last_gameweek=last,
    )


def compare(
    season: pd.DataFrame,
    predictors: list[Predictor],
    first_gameweek: int = DEFAULT_FIRST_GAMEWEEK,
    last_gameweek: int | None = None,
) -> pd.DataFrame:
    """Replay several predictors over the same season and tabulate them.

    Sorted by the metric that reflects the real question -- what the predicted
    top 15 actually scored -- rather than by error.
    """
    from fpl.backtest.metrics import DEFAULT_TOP_N, summarise

    rows = []
    for predictor in predictors:
        result = replay(season, predictor, first_gameweek, last_gameweek)
        summary = summarise(result.predictions)
        if not summary:
            continue
        summary["model"] = result.model
        rows.append(summary)

    if not rows:
        return pd.DataFrame()

    table = pd.DataFrame(rows).set_index("model")
    sort_column = f"top_{DEFAULT_TOP_N}_mean_actual"
    if sort_column in table.columns:
        table = table.sort_values(sort_column, ascending=False)
    return table
