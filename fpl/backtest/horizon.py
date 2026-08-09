"""Scoring a prediction over the horizon you actually commit for.

Every comparison so far has scored the predicted top fifteen against *one*
gameweek. That is not the decision being made. You buy a squad and keep it for
five to seven weeks, paying four points to change your mind, so what matters is
what those players score over the whole run.

The distinction is not cosmetic. A single gameweek is dominated by hauls — one
player returning fifteen decides the week — and hauls are close to unpredictable.
Averaged over six weeks, hauls partly cancel and the underlying rate shows
through. If a model has genuine ranking skill and it has been invisible on the
one-week metric, this is where it should appear.

That is the hypothesis this module exists to test, and it is a real hypothesis:
the component model ranks better than the season mean in 129 of 131 gameweeks
and has converted that into nothing on three separate selection measures. Either
the horizon reveals it or the ranking genuinely does not translate.

**The point-in-time rule is unchanged and slightly stricter here.** A prediction
made before gameweek G is scored against gameweeks G to G+H-1, so the harness
must not let a model see any of them. The window is the *future*, entirely.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from fpl.backtest.harness import DEFAULT_FIRST_GAMEWEEK, known_fixtures, prepare_season
from fpl.models.base import Predictor

DEFAULT_HORIZON = 6


@dataclass
class HorizonResult:
    """Predictions joined to what the players went on to score."""

    model: str
    horizon: int
    predictions: pd.DataFrame

    @property
    def gameweeks(self) -> list[int]:
        if self.predictions.empty:
            return []
        return sorted(self.predictions["gameweek"].unique().tolist())


def horizon_actuals(season: pd.DataFrame, start: int, horizon: int) -> pd.DataFrame:
    """Points each player scored across ``horizon`` gameweeks from ``start``.

    Players who appear in none of the window are absent rather than zero: a
    prediction for someone who never played again should be dropped, not
    scored as a failure, since the model was answering a different question.
    """
    window = season[season["gameweek"].between(start, start + horizon - 1)]
    if window.empty:
        return pd.DataFrame(columns=["element", "total_points", "gameweeks_played"])

    return (
        window.groupby("element", as_index=False)
        .agg(
            total_points=("total_points", "sum"),
            gameweeks_played=("gameweek", "nunique"),
        )
        .reset_index(drop=True)
    )


def replay_horizon(
    season: pd.DataFrame,
    predictor: Predictor,
    horizon: int = DEFAULT_HORIZON,
    first_gameweek: int = DEFAULT_FIRST_GAMEWEEK,
    last_gameweek: int | None = None,
) -> HorizonResult:
    """Replay a season, scoring each prediction over the following ``horizon``.

    Only windows that fit entirely inside the season are scored. A six-week
    horizon starting at gameweek 36 would be judged on three weeks of football
    and look artificially volatile, so those starts are skipped.
    """
    season = prepare_season(season)
    if season.empty:
        return HorizonResult(predictor.name, horizon, pd.DataFrame())

    available = sorted(season["gameweek"].unique())
    last = last_gameweek if last_gameweek is not None else max(available)
    # The window must close before the season does.
    targets = [gw for gw in available if first_gameweek <= gw <= last - horizon + 1]

    frames = []
    for gameweek in targets:
        history = season[season["gameweek"] < gameweek]
        if history.empty:
            continue

        predictions = predictor.predict(history, gameweek, known_fixtures(season, gameweek))
        if predictions.empty:
            continue

        actual = horizon_actuals(season, gameweek, horizon)
        joined = predictions.merge(actual, on="element", how="inner")
        joined["gameweek"] = gameweek
        frames.append(joined)

    if not frames:
        return HorizonResult(predictor.name, horizon, pd.DataFrame())

    return HorizonResult(
        model=predictor.name,
        horizon=horizon,
        predictions=pd.concat(frames, ignore_index=True),
    )


def squad_turnover(result: HorizonResult, n: int = 15) -> float:
    """How much the predicted top ``n`` changes between consecutive starts.

    A model whose top fifteen is different every week is unusable regardless
    of accuracy: acting on it costs four points a transfer. Reported as the
    share of the squad replaced, so 0.2 means three of fifteen changed.

    This is the cost side of the ledger the one-gameweek metric ignores
    entirely.
    """
    if result.predictions.empty:
        return float("nan")

    picks = {
        gameweek: set(group.nlargest(n, "expected_points")["element"])
        for gameweek, group in result.predictions.groupby("gameweek")
    }
    gameweeks = sorted(picks)
    if len(gameweeks) < 2:
        return float("nan")

    # Deliberately not strict: consecutive pairs are one shorter than the list.
    changes = [
        len(picks[later] - picks[earlier]) / n
        for earlier, later in zip(gameweeks, gameweeks[1:], strict=False)
    ]
    return float(sum(changes) / len(changes))


def evaluate_horizon(result: HorizonResult, n: int = 15) -> dict[str, float]:
    """Metrics for a horizon replay.

    ``top_n_mean_actual`` is per gameweek of the window, so it is directly
    comparable with the one-week figure in ``docs/model-results.md`` rather
    than being ``horizon`` times larger.
    """
    from fpl.backtest.metrics import evaluate_by_gameweek

    if result.predictions.empty:
        return {}

    per_start = evaluate_by_gameweek(result.predictions, n)
    if per_start.empty:
        return {}

    summary = per_start.mean(numeric_only=True).to_dict()
    # Convert "points over the whole window" back to a per-gameweek rate.
    for key in list(summary):
        if key.endswith("mean_actual"):
            summary[key] = summary[key] / result.horizon

    summary["horizon"] = float(result.horizon)
    summary["starts"] = float(len(per_start))
    summary["squad_turnover"] = squad_turnover(result, n)
    return summary


def compare_horizons(
    season: pd.DataFrame,
    predictors: list[Predictor],
    horizons: tuple[int, ...] = (1, 3, 6),
    first_gameweek: int = DEFAULT_FIRST_GAMEWEEK,
    n: int = 15,
) -> pd.DataFrame:
    """Every predictor at every horizon, one row each.

    Including horizon 1 on purpose: it reproduces the existing one-week metric,
    so any difference at longer horizons can be attributed to the horizon
    rather than to a change in how the metric is computed.
    """
    rows = []
    for predictor in predictors:
        for horizon in horizons:
            result = replay_horizon(
                season, predictor, horizon=horizon, first_gameweek=first_gameweek
            )
            summary = evaluate_horizon(result, n)
            if not summary:
                continue
            summary["model"] = predictor.name
            rows.append(summary)

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).set_index(["model", "horizon"]).sort_index()
