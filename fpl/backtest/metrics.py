"""How to tell whether a predictor is any good.

FPL is a ranking-and-selection problem, not a regression problem. You never
need to know that Salah will score 6.3 rather than 5.9; you need to know he
will out-score the alternative. So error metrics are reported, but the ones
that decide anything are the ranking ones.

``mean_actual_of_top_n`` is the metric closest to the real question: if you
picked this model's top 15 every week, what would they actually have scored?
Everything else is diagnostic.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

PREDICTED = "expected_points"
ACTUAL = "total_points"

# A squad is 15 players, so the top 15 is the natural selection horizon.
DEFAULT_TOP_N = 15


def mean_absolute_error(joined: pd.DataFrame) -> float:
    """Average size of the miss, in points."""
    if joined.empty:
        return float("nan")
    return float((joined[PREDICTED] - joined[ACTUAL]).abs().mean())


def root_mean_squared_error(joined: pd.DataFrame) -> float:
    """Like MAE but punishes big misses harder."""
    if joined.empty:
        return float("nan")
    return float(np.sqrt(((joined[PREDICTED] - joined[ACTUAL]) ** 2).mean()))


def rank_correlation(joined: pd.DataFrame) -> float:
    """Spearman correlation between predicted and actual points.

    The headline diagnostic: are we ordering players correctly? Returns NaN
    when there is nothing to correlate, which includes the case where every
    prediction is identical -- a constant has no ranking information, and
    reporting 0 there would imply we had measured something.
    """
    if len(joined) < 2:
        return float("nan")
    if joined[PREDICTED].nunique() < 2 or joined[ACTUAL].nunique() < 2:
        return float("nan")
    # Spearman is Pearson on ranks. Computing it that way avoids depending on
    # scipy, which pandas' method="spearman" would pull in for this one call.
    predicted_ranks = joined[PREDICTED].rank()
    actual_ranks = joined[ACTUAL].rank()
    return float(predicted_ranks.corr(actual_ranks))


def top_n_precision(joined: pd.DataFrame, n: int = DEFAULT_TOP_N) -> float:
    """Fraction of the predicted top ``n`` that were in the actual top ``n``."""
    if joined.empty:
        return float("nan")
    n = min(n, len(joined))
    predicted_top = set(joined.nlargest(n, PREDICTED)["element"])
    actual_top = set(joined.nlargest(n, ACTUAL)["element"])
    return len(predicted_top & actual_top) / n


def mean_actual_of_top_n(joined: pd.DataFrame, n: int = DEFAULT_TOP_N) -> float:
    """Average points actually scored by the predicted top ``n``.

    The metric that matters: it is what you would have got by following the
    model. Compare against ``mean_actual_overall`` -- beating the field average
    is the minimum bar for a model being worth using at all.
    """
    if joined.empty:
        return float("nan")
    n = min(n, len(joined))
    return float(joined.nlargest(n, PREDICTED)[ACTUAL].mean())


def mean_actual_overall(joined: pd.DataFrame) -> float:
    """Average points across every player -- the do-nothing comparison."""
    if joined.empty:
        return float("nan")
    return float(joined[ACTUAL].mean())


def evaluate(joined: pd.DataFrame, n: int = DEFAULT_TOP_N) -> dict[str, float]:
    """Every metric for one set of predictions joined to actuals."""
    return {
        "rows": float(len(joined)),
        "mae": mean_absolute_error(joined),
        "rmse": root_mean_squared_error(joined),
        "rank_correlation": rank_correlation(joined),
        f"top_{n}_precision": top_n_precision(joined, n),
        f"top_{n}_mean_actual": mean_actual_of_top_n(joined, n),
        "field_mean_actual": mean_actual_overall(joined),
    }


def evaluate_by_gameweek(joined: pd.DataFrame, n: int = DEFAULT_TOP_N) -> pd.DataFrame:
    """Metrics computed per gameweek, then returned as one row each.

    Per-gameweek is the honest unit for the selection metrics: you pick a
    squad each week, so precision has to be measured each week. Pooling every
    gameweek first would let a good week paper over a bad one.
    """
    if joined.empty:
        return pd.DataFrame()

    rows = []
    for gameweek, group in joined.groupby("gameweek"):
        metrics = evaluate(group, n)
        metrics["gameweek"] = int(gameweek)
        rows.append(metrics)
    return pd.DataFrame(rows).set_index("gameweek").sort_index()


def summarise(joined: pd.DataFrame, n: int = DEFAULT_TOP_N) -> dict[str, float]:
    """Per-gameweek metrics averaged into one row per model."""
    per_gameweek = evaluate_by_gameweek(joined, n)
    if per_gameweek.empty:
        return {}
    summary = per_gameweek.mean(numeric_only=True).to_dict()
    summary["gameweeks"] = float(len(per_gameweek))
    summary["rows"] = float(len(joined))
    return summary
