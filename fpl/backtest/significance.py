"""Is the difference between two models real, or is it noise?

A season is 33 scored gameweeks. Gameweek-to-gameweek variance in FPL is
enormous -- one captain haul swings a week -- so a model can beat another by a
few percent over a whole season purely by luck. Without a check like this,
every new model looks like an improvement, and the results table becomes a
record of which model got the luckier season.

The comparison is *paired*: both models are scored on the same gameweeks, and
the difference is taken per gameweek. Pairing removes the shared variance --
a week where everybody scored well helps both models equally -- which is the
only reason a signal is detectable at n=33 at all.

The statistic is deliberately crude (mean difference over its standard error,
a paired t-statistic in all but name). Precision here would be false comfort:
gameweek results are not independent or normal, and the honest use of this
number is as a smell test, not a p-value.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from fpl.backtest.harness import DEFAULT_FIRST_GAMEWEEK, replay
from fpl.backtest.metrics import DEFAULT_TOP_N, evaluate_by_gameweek
from fpl.models.base import Predictor

DEFAULT_METRIC = f"top_{DEFAULT_TOP_N}_mean_actual"

# Most metrics here are "more is better". Error metrics are not, and a verdict
# that assumes otherwise reports a genuinely more accurate model as worse.
LOWER_IS_BETTER = frozenset({"mae", "rmse"})

# |t| below this is noise; above it is worth a second look. Two is the
# conventional rough threshold and is about right for a smell test.
SIGNIFICANCE_THRESHOLD = 2.0


@dataclass
class Comparison:
    """One model measured against a benchmark, gameweek by gameweek."""

    model: str
    benchmark: str
    metric: str
    gameweeks: int
    wins: int
    mean_difference: float
    t_statistic: float

    @property
    def is_distinguishable(self) -> bool:
        """Whether the difference stands out from gameweek noise at all."""
        return abs(self.t_statistic) >= SIGNIFICANCE_THRESHOLD

    @property
    def is_improvement(self) -> bool:
        """Whether the difference points the right way for this metric.

        Error metrics improve downwards, so a raw ``mean_difference > 0`` test
        would report a more accurate model as worse.
        """
        if self.metric in LOWER_IS_BETTER:
            return self.mean_difference < 0
        return self.mean_difference > 0

    @property
    def verdict(self) -> str:
        if not self.is_distinguishable:
            return "indistinguishable from the benchmark"
        return "better than the benchmark" if self.is_improvement else "worse than the benchmark"


def compare_to_benchmark(
    season: pd.DataFrame,
    predictor: Predictor,
    benchmark: Predictor,
    metric: str = DEFAULT_METRIC,
    first_gameweek: int = DEFAULT_FIRST_GAMEWEEK,
) -> Comparison:
    """Score ``predictor`` against ``benchmark`` on the same gameweeks."""
    model_scores = _per_gameweek(season, predictor, metric, first_gameweek)
    benchmark_scores = _per_gameweek(season, benchmark, metric, first_gameweek)

    paired = pd.concat(
        [model_scores.rename("model"), benchmark_scores.rename("benchmark")], axis=1
    ).dropna()

    if paired.empty:
        return Comparison(predictor.name, benchmark.name, metric, 0, 0, float("nan"), float("nan"))

    difference = paired["model"] - paired["benchmark"]
    standard_error = difference.std() / (len(difference) ** 0.5)

    if standard_error:
        t_statistic = difference.mean() / standard_error
    elif difference.mean() == 0:
        # Identical every week: no difference at all.
        t_statistic = 0.0
    else:
        # The same non-zero gap every single week. Zero variance means this is
        # as conclusive as evidence gets, not unmeasurable -- returning NaN
        # here would file the most consistent result possible under "unknown".
        t_statistic = float("inf") * (1 if difference.mean() > 0 else -1)

    return Comparison(
        model=predictor.name,
        benchmark=benchmark.name,
        metric=metric,
        gameweeks=len(difference),
        wins=int((difference > 0).sum()),
        mean_difference=float(difference.mean()),
        t_statistic=float(t_statistic),
    )


def _per_gameweek(
    season: pd.DataFrame, predictor: Predictor, metric: str, first_gameweek: int
) -> pd.Series:
    result = replay(season, predictor, first_gameweek=first_gameweek)
    per_gameweek = evaluate_by_gameweek(result.predictions)
    if per_gameweek.empty or metric not in per_gameweek.columns:
        return pd.Series(dtype="float64")
    return per_gameweek[metric]


def compare_across_seasons(
    per_gameweek: pd.DataFrame,
    model: str,
    benchmark: str,
    metric: str = DEFAULT_METRIC,
) -> Comparison:
    """Compare two models over pooled per-(season, gameweek) metrics.

    Takes the output of :func:`fpl.backtest.seasons.compare_many` rather than
    replaying, so several seasons cost one pass. Pairing is on
    ``(season, gameweek)``: comparing a 2022-23 gameweek against a 2025-26 one
    would reintroduce exactly the shared variance that pairing removes.
    """
    if per_gameweek.empty or metric not in per_gameweek.columns:
        return Comparison(model, benchmark, metric, 0, 0, float("nan"), float("nan"))

    keys = ["season", "gameweek"]
    left = per_gameweek[per_gameweek["model"] == model].set_index(keys)[metric]
    right = per_gameweek[per_gameweek["model"] == benchmark].set_index(keys)[metric]

    paired = pd.concat(
        [left.rename("model"), right.rename("benchmark")], axis=1, join="inner"
    ).dropna()
    if paired.empty:
        return Comparison(model, benchmark, metric, 0, 0, float("nan"), float("nan"))

    difference = paired["model"] - paired["benchmark"]
    standard_error = difference.std() / (len(difference) ** 0.5)

    if standard_error:
        t_statistic = difference.mean() / standard_error
    elif difference.mean() == 0:
        t_statistic = 0.0
    else:
        t_statistic = float("inf") * (1 if difference.mean() > 0 else -1)

    return Comparison(
        model=model,
        benchmark=benchmark,
        metric=metric,
        gameweeks=len(difference),
        wins=int((difference > 0).sum()),
        mean_difference=float(difference.mean()),
        t_statistic=float(t_statistic),
    )


def season_comparison_table(
    per_gameweek: pd.DataFrame, benchmark: str, metric: str = DEFAULT_METRIC
) -> pd.DataFrame:
    """Every model against the benchmark, pooled across seasons."""
    if per_gameweek.empty:
        return pd.DataFrame()

    rows = []
    for model in per_gameweek["model"].unique():
        if model == benchmark:
            continue
        comparison = compare_across_seasons(per_gameweek, model, benchmark, metric)
        rows.append(
            {
                "model": comparison.model,
                "wins": f"{comparison.wins}/{comparison.gameweeks}",
                "mean_difference": comparison.mean_difference,
                "t_statistic": comparison.t_statistic,
                "verdict": comparison.verdict,
            }
        )

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("mean_difference", ascending=False).set_index("model")


def comparison_table(
    season: pd.DataFrame,
    predictors: list[Predictor],
    benchmark: Predictor,
    metric: str = DEFAULT_METRIC,
    first_gameweek: int = DEFAULT_FIRST_GAMEWEEK,
) -> pd.DataFrame:
    """Every predictor measured against the benchmark, best difference first."""
    rows = []
    for predictor in predictors:
        if predictor.name == benchmark.name:
            continue
        comparison = compare_to_benchmark(season, predictor, benchmark, metric, first_gameweek)
        rows.append(
            {
                "model": comparison.model,
                "wins": f"{comparison.wins}/{comparison.gameweeks}",
                "mean_difference": comparison.mean_difference,
                "t_statistic": comparison.t_statistic,
                "verdict": comparison.verdict,
            }
        )

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("mean_difference", ascending=False).set_index("model")
