"""Replaying season-opening squad selection against what actually happened.

The test the whole season-opening design rests on. For a season S, build a
squad using **only seasons before S** and S's own opening prices, then score
what those fifteen actually went on to do.

Possible at all because the archive records a price for every player in
gameweek 1 of every season, which is the real budget they had to be bought
within.

This module now only *replays*. Pool assembly lives in
``features/preseason_pool.py``, the analytical models in
``models/preseason_strategies.py``, and squad construction in
``optimise/preseason.py`` — so a strategy can be added, or the constructor
changed, without touching the harness that scores them.

**Three seasons are testable, so each strategy yields three squads.** That is a
very small sample and no amount of statistics rescues it. This can show that a
model is bad; it cannot show that one is good. Read a positive result as
permission to proceed, not as evidence of skill.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from fpl.domain.rules import season_scores_defensive_contributions
from fpl.features.preseason_pool import OPENING_GAMEWEEK, build_pool, opening_prices
from fpl.models.preseason_strategies import (
    PreseasonContext,
    PreseasonStrategy,
    strategies,
)
from fpl.optimise.preseason import construct_squad
from fpl.optimise.squad import SquadConstraints

# The windows the opening squad is actually judged over. Three is what you are
# certain to hold it for, seven is roughly where a free transfer a week has
# rebuilt it, and five is the middle. Scoring at one horizon hides whether an
# edge is real or just early.
SCORING_HORIZONS = (3, 5, 7)
DEFAULT_HORIZON = 7

__all__ = [
    "DEFAULT_HORIZON",
    "OPENING_GAMEWEEK",
    "SCORING_HORIZONS",
    "PreseasonResult",
    "actual_points",
    "build_pool",
    "compare_horizons",
    "compare_strategies",
    "defensive_forecast_status",
    "hindsight_squad",
    "horizon_table",
    "opening_prices",
    "run_strategy",
]


@dataclass
class PreseasonResult:
    """A squad picked before a season, and what it went on to score."""

    season: str
    strategy: str
    squad: pd.DataFrame
    opening_points: float
    season_points: float
    cost: float
    defensive: str = ""

    def summary(self) -> str:
        return (
            f"{self.strategy} in {self.season}: {self.opening_points:.0f} over the "
            f"opening run, {self.season_points:.0f} across the season, "
            f"£{self.cost:.1f}m spent"
        )


def actual_points(
    season: pd.DataFrame, elements: list[int], horizon: int = DEFAULT_HORIZON
) -> tuple[float, float]:
    """What a set of players scored over the opening run, and over the season.

    Counts every player in the squad rather than a chosen eleven. Picking the
    best eleven each week is a separate skill and folding it in here would
    flatter the squad selection being tested.
    """
    owned = season[season["element"].isin(elements)]
    if owned.empty:
        return 0.0, 0.0

    opening = owned[owned["gameweek"] <= horizon]["total_points"].sum()
    return float(opening), float(owned["total_points"].sum())


def defensive_forecast_status(target: str, pool: pd.DataFrame) -> str:
    """Whether defensive contributions can be forecast for ``target``.

    Three genuinely different states, and collapsing them is how a backtest
    comes to report a number that means nothing:

    ``"not scored"``     the season predates the rule. Scoring none is correct.
    ``"forecast"``       the rule applies and prior seasons carry the data.
    ``"blind"``          the rule applies and prior seasons do **not** carry the
                         data. The squad is being judged against points it had
                         no way to see coming.
    """
    if not season_scores_defensive_contributions(target):
        return "not scored"
    if "defensive_rate" in pool.columns and pool["defensive_rate"].notna().any():
        return "forecast"
    return "blind"


def run_strategy(
    target: str,
    season: pd.DataFrame,
    prior_seasons: dict[str, pd.DataFrame],
    strategy: PreseasonStrategy,
    horizon: int = DEFAULT_HORIZON,
    constraints: SquadConstraints | None = None,
    pool: pd.DataFrame | None = None,
) -> PreseasonResult | None:
    """Build a squad for ``target`` with one strategy, and score it."""
    candidates = build_pool(prior_seasons, opening_prices(season)) if pool is None else pool
    if candidates.empty:
        return None

    context = PreseasonContext(target=target, prior_seasons=prior_seasons, horizon=horizon)
    squad = construct_squad(candidates, strategy.expected_points(candidates, context), constraints)
    if squad is None:
        return None

    players = squad.players
    opening, whole = actual_points(season, players["element"].tolist(), horizon)
    return PreseasonResult(
        season=target,
        strategy=strategy.name,
        squad=players,
        opening_points=opening,
        season_points=whole,
        cost=float(players["price"].sum()),
        defensive=defensive_forecast_status(target, candidates),
    )


def hindsight_squad(
    target: str, season: pd.DataFrame, horizon: int = DEFAULT_HORIZON
) -> PreseasonResult | None:
    """The best squad it was possible to buy, knowing everything. The ceiling.

    Reported because a score without a ceiling is uninterpretable. Nobody can
    reach this; the useful number is the share of it a strategy captures.

    Not a strategy in the registry, deliberately — it reads the target season,
    which every real strategy is forbidden from doing.
    """
    prices = opening_prices(season)
    if prices.empty:
        return None

    scored = (
        season[season["gameweek"] <= horizon]
        .groupby("element", as_index=False)["total_points"]
        .sum()
    )
    pool = prices.merge(scored, on="element", how="left")
    pool["total_points"] = pool["total_points"].fillna(0)

    squad = construct_squad(pool, pool["total_points"])
    if squad is None:
        return None

    players = squad.players
    opening, whole = actual_points(season, players["element"].tolist(), horizon)
    return PreseasonResult(
        target, "Hindsight", players, opening, whole, float(players["price"].sum())
    )


def compare_strategies(
    season_data: dict[str, pd.DataFrame],
    target: str,
    horizon: int = DEFAULT_HORIZON,
    catalogue: list[PreseasonStrategy] | None = None,
) -> pd.DataFrame:
    """Every registered strategy plus the ceiling, for one season and horizon."""
    if target not in season_data:
        return pd.DataFrame()

    season = season_data[target]
    prior = {name: data for name, data in season_data.items() if name < target}
    if not prior:
        return pd.DataFrame()

    # Built once and shared: assembling it per strategy repeated the most
    # expensive step in the run for no benefit.
    pool = build_pool(prior, opening_prices(season))
    if pool.empty:
        return pd.DataFrame()

    results = [
        run_strategy(target, season, prior, strategy, horizon, pool=pool)
        for strategy in (catalogue or strategies())
    ]
    results.append(hindsight_squad(target, season, horizon))

    rows = [
        {
            "season": result.season,
            "strategy": result.strategy,
            "opening_points": result.opening_points,
            "season_points": result.season_points,
            "cost": result.cost,
        }
        for result in results
        if result is not None
    ]
    if not rows:
        return pd.DataFrame()

    frame = pd.DataFrame(rows).sort_values("opening_points", ascending=False)

    ceiling = frame[frame["strategy"] == "Hindsight"]["opening_points"]
    if not ceiling.empty and ceiling.iloc[0] > 0:
        frame["share_of_ceiling"] = frame["opening_points"] / ceiling.iloc[0]

    # Carried on every row so a reader cannot see the score without seeing
    # whether the model could account for defensive contributions at all.
    frame["defensive"] = defensive_forecast_status(target, pool)
    return frame.reset_index(drop=True)


def compare_horizons(
    season_data: dict[str, pd.DataFrame],
    targets: tuple[str, ...] | None = None,
    horizons: tuple[int, ...] = SCORING_HORIZONS,
    catalogue: list[PreseasonStrategy] | None = None,
) -> pd.DataFrame:
    """Every strategy, season and horizon in one frame.

    The opening squad is scored over three, five and seven gameweeks rather
    than a single window. A model that looks good only at three gameweeks has
    found something that decays; one that only looks good at seven has not
    helped with the part of the season the squad was actually chosen for.

    Returns one row per season, horizon and strategy, carrying the share of
    that horizon's own achievable ceiling. Shares are comparable across
    horizons; raw points are not, since a seven-gameweek total is larger for
    reasons that have nothing to do with skill.
    """
    seasons = targets or tuple(sorted(season_data)[1:])

    frames = []
    for horizon in horizons:
        for target in seasons:
            frame = compare_strategies(season_data, target, horizon, catalogue)
            if frame.empty:
                continue
            frames.append(frame.assign(horizon=horizon))

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def horizon_table(comparison: pd.DataFrame) -> pd.DataFrame:
    """Strategies down, horizons across, mean share of ceiling in the cells."""
    if comparison.empty or "share_of_ceiling" not in comparison.columns:
        return pd.DataFrame()

    table = comparison.pivot_table(
        index="strategy", columns="horizon", values="share_of_ceiling", aggfunc="mean"
    )
    return table.sort_values(table.columns[-1], ascending=False)
