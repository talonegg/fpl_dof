"""Picking a squad before a season starts, and finding out how it did.

The test the whole season-opening design rests on. For a season S, build a
squad using **only seasons before S** and S's own opening prices, then score
what those fifteen actually went on to do.

Possible at all because the archive records a price for every player in
gameweek 1 of every season, which is the real budget they had to be bought
within.

**Three seasons are testable, so this yields three squads.** That is a very
small sample and no amount of statistics rescues it. This can show that a
model is bad; it cannot show that one is good. Read a positive result as
permission to proceed, not as evidence of skill.

Benchmarks matter more than the score for that reason. "Our squad scored 620"
means nothing on its own; "620 against a template's 585 and a hindsight-perfect
760" is a result.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from fpl.domain.identity import add_match_key
from fpl.domain.positions import display_name
from fpl.features.career import blend_career_rates, shrink_towards_prior
from fpl.optimise.squad import InfeasibleSquad, SquadConstraints, optimise_squad

OPENING_GAMEWEEK = 1
DEFAULT_HORIZON = 10


@dataclass
class PreseasonResult:
    """A squad picked before a season, and what it went on to score."""

    season: str
    strategy: str
    squad: pd.DataFrame
    opening_points: float
    season_points: float
    cost: float

    def summary(self) -> str:
        return (
            f"{self.strategy} in {self.season}: {self.opening_points:.0f} over the "
            f"opening run, {self.season_points:.0f} across the season, "
            f"£{self.cost:.1f}m spent"
        )


def opening_prices(season: pd.DataFrame) -> pd.DataFrame:
    """Each player's price, club and position at gameweek 1.

    The budget that actually applied. Using end-of-season prices would let a
    model buy a player who rose from £4.5m to £7m for the price he ended at,
    which is a form of hindsight.
    """
    if season.empty:
        return pd.DataFrame()

    opening = season[season["gameweek"] == OPENING_GAMEWEEK]
    if opening.empty:
        return pd.DataFrame()

    columns = [
        column
        for column in ("element", "player_name", "position", "team_name", "price")
        if column in opening.columns
    ]
    prices = opening[columns].drop_duplicates(subset="element").copy()
    if "position" in prices.columns:
        # The optimiser keys on the long names in domain/rules.py; the archive
        # uses short codes. Translate towards the rules, not away from them.
        prices["position"] = prices["position"].map(display_name)
    prices["team"] = prices["team_name"]
    return add_match_key(prices, "player_name")


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


def build_pool(prior_seasons: dict[str, pd.DataFrame], prices: pd.DataFrame) -> pd.DataFrame:
    """Join blended career rates onto the opening prices.

    Players with no prior history keep a null rate rather than a guessed one —
    the caller decides. That is 15% of the list on real data, and inventing
    numbers for them here would hide the gap the design is explicit about.
    """
    if prices.empty:
        return pd.DataFrame()

    career = blend_career_rates(prior_seasons)
    if career.empty:
        return pd.DataFrame()

    career = shrink_towards_prior(career)
    columns = [
        "match_key",
        "total_points_per_90",
        "confidence",
        "career_minutes",
        "seasons_seen",
    ]
    available = [column for column in columns if column in career.columns]
    return prices.merge(career[available], on="match_key", how="left")


def expected_points_from_history(
    pool: pd.DataFrame, horizon: int = DEFAULT_HORIZON, minutes_prior: float = 60.0
) -> pd.Series:
    """Points expected over the opening run, from blended history alone.

    Deliberately the simplest thing that could work: a per-90 rate times an
    assumed minutes share times the horizon. It exists to test the *pipeline*
    before any real model does, so that a failure can be attributed to the
    plumbing rather than to the modelling.
    """
    if pool.empty:
        return pd.Series(dtype="float64")

    rate = pool.get("total_points_per_90")
    if rate is None:
        return pd.Series(0.0, index=pool.index)

    return (rate.fillna(0.0) * minutes_prior / 90.0 * horizon).clip(lower=0)


def expected_points_with_minutes(
    pool: pd.DataFrame, horizon: int = DEFAULT_HORIZON, gameweeks_per_season: int = 38
) -> pd.Series:
    """Points expected over the opening run, using a minutes forecast.

    The correction to :func:`expected_points_from_history`, which assumed every
    player would play the same 60 minutes. Measured on 2025-26 that assumption
    was catastrophic: it bought a squad that played 1,121 minutes across the
    opening run where the obvious heuristic's squad played 10,234, and scored
    11% of the achievable ceiling against that heuristic's 55%.

    A per-90 rate says how good a player is *while on the pitch*. Multiplying it
    by a constant treats a 20-minute substitute as a starter. Prior-season
    totals implicitly carry minutes, which is exactly why the naive heuristic
    beat the sophisticated blend.

    Expected minutes here come from history: total minutes over the seasons
    seen, divided by the gameweeks those seasons contained.
    """
    if pool.empty:
        return pd.Series(dtype="float64")

    rate = pool.get("total_points_per_90")
    if rate is None:
        return pd.Series(0.0, index=pool.index)

    seasons = pool.get("seasons_seen")
    minutes = pool.get("career_minutes")
    if minutes is None:
        return expected_points_from_history(pool, horizon)

    span = (seasons.fillna(1).clip(lower=1) if seasons is not None else 1) * gameweeks_per_season
    minutes_per_gameweek = (minutes.fillna(0) / span).clip(0, 90)

    return (rate.fillna(0.0) * minutes_per_gameweek / 90.0 * horizon).clip(lower=0)


def pick_squad(
    pool: pd.DataFrame, expected: pd.Series, constraints: SquadConstraints | None = None
) -> pd.DataFrame | None:
    """Optimise a squad from a priced pool and an expected-points vector."""
    if pool.empty:
        return None

    candidates = pool.assign(expected_points=expected).dropna(
        subset=["price", "position", "team", "expected_points"]
    )
    if candidates.empty:
        return None

    try:
        squad = optimise_squad(candidates, constraints)
    except InfeasibleSquad:
        return None
    return squad.players


def run_strategy(
    season_name: str,
    season: pd.DataFrame,
    prior_seasons: dict[str, pd.DataFrame],
    strategy: str,
    expected: pd.Series | None = None,
    horizon: int = DEFAULT_HORIZON,
) -> PreseasonResult | None:
    """Build a squad for ``season_name`` and score it."""
    prices = opening_prices(season)
    pool = build_pool(prior_seasons, prices)
    if pool.empty:
        return None

    vector = expected if expected is not None else expected_points_from_history(pool, horizon)
    squad = pick_squad(pool, vector)
    if squad is None:
        return None

    opening, whole = actual_points(season, squad["element"].tolist(), horizon)
    return PreseasonResult(
        season=season_name,
        strategy=strategy,
        squad=squad,
        opening_points=opening,
        season_points=whole,
        cost=float(squad["price"].sum()),
    )


# -- Benchmarks -----------------------------------------------------------


def hindsight_squad(
    season_name: str, season: pd.DataFrame, horizon: int = DEFAULT_HORIZON
) -> PreseasonResult | None:
    """The best squad it was possible to buy, knowing everything. The ceiling.

    Reported because a score without a ceiling is uninterpretable. Nobody can
    reach this; the useful number is the share of it a strategy captures.
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

    squad = pick_squad(pool, pool["total_points"])
    if squad is None:
        return None

    opening, whole = actual_points(season, squad["element"].tolist(), horizon)
    return PreseasonResult(
        season_name, "Hindsight", squad, opening, whole, float(squad["price"].sum())
    )


def cheapest_squad(
    season_name: str, season: pd.DataFrame, horizon: int = DEFAULT_HORIZON
) -> PreseasonResult | None:
    """A legal squad chosen with no information at all. The floor."""
    prices = opening_prices(season)
    if prices.empty:
        return None

    # Every player equally good: the optimiser returns a legal squad, nothing more.
    squad = pick_squad(prices, pd.Series(1.0, index=prices.index))
    if squad is None:
        return None

    opening, whole = actual_points(season, squad["element"].tolist(), horizon)
    return PreseasonResult(
        season_name, "Uninformed", squad, opening, whole, float(squad["price"].sum())
    )


def prior_points_squad(
    season_name: str,
    season: pd.DataFrame,
    prior_seasons: dict[str, pd.DataFrame],
    horizon: int = DEFAULT_HORIZON,
) -> PreseasonResult | None:
    """Buy last season's highest scorers within budget. The obvious heuristic.

    This is what a person does without a model, so it is the benchmark that
    actually matters: beating it is the minimum bar for the work being worth
    anything.
    """
    prices = opening_prices(season)
    if prices.empty or not prior_seasons:
        return None

    latest = max(prior_seasons)
    totals = prior_seasons[latest].groupby("player_name", as_index=False)["total_points"].sum()
    totals = add_match_key(totals, "player_name")

    pool = prices.merge(totals[["match_key", "total_points"]], on="match_key", how="left")
    pool["total_points"] = pool["total_points"].fillna(0)

    squad = pick_squad(pool, pool["total_points"])
    if squad is None:
        return None

    opening, whole = actual_points(season, squad["element"].tolist(), horizon)
    return PreseasonResult(
        season_name, "PriorSeasonPoints", squad, opening, whole, float(squad["price"].sum())
    )


def compare_strategies(
    season_data: dict[str, pd.DataFrame], target: str, horizon: int = DEFAULT_HORIZON
) -> pd.DataFrame:
    """Every strategy and benchmark for one season, best opening run first."""
    if target not in season_data:
        return pd.DataFrame()

    season = season_data[target]
    prior = {name: data for name, data in season_data.items() if name < target}
    if not prior:
        return pd.DataFrame()

    results = [
        run_strategy(target, season, prior, "BlendedCareer", horizon=horizon),
        run_strategy(
            target,
            season,
            prior,
            "BlendedCareer+Minutes",
            expected=expected_points_with_minutes(
                build_pool(prior, opening_prices(season)), horizon
            ),
            horizon=horizon,
        ),
        prior_points_squad(target, season, prior, horizon),
        cheapest_squad(target, season, horizon),
        hindsight_squad(target, season, horizon),
    ]

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
    return frame.reset_index(drop=True)
