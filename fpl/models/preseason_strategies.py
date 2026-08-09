"""The catalogue of ways to value a player before a season starts.

Every season-opening model reduces to one question — given a pool of priced
players and the seasons behind them, how many points is each worth over the
opening run — and they differ only in how they answer it. So they are
registered here as interchangeable strategies rather than written as separate
scripts, which is what makes "which of these is best" a measurement instead of
an argument.

Each strategy declares what it *uses*, not just what it is called. That matters
because the honest reading of the results depends on it: a strategy that
ignores minutes is not a slightly worse version of one that uses them, it is a
different claim about the problem, and the measured gap between those two
claims is the largest finding this project has.

Adding a strategy means adding it to :func:`strategies` — a call site should
never construct one directly, or the comparison silently stops being complete.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import pandas as pd

from fpl.domain.identity import add_match_key
from fpl.domain.rules import season_scores_defensive_contributions
from fpl.features.team_strength import (
    DEFAULT_HORIZON,
    blend_team_defence,
    estimate_promoted_prior,
    opening_run_difficulty,
)
from fpl.models.preseason import PreseasonPredictor

MINUTES_PER_MATCH = 90
GAMEWEEKS_PER_SEASON = 38


@dataclass
class PreseasonContext:
    """Everything a strategy may look at, and nothing it may not.

    The point-in-time guarantee lives here rather than in each strategy. A
    strategy receives prior seasons and the target season's *opening prices*;
    it never receives the target season itself, so it cannot read the result it
    is about to be scored against.
    """

    target: str
    prior_seasons: dict[str, pd.DataFrame]
    horizon: int = DEFAULT_HORIZON

    _defence: pd.DataFrame | None = field(default=None, repr=False)

    @property
    def team_defence(self) -> pd.DataFrame:
        """Blended per-club concession, computed once and reused."""
        if self._defence is None:
            self._defence = blend_team_defence(self.prior_seasons)
        return self._defence

    @property
    def promoted_prior(self) -> float:
        return estimate_promoted_prior(self.prior_seasons)

    @property
    def scores_defensive_contributions(self) -> bool:
        """Whether the *target season's rules* pay for defensive contributions.

        Deliberately a question about the season, not about the data. See
        ``domain/rules.py``.
        """
        return season_scores_defensive_contributions(self.target)

    @property
    def latest_prior(self) -> pd.DataFrame:
        if not self.prior_seasons:
            return pd.DataFrame()
        return self.prior_seasons[max(self.prior_seasons)]


@runtime_checkable
class PreseasonStrategy(Protocol):
    """Values every player in a pool over the opening run."""

    name: str
    uses: tuple[str, ...]

    def expected_points(self, pool: pd.DataFrame, context: PreseasonContext) -> pd.Series:
        """Points expected over ``context.horizon``, indexed like ``pool``."""
        ...


@dataclass
class Uninformed:
    """Every player equally good. The floor.

    Returns a legal squad and nothing more. Its purpose is to make the other
    numbers interpretable: a strategy that cannot beat this has no information
    in it at all.
    """

    name: str = "Uninformed"
    uses: tuple[str, ...] = ()

    def expected_points(self, pool: pd.DataFrame, context: PreseasonContext) -> pd.Series:
        return pd.Series(1.0, index=pool.index)


@dataclass
class PriorSeasonPoints:
    """Buy last season's highest scorers within budget. The obvious heuristic.

    This is what a person does without a model, so it is the benchmark that
    actually matters — beating it is the minimum bar for any of the work being
    worth anything. It has repeatedly not been easy to beat, because a
    prior-season points total implicitly carries minutes, and minutes turn out
    to be most of the problem.
    """

    name: str = "PriorSeasonPoints"
    uses: tuple[str, ...] = ("prior season totals",)

    def expected_points(self, pool: pd.DataFrame, context: PreseasonContext) -> pd.Series:
        latest = context.latest_prior
        if latest.empty or "player_name" not in latest.columns:
            return pd.Series(0.0, index=pool.index)

        totals = latest.groupby("player_name", as_index=False)["total_points"].sum()
        totals = add_match_key(totals, "player_name")
        lookup = totals.set_index("match_key")["total_points"]
        return pool["match_key"].map(lookup).fillna(0.0)


@dataclass
class BlendedRates:
    """Blended per-90 rates times an assumed constant minutes share.

    Kept in the catalogue because of what it demonstrates rather than what it
    achieves. It scores **below a randomly chosen legal squad** in some seasons:
    a per-90 rate says how good a player is while on the pitch, and multiplying
    it by a constant treats a twenty-minute substitute as a starter. Removing it
    would delete the evidence for the project's largest finding.
    """

    name: str = "BlendedCareer"
    uses: tuple[str, ...] = ("career rates",)
    minutes_prior: float = 60.0

    def expected_points(self, pool: pd.DataFrame, context: PreseasonContext) -> pd.Series:
        rate = pool.get("total_points_per_90")
        if rate is None:
            return pd.Series(0.0, index=pool.index)
        return (rate.fillna(0.0) * self.minutes_prior / MINUTES_PER_MATCH * context.horizon).clip(
            lower=0
        )


@dataclass
class BlendedRatesWithMinutes:
    """The same rates, scaled by minutes actually played per gameweek.

    The single correction that matters most. Adding this term took the same
    model from 19% of the achievable ceiling to 55%.
    """

    name: str = "BlendedCareer+Minutes"
    uses: tuple[str, ...] = ("career rates", "minutes")
    gameweeks_per_season: int = GAMEWEEKS_PER_SEASON

    def expected_points(self, pool: pd.DataFrame, context: PreseasonContext) -> pd.Series:
        rate = pool.get("total_points_per_90")
        minutes = pool.get("career_minutes")
        if rate is None or minutes is None:
            return pd.Series(0.0, index=pool.index)

        seasons = pool.get("seasons_seen")
        span = (
            seasons.fillna(1).clip(lower=1) if seasons is not None else 1
        ) * self.gameweeks_per_season
        per_gameweek = (minutes.fillna(0) / span).clip(0, MINUTES_PER_MATCH)
        return (rate.fillna(0.0) * per_gameweek / MINUTES_PER_MATCH * context.horizon).clip(lower=0)


@dataclass
class Components:
    """Score each route to points separately and add them up.

    Appearance, goals valued by position, assists, clean sheets taken from the
    club the player is *joining*, bonus, and — where the season's rules pay for
    them — defensive contributions.

    ``use_fixtures`` applies opening-run difficulty. Off by default: measured, it
    does not point the same way twice across the three testable seasons.
    """

    name: str = "Components"
    uses: tuple[str, ...] = ("career rates", "minutes", "team defence", "defensive contributions")
    use_fixtures: bool = False

    def expected_points(self, pool: pd.DataFrame, context: PreseasonContext) -> pd.Series:
        if pool.empty:
            return pd.Series(dtype="float64")

        model = PreseasonPredictor(
            horizon=context.horizon,
            team_defence=context.team_defence,
            promoted_prior=context.promoted_prior,
            score_defensive_contributions=context.scores_defensive_contributions,
        )

        difficulty = None
        if self.use_fixtures:
            ratings = opening_run_difficulty(
                context.latest_prior, context.team_defence, horizon=context.horizon
            )
            if not ratings.empty:
                lookup = ratings.set_index("team_name")["opening_difficulty"]
                difficulty = pool["team"].map(lookup).fillna(1.0)

        predictions = model.predict(pool, fixture_difficulty=difficulty)
        if predictions.empty:
            return pd.Series(0.0, index=pool.index)
        return pd.Series(predictions["expected_points"].to_numpy(), index=pool.index)


@dataclass
class ComponentsWithFixtures(Components):
    name: str = "Components+Fixtures"
    uses: tuple[str, ...] = (
        "career rates",
        "minutes",
        "team defence",
        "defensive contributions",
        "fixtures",
    )
    use_fixtures: bool = True


def strategies() -> list[PreseasonStrategy]:
    """The standard comparison set, floor first.

    Ordered by how much they claim to know, so a results table read top to
    bottom shows what each addition bought.
    """
    return [
        Uninformed(),
        BlendedRates(),
        PriorSeasonPoints(),
        BlendedRatesWithMinutes(),
        Components(),
        ComponentsWithFixtures(),
    ]


def strategy_by_name(name: str) -> PreseasonStrategy:
    """Look one up, failing loudly rather than returning a default.

    A silent fallback here would mean a backtest reporting results for a
    strategy nobody asked for, under the name of one that does not exist.
    """
    for strategy in strategies():
        if strategy.name == name:
            return strategy
    known = ", ".join(strategy.name for strategy in strategies())
    raise KeyError(f"no preseason strategy named {name!r}. Known: {known}")
