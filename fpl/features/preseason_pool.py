"""Assembling the candidate pool for a season that has not started.

The pool is the join every season-opening model works from: who is buyable, at
what price, for which club, with what history behind them. It is a *feature*
step rather than a backtest one — the live recommender and the historical
replay need exactly the same frame, and building it twice is how the two
quietly diverge.

Three sources meet here:

**Prices** are the season's own gameweek-1 prices. Using anything later is
hindsight: end-of-season prices would let a model buy a player who rose from
£4.5m to £7m at the price he finished at.

**Career rates** are blended across the prior seasons, weighted by recency and
by minutes (see :mod:`fpl.features.career`).

**Defensive-contribution rates** come from whichever prior seasons recorded the
underlying actions — which, before 2025-26, is none of them. The column is
absent rather than zero in that case, so a caller can tell "will not clear the
threshold" from "nobody counted".
"""

from __future__ import annotations

import pandas as pd

from fpl.domain.identity import add_match_key
from fpl.domain.positions import display_name
from fpl.features.career import blend_career_rates, shrink_towards_prior
from fpl.features.defensive import defensive_contribution_rate

OPENING_GAMEWEEK = 1


def opening_prices(season: pd.DataFrame) -> pd.DataFrame:
    """Each player's price, club and position at gameweek 1.

    The budget that actually applied.
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


def build_pool(prior_seasons: dict[str, pd.DataFrame], prices: pd.DataFrame) -> pd.DataFrame:
    """Join blended career rates and defensive rates onto the opening prices.

    Players with no prior history keep a null rate rather than a guessed one —
    the caller decides. That is a third of the priced list on real data, and
    inventing numbers for them here would hide a gap the design is explicit
    about.
    """
    if prices.empty:
        return pd.DataFrame()

    career = blend_career_rates(prior_seasons)
    if career.empty:
        return pd.DataFrame()

    career = shrink_towards_prior(career)
    # Carry every career column, not a chosen subset. A component model needs
    # the underlying rates and the appearance counts, and picking columns here
    # silently starved the minutes forecaster of its inputs.
    drop = [column for column in ("player_name", "last_club") if column in career.columns]
    pool = prices.merge(career.drop(columns=drop), on="match_key", how="left")

    defensive = defensive_contribution_rate(prior_seasons)
    if not defensive.empty:
        pool = pool.merge(
            defensive.drop(columns=["player_name"], errors="ignore"),
            on="match_key",
            how="left",
        )
    return pool


def pool_coverage(pool: pd.DataFrame) -> dict[str, int]:
    """How much of the pool the models can actually see.

    Reported rather than assumed, because the answer is worse than it looks: a
    player with no prior minutes gets no expected points and is dropped before
    the optimiser runs, and most of those are cheap enough to be exactly the
    bench slots a real squad has to fill.
    """
    if pool.empty:
        return {"priced": 0, "with_history": 0, "without_history": 0, "cheap_without": 0}

    known = pool["career_minutes"].notna() if "career_minutes" in pool.columns else pool.index == -1
    cheap = pool["price"] < 5.0 if "price" in pool.columns else pool.index == -1
    return {
        "priced": int(len(pool)),
        "with_history": int(known.sum()),
        "without_history": int((~known).sum()),
        "cheap_without": int((~known & cheap).sum()),
    }
