"""Blending a player's rates across the seasons behind them.

Within a season, a model can read a player's form directly. Before a season
starts there is no form, only history — and history has to be weighted, because
three years ago is not evidence about now in the way last year is.

Two weightings are combined and they do different jobs:

**Season weight** decays with age: the most recent season counts most.

**Minutes weight** decays with sample size: a rate from 400 minutes is a worse
estimate than the same rate from 3,000, whichever season it came from.

Multiplying them is what stops the obvious failure — a player who barely played
last season having their whole projection set by that fragment, or a player
with one enormous old season being rated on it forever.

Identity is by normalised name, because ``element`` ids are reassigned each
season and the archive carries no stable code. That is fuzzy, and
:func:`blend_career_rates` reports how many seasons each player was found in so
a one-season estimate can be told from a four-season one.
"""

from __future__ import annotations

import pandas as pd

from fpl.domain.identity import add_match_key

# Geometric decay across seasons. 0.55 gives roughly 0.50 / 0.28 / 0.15 / 0.07
# once normalised over four seasons -- the most recent worth about half.
DEFAULT_SEASON_DECAY = 0.55

# Rates worth carrying across seasons. All are per-90 quantities once divided
# by minutes, and all exist in every season from 2022-23.
RATE_COLUMNS = (
    "total_points",
    "goals_scored",
    "assists",
    "expected_goals",
    "expected_assists",
    "bps",
    "bonus",
    "clean_sheets",
    "goals_conceded",
    "saves",
    "yellow_cards",
    "minutes",
)

MINUTES_PER_MATCH = 90

# Below this a player's rates are dominated by noise, so confidence is scaled
# down rather than the estimate being thrown away.
RELIABLE_MINUTES = 900


def season_weights(seasons: list[str], decay: float = DEFAULT_SEASON_DECAY) -> dict[str, float]:
    """Weight per season, most recent highest, normalised to sum to 1.

    Ordering is by the season label, which sorts correctly for ``YYYY-YY``.
    """
    if not seasons:
        return {}

    ordered = sorted(seasons, reverse=True)
    raw = {season: decay**index for index, season in enumerate(ordered)}
    total = sum(raw.values())
    return {season: weight / total for season, weight in raw.items()}


def season_totals(season_data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """One row per player per season, with their totals and a match key."""
    frames = []
    for season, data in season_data.items():
        if data.empty or "player_name" not in data.columns:
            continue

        columns = [column for column in RATE_COLUMNS if column in data.columns]
        totals = data.groupby("player_name", as_index=False)[columns].sum()
        totals["season"] = season
        totals["appearances"] = (
            data[data["minutes"] > 0]
            .groupby("player_name")
            .size()
            .reindex(totals["player_name"])
            .fillna(0)
            .to_numpy()
        )
        # Last known club, which is what a transfer check compares against.
        if "team_name" in data.columns:
            last_club = data.sort_values("gameweek").groupby("player_name")["team_name"].last()
            totals["team_name"] = totals["player_name"].map(last_club)
        frames.append(totals)

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    return add_match_key(combined, "player_name")


def blend_career_rates(
    season_data: dict[str, pd.DataFrame], decay: float = DEFAULT_SEASON_DECAY
) -> pd.DataFrame:
    """Per-90 rates for each player, blended across the seasons supplied.

    Returns one row per player with ``<column>_per_90`` for each rate, plus:

    ``seasons_seen``    how many seasons the player appears in
    ``career_minutes``  total minutes behind the estimate
    ``confidence``      0 to 1, from minutes played and seasons seen
    ``last_club``       most recent club, for detecting transfers

    A player found in no season is simply absent — the caller decides what to
    do about the 15% with no history, and inventing a rate here would hide them.
    """
    totals = season_totals(season_data)
    if totals.empty:
        return pd.DataFrame()

    weights = season_weights(list(season_data), decay)
    totals = totals.copy()
    totals["season_weight"] = totals["season"].map(weights).fillna(0.0)

    # The weight on a season's rate is its recency times its sample size.
    totals["blend_weight"] = totals["season_weight"] * totals["minutes"]

    rows = []
    for match_key, group in totals.groupby("match_key"):
        weight = group["blend_weight"].sum()
        if weight <= 0:
            continue

        row = {
            "match_key": match_key,
            "player_name": group.sort_values("season")["player_name"].iloc[-1],
            "seasons_seen": int(group["season"].nunique()),
            "career_minutes": float(group["minutes"].sum()),
            "career_appearances": float(group["appearances"].sum()),
        }
        if "team_name" in group.columns:
            row["last_club"] = group.sort_values("season")["team_name"].iloc[-1]

        for column in RATE_COLUMNS:
            if column == "minutes" or column not in group.columns:
                continue
            # Weighted mean of the per-90 rates, where each season's rate is
            # weighted by recency times minutes.
            per_90 = group[column] * MINUTES_PER_MATCH / group["minutes"].replace(0, pd.NA)
            valid = per_90.notna()
            if not valid.any():
                continue
            row[f"{column}_per_90"] = float(
                (per_90[valid] * group["blend_weight"][valid]).sum()
                / group["blend_weight"][valid].sum()
            )

        rows.append(row)

    if not rows:
        return pd.DataFrame()

    career = pd.DataFrame(rows)
    career["confidence"] = confidence(career)
    return career.sort_values("career_minutes", ascending=False).reset_index(drop=True)


def confidence(career: pd.DataFrame) -> pd.Series:
    """How much to trust a player's blended rates, from 0 to 1.

    Minutes dominate, because a rate is only as good as the sample under it.
    Seasons seen contributes separately: four hundred minutes spread over two
    seasons is slightly better evidence than the same in one, since it is less
    likely to be a single hot streak.
    """
    if career.empty:
        return pd.Series(dtype="float64")

    minutes_term = (career["career_minutes"] / RELIABLE_MINUTES).clip(0, 1)
    seasons_term = (career["seasons_seen"] / 4).clip(0, 1)
    return (0.75 * minutes_term + 0.25 * seasons_term).clip(0, 1)


def finishing_multiplier(
    career: pd.DataFrame, shrink: float = 0.25, bounds: tuple[float, float] = (0.85, 1.15)
) -> pd.Series:
    """Adjustment for persistent over- or under-performance against xG.

    Deliberately heavily shrunk. This project established that finishing
    over-performance is a *sell* signal — shot quality persists, finishing
    streaks do not — so a player 40% above their expected goals is credited
    about 10%, not 40%.
    """
    if career.empty:
        return pd.Series(dtype="float64")

    # "No adjustment" is a multiplier of 1, not an empty series. Returning
    # nothing here silently turned every downstream product into NaN.
    expected = career.get("expected_goals_per_90")
    scored = career.get("goals_scored_per_90")
    if expected is None or scored is None:
        return pd.Series(1.0, index=career.index)

    safe = expected.where(expected > 0)
    ratio = (scored - safe) / safe
    return (1 + shrink * ratio.fillna(0.0)).clip(*bounds)


def shrink_towards_prior(
    career: pd.DataFrame, columns: tuple[str, ...] | None = None
) -> pd.DataFrame:
    """Pull unreliable rates towards the population mean, in proportion to doubt.

    Without this the blend produces monsters: on real data the top scorer by
    points per 90 is a player with three career minutes and one goal, rating
    90 points per 90. Confidence correctly marks him untrustworthy, but the
    *rate* still reaches whatever consumes it, and an optimiser maximising
    expected points would fill a squad with such players.

    Standard shrinkage: ``confidence × observed + (1 − confidence) × prior``.
    A player with 3,000 minutes keeps their own rate almost exactly; a player
    with 90 minutes is pulled almost entirely to the mean.

    The prior is the minutes-weighted population mean, so it reflects a typical
    *player* rather than a typical row.
    """
    if career.empty or "confidence" not in career.columns:
        return career

    df = career.copy()
    targets = columns or tuple(c for c in df.columns if c.endswith("_per_90"))

    for column in targets:
        if column not in df.columns:
            continue
        weights = df["career_minutes"].clip(lower=0)
        if weights.sum() <= 0:
            continue
        prior = float((df[column] * weights).sum() / weights.sum())
        df[column] = df["confidence"] * df[column] + (1 - df["confidence"]) * prior

    return df
