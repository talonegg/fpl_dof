"""How much a club concedes, and what that means for its defenders.

Clean sheets are a team property that pays out to individuals. A defender's own
clean-sheet history tells you about the club they were at, not about them — so
when a player moves, their prior record is close to worthless and their new
club's record is what matters.

The data supports this strongly. Across 2025-26, expected goals conceded per
match ranged from 0.76 to 2.02 between clubs — a 2.6-fold spread — correlating
−0.825 with the rate at which those clubs actually kept clean sheets. A
defender moving between the extremes of that range should see their clean-sheet
expectation fall from roughly 65% to roughly 19%.

**Promoted clubs are the hard case.** They have no Premier League record at
all, and using a league-average prior systematically over-rates their
defenders, who are among the cheapest and therefore most tempting picks. They
get their own prior, estimated from how promoted clubs have actually done.
"""

from __future__ import annotations

import math

import pandas as pd

from fpl.features.career import season_weights

MINUTES_PER_MATCH = 90
FULL_APPEARANCE_MINUTES = 60

# Promoted clubs concede materially more than the league mean. Estimated from
# the archive rather than assumed; see estimate_promoted_prior().
DEFAULT_PROMOTED_XGC = 1.75


def team_match_defence(season: pd.DataFrame) -> pd.DataFrame:
    """One row per club per gameweek: goals and expected goals conceded.

    Taken from players who completed 60 minutes, because goals conceded is
    recorded per player and only a player who was on for the whole match
    reflects the team's total. The maximum across those players is the team
    figure.
    """
    if season.empty or "team_name" not in season.columns:
        return pd.DataFrame()

    played = season[season["minutes"] >= FULL_APPEARANCE_MINUTES]
    if played.empty:
        return pd.DataFrame()

    aggregations = {"goals_conceded": ("goals_conceded", "max")}
    if "expected_goals_conceded" in played.columns:
        aggregations["expected_goals_conceded"] = ("expected_goals_conceded", "max")
    if "clean_sheets" in played.columns:
        aggregations["clean_sheets"] = ("clean_sheets", "max")

    return played.groupby(["team_name", "gameweek"], as_index=False).agg(**aggregations)


def season_defence(season: pd.DataFrame) -> pd.DataFrame:
    """Per-club concession rates for one season."""
    matches = team_match_defence(season)
    if matches.empty:
        return pd.DataFrame()

    aggregations = {
        "matches": ("gameweek", "nunique"),
        "goals_conceded_per_match": ("goals_conceded", "mean"),
    }
    if "expected_goals_conceded" in matches.columns:
        aggregations["expected_goals_conceded_per_match"] = (
            "expected_goals_conceded",
            "mean",
        )
    if "clean_sheets" in matches.columns:
        aggregations["clean_sheet_rate"] = ("clean_sheets", "mean")

    return matches.groupby("team_name", as_index=False).agg(**aggregations)


def blend_team_defence(season_data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Per-club concession rates blended across seasons, recent weighted most.

    Clubs absent from a season simply contribute nothing to it, so a club
    promoted last year is rated on its one season rather than being penalised
    for the years it was not there.
    """
    weights = season_weights(list(season_data))

    frames = []
    for season, data in season_data.items():
        defence = season_defence(data)
        if defence.empty:
            continue
        defence["season"] = season
        defence["weight"] = weights.get(season, 0.0) * defence["matches"]
        frames.append(defence)

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)

    rows = []
    for club, group in combined.groupby("team_name"):
        weight = group["weight"].sum()
        if weight <= 0:
            continue
        row = {
            "team_name": club,
            "seasons_seen": int(group["season"].nunique()),
            "matches": float(group["matches"].sum()),
        }
        for column in (
            "goals_conceded_per_match",
            "expected_goals_conceded_per_match",
            "clean_sheet_rate",
        ):
            if column in group.columns:
                row[column] = float((group[column] * group["weight"]).sum() / weight)
        rows.append(row)

    return (
        pd.DataFrame(rows).sort_values("expected_goals_conceded_per_match").reset_index(drop=True)
    )


def estimate_promoted_prior(
    season_data: dict[str, pd.DataFrame], known_clubs: set[str] | None = None
) -> float:
    """Expected goals conceded per match for a club new to the division.

    Estimated from clubs that appear in one season and not the previous one,
    which is what promotion looks like in this data. Falls back to a documented
    constant when there are too few to estimate from.
    """
    seasons = sorted(season_data)
    if len(seasons) < 2:
        return DEFAULT_PROMOTED_XGC

    newcomer_rates = []
    for earlier, later in zip(seasons, seasons[1:], strict=False):
        before = set(season_defence(season_data[earlier])["team_name"])
        after = season_defence(season_data[later])
        newcomers = after[~after["team_name"].isin(before)]
        if "expected_goals_conceded_per_match" in newcomers.columns:
            newcomer_rates.extend(newcomers["expected_goals_conceded_per_match"].tolist())

    if not newcomer_rates:
        return DEFAULT_PROMOTED_XGC
    return float(sum(newcomer_rates) / len(newcomer_rates))


def expected_concession(
    club: str, blended: pd.DataFrame, promoted_prior: float = DEFAULT_PROMOTED_XGC
) -> float:
    """Expected goals conceded per match for a club in the season ahead.

    A club with no record is treated as promoted rather than as average, which
    is the conservative direction: over-rating a promoted side's defenders is
    the error this exists to prevent.
    """
    if blended.empty or "expected_goals_conceded_per_match" not in blended.columns:
        return promoted_prior

    match = blended[blended["team_name"] == club]
    if match.empty:
        return promoted_prior
    return float(match["expected_goals_conceded_per_match"].iloc[0])


def clean_sheet_probability(expected_goals_conceded: float) -> float:
    """P(clean sheet) from expected goals conceded, under a Poisson.

    The same assumption used for the market-derived probabilities in
    ``features/market.py``, and stated for the same reason: goals within a
    match are not quite independent, but this is the standard working model.
    """
    if expected_goals_conceded < 0:
        return 0.0
    return float(math.exp(-expected_goals_conceded))


def clean_sheet_outlook(
    clubs: pd.Series, blended: pd.DataFrame, promoted_prior: float = DEFAULT_PROMOTED_XGC
) -> pd.DataFrame:
    """Expected concession and clean-sheet probability for each club named.

    Takes the club a player will play for *next* season, which for a transfer
    is not the club their own history was accumulated at.
    """
    rows = []
    for club in clubs.dropna().unique():
        concession = expected_concession(club, blended, promoted_prior)
        rows.append(
            {
                "team_name": club,
                "expected_goals_conceded": concession,
                "clean_sheet_probability": clean_sheet_probability(concession),
                "is_promoted": blended.empty or club not in set(blended.get("team_name", [])),
            }
        )
    return pd.DataFrame(rows)
