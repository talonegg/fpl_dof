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

# Fixture weighting across the opening run: a flat plateau over the gameweeks
# you are certain to hold the squad for, then a decay across the ones you are
# not. Three at full weight, then four diminishing, then nothing.
#
# Lives here rather than in the predictor because two consumers must agree on
# it. The predictor's weights are a uniform scalar and cannot change which
# players it prefers; opening_run_difficulty's weights decide *which opponents
# count*, and that does. Two curves would mean the shape being asked for
# applied to only one of them.
PLATEAU_GAMEWEEKS = 3
DEFAULT_FIXTURE_DECAY = 0.7
DEFAULT_HORIZON = 7


def fixture_weights(
    horizon: int = DEFAULT_HORIZON,
    decay: float = DEFAULT_FIXTURE_DECAY,
    plateau: int = PLATEAU_GAMEWEEKS,
) -> list[float]:
    """Weight per gameweek of the opening run: flat, then diminishing, then off.

    The opening three gameweeks are the ones the squad is actually held for —
    a free transfer a week means gameweek 8 will be re-decided with better
    information than exists now — so they carry full weight rather than a decay
    that already discounts gameweek 3 to 0.6.

    Gameweeks 4 to 7 decay geometrically: a fixture you will probably still be
    holding these players for, weighted below one you certainly will.

    With the defaults: ``[1.0, 1.0, 1.0, 0.70, 0.49, 0.34, 0.24]``.
    """
    return [1.0 if index < plateau else decay ** (index - plateau + 1) for index in range(horizon)]


def team_match_defence(season: pd.DataFrame) -> pd.DataFrame:
    """One row per club per gameweek: goals and expected goals conceded.

    Taken from players who completed 60 minutes, because goals conceded is
    recorded per player and only a player who was on for the whole match
    reflects the team's total. The maximum across those players is the team
    figure.
    """
    required = {"team_name", "gameweek", "minutes", "goals_conceded"}
    if season.empty or not required <= set(season.columns):
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
        previous = season_defence(season_data[earlier])
        after = season_defence(season_data[later])
        # A season the archive cannot supply a defence table for tells us
        # nothing about who was promoted into the next one. Skip the pair
        # rather than reading a column that is not there.
        if previous.empty or after.empty:
            continue

        before = set(previous["team_name"])
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


def opponent_names(season: pd.DataFrame) -> pd.DataFrame:
    """Resolve each club's opponent per gameweek to a club *name*.

    ``opponent_team`` is a numeric id whose mapping to names is not published
    in the archive. It is recoverable anyway: both clubs in a match share a
    ``fixture`` id, so a club's opponent is simply the other name against that
    fixture.
    """
    if season.empty or "fixture" not in season.columns:
        return pd.DataFrame(columns=["team_name", "gameweek", "opponent_name"])

    sides = season.groupby(["fixture", "team_name"], as_index=False)["gameweek"].first()

    rows = []
    for fixture, group in sides.groupby("fixture"):
        clubs = group["team_name"].tolist()
        if len(clubs) != 2:
            continue
        gameweek = int(group["gameweek"].iloc[0])
        rows.append(
            {
                "team_name": clubs[0],
                "gameweek": gameweek,
                "opponent_name": clubs[1],
                "fixture": fixture,
            }
        )
        rows.append(
            {
                "team_name": clubs[1],
                "gameweek": gameweek,
                "opponent_name": clubs[0],
                "fixture": fixture,
            }
        )

    return pd.DataFrame(rows)


def opening_run_difficulty(
    season: pd.DataFrame,
    attack: pd.DataFrame,
    horizon: int = DEFAULT_HORIZON,
    decay: float = DEFAULT_FIXTURE_DECAY,
    plateau: int = PLATEAU_GAMEWEEKS,
) -> pd.DataFrame:
    """How kind each club's opening fixtures are, weighted towards the near ones.

    Returns a multiplier centred on 1.0: above 1 means an easier-than-average
    start. Weighted by :func:`fixture_weights` — the opening three gameweeks
    flat at full weight, four to seven diminishing, nothing after — so the
    opponents a squad is certainly held against count for more than the ones it
    probably is not.

    Difficulty is the opponent's *attacking* strength: the goals a club
    concedes to an average side, taken from ``attack`` keyed by club name.
    """
    columns = ["team_name", "opening_difficulty"]
    schedule = opponent_names(season)
    if schedule.empty or attack.empty:
        return pd.DataFrame(columns=columns)

    window = schedule[schedule["gameweek"] <= horizon].copy()
    if window.empty:
        return pd.DataFrame(columns=columns)

    strength = attack.set_index("team_name")["expected_goals_conceded_per_match"]
    league_mean = float(strength.mean())
    if league_mean <= 0:
        return pd.DataFrame(columns=columns)

    # An opponent who concedes a lot is an easy fixture.
    window["opponent_leakiness"] = window["opponent_name"].map(strength).fillna(league_mean)

    curve = fixture_weights(horizon, decay, plateau)
    window["weight"] = window["gameweek"].map(
        {gameweek: curve[gameweek - 1] for gameweek in range(1, horizon + 1)}
    )

    rows = []
    for club, group in window.groupby("team_name"):
        weight = group["weight"].sum()
        if weight <= 0:
            continue
        weighted = float((group["opponent_leakiness"] * group["weight"]).sum() / weight)
        rows.append({"team_name": club, "opening_difficulty": weighted / league_mean})

    return (
        pd.DataFrame(rows).sort_values("opening_difficulty", ascending=False).reset_index(drop=True)
    )
