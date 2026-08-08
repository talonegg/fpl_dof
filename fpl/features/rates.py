"""Rate and value metrics the raw API does not provide.

The API gives totals. Totals reward players who happen to have played more,
which is exactly the wrong signal when deciding who to buy: a defender with
120 points from 38 starts and one with 120 points from 20 are not the same
player. Everything here converts totals into rates, so players on different
amounts of pitch time can be compared honestly.

Every function is pure and returns a new frame. Division guards against zero
minutes by producing NaN, never 0 -- "we cannot know" and "is bad" are
different answers, and collapsing them makes a player who has not played look
like a player who played badly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MINUTES_PER_MATCH = 90

# Below this, per-90 figures are noise: one lucky cameo turns into a
# spectacular-looking rate. Used for flagging, not filtering -- the caller
# decides what to do about it.
LOW_MINUTES_THRESHOLD = 270


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Element-wise division where dividing by zero yields NaN, not an error."""
    denominator = denominator.replace(0, np.nan)
    return numerator / denominator


def per_90(df: pd.DataFrame, column: str, minutes_column: str = "minutes") -> pd.Series:
    """Rate of ``column`` per 90 minutes played."""
    return _safe_divide(df[column] * MINUTES_PER_MATCH, df[minutes_column])


def add_per_90_columns(
    df: pd.DataFrame, columns: list[str], minutes_column: str = "minutes"
) -> pd.DataFrame:
    """Add a ``<column>_per_90`` for each requested column that is present."""
    df = df.copy()
    for column in columns:
        if column in df.columns:
            df[f"{column}_per_90"] = per_90(df, column, minutes_column)
    return df


def points_per_million(df: pd.DataFrame) -> pd.Series:
    """Total points per £m of price -- the crude but useful value measure.

    Biased towards cheap players by construction: a 4.0m defender only needs
    40 points to look better than a 12.0m forward on 120. Read it alongside
    raw points, never instead of them.
    """
    return _safe_divide(df["total_points"], df["price"])


def minutes_share(df: pd.DataFrame, team_matches_played: pd.Series) -> pd.Series:
    """Fraction of available minutes a player has been on the pitch for.

    ``team_matches_played`` is indexed the same as ``df`` and holds how many
    matches that player's *team* has played -- which is not the gameweek
    number, because of blanks, doubles and postponements.

    This is the single best cheap proxy for whether a player is a starter, and
    it is what separates a genuine 6-points-per-90 asset from a substitute who
    will never accumulate.
    """
    return _safe_divide(df["minutes"], team_matches_played * MINUTES_PER_MATCH)


def team_matches_played(schedule: pd.DataFrame) -> pd.Series:
    """Matches each team has actually completed, indexed by team id."""
    if schedule.empty:
        return pd.Series(dtype="int64")
    finished = schedule[schedule["finished"]]
    return finished.groupby("team").size()


def add_scouting_metrics(
    players: pd.DataFrame, schedule: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Add the standard derived columns used across the scouting views.

    Adds per-90 rates for the counting stats, ``points_per_million``,
    ``minutes_share`` (when a schedule is supplied) and a ``low_minutes`` flag
    so views can mark rates that are not yet trustworthy.
    """
    rate_columns = [
        "total_points",
        "goals_scored",
        "assists",
        "expected_goals",
        "expected_assists",
        "expected_goal_involvements",
        "bonus",
    ]
    df = add_per_90_columns(players, rate_columns)
    df["points_per_million"] = points_per_million(df)
    df["low_minutes"] = df["minutes"] < LOW_MINUTES_THRESHOLD

    if schedule is not None and not schedule.empty:
        played = team_matches_played(schedule)
        df["minutes_share"] = minutes_share(df, df["team"].map(played))
    else:
        df["minutes_share"] = np.nan

    return df
