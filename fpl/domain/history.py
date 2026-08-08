"""Per-player history, from ``element-summary``.

Two different shapes come back from that endpoint and they are easy to
confuse:

``history``
    One row per gameweek *this* season. This is the modelling substrate --
    but it is empty until a player has actually played, so it yields nothing
    before the season opens.

``history_past``
    One row per *previous season*, aggregated. Useful as a prior on a player's
    ceiling, useless for gameweek-level modelling.

Per-gameweek data for past seasons comes from :mod:`fpl.sources.archive`.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from fpl.domain.players import coerce_numeric_columns

GAMEWEEK_COLUMNS = [
    "element",
    "round",
    "fixture",
    "opponent_team",
    "was_home",
    "kickoff_time",
    "minutes",
    "total_points",
    "goals_scored",
    "assists",
    "clean_sheets",
    "goals_conceded",
    "bonus",
    "bps",
    "value",
    "selected",
]


def build_gameweek_history(summary: dict[str, Any]) -> pd.DataFrame:
    """Per-gameweek rows for one player from an ``element-summary`` payload.

    Adds a ``price`` column (``value`` is in integer tenths, like ``now_cost``)
    and a ``gameweek`` alias for ``round``, so the column means the same thing
    here as everywhere else in the codebase.

    Returns an empty frame with the expected columns when the player has not
    played yet, so callers can concatenate without special-casing.
    """
    history = summary.get("history", [])
    if not history:
        return pd.DataFrame(columns=[*GAMEWEEK_COLUMNS, "gameweek", "price"])

    df = pd.DataFrame(history)
    df = coerce_numeric_columns(df)
    df["gameweek"] = df["round"].astype(int)
    if "value" in df.columns:
        df["price"] = df["value"] / 10
    return df.sort_values("gameweek").reset_index(drop=True)


def build_past_seasons(summary: dict[str, Any]) -> pd.DataFrame:
    """One row per previous season for a player, as a coarse prior.

    Note the identifier here is ``element_code``, not ``element``: element ids
    are reassigned every season, codes are not.
    """
    past = summary.get("history_past", [])
    if not past:
        return pd.DataFrame(columns=["element_code", "season_name", "total_points"])

    df = coerce_numeric_columns(pd.DataFrame(past))
    for column, source in (("start_price", "start_cost"), ("end_price", "end_cost")):
        if source in df.columns:
            df[column] = df[source] / 10
    return df.sort_values("season_name").reset_index(drop=True)
