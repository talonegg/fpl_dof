"""Build the canonical player table from a raw ``bootstrap-static`` payload."""

from __future__ import annotations

from typing import Any

import pandas as pd

from fpl.domain.reference import add_readable_columns

# The FPL API returns these as JSON strings ("3.5", "12.7") rather than numbers.
# Left alone they sort and aggregate lexicographically, which silently produces
# wrong answers -- "9.0" sorts above "12.0". Coerce them once, here.
NUMERIC_STRING_COLUMNS = (
    "form",
    "points_per_game",
    "selected_by_percent",
    "value_form",
    "value_season",
    "ep_this",
    "ep_next",
    "expected_goals",
    "expected_assists",
    "expected_goal_involvements",
    "expected_goals_conceded",
    "expected_goals_per_90",
    "expected_assists_per_90",
    "expected_goal_involvements_per_90",
    "expected_goals_conceded_per_90",
    "influence",
    "creativity",
    "threat",
    "ict_index",
    "starts_per_90",
    "clean_sheets_per_90",
    "saves_per_90",
    "defensive_contribution_per_90",
)


def coerce_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with known numeric-as-string columns converted to floats.

    Unparseable values become NaN rather than raising, so a single malformed
    row from the API cannot take the whole app down.
    """
    df = df.copy()
    for column in NUMERIC_STRING_COLUMNS:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    return df


def build_players_frame(bootstrap: dict[str, Any]) -> pd.DataFrame:
    """Turn a raw ``bootstrap-static`` payload into the canonical player table.

    Adds readable team/position/price columns and normalises numeric types.
    This is the boundary where raw API shapes stop and domain types begin --
    nothing downstream should touch ``bootstrap["elements"]`` directly.

    Also aliases ``id`` to ``element``. Bootstrap calls the player id ``id``,
    while every other endpoint and the archive call the same number
    ``element``; carrying both names means joins do not need to remember which
    source a frame came from.
    """
    df = pd.DataFrame(bootstrap["elements"])
    df = add_readable_columns(
        df,
        teams=bootstrap["teams"],
        element_types=bootstrap["element_types"],
    )
    df = coerce_numeric_columns(df)
    if "element" not in df.columns and "id" in df.columns:
        df["element"] = df["id"]
    return df
