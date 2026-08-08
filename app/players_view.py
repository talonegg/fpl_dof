"""The player explorer page: filter, choose columns, sort, display."""

from __future__ import annotations

import pandas as pd
import streamlit as st

MANDATORY_COLUMNS = ["web_name", "team_name", "position", "price"]

OPTIONAL_COLUMNS = [
    "total_points",
    "form",
    "points_per_game",
    "selected_by_percent",
    "minutes",
    "goals_scored",
    "assists",
    "clean_sheets",
    "expected_goals",
    "expected_assists",
    "expected_goal_involvements",
    "bonus",
    "ict_index",
]

DEFAULT_COLUMNS = ["total_points", "form"]

COLUMN_LABELS = {
    "web_name": "Player",
    "team_name": "Team",
    "position": "Pos",
    "price": "£m",
    "total_points": "Pts",
    "points_per_game": "PPG",
    "selected_by_percent": "TSB%",
    "expected_goals": "xG",
    "expected_assists": "xA",
    "expected_goal_involvements": "xGI",
    "ict_index": "ICT",
    "goals_scored": "Goals",
    "clean_sheets": "CS",
}


def _column_controls() -> list[str]:
    """Render the column picker and return the columns to display."""
    st.sidebar.header("Columns")
    selected = st.sidebar.multiselect(
        "Additional columns", options=OPTIONAL_COLUMNS, default=DEFAULT_COLUMNS
    )
    return MANDATORY_COLUMNS + selected


def _sort_controls(df: pd.DataFrame, display_columns: list[str]) -> pd.DataFrame:
    """Render the multi-column sort controls and return the sorted frame."""
    st.sidebar.header("Sort")
    sort_columns = st.sidebar.multiselect(
        "Sort by (in priority order)", options=display_columns, default=["price"]
    )
    if not sort_columns:
        return df

    ascending = [
        st.sidebar.toggle(f"Ascending: {column}", value=False, key=f"sort_asc_{column}")
        for column in sort_columns
    ]
    return df.sort_values(by=sort_columns, ascending=ascending)


def render(players: pd.DataFrame, total: int | None = None) -> None:
    """Render the player explorer.

    ``players`` arrives already filtered by the shared sidebar controls;
    ``total`` is the unfiltered count, so the caption can say what was left out.
    """
    st.title("FPL Data Explorer")

    if players.empty:
        st.warning("No players match those filters.")
        return

    display_columns = _column_controls()
    sorted_players = _sort_controls(players, display_columns)

    total = len(players) if total is None else total
    st.caption(f"Showing {len(sorted_players)} of {total} players.")
    st.dataframe(
        sorted_players[display_columns].rename(columns=COLUMN_LABELS),
        width="stretch",
        hide_index=True,
    )
