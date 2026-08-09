"""The watchlist sidebar.

Previously twenty lines of state handling inline in the entry point, which is
the one place that should contain no logic at all.

Deliberately not filtered. A watchlist is a standing list of players you have
chosen to follow, and hiding an entry because of a price slider would defeat
the purpose — you watch a player precisely so you notice when their situation
changes.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from fpl.features import watchlist

SESSION_KEY = "watchlist"
WIDGET_KEY = "watchlist_select"
DISPLAY_COLUMNS = ["web_name", "team_name", "price", "form"]


def initialise(path: Path) -> None:
    """Load the persisted watchlist into session state once per session."""
    if SESSION_KEY not in st.session_state:
        st.session_state[SESSION_KEY] = watchlist.load(path)


def render(players: pd.DataFrame, path: Path) -> None:
    """Render the picker and the watched players, persisting any change."""
    st.header("Watchlist")

    if players.empty:
        st.caption("No players available.")
        return

    watched = watchlist.filter_players(players, st.session_state[SESSION_KEY])
    labels = {row.element: f"{row.web_name} ({row.team_name})" for row in players.itertuples()}

    chosen = st.multiselect(
        "Players to keep an eye on",
        options=list(labels),
        default=watched["element"].tolist(),
        format_func=lambda value: labels[value],
        key=WIDGET_KEY,
    )

    # Compare as sets: the widget returns codes in element order while the
    # stored list is sorted, so a list comparison differs on the first render
    # of every session and rewrites a file the user never touched.
    codes = players[players["element"].isin(chosen)]["code"].tolist()
    if set(codes) != set(st.session_state[SESSION_KEY]):
        st.session_state[SESSION_KEY] = codes
        watchlist.save(path, codes)

    if watched.empty:
        st.caption("Nothing watched yet.")
        return

    columns = [column for column in DISPLAY_COLUMNS if column in watched.columns]
    st.dataframe(watched[columns], width="stretch", hide_index=True)
