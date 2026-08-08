"""Entry point for the FPL DOF app.

Deliberately lives at the repository root: Streamlit puts the entry script's
directory on ``sys.path``, so this is what makes ``import fpl`` work both
locally and on Streamlit Community Cloud without an editable install.
"""

import os
from pathlib import Path

import streamlit as st

from app import filters_view, fixtures_view, players_view, scouting_view
from app.data import (
    ARCHIVE_SEASON,
    load_archive_history,
    load_next_gameweek,
    load_schedule,
    load_scouting_players,
)
from fpl.features import watchlist
from fpl.features.filters import apply_filter

# Overridable so tests can redirect it, and because a hosted deployment has an
# ephemeral filesystem and may want to point this elsewhere.
DATA_DIR = Path(os.environ.get("FPL_DOF_DATA_DIR", Path(__file__).resolve().parent / "data"))
WATCHLIST_PATH = DATA_DIR / "watchlist.json"

st.set_page_config(
    page_title="FPL DOF",
    page_icon="⚽",
    layout="wide",
    # "auto" keeps the sidebar open on laptops but collapsed on phones.
    initial_sidebar_state="auto",
)

players = load_scouting_players()

if "watchlist" not in st.session_state:
    st.session_state.watchlist = watchlist.load(WATCHLIST_PATH)

# Filters are rendered before the tabs so one set of controls drives them all.
player_filter = filters_view.render(players)
filtered_players = apply_filter(players, player_filter)

# Tabs rather than columns: they stack cleanly on a phone.
players_tab, scouting_tab, fixtures_tab = st.tabs(["Players", "Scouting", "Fixtures"])

with players_tab:
    players_view.render(filtered_players, total=len(players))

with scouting_tab:
    st.caption(filters_view.caption(player_filter, len(filtered_players), len(players)))
    history = load_archive_history()
    scouting_view.render_detail(filtered_players, history, season_label=ARCHIVE_SEASON)
    st.divider()
    scouting_view.render_comparison(filtered_players, history, season_label=ARCHIVE_SEASON)

with fixtures_tab:
    gameweek = load_next_gameweek()
    if gameweek is None:
        st.info("The season is over — no upcoming fixtures.")
    else:
        # Only the club filter reaches here; a fixture has no position or price.
        schedule = filters_view.filter_schedule(load_schedule(), player_filter)
        if schedule.empty:
            st.warning("No fixtures for the selected clubs.")
        else:
            fixtures_view.render(schedule, from_gameweek=gameweek)

with st.sidebar:
    st.header("Watchlist")
    watched = watchlist.filter_players(players, st.session_state.watchlist)
    labels = {row.element: f"{row.web_name} ({row.team_name})" for row in players.itertuples()}

    chosen = st.multiselect(
        "Players to keep an eye on",
        options=list(labels),
        default=watched["element"].tolist(),
        format_func=lambda value: labels[value],
        key="watchlist_select",
    )
    codes = players[players["element"].isin(chosen)]["code"].tolist()
    if codes != st.session_state.watchlist:
        st.session_state.watchlist = codes
        watchlist.save(WATCHLIST_PATH, codes)

    if watched.empty:
        st.caption("Nothing watched yet.")
    else:
        st.dataframe(
            watched[["web_name", "team_name", "price", "form"]],
            width="stretch",
            hide_index=True,
        )
