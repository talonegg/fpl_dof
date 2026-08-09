"""Entry point for the FPL DOF app.

Deliberately lives at the repository root: Streamlit puts the entry script's
directory on ``sys.path``, so this is what makes ``import fpl`` work both
locally and on Streamlit Community Cloud without an editable install.

This file composes; it does not decide. Tabs come from ``app/registry.py`` and
each one declares which filters it honours, so adding a feature does not touch
this file.
"""

import os
from dataclasses import replace
from pathlib import Path

import streamlit as st

from app import filters_view, scouting_view, watchlist_view
from app.data import (
    ARCHIVE_SEASON,
    load_archive_history,
    load_next_gameweek,
    load_schedule,
    load_scouting_players,
)
from app.registry import build_registry
from app.views import ViewContext
from fpl.features.filters import apply_filter

DATA_DIR = Path(os.environ.get("FPL_DOF_DATA_DIR", "data"))
WATCHLIST_PATH = DATA_DIR / "watchlist.json"

st.set_page_config(
    page_title="FPL DOF",
    page_icon="⚽",
    layout="wide",
    # "auto" keeps the sidebar open on laptops but collapsed on phones.
    initial_sidebar_state="auto",
)

players = load_scouting_players()
watchlist_view.initialise(WATCHLIST_PATH)

# Filters are rendered before the tabs so one set of controls drives them all.
player_filter = filters_view.render(players)

context = ViewContext(
    players=players,
    all_players=players,
    player_filter=player_filter,
    schedule=load_schedule,
    history=load_archive_history,
    next_gameweek=load_next_gameweek,
    season_label=ARCHIVE_SEASON,
)

# Tabs rather than columns: they stack cleanly on a phone.
registry = build_registry()
for tab, view in zip(st.tabs(registry.names()), registry, strict=True):
    with tab:
        view.render(view.context(context))

with st.sidebar:
    watchlist_view.render(players, WATCHLIST_PATH)

# Below the tabs, so it is on every page: the availability table is what you
# check when a filter has hidden someone, and it is reference rather than
# analysis. It honours club, position and price but not the availability band --
# filtering it by that control would empty it precisely when it is needed.
st.divider()
scouting_view.render_availability(
    apply_filter(players, replace(player_filter, availability_bands=None))
)
