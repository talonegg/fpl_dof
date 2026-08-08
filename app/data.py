"""Cached data loading for the UI.

The only place in ``app/`` allowed to know about Streamlit's cache. Everything
here is a thin wrapper around ``fpl/`` -- no calculations.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from fpl.domain.fixtures import build_team_schedule, next_gameweek
from fpl.domain.players import build_players_frame
from fpl.sources.fpl_api import fetch_bootstrap, fetch_fixtures

# The FPL API only changes meaningfully at gameweek deadlines and price
# changes. An hour is a safe compromise; without this the app refetched on
# every widget interaction.
CACHE_TTL_SECONDS = 3600


@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner="Loading FPL data...")
def load_players() -> pd.DataFrame:
    """Load the canonical player table, cached for an hour."""
    return build_players_frame(fetch_bootstrap())


@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner="Loading fixtures...")
def load_schedule() -> pd.DataFrame:
    """Load the team-perspective fixture schedule, cached for an hour."""
    return build_team_schedule(fetch_fixtures(), fetch_bootstrap()["teams"])


@st.cache_data(ttl=CACHE_TTL_SECONDS)
def load_next_gameweek() -> int | None:
    """The gameweek currently accepting transfers."""
    return next_gameweek(fetch_bootstrap()["events"])
