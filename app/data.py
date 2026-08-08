"""Cached data loading for the UI.

The only place in ``app/`` allowed to know about Streamlit's cache. Everything
here is a thin wrapper around ``fpl/`` -- no calculations.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from fpl.domain.players import build_players_frame
from fpl.sources.fpl_api import fetch_bootstrap

# The FPL API only changes meaningfully at gameweek deadlines and price
# changes. An hour is a safe compromise; without this the app refetched on
# every widget interaction.
CACHE_TTL_SECONDS = 3600


@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner="Loading FPL data...")
def load_players() -> pd.DataFrame:
    """Load the canonical player table, cached for an hour."""
    return build_players_frame(fetch_bootstrap())
