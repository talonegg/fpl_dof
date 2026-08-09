"""Cached data loading for the UI.

The only place in ``app/`` allowed to know about Streamlit's cache. Everything
here is a thin wrapper around ``fpl/`` -- no calculations.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from fpl.domain.fixtures import build_team_schedule, next_gameweek
from fpl.domain.identity import match_to_current_players
from fpl.domain.players import build_players_frame
from fpl.features.availability import add_availability
from fpl.features.rates import add_scouting_metrics
from fpl.sources.archive import fetch_season_gameweeks
from fpl.sources.fpl_api import fetch_bootstrap, fetch_fixtures

# The most recent completed season, used for history until the current one has
# gameweeks of its own.
ARCHIVE_SEASON = "2025-26"

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


@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner="Adding scouting metrics...")
def load_scouting_players() -> pd.DataFrame:
    """The player table with rates, value, minutes share and availability."""
    return add_availability(add_scouting_metrics(load_players(), load_schedule()))


# Past seasons never change, so this is cached for the life of the process
# rather than on a clock.
@st.cache_data(show_spinner="Loading last season's gameweek history...")
def load_archive_history(season: str = ARCHIVE_SEASON) -> pd.DataFrame:
    """Last completed season's per-gameweek rows, keyed to current player ids.

    Players who have left the league keep a null ``current_element`` rather
    than being dropped, so counts stay honest.
    """
    archive = fetch_season_gameweeks(season)
    return match_to_current_players(archive, load_players())
