"""Cached data loading for the UI.

The only place in ``app/`` allowed to know about Streamlit's cache. Everything
here is a thin wrapper around ``fpl/`` -- no calculations.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from fpl.domain.fixtures import build_team_schedule, next_gameweek
from fpl.domain.identity import add_match_key, match_to_current_players
from fpl.domain.players import build_players_frame
from fpl.features.preseason_pool import build_pool
from fpl.features.registry import enrich
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
    """The player table with every applicable derivation attached.

    Goes through the catalogue rather than chaining the derivations by hand,
    so adding one is an entry in ``fpl/features/registry.py`` rather than an
    edit here that is easy to forget.
    """
    result = enrich(load_players(), rates={"schedule": load_schedule()})
    return result.frame


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


# The season being picked for. Its squad is chosen from the seasons behind it.
UPCOMING_SEASON = "2026-27"

# Seasons behind the one being picked. Four is what the blend weights are
# calibrated for; older seasons carry almost no weight and cost a download.
PRESEASON_PRIOR_SEASONS = ("2022-23", "2023-24", "2024-25", "2025-26")


@st.cache_data(show_spinner="Loading prior seasons...")
def load_prior_seasons(seasons: tuple[str, ...] = PRESEASON_PRIOR_SEASONS) -> dict:
    """Per-gameweek rows for each completed season behind the current one.

    A season that fails to download is **omitted and reported**, never silently
    skipped — dropping one quietly changes every blended rate downstream, which
    has happened here before and was invisible until the numbers were audited.
    """
    loaded: dict[str, pd.DataFrame] = {}
    failures: list[str] = []
    for season in seasons:
        try:
            data = fetch_season_gameweeks(season)
        except Exception:  # noqa: BLE001 - the reason varies; the reporting does not
            failures.append(season)
            continue
        if data.empty:
            failures.append(season)
            continue
        loaded[season] = data

    loaded["__failures__"] = failures  # type: ignore[assignment]
    return loaded


@st.cache_data(show_spinner="Building the candidate pool...")
def load_preseason_pool() -> pd.DataFrame:
    """Every currently priced player, with the history behind them attached.

    Prices come from the live API — they are this season's real opening prices
    — while rates come from the completed seasons behind it.

    The name join is the fragile part and is done the same way as everywhere
    else: full name rather than ``web_name``, because the archive carries full
    names and ``web_name`` is a display abbreviation that collides.
    """
    prior = dict(load_prior_seasons())
    prior.pop("__failures__", None)
    if not prior:
        return pd.DataFrame()

    players = load_players()
    if players.empty:
        return pd.DataFrame()

    prices = pd.DataFrame(
        {
            "element": players["element"],
            "player_name": players["first_name"].fillna("")
            + " "
            + players["second_name"].fillna(""),
            # Already the long names the optimiser keys on -- bootstrap's
            # element_types carry "Goalkeeper", not "GKP".
            "position": players["position"],
            "team": players["team_name"],
            "price": players["price"],
        }
    )
    return build_pool(prior, add_match_key(prices, "player_name"))
