"""Access to the official Fantasy Premier League API.

Every function takes an injectable ``fetcher`` so that tests can supply frozen
JSON snapshots instead of hitting the network. Production callers use the
default :func:`http_fetcher`.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import requests

BASE_URL = "https://fantasy.premierleague.com/api"

# The FPL API rejects some default client user agents, and identifying the
# client is basic good citizenship for an unofficial API.
USER_AGENT = "fpl-dof/0.1 (personal FPL analytics tool)"

REQUEST_TIMEOUT_SECONDS = 20

Fetcher = Callable[[str], Any]


def http_fetcher(url: str) -> Any:
    """Fetch and decode JSON from ``url`` over HTTP."""
    response = requests.get(
        url,
        headers={"User-Agent": USER_AGENT},
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    return response.json()


def fetch_bootstrap(fetcher: Fetcher = http_fetcher) -> dict[str, Any]:
    """Fetch ``bootstrap-static``: players, teams, positions and gameweeks.

    This is the single endpoint carrying the current state of the whole game,
    so nearly everything downstream starts here.
    """
    return fetcher(f"{BASE_URL}/bootstrap-static/")


def fetch_fixtures(fetcher: Fetcher = http_fetcher) -> list[dict[str, Any]]:
    """Fetch all 380 fixtures for the season, played and unplayed."""
    return fetcher(f"{BASE_URL}/fixtures/")


def fetch_player_summary(element_id: int, fetcher: Fetcher = http_fetcher) -> dict[str, Any]:
    """Fetch one player's gameweek history, past seasons and remaining fixtures.

    Note ``history`` is empty until the player has actually played, so before
    the season starts this returns per-season summaries only. Historical
    per-gameweek data for modelling comes from :mod:`fpl.sources.archive`.
    """
    return fetcher(f"{BASE_URL}/element-summary/{element_id}/")
