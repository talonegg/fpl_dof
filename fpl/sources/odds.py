"""Betting odds from The Odds API.

The market is the strongest freely available prior on football outcomes. It
aggregates far more information than any model here will, and it is priced by
people with money at stake — which is a considerably harder test than a
backtest.

Two things make it usable rather than merely interesting:

**Odds are not probabilities.** A bookmaker's prices sum to more than 1; the
excess is their margin. Converting naively gives probabilities that are all
too high, in a way that looks plausible and is systematically wrong. See
:mod:`fpl.features.market` for the de-vigging.

**Quota is the binding constraint.** The free tier is 500 requests a month —
about 16 a day for a whole season. Everything here is built around fetching
rarely and caching hard, because losing the source is worse than having
slightly stale odds.

This module only fetches and flattens. It deliberately knows nothing about
FPL: no players, no gameweeks, no scoring.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import pandas as pd

from fpl.config import Config, load_config
from fpl.sources.base import RateLimiter, SourceResult, guarded

BASE_URL = "https://api.the-odds-api.com/v4"
SPORT_KEY = "soccer_epl"

# Head-to-head is the market every bookmaker prices and the free tier includes.
# Totals give the goals expectation a clean-sheet prior needs.
MARKET_H2H = "h2h"
MARKET_TOTALS = "totals"
DEFAULT_MARKETS = (MARKET_H2H, MARKET_TOTALS)

DEFAULT_REGION = "uk"
DRAW = "Draw"

# One request every few seconds is far below anything that would trouble the
# service, and quota runs out long before rate limits do.
MIN_REQUEST_INTERVAL_SECONDS = 2.0

Fetcher = Callable[[str, dict[str, Any]], Any]


def http_fetcher(url: str, params: dict[str, Any]) -> Any:
    """Fetch JSON, keeping the API key out of any raised message."""
    import requests

    response = requests.get(url, params=params, timeout=20)
    if response.status_code == 401:
        raise PermissionError("odds API rejected the key (401)")
    if response.status_code == 429:
        raise RuntimeError("odds API quota or rate limit exhausted (429)")
    response.raise_for_status()
    return response.json()


@dataclass
class OddsSource:
    """Match odds for the Premier League."""

    config: Config | None = None
    fetcher: Fetcher = http_fetcher
    markets: tuple[str, ...] = DEFAULT_MARKETS
    region: str = DEFAULT_REGION
    name: str = "odds"

    def __post_init__(self):
        self.config = self.config or load_config()
        self._limiter = RateLimiter(MIN_REQUEST_INTERVAL_SECONDS)

    @property
    def available(self) -> bool:
        """Whether a key is configured. Without one this source is simply off."""
        return bool(self.config and self.config.has_odds)

    def fetch(self) -> SourceResult:
        """Fetch current odds, returning failure as data rather than raising."""
        if not self.available:
            return SourceResult(
                name=self.name,
                error="no ODDS_API_KEY configured — odds are unavailable",
            )
        return guarded(self.name, self._fetch_frame)

    def _fetch_frame(self) -> pd.DataFrame:
        self._limiter.wait()
        payload = self.fetcher(
            f"{BASE_URL}/sports/{SPORT_KEY}/odds",
            {
                "apiKey": self.config.odds_api_key,
                "regions": self.region,
                "markets": ",".join(self.markets),
                "oddsFormat": "decimal",
            },
        )
        return flatten_odds(payload)


def flatten_odds(payload: Any) -> pd.DataFrame:
    """Turn the nested odds payload into one row per price.

    Columns: ``match_id``, ``commence_time``, ``home_team``, ``away_team``,
    ``bookmaker``, ``market``, ``outcome``, ``point``, ``price``.

    One row per (bookmaker, market, outcome) rather than per match, because
    de-vigging has to happen within a single bookmaker's book — mixing prices
    across bookmakers before removing the margin produces a number that is not
    a probability of anything.
    """
    if not isinstance(payload, list):
        return pd.DataFrame()

    rows = []
    for event in payload:
        if not isinstance(event, dict):
            continue
        for bookmaker in event.get("bookmakers") or []:
            for market in bookmaker.get("markets") or []:
                for outcome in market.get("outcomes") or []:
                    price = outcome.get("price")
                    if not price or price <= 1:
                        # A decimal price of 1 or less is not a real quote.
                        continue
                    rows.append(
                        {
                            "match_id": event.get("id"),
                            "commence_time": event.get("commence_time"),
                            "home_team": event.get("home_team"),
                            "away_team": event.get("away_team"),
                            "bookmaker": bookmaker.get("key"),
                            "market": market.get("key"),
                            "outcome": outcome.get("name"),
                            "point": outcome.get("point"),
                            "price": float(price),
                        }
                    )

    if not rows:
        return pd.DataFrame(
            columns=[
                "match_id",
                "commence_time",
                "home_team",
                "away_team",
                "bookmaker",
                "market",
                "outcome",
                "point",
                "price",
            ]
        )
    return pd.DataFrame(rows)
