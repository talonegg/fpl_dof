"""Configuration, read from the environment.

Keys live in environment variables and never in the repository. On Streamlit
Community Cloud they are set as secrets; locally they come from the shell or a
gitignored ``.env``.

Nothing here raises on import. A missing key means the source that needs it is
unavailable, which the app should degrade around rather than refuse to start
for -- one unconfigured external source must not take the whole app down.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

ODDS_API_KEY_VAR = "ODDS_API_KEY"

# Free tiers are small (The Odds API allows 500 requests a month), so the
# default cache is deliberately long: odds barely move outside the hours
# before kick-off, and burning quota on refreshes is the easiest way to lose
# access to the source entirely.
DEFAULT_ODDS_CACHE_SECONDS = 6 * 3600


@dataclass(frozen=True)
class Config:
    """Everything the app reads from its environment."""

    odds_api_key: str | None = None
    odds_cache_seconds: int = DEFAULT_ODDS_CACHE_SECONDS

    @property
    def has_odds(self) -> bool:
        """Whether the betting-odds source can be used at all."""
        return bool(self.odds_api_key)


def load_config(environment: dict[str, str] | None = None) -> Config:
    """Read configuration from ``environment`` (defaults to ``os.environ``)."""
    env = os.environ if environment is None else environment

    raw_ttl = env.get("ODDS_CACHE_SECONDS", "")
    try:
        cache_seconds = int(raw_ttl) if raw_ttl else DEFAULT_ODDS_CACHE_SECONDS
    except ValueError:
        # A malformed value should not be silently treated as "no caching",
        # which would drain a free-tier quota in minutes.
        cache_seconds = DEFAULT_ODDS_CACHE_SECONDS

    key = env.get(ODDS_API_KEY_VAR, "").strip()
    return Config(odds_api_key=key or None, odds_cache_seconds=cache_seconds)


def redact(value: str | None, keep: int = 4) -> str:
    """Render a secret safely for logs and error messages."""
    if not value:
        return "<unset>"
    if len(value) <= keep:
        return "*" * len(value)
    return f"{'*' * (len(value) - keep)}{value[-keep:]}"
