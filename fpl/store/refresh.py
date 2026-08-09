"""When each dataset is refreshed, and why that cadence.

Refresh policy was previously spread across three places: a cron line in a
workflow, ``@st.cache_data(ttl=...)`` decorators in the app, and a
``max_age_seconds`` argument at each cache call site. Nothing stated the
*reason* for any of it, so nobody could tell a considered choice from a
default.

The cadences here follow from three measured facts:

**Deadlines are early afternoon.** Measured across the 38 gameweeks of
2026-27: the earliest is 12:30 UTC, 29 of 38 fall between 12:00 and 14:00, and
the latest is 18:30. A capture at 06:00 is therefore 6.5 hours stale by the
time the earliest deadline passes — and the hours immediately before a deadline
are exactly when team news lands.

**Prices move overnight, once.** A single daily capture is enough to
reconstruct the price series; more would add rows without adding information.

**Past seasons never change.** Archive data is immutable once a season ends,
so re-fetching it is pure waste. It is the one dataset with no expiry.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

HOUR = 3600
DAY = 24 * HOUR


class Cadence(Enum):
    """How often a dataset is refreshed."""

    ON_DEMAND = "on demand"
    """Fetched when the app asks, subject to a cache TTL."""

    DAILY = "daily"
    """Captured once a day by the scheduled workflow."""

    PRE_DEADLINE = "pre-deadline"
    """Captured again shortly before the gameweek deadline."""

    IMMUTABLE = "immutable"
    """Never changes once written; fetch once and keep."""


# The scheduled runs, in UTC. Both are daily; the second exists because the
# first is hours stale by the time a deadline arrives.
MORNING_RUN_UTC = "06:00"
PRE_DEADLINE_RUN_UTC = "11:30"

# Measured from the 2026-27 fixture list.
EARLIEST_DEADLINE_UTC = "12:30"
LATEST_DEADLINE_UTC = "18:30"


@dataclass(frozen=True)
class RefreshPolicy:
    """How one dataset stays current."""

    dataset: str
    cadence: Cadence
    max_age_seconds: float
    reason: str
    orchestrated_by: str

    @property
    def max_age_hours(self) -> float:
        return self.max_age_seconds / HOUR


POLICIES: tuple[RefreshPolicy, ...] = (
    RefreshPolicy(
        dataset="bootstrap (players, teams, events)",
        cadence=Cadence.ON_DEMAND,
        max_age_seconds=1 * HOUR,
        reason=(
            "prices and injury news change through the day; an hour bounds how "
            "stale the app can be without refetching on every widget click"
        ),
        orchestrated_by="app/data.py @st.cache_data",
    ),
    RefreshPolicy(
        dataset="fixtures",
        cadence=Cadence.ON_DEMAND,
        max_age_seconds=1 * HOUR,
        reason=(
            "the schedule is published months ahead and changes rarely, but it "
            "shares a cache with bootstrap and the cost of an hour is nothing"
        ),
        orchestrated_by="app/data.py @st.cache_data",
    ),
    RefreshPolicy(
        dataset="odds",
        cadence=Cadence.ON_DEMAND,
        max_age_seconds=6 * HOUR,
        reason=(
            "the free tier allows 500 requests a month, about 16 a day across a "
            "season; odds barely move outside the hours before kick-off, and "
            "losing the source to quota exhaustion is worse than stale prices"
        ),
        orchestrated_by="fpl/config.py ODDS_CACHE_SECONDS",
    ),
    RefreshPolicy(
        dataset="archive (past seasons)",
        cadence=Cadence.IMMUTABLE,
        max_age_seconds=float("inf"),
        reason="a finished season cannot change; refetching it is pure waste",
        orchestrated_by="fpl/store/cache.py NEVER_STALE",
    ),
    RefreshPolicy(
        dataset="daily signals",
        cadence=Cadence.DAILY,
        max_age_seconds=1 * DAY,
        reason=(
            "injury status, set-piece duty, price and ownership are published "
            "only for now and never retrospectively; one capture a day is the "
            "record that will exist of them, and a second would overwrite the "
            "morning's news with the afternoon's"
        ),
        orchestrated_by=".github/workflows/snapshot.yml (06:00 UTC)",
    ),
    RefreshPolicy(
        dataset="gameweek snapshot",
        cadence=Cadence.PRE_DEADLINE,
        max_age_seconds=12 * HOUR,
        reason=(
            "a point-in-time backtest replays the state as it was at the "
            "deadline, so this must be captured as close to it as possible; "
            "the 06:00 run alone is 6.5 hours stale by the earliest deadline"
        ),
        orchestrated_by=".github/workflows/snapshot.yml (06:00 and 11:30 UTC)",
    ),
)


def by_dataset() -> dict[str, RefreshPolicy]:
    return {policy.dataset: policy for policy in POLICIES}


def by_cadence(cadence: Cadence) -> tuple[RefreshPolicy, ...]:
    return tuple(policy for policy in POLICIES if policy.cadence is cadence)


def scheduled_runs() -> tuple[str, ...]:
    """The times the workflow fires, in UTC."""
    return (MORNING_RUN_UTC, PRE_DEADLINE_RUN_UTC)
