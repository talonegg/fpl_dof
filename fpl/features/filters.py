"""Selecting a subset of players.

Pure and shared, because the same selection has to drive every view. When the
filtering lived inside one page, that page was the only thing it could affect,
and a sidebar control that silently applies to one tab out of three is worse
than no control at all.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class PlayerFilter:
    """What the user has narrowed the player list down to.

    ``None`` means "no constraint", which is deliberately different from an
    empty sequence: an empty list of positions means the user has deselected
    everything and should see nothing, not everything.
    """

    positions: tuple[str, ...] | None = None
    teams: tuple[str, ...] | None = None
    price_range: tuple[float, float] | None = None

    @property
    def is_empty(self) -> bool:
        """Whether this filter can only ever match nothing."""
        return self.positions == () or self.teams == ()

    def describe(self) -> str:
        """A short human-readable summary, for showing on pages the filter affects."""
        parts = []
        if self.positions is not None:
            parts.append(f"{len(self.positions)} position(s)")
        if self.teams is not None:
            parts.append(f"{len(self.teams)} team(s)")
        if self.price_range is not None:
            low, high = self.price_range
            parts.append(f"£{low:.1f}m–£{high:.1f}m")
        return ", ".join(parts) if parts else "no filters"


def apply_filter(players: pd.DataFrame, player_filter: PlayerFilter) -> pd.DataFrame:
    """Return the rows of ``players`` matching ``player_filter``."""
    if players.empty:
        return players

    mask = pd.Series(True, index=players.index)

    if player_filter.positions is not None and "position" in players.columns:
        mask &= players["position"].isin(list(player_filter.positions))

    if player_filter.teams is not None and "team_name" in players.columns:
        mask &= players["team_name"].isin(list(player_filter.teams))

    if player_filter.price_range is not None and "price" in players.columns:
        low, high = player_filter.price_range
        mask &= players["price"].between(low, high)

    return players[mask]


def price_bounds(players: pd.DataFrame) -> tuple[float, float] | None:
    """Lowest and highest price present, or ``None`` when there is no range."""
    if players.empty or "price" not in players.columns:
        return None
    prices = players["price"].dropna()
    if prices.empty:
        return None
    return float(prices.min()), float(prices.max())


def options(players: pd.DataFrame, column: str) -> list[str]:
    """Sorted unique values of ``column``, for populating a filter control."""
    if players.empty or column not in players.columns:
        return []
    return sorted(players[column].dropna().unique().tolist())


def restrict_to_available(values: Iterable[str], available: Iterable[str]) -> list[str]:
    """Keep only values that still exist in ``available``, preserving order.

    A selection can outlive the data behind it -- a team filter chosen before a
    reload, say -- and passing a stale value to a widget raises.
    """
    available = set(available)
    return [value for value in values if value in available]
