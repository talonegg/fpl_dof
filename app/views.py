"""The catalogue of tabs, and what each one is given.

Tabs used to be hard-coded in ``streamlit_app.py``: adding one meant editing
the entry point and *remembering* to apply the filters, add the caption and
handle the empty case. `CLAUDE.md` says every new tab must consume the global
filter or say why it cannot — which was a convention with nothing enforcing it.

A view now declares which filters it honours, and the context it is handed is
already filtered accordingly. Forgetting is no longer possible, because the
view never sees the unfiltered frame unless it asks for it.

The shapes differ for a reason. Players and Scouting are player-shaped, so
club, position, price and availability all apply. Fixtures is team-shaped: a
fixture has no position, price or injury, so only the club filter reaches it.
That asymmetry is data, declared per view, rather than four lines of special
casing at the call site.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field, replace

import pandas as pd

from fpl.features.filters import PlayerFilter, apply_filter


@dataclass
class ViewContext:
    """Everything a tab may need, already filtered as that tab declared.

    Loaders are passed as callables rather than data so a tab that is never
    opened does not pay to fetch a season of history. Streamlit renders every
    tab's body regardless of which is selected, so this matters less than it
    would elsewhere — but it keeps the cost at the point of use.
    """

    players: pd.DataFrame
    """Filtered according to the view's declaration."""

    all_players: pd.DataFrame
    """Unfiltered, for "showing X of Y" counts and reference tables."""

    player_filter: PlayerFilter
    schedule: Callable[[], pd.DataFrame]
    history: Callable[[], pd.DataFrame]
    next_gameweek: Callable[[], int | None]
    season_label: str = ""

    @property
    def filtered_count(self) -> int:
        return len(self.players)

    @property
    def total_count(self) -> int:
        return len(self.all_players)


@dataclass(frozen=True)
class View:
    """One tab: a label, a renderer, and which filters it honours."""

    name: str
    render: Callable[[ViewContext], None]

    # Which parts of the global filter apply. A view that cannot honour one
    # says so here rather than silently ignoring it.
    honours_club: bool = True
    honours_position: bool = True
    honours_price: bool = True
    honours_availability: bool = True

    note: str = ""

    @property
    def ignored_filters(self) -> tuple[str, ...]:
        """Filters this view cannot apply, for documenting and testing."""
        ignored = []
        if not self.honours_club:
            ignored.append("club")
        if not self.honours_position:
            ignored.append("position")
        if not self.honours_price:
            ignored.append("price")
        if not self.honours_availability:
            ignored.append("availability")
        return tuple(ignored)

    def filter_for(self, player_filter: PlayerFilter) -> PlayerFilter:
        """The filter as this view honours it, with the rest neutralised."""
        return replace(
            player_filter,
            teams=player_filter.teams if self.honours_club else None,
            positions=player_filter.positions if self.honours_position else None,
            price_range=player_filter.price_range if self.honours_price else None,
            availability_bands=(
                player_filter.availability_bands if self.honours_availability else None
            ),
        )

    def context(self, base: ViewContext) -> ViewContext:
        """A context whose ``players`` respect exactly this view's declaration."""
        return replace(
            base,
            players=apply_filter(base.all_players, self.filter_for(base.player_filter)),
        )


@dataclass
class Registry:
    """The ordered set of tabs the app renders."""

    views: list[View] = field(default_factory=list)

    def add(self, view: View) -> View:
        if any(existing.name == view.name for existing in self.views):
            raise ValueError(f"a view named {view.name!r} is already registered")
        self.views.append(view)
        return view

    def names(self) -> list[str]:
        return [view.name for view in self.views]

    def __iter__(self):
        return iter(self.views)

    def __len__(self) -> int:
        return len(self.views)
