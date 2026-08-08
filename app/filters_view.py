"""The sidebar filter controls, rendered once for the whole app.

Rendered before any tab so that a single set of controls drives all of them.
Which filter reaches which tab is a property of the data, not a preference:

- **Club** applies everywhere, including Fixtures, because a fixture belongs to
  a team.
- **Position and price** apply to the player-shaped tabs only. A fixture has
  neither, so offering them there would imply a filter that cannot do anything.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from fpl.features.filters import PlayerFilter, apply_filter, options, price_bounds

PRICE_STEP = 0.1


def render(players: pd.DataFrame) -> PlayerFilter:
    """Render the filter controls and return what the user selected."""
    st.sidebar.header("Filters")

    if players.empty:
        return PlayerFilter()

    positions = options(players, "position")
    selected_positions = st.sidebar.multiselect(
        "Position", options=positions, default=positions, key="filter_positions"
    )

    teams = options(players, "team_name")
    selected_teams = st.sidebar.multiselect(
        "Club",
        options=teams,
        default=teams,
        key="filter_teams",
        help="Applies to every tab, including Fixtures.",
    )

    # Price bounds come from what the other filters leave behind, so the slider
    # never spans a range with nothing in it.
    narrowed = apply_filter(
        players,
        PlayerFilter(positions=tuple(selected_positions), teams=tuple(selected_teams)),
    )
    bounds = price_bounds(narrowed)

    price_range = None
    if bounds is not None and bounds[0] != bounds[1]:
        price_range = st.sidebar.slider(
            "Price range (£m)",
            min_value=bounds[0],
            max_value=bounds[1],
            value=bounds,
            step=PRICE_STEP,
            key="filter_price",
        )

    return PlayerFilter(
        positions=tuple(selected_positions),
        teams=tuple(selected_teams),
        price_range=tuple(price_range) if price_range else None,
    )


def filter_schedule(schedule: pd.DataFrame, player_filter: PlayerFilter) -> pd.DataFrame:
    """Narrow a fixture schedule to the selected clubs.

    Only the club filter applies: position and price are properties of players,
    and a fixture has neither.
    """
    if schedule.empty or player_filter.teams is None:
        return schedule
    return schedule[schedule["team_name"].isin(list(player_filter.teams))]


def caption(player_filter: PlayerFilter, shown: int, total: int) -> str:
    """A consistent "what am I looking at" line for every filtered view."""
    if shown == total:
        return f"Showing all {total} players."
    return f"Showing {shown} of {total} players — {player_filter.describe()}."
