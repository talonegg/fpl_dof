"""Which tabs the app has, and what each one honours.

Adding a feature is an entry here plus a render function. The entry point does
not change, and the filter contract is declared rather than remembered.
"""

from __future__ import annotations

import streamlit as st

from app import filters_view, fixtures_view, players_view, scouting_view
from app.views import Registry, View, ViewContext


def _render_players(context: ViewContext) -> None:
    players_view.render(context.players, total=context.total_count)


def _render_scouting(context: ViewContext) -> None:
    st.caption(
        filters_view.caption(context.player_filter, context.filtered_count, context.total_count)
    )
    history = context.history()
    scouting_view.render_detail(context.players, history, context.season_label)
    st.divider()
    scouting_view.render_comparison(context.players, history, context.season_label)


def _render_fixtures(context: ViewContext) -> None:
    gameweek = context.next_gameweek()
    if gameweek is None:
        st.info("The season is over — no upcoming fixtures.")
        return

    schedule = filters_view.filter_schedule(context.schedule(), context.player_filter)
    if schedule.empty:
        st.warning("No fixtures for the selected clubs.")
        return

    fixtures_view.render(schedule, from_gameweek=gameweek)


def build_registry() -> Registry:
    """The tabs, in the order they appear."""
    registry = Registry()

    registry.add(
        View(
            name="Players",
            render=_render_players,
            note="the sortable table; honours every filter",
        )
    )
    registry.add(
        View(
            name="Scouting",
            render=_render_scouting,
            note="player detail and comparison; honours every filter",
        )
    )
    registry.add(
        View(
            name="Fixtures",
            render=_render_fixtures,
            # A fixture belongs to a club but has no position, price or injury,
            # so three of the four filters cannot mean anything here.
            honours_position=False,
            honours_price=False,
            honours_availability=False,
            note="team-shaped: only the club filter applies",
        )
    )

    return registry
