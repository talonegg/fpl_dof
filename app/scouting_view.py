"""Player scouting: one player in depth, or a few side by side.

Two deliberate choices about the charts here:

Points and minutes get separate charts rather than one chart with two y-axes.
A dual-axis chart lets you slide the two scales until any story you like
appears, which is exactly the wrong property for something you are using to
spend money.

Comparisons ship the numbers as a table underneath. Two of the four series
colours fall below 3:1 contrast on a light background, so the chart alone
cannot be the only way to read the values.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from app.theme import MAX_COMPARISON_SERIES, series_colours
from fpl.domain.history import collapse_to_gameweeks
from fpl.features.availability import flagged
from fpl.features.filters import restrict_to_available
from fpl.features.rates import LOW_MINUTES_THRESHOLD

POINTS_LABEL = "Points"
MINUTES_LABEL = "Minutes"


def _prune_selection(key: str, valid: set) -> None:
    """Drop remembered selections that the current filter has removed.

    Streamlit keeps widget state under its key, so a player selected before a
    filter change would otherwise be handed back to a widget that no longer
    offers them.
    """
    if key not in st.session_state:
        return
    current = st.session_state[key]
    if isinstance(current, list):
        st.session_state[key] = restrict_to_available(current, valid)
    elif current not in valid:
        del st.session_state[key]


def _player_label(row) -> str:
    return f"{row.web_name} ({row.team_name}, {row.position}, £{row.price}m)"


def _history_for(history: pd.DataFrame, element: int) -> pd.DataFrame:
    """One player's gameweek rows, collapsed to one row per gameweek.

    In a double gameweek a player has two rows. Points and minutes sum; price
    is a snapshot, so take the last. Without this the frame has duplicate
    gameweeks, which quietly double-plots a bar chart and outright breaks a
    pivot.
    """
    if history.empty:
        return history

    player = history[history["current_element"] == element]
    if player.empty:
        return player

    return collapse_to_gameweeks(player, ["gameweek"]).sort_values("gameweek")


def render_availability(players: pd.DataFrame) -> None:
    """Who is carrying an injury, suspension or doubt.

    Shown before anything else in the scouting tab: no amount of expected
    points matters if the player is not going to be on the pitch.
    """
    st.subheader("Availability")

    concerns = flagged(players)
    if concerns.empty:
        st.caption("No availability concerns among the filtered players.")
        return

    display = concerns.assign(
        Fit=(concerns["availability"] * 100).round().astype(int).astype(str) + "%"
    )
    columns = {"web_name": "Player", "team_name": "Team", "news": "News"}
    available = [column for column in columns if column in display.columns]

    st.caption(f"{len(concerns)} player(s) with news. Check before transferring in.")
    st.dataframe(
        display[[*available, "Fit"]].rename(columns=columns),
        width="stretch",
        hide_index=True,
    )


def _summary_metrics(player: pd.Series) -> None:
    """The headline numbers, as stat tiles rather than a chart."""
    columns = st.columns(4)
    columns[0].metric("Points", int(player.get("total_points", 0)))
    columns[1].metric("Form", f"{player.get('form', 0):.1f}")
    columns[2].metric("Price", f"£{player.get('price', 0):.1f}m")

    per_90 = player.get("total_points_per_90")
    columns[3].metric(
        "Points per 90",
        "—" if pd.isna(per_90) else f"{per_90:.2f}",
        help="Unavailable until the player has minutes on the board.",
    )

    if player.get("low_minutes", False):
        st.caption(
            f"Fewer than {LOW_MINUTES_THRESHOLD} minutes played — per-90 rates are still noise."
        )


def _render_history_charts(history: pd.DataFrame, season_label: str) -> None:
    """Gameweek-by-gameweek points and minutes, as two separate charts."""
    if history.empty:
        st.info(
            "No gameweek history yet. The season has not started, and this "
            "player has no matched history in the archive."
        )
        return

    st.caption(f"Gameweek history — {season_label}")
    colour = series_colours(1)

    st.markdown(f"**{POINTS_LABEL} per gameweek**")
    st.bar_chart(
        history.set_index("gameweek")[["total_points"]].rename(
            columns={"total_points": POINTS_LABEL}
        ),
        color=colour,
        height=220,
    )

    st.markdown(f"**{MINUTES_LABEL} per gameweek**")
    st.caption(
        "Shown separately on purpose — points and minutes on shared axes would "
        "let the scales tell any story you wanted."
    )
    st.bar_chart(
        history.set_index("gameweek")[["minutes"]].rename(columns={"minutes": MINUTES_LABEL}),
        color=colour,
        height=220,
    )

    if "price" in history.columns and history["price"].notna().any():
        st.markdown("**Price**")
        st.line_chart(
            history.set_index("gameweek")[["price"]].rename(columns={"price": "£m"}),
            color=colour,
            height=200,
        )


def render_detail(players: pd.DataFrame, history: pd.DataFrame, season_label: str) -> None:
    """One player, in depth."""
    st.subheader("Player detail")

    if players.empty:
        st.warning("No players match those filters.")
        return

    labels = {row.element: _player_label(row) for row in players.itertuples()}
    _prune_selection("detail_player", set(labels))
    element = st.selectbox(
        "Player",
        options=list(labels),
        format_func=lambda value: labels[value],
        key="detail_player",
    )

    player = players[players["element"] == element].iloc[0]
    _summary_metrics(player)
    _render_history_charts(_history_for(history, element), season_label)


def render_comparison(players: pd.DataFrame, history: pd.DataFrame, season_label: str) -> None:
    """Two to four players on shared axes."""
    st.subheader("Compare players")

    if players.empty:
        st.warning("No players match those filters.")
        return

    labels = {row.element: _player_label(row) for row in players.itertuples()}
    _prune_selection("comparison_players", set(labels))
    selected = st.multiselect(
        "Players",
        options=list(labels),
        format_func=lambda value: labels[value],
        max_selections=MAX_COMPARISON_SERIES,
        key="comparison_players",
        help=f"Up to {MAX_COMPARISON_SERIES}. Beyond that the colours stop being "
        "reliably distinguishable.",
    )

    if len(selected) < 2:
        st.info("Pick at least two players to compare.")
        return

    # Series names must be unique or the pivot collides: several players share
    # a web_name ("Gabriel"), so the team goes in the label too.
    names = {
        row.element: f"{row.web_name} ({row.team_name})"
        for row in players[players["element"].isin(selected)].itertuples()
    }

    frames = []
    for element in selected:
        player_history = _history_for(history, element)
        if player_history.empty:
            continue
        frames.append(
            player_history.assign(player=names[element])[["gameweek", "player", "total_points"]]
        )

    if not frames:
        st.info(f"No gameweek history for these players in {season_label}.")
        return

    combined = pd.concat(frames)
    # Cumulative points, because that is the question actually being asked:
    # who has delivered more over the run, not who spiked once.
    combined["cumulative"] = combined.groupby("player")["total_points"].cumsum()
    wide = combined.pivot(index="gameweek", columns="player", values="cumulative")
    wide = wide.ffill()

    # Hold column order to the order players were selected. `pivot` sorts
    # alphabetically, and colours are assigned by position, so removing one
    # player would otherwise recolour the others -- the exact repaint the
    # fixed-slot palette exists to prevent.
    ordered = [
        labels[element].split(" (")[0]
        for element in selected
        if labels[element].split(" (")[0] in wide.columns
    ]
    wide = wide[ordered]

    st.markdown(f"**Cumulative points — {season_label}**")
    st.line_chart(wide, color=series_colours(len(wide.columns)), height=320)

    # The table is not optional: it is how the values stay readable for anyone
    # the low-contrast series colours fail.
    with st.expander("Show the numbers"):
        st.dataframe(wide.tail(10), width="stretch")
