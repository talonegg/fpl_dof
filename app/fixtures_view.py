"""The fixture ticker: which teams have the kindest run coming up."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from fpl.domain.fixtures import blanks_and_doubles, difficulty_summary, upcoming

DEFAULT_HORIZON = 6

# FPL's own 1-5 difficulty scale, green (easy) through red (hard).
DIFFICULTY_COLOURS = {
    1: "#0e8a3e",
    2: "#4fb06d",
    3: "#8f9497",
    4: "#e07a5f",
    5: "#c1352b",
}


def _ticker_grid(window: pd.DataFrame) -> pd.DataFrame:
    """Pivot to teams as rows, gameweeks as columns, opponents in the cells.

    Doubles put both opponents in one cell rather than adding a column, which
    keeps every team's row the same width.
    """

    def label(group: pd.DataFrame) -> str:
        parts = []
        for row in group.itertuples():
            # An opponent outside the supplied team list has no name; show the
            # id rather than crashing the whole ticker over one odd fixture.
            name = row.opponent_name if isinstance(row.opponent_name, str) else None
            short = name[:3].upper() if name else f"#{row.opponent}"
            parts.append(f"{short} ({'H' if row.is_home else 'A'})")
        return " + ".join(parts)

    cells = (
        window.groupby(["team_name", "gameweek"])
        .apply(label, include_groups=False)
        .reset_index(name="cell")
    )
    return cells.pivot(index="team_name", columns="gameweek", values="cell").fillna("—")


def _mean_difficulty_grid(window: pd.DataFrame) -> pd.DataFrame:
    """The same grid, but holding difficulty, for colouring."""
    return window.groupby(["team_name", "gameweek"])["difficulty"].mean().unstack()


def render(schedule: pd.DataFrame, from_gameweek: int) -> None:
    """Render the fixture ticker for the gameweeks after ``from_gameweek``."""
    st.subheader("Fixture ticker")

    if schedule.empty:
        st.info("No fixture data available.")
        return

    horizon = st.slider(
        "Gameweeks ahead",
        min_value=3,
        max_value=10,
        value=DEFAULT_HORIZON,
        help="How far ahead to look. 5-7 is the usual planning horizon.",
    )

    window = upcoming(schedule, from_gameweek, horizon)
    if window.empty:
        st.info("No fixtures in that window — the season may be over.")
        return

    summary = difficulty_summary(schedule, from_gameweek, horizon)
    easiest = summary.head(3)["team_name"].tolist()
    st.caption(
        f"Gameweeks {from_gameweek}–{from_gameweek + horizon - 1}. "
        f"Kindest run: {', '.join(easiest)}."
    )

    grid = _ticker_grid(window)
    difficulty = _mean_difficulty_grid(window).reindex(index=grid.index, columns=grid.columns)
    # Order rows by how easy the run is, so the top of the table is the answer.
    order = [name for name in summary["team_name"] if name in grid.index]
    grid = grid.loc[order]
    difficulty = difficulty.loc[order]

    def colour(_value, row, column):
        score = difficulty.loc[row, column]
        if pd.isna(score):
            return ""
        return f"background-color: {DIFFICULTY_COLOURS[round(score)]}; color: white"

    styled = grid.style.apply(
        lambda column: [colour(v, row, column.name) for row, v in column.items()],
        axis=0,
    )
    st.dataframe(styled, width="stretch")

    notable = blanks_and_doubles(schedule, from_gameweek, horizon)
    if notable.empty:
        st.caption("No blank or double gameweeks in this window.")
    else:
        st.caption("Blanks and doubles — check these before planning transfers.")
        st.dataframe(
            notable.rename(
                columns={
                    "team_name": "Team",
                    "gameweek": "GW",
                    "fixture_count": "Matches",
                    "kind": "",
                }
            )[["Team", "GW", "Matches", ""]],
            width="stretch",
            hide_index=True,
        )
