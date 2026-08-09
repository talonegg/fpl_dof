"""The season-opening squad tab: twenty ranked squads, and what separates them.

Presentation only. Every number here is computed in ``fpl/`` — this module
chooses what to show and how honest to be about it.

The design decision that shapes the whole tab: **show twenty squads, not one.**
A single optimal squad reads as an answer. Twenty squads separated by a point
and a half read as what they actually are — a set of near-equivalent options
where the ranking is inside the noise of the prediction. The second is true, so
it is what gets rendered.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from fpl.backtest.preseason import defensive_forecast_status
from fpl.models.preseason_strategies import PreseasonContext, strategies
from fpl.optimise.ranking import Shortlist, rank_squads, squad_differences
from fpl.optimise.squad import SquadConstraints

SHORTLIST_SIZE = 20

# Phone-first: the wide table is for laptops, these columns for everything else.
NARROW_COLUMNS = ["rank", "score", "gap", "cost"]
SQUAD_COLUMNS = ["player_name", "position", "team", "price", "expected_points", "role"]


def _shortlist_table(shortlist: Shortlist, narrow: bool) -> pd.DataFrame:
    table = shortlist.table()
    if table.empty:
        return table

    display = table.copy()
    display["score"] = display["score"].round(1)
    display["xi_points"] = display["xi_points"].round(1)
    display["gap"] = display["gap"].round(1)
    display["cost"] = display["cost"].round(1)
    if narrow:
        return display[[column for column in NARROW_COLUMNS if column in display.columns]]
    return display


def _render_confidence(shortlist: Shortlist) -> None:
    """Say plainly how much the ranking means before showing the ranking."""
    if len(shortlist) < 2:
        return

    best = shortlist.squads[0].score
    spread = shortlist.spread
    share = spread / best if best else 0.0

    if share < 0.02:
        st.warning(
            f"The twenty squads are separated by {spread:.1f} expected points "
            f"({share:.1%} of the top squad's total). That is well inside the "
            "error of the prediction — treat these as near-equivalent options, "
            "not as an order of merit."
        )
    else:
        st.info(
            f"Top to bottom the shortlist spans {spread:.1f} expected points "
            f"({share:.1%}). The ordering carries some information, but the gap "
            "between neighbouring squads does not."
        )


def render(
    pool: pd.DataFrame,
    target_season: str,
    prior_seasons: dict[str, pd.DataFrame],
    failures: tuple[str, ...] = (),
    narrow: bool = False,
) -> None:
    """The tab body."""
    st.subheader(f"Season-opening squad for {target_season}")

    if failures:
        st.error(
            "Could not load "
            + ", ".join(failures)
            + ". Every blended rate below is computed without those seasons, so "
            "the numbers are not the ones the model is calibrated on."
        )

    if pool.empty:
        st.warning("No candidate pool could be built — prior seasons are unavailable.")
        return

    catalogue = strategies()
    names = [strategy.name for strategy in catalogue]
    default = names.index("BlendedCareer+Minutes") if "BlendedCareer+Minutes" in names else 0

    chosen = st.selectbox(
        "Model",
        names,
        index=default,
        help=(
            "BlendedCareer+Minutes is the default because it is the best measured "
            "performer across three backtested seasons — not because it is the "
            "most sophisticated. It is not."
        ),
    )
    strategy = catalogue[names.index(chosen)]
    st.caption("Uses: " + (", ".join(strategy.uses) if strategy.uses else "nothing at all"))

    horizon = st.select_slider(
        "Score over",
        options=[3, 5, 7],
        value=7,
        format_func=lambda value: f"first {value} gameweeks",
        help=(
            "The window the squad is judged over. The model's advantage over the "
            "naive heuristic is largest at three gameweeks and gone by seven."
        ),
    )

    context = PreseasonContext(
        target=target_season, prior_seasons=prior_seasons, horizon=int(horizon)
    )

    with st.spinner(f"Ranking the best {SHORTLIST_SIZE} squads..."):
        expected = strategy.expected_points(pool, context)
        candidates = pool.assign(expected_points=expected).dropna(
            subset=["price", "position", "team", "expected_points"]
        )
        shortlist = rank_squads(candidates, count=SHORTLIST_SIZE, constraints=SquadConstraints())

    if not len(shortlist):
        st.error("No legal squad could be built from the available players.")
        return

    _render_coverage(pool, candidates, target_season)
    _render_confidence(shortlist)

    st.dataframe(_shortlist_table(shortlist, narrow), width="stretch", hide_index=True)

    ranks = [entry.rank for entry in shortlist.squads]
    selected = st.selectbox("Inspect squad", ranks, format_func=lambda value: f"Rank {value}")
    entry = shortlist.squads[ranks.index(selected)]

    st.caption(
        f"{entry.squad.formation} · £{entry.cost:.1f}m · "
        f"{entry.expected_points:.1f} expected points from the XI and captain "
        f"(score {entry.score:.1f} including the bench) · "
        f"{entry.changes_from_best} changes from the top squad"
    )
    st.dataframe(_squad_frame(entry, candidates), width="stretch", hide_index=True)

    differences = squad_differences(shortlist)
    if not differences.empty:
        with st.expander("Which players the twenty squads disagree about"):
            counts = (
                differences[differences["change"] == "in"]["player"]
                .value_counts()
                .rename_axis("player")
                .reset_index(name="appears in")
            )
            st.caption(
                "Players entering the alternatives most often — the real decisions "
                "behind a shortlist this tightly packed."
            )
            st.dataframe(counts.head(15), width="stretch", hide_index=True)


def _squad_frame(entry, candidates: pd.DataFrame) -> pd.DataFrame:
    """One squad's fifteen, starters first, with what each was valued at."""
    players = entry.players.copy()
    starters = set(entry.squad.starting["element"])
    players["role"] = players["element"].map(
        lambda element: "start" if element in starters else "bench"
    )
    if "expected_points" in players.columns:
        players["expected_points"] = players["expected_points"].round(1)

    columns = [column for column in SQUAD_COLUMNS if column in players.columns]
    return players[columns].sort_values(["role", "expected_points"], ascending=[True, False])


def _render_coverage(pool: pd.DataFrame, candidates: pd.DataFrame, target_season: str) -> None:
    """What the model could not see. Shown above the answer, not below it."""
    excluded = len(pool) - len(candidates)
    messages = []
    if excluded > 0:
        cheap = 0
        if "price" in pool.columns:
            missing = pool.index.difference(candidates.index)
            cheap = int((pool.loc[missing, "price"] < 5.0).sum())
        messages.append(
            f"{excluded} of {len(pool)} priced players have no Premier League history "
            f"and were excluded before optimising ({cheap} under £5.0m, where the "
            "bench is). They are not rated as poor — they are unrated."
        )

    if defensive_forecast_status(target_season, pool) == "blind":
        messages.append(
            "This season scores defensive contributions, but no prior season "
            "recorded the underlying actions — that route to points is invisible "
            "here, and it was worth 8.2% of all points in 2025-26."
        )

    for message in messages:
        st.caption(f":grey[{message}]")
