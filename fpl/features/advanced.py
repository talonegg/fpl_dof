"""Advanced player signals, from the official API rather than a scraper.

The roadmap named Understat and FBref for this. Neither can be used:
Understat's ``robots.txt`` is ``User-agent: * / Disallow: /``, and FBref sits
behind a Cloudflare challenge that returns 403 even for ``robots.txt``.
Respecting those is a project non-negotiable, and working around a bot
challenge would be exactly the evasion the rule exists to prevent.

It turns out the two things worth having are published officially anyway:

**Set-piece duties.** ``penalties_order``, ``corners_and_indirect_freekicks_order``
and ``direct_freekicks_order`` name the designated takers, ranked. This is the
single most actionable non-obvious signal in FPL — a first-choice penalty taker
carries a standing points premium that no amount of xG modelling recovers,
because the chance only exists for the one player assigned to it.

**Finishing over- and under-performance.** Goals minus expected goals says
whether a player's return is being flattered or suppressed by finishing luck.
It is a *regression* signal and reads backwards to intuition: a striker well
above his xG is a sell indicator, not a buy one.

Both are live-only. The archive carries neither, so like availability these
must never be evaluated on historical seasons — see
:mod:`fpl.features.availability` for why that matters.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

PENALTIES = "penalties_order"
CORNERS = "corners_and_indirect_freekicks_order"
FREE_KICKS = "direct_freekicks_order"

SET_PIECE_COLUMNS = (PENALTIES, CORNERS, FREE_KICKS)

# "First choice" cannot be hardcoded to order 1: the API does not use the same
# scale for every duty. On live data penalties and direct free kicks start at
# 1, but corner orders start at 2 and run to 12 -- testing for 1 flags nobody
# for corners at all, silently. So first choice means the lowest order within
# that player's own team, which is what "designated taker" actually means and
# is independent of the scale.
TEAM_COLUMN = "team"

MINUTES_PER_MATCH = 90

# Below this, a finishing delta is noise: a single goal moves it wildly.
MIN_MINUTES_FOR_FINISHING = 450


class AdvancedDataUnavailable(ValueError):
    """The frame carries no set-piece data, so the question cannot be asked."""


def has_set_piece_data(players: pd.DataFrame) -> bool:
    """Whether this frame carries set-piece duties at all.

    False for every archive season: the historical files record what happened,
    not who was assigned to take corners.
    """
    return any(column in players.columns for column in SET_PIECE_COLUMNS)


def set_piece_duties(players: pd.DataFrame) -> pd.DataFrame:
    """Flag the designated takers, and how senior they are.

    Adds ``takes_penalties``, ``takes_corners``, ``takes_free_kicks`` (all
    first-choice only) and ``set_piece_duties``, a count of how many of the
    three a player is first choice for.

    A null order means "not on the list", which is the overwhelming majority
    of players and is a real answer rather than missing data.
    """
    if players.empty:
        return players

    if not has_set_piece_data(players):
        raise AdvancedDataUnavailable(
            "this data carries no set-piece orders — they are published only "
            "for the current season and were never recorded historically"
        )

    df = players.copy()
    for name, column in (
        ("takes_penalties", PENALTIES),
        ("takes_corners", CORNERS),
        ("takes_free_kicks", FREE_KICKS),
    ):
        if column not in df.columns:
            df[name] = False
            continue

        order = pd.to_numeric(df[column], errors="coerce")
        if TEAM_COLUMN in df.columns:
            best = order.groupby(df[TEAM_COLUMN]).transform("min")
        else:
            # No team to group by: fall back to the best order in the frame,
            # which is right for a single-club selection and never worse than
            # assuming a scale.
            best = order.min()
        df[name] = order.notna() & (order == best)

    df["set_piece_duties"] = (
        df["takes_penalties"].astype(int)
        + df["takes_corners"].astype(int)
        + df["takes_free_kicks"].astype(int)
    )
    return df


def finishing_delta(players: pd.DataFrame) -> pd.Series:
    """Goals scored minus expected goals.

    Positive means finishing above expectation. Read it as a warning rather
    than a recommendation: shot quality persists, finishing streaks do not, so
    a large positive delta more often precedes a fall than a continuation.
    """
    if players.empty:
        return pd.Series(dtype="float64")
    if "goals_scored" not in players.columns or "expected_goals" not in players.columns:
        return pd.Series(np.nan, index=players.index)

    goals = pd.to_numeric(players["goals_scored"], errors="coerce")
    expected = pd.to_numeric(players["expected_goals"], errors="coerce")

    # A scorer with no expected goals recorded is missing data, not a
    # miraculous finisher. On live data one player shows 11 goals against 0.00
    # xG -- reporting that as +11 over-performance would put him top of every
    # sell list on the strength of an absent column.
    missing = (expected <= 0) & (goals > 0)
    return (goals - expected).where(~missing)


def finishing_delta_per_90(players: pd.DataFrame) -> pd.Series:
    """Finishing delta at a common rate, NaN below a usable sample.

    Deliberately NaN rather than a number for players with few minutes: one
    goal in 90 minutes produces a spectacular-looking rate that means nothing.
    """
    delta = finishing_delta(players)
    if delta.empty or "minutes" not in players.columns:
        return delta

    minutes = pd.to_numeric(players["minutes"], errors="coerce")
    rate = delta * MINUTES_PER_MATCH / minutes.replace(0, np.nan)
    return rate.where(minutes >= MIN_MINUTES_FOR_FINISHING)


def add_advanced_metrics(players: pd.DataFrame) -> pd.DataFrame:
    """Attach set-piece duties and finishing signals.

    Returns the frame unchanged when the data cannot support it, so a caller
    working with historical rows gets no columns rather than misleading ones.
    """
    if players.empty or not has_set_piece_data(players):
        return players

    df = set_piece_duties(players)
    df["finishing_delta"] = finishing_delta(df)
    df["finishing_delta_per_90"] = finishing_delta_per_90(df)
    return df


def set_piece_takers(players: pd.DataFrame) -> pd.DataFrame:
    """Every first-choice set-piece taker, most duties first.

    A scouting shortlist in its own right: these players have a route to
    points that their underlying numbers do not show.
    """
    df = add_advanced_metrics(players)
    if "set_piece_duties" not in df.columns:
        return players.iloc[0:0]

    takers = df[df["set_piece_duties"] > 0]
    return takers.sort_values(
        ["set_piece_duties", "takes_penalties", "total_points"],
        ascending=[False, False, False],
    )
