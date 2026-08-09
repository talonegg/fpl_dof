"""The Bonus Points System, and how much of it this project can see.

Bonus points are awarded 3/2/1 to the top three BPS scores in a match, so BPS
is worth modelling — but it is scored from Opta events, most of which the FPL
API does not publish.

The table below is the official one (Premier League, 2025-26). It is recorded
in full, including the actions we cannot observe, because the *shape* of the
gap is the useful thing: knowing that key passes are worth 1 and big chances
created 3 tells you what kind of player this project will systematically
under-rate.

Rather than fit coefficients and quote an R², :func:`reconstruct` applies the
published values to the inputs that are available. What it misses is then a
direct measurement of the unpublished data rather than an artefact of a model.

2025-26 changed several values: penalty goals became 12 for every position,
saves split into 3 inside the box and 2 outside, goalline clearances went from
3 to 9, penalty saves from 9 to 8, and tackles moved to 2 per tackle won.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from fpl.domain.positions import canonical_position


@dataclass(frozen=True)
class BpsAction:
    """One scoring action in the BPS table."""

    name: str
    value: float
    available: bool
    note: str = ""


# The official table. `available` records whether the FPL API publishes the
# input, not whether the action exists.
BPS_ACTIONS = (
    # -- Appearance ---------------------------------------------------------
    BpsAction("Playing 1-60 minutes", 3, True),
    BpsAction("Playing over 60 minutes", 6, True),
    # -- Goals and assists --------------------------------------------------
    BpsAction("Goal (penalty, any position)", 12, False, "no penalty/open-play split"),
    BpsAction("Goal (goalkeeper or defender)", 12, True),
    BpsAction("Goal (midfielder)", 18, True),
    BpsAction("Goal (forward)", 24, True),
    BpsAction("Assist", 9, True),
    BpsAction("Match-winning goal", 3, False, "needs the match scoreline"),
    # -- Goalkeeping and defending -----------------------------------------
    BpsAction("Clean sheet (goalkeeper or defender)", 12, True),
    BpsAction("Penalty save", 8, True),
    BpsAction("Save (shot inside the box)", 3, False, "saves are not split by location"),
    BpsAction("Save (shot outside the box)", 2, False, "saves are not split by location"),
    BpsAction("Goalline clearance", 9, False),
    BpsAction("Clearances, blocks and interceptions (per 2)", 1, True),
    BpsAction("Recovery (per 3)", 1, True),
    BpsAction("Successful tackle", 2, True),
    # -- Creating and carrying ---------------------------------------------
    BpsAction("Key pass", 1, False),
    BpsAction("Big chance created", 3, False),
    BpsAction("Successful open-play cross", 1, False),
    BpsAction("Successful dribble", 1, False),
    BpsAction("Shot on target", 2, False),
    BpsAction("Foul won", 1, False),
    BpsAction("Pass completion 70-79% (min 30 passes)", 2, False),
    BpsAction("Pass completion 80-89% (min 30 passes)", 4, False),
    BpsAction("Pass completion 90%+ (min 30 passes)", 6, False),
    # -- Penalties ----------------------------------------------------------
    BpsAction("Goal conceded (goalkeeper or defender)", -4, True),
    BpsAction("Yellow card", -3, True),
    BpsAction("Red card", -9, True),
    BpsAction("Own goal", -6, True),
    BpsAction("Missed penalty", -6, True),
    BpsAction("Conceded penalty", -3, False),
    BpsAction("Missed big chance", -3, False),
    BpsAction("Error leading to a goal", -3, False),
    BpsAction("Error leading to an attempt", -1, False),
    BpsAction("Being tackled", -1, False),
    BpsAction("Conceded foul", -1, False),
    BpsAction("Offside", -1, False),
    BpsAction("Shot off target", -1, False),
)

APPEARANCE_PARTIAL = 3
APPEARANCE_FULL = 6
FULL_APPEARANCE_MINUTES = 60
GOAL_BPS = {"GK": 12, "DEF": 12, "MID": 18, "FWD": 24}
CLEAN_SHEET_BPS = 12
ASSIST_BPS = 9
PENALTY_SAVE_BPS = 8
TACKLE_BPS = 2
CBI_PER_POINT = 2
RECOVERIES_PER_POINT = 3
GOAL_CONCEDED_BPS = -4
YELLOW_BPS = -3
RED_BPS = -9
OWN_GOAL_BPS = -6
MISSED_PENALTY_BPS = -6

DEFENSIVE_POSITIONS = ("GK", "DEF")


def observable_actions() -> tuple[BpsAction, ...]:
    """Actions whose inputs the FPL API publishes."""
    return tuple(action for action in BPS_ACTIONS if action.available)


def unobservable_actions() -> tuple[BpsAction, ...]:
    """Actions the API gives us no way to see."""
    return tuple(action for action in BPS_ACTIONS if not action.available)


PENALTY_GOAL_BPS = 12


def penalty_correction(appearances: pd.DataFrame, penalty_goals: pd.Series) -> pd.Series:
    """How much a naive reconstruction over-credits penalty goals.

    Since 2025-26 a penalty scores 12 BPS whatever the position, but the API
    reports only ``goals_scored``, so a midfielder's penalty is credited 18 and
    a forward's 24. The excess is ``position value − 12`` per penalty.

    Returns a positive number to be *subtracted*. Zero for goalkeepers and
    defenders, whose goals are already worth 12.
    """
    if appearances.empty:
        return pd.Series(dtype="float64")

    positions = appearances["position"].map(canonical_position)
    excess = positions.map(GOAL_BPS).fillna(0) - PENALTY_GOAL_BPS
    return penalty_goals.fillna(0) * excess.clip(lower=0)


def reconstruct(appearances: pd.DataFrame, penalty_goals: pd.Series | None = None) -> pd.Series:
    """BPS rebuilt from the published coefficients and available inputs only.

    Deliberately *not* fitted. Applying the real values and seeing what is
    left over measures the unpublished data; fitting coefficients would let
    the model absorb the missing components and hide the gap.

    Always an underestimate: every unobservable action in the table is
    positive far more often than not.

    ``penalty_goals`` optionally supplies how many of each player's goals were
    penalties, so they can be credited at 12 rather than the position value.
    Supplying it needs ``penalties_order``, which is live-only — the archive
    identifies a taker only when they miss — so historical reconstructions
    leave it out and over-credit takers by a measured 0.5 to 1.3 BPS per
    goal-scoring appearance.
    """
    if appearances.empty:
        return pd.Series(dtype="float64")

    def column(name: str) -> pd.Series:
        if name in appearances.columns:
            return pd.to_numeric(appearances[name], errors="coerce").fillna(0)
        return pd.Series(0.0, index=appearances.index)

    positions = appearances["position"].map(canonical_position)
    minutes = column("minutes")

    appearance = pd.Series(0.0, index=appearances.index)
    appearance = appearance.mask(minutes > 0, APPEARANCE_PARTIAL)
    appearance = appearance.mask(minutes >= FULL_APPEARANCE_MINUTES, APPEARANCE_FULL)

    is_defensive = positions.isin(DEFENSIVE_POSITIONS)

    total = appearance
    total = total + column("goals_scored") * positions.map(GOAL_BPS).fillna(0)
    total = total + column("assists") * ASSIST_BPS
    total = total + column("clean_sheets") * CLEAN_SHEET_BPS * is_defensive
    total = total + column("penalties_saved") * PENALTY_SAVE_BPS
    total = total + column("tackles") * TACKLE_BPS
    # Integer division: the points accrue per completed pair or triple.
    total = total + (column("clearances_blocks_interceptions") // CBI_PER_POINT)
    total = total + (column("recoveries") // RECOVERIES_PER_POINT)
    total = total + column("goals_conceded") * GOAL_CONCEDED_BPS * is_defensive
    total = total + column("yellow_cards") * YELLOW_BPS
    total = total + column("red_cards") * RED_BPS
    total = total + column("own_goals") * OWN_GOAL_BPS
    total = total + column("penalties_missed") * MISSED_PENALTY_BPS

    if penalty_goals is not None:
        total = total - penalty_correction(appearances, penalty_goals)

    return total


def reconstruction_gap(appearances: pd.DataFrame) -> pd.Series:
    """Published BPS minus what the available inputs account for.

    The residue is the unobservable actions: key passes, big chances,
    dribbles, crosses, shots, fouls, pass completion and the error penalties.
    """
    if appearances.empty or "bps" not in appearances.columns:
        return pd.Series(dtype="float64")
    return pd.to_numeric(appearances["bps"], errors="coerce") - reconstruct(appearances)


def action_table() -> pd.DataFrame:
    """The BPS table as data: one row per action, with its value and availability.

    Exposed as a frame rather than only as constants so it can be joined,
    filtered and displayed like anything else in the model — "which scoring
    actions can we not see, and what are they worth" is a query, not a
    docstring.
    """
    return pd.DataFrame(
        [
            {
                "action": action.name,
                "bps": action.value,
                "observable": action.available,
                "note": action.note,
            }
            for action in BPS_ACTIONS
        ]
    )


def unobservable_weight() -> float:
    """Total absolute BPS tied up in actions we cannot observe.

    A crude but honest sense of scale: how much of the table is invisible,
    regardless of how often each action occurs.
    """
    return float(sum(abs(action.value) for action in unobservable_actions()))
