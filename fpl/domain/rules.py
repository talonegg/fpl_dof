"""The rules of Fantasy Premier League.

Every constraint the optimiser and validators rely on lives here. If a number
about squad shape, budget or transfers appears anywhere else in the codebase,
it is a bug waiting to happen when the rules change between seasons.
"""

from __future__ import annotations

from dataclasses import dataclass

SQUAD_SIZE = 15
STARTING_XI_SIZE = 11
BUDGET_MILLIONS = 100.0
MAX_PLAYERS_PER_CLUB = 3

TRANSFER_HIT_POINTS = 4
FREE_TRANSFERS_PER_GAMEWEEK = 1
MAX_ROLLED_FREE_TRANSFERS = 5

GOALKEEPER = "Goalkeeper"
DEFENDER = "Defender"
MIDFIELDER = "Midfielder"
FORWARD = "Forward"

POSITIONS = (GOALKEEPER, DEFENDER, MIDFIELDER, FORWARD)

# Exact squad composition: 2 GK, 5 DEF, 5 MID, 3 FWD.
SQUAD_COMPOSITION: dict[str, int] = {
    GOALKEEPER: 2,
    DEFENDER: 5,
    MIDFIELDER: 5,
    FORWARD: 3,
}


@dataclass(frozen=True)
class PositionLimits:
    """Min and max of a position allowed in a valid starting XI."""

    minimum: int
    maximum: int


# A starting XI must have exactly one keeper, at least three defenders and at
# least one forward; the rest is free. These bounds generate every legal
# formation without enumerating them.
STARTING_XI_LIMITS: dict[str, PositionLimits] = {
    GOALKEEPER: PositionLimits(1, 1),
    DEFENDER: PositionLimits(3, 5),
    MIDFIELDER: PositionLimits(2, 5),
    FORWARD: PositionLimits(1, 3),
}


def valid_formations() -> list[tuple[int, int, int]]:
    """Every legal (defenders, midfielders, forwards) split for a starting XI."""
    outfield = STARTING_XI_SIZE - STARTING_XI_LIMITS[GOALKEEPER].minimum
    formations = []
    for defenders in range(
        STARTING_XI_LIMITS[DEFENDER].minimum, STARTING_XI_LIMITS[DEFENDER].maximum + 1
    ):
        for midfielders in range(
            STARTING_XI_LIMITS[MIDFIELDER].minimum,
            STARTING_XI_LIMITS[MIDFIELDER].maximum + 1,
        ):
            forwards = outfield - defenders - midfielders
            limits = STARTING_XI_LIMITS[FORWARD]
            if limits.minimum <= forwards <= limits.maximum:
                formations.append((defenders, midfielders, forwards))
    return formations


# -- Scoring rules that changed between seasons ---------------------------

# Defensive contributions arrived in 2025-26 and continue in 2026-27. The three
# seasons before it had no such route to points at all.
#
# This is a property of the **season being played**, not of whether a data file
# happens to carry the column, and the distinction is load-bearing. For a
# 2024-25 squad, scoring no defensive contributions is *correct* -- they did not
# exist. For a 2025-26 squad they very much did, and a model that scores none
# because its prior seasons lack the data is not correct, it is blind. Keying
# off the column would make those two cases indistinguishable.
DEFENSIVE_CONTRIBUTION_FIRST_SEASON = "2025-26"


def season_scores_defensive_contributions(season: str) -> bool:
    """Whether ``season`` was played under defensive-contribution scoring.

    Season labels are ``YYYY-YY``, which sorts chronologically as a string.
    """
    return bool(season) and season >= DEFENSIVE_CONTRIBUTION_FIRST_SEASON
