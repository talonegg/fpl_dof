"""The vocabulary of playing positions.

Four positions, spelled at least three different ways depending on where the
data came from: the API says ``Goalkeeper``, the archive says ``GK``, and
occasionally ``GKP``. Scoring rules, squad composition and defensive
thresholds all key on position, so an unrecognised spelling does not fail
loudly — it silently drops whichever rule it was looking up.

This lives in ``domain`` because it is a fact about football, not about any
model. It previously sat inside ``models/components.py``, which meant both
``domain`` and ``features`` had to import *upwards* into ``models`` to ask
what a goalkeeper was.
"""

from __future__ import annotations

GOALKEEPER = "GK"
DEFENDER = "DEF"
MIDFIELDER = "MID"
FORWARD = "FWD"

POSITIONS = (GOALKEEPER, DEFENDER, MIDFIELDER, FORWARD)

# Every spelling seen in the wild, mapped to the short form used internally.
POSITION_ALIASES = {
    "GKP": GOALKEEPER,
    "GOALKEEPER": GOALKEEPER,
    "DEFENDER": DEFENDER,
    "MIDFIELDER": MIDFIELDER,
    "FORWARD": FORWARD,
}

# The long form, for anything user-facing.
POSITION_NAMES = {
    GOALKEEPER: "Goalkeeper",
    DEFENDER: "Defender",
    MIDFIELDER: "Midfielder",
    FORWARD: "Forward",
}

# Positions that concede goals and keep clean sheets for scoring purposes.
DEFENSIVE_POSITIONS = (GOALKEEPER, DEFENDER)


def canonical_position(position: object) -> object:
    """Map any spelling onto the short form the scoring tables use.

    Non-strings pass through untouched so a NaN stays a NaN rather than
    becoming a string that silently fails every lookup.
    """
    if not isinstance(position, str):
        return position
    upper = position.upper()
    return POSITION_ALIASES.get(upper, upper)


def is_defensive(position: object) -> bool:
    """Whether the position concedes goals and earns clean sheets."""
    return canonical_position(position) in DEFENSIVE_POSITIONS


def display_name(position: object) -> str:
    """The long form, falling back to whatever was given."""
    canonical = canonical_position(position)
    return POSITION_NAMES.get(canonical, str(position))
