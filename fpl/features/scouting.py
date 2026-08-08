"""A fixture-aware scouting heuristic.

**This is not a model.** It is a transparent arithmetic aid for the human
scouting views: recent form, scaled by how kind each upcoming fixture looks,
summed over the horizon. It has no training, no validation and no claim to
accuracy, and Phase 3's predictors in ``fpl/models/`` are meant to replace it
for anything that actually picks a team.

It is kept deliberately simple so that a number on screen can always be
explained: "three fixtures, two of them easy, against a player averaging 5.0".
"""

from __future__ import annotations

import pandas as pd

from fpl.domain.fixtures import upcoming

# FPL rates fixtures 1 (easiest) to 5 (hardest). A neutral fixture is 3, so
# each step away from neutral moves the expectation by 10%. The size of that
# step is a guess; its direction is not.
NEUTRAL_DIFFICULTY = 3
DIFFICULTY_STEP = 0.1


def difficulty_multiplier(difficulty: float) -> float:
    """Scale factor for a fixture of the given difficulty.

    1.2 for the easiest fixture down to 0.8 for the hardest.
    """
    return 1 + (NEUTRAL_DIFFICULTY - difficulty) * DIFFICULTY_STEP


def team_outlook(schedule: pd.DataFrame, from_gameweek: int, horizon: int) -> pd.DataFrame:
    """Per-team summed fixture multiplier over the horizon.

    Summing rather than averaging is the point: a team with a double gameweek
    gets two terms and a team with a blank gets one fewer, so the score
    reflects *how many chances to score points* a player has, not just how
    hard each one is.
    """
    window = upcoming(schedule, from_gameweek, horizon)
    columns = ["team", "fixture_count", "fixture_multiplier", "mean_difficulty"]
    if window.empty:
        return pd.DataFrame(columns=columns)

    window = window.copy()
    window["multiplier"] = window["difficulty"].map(difficulty_multiplier)

    return window.groupby("team", as_index=False).agg(
        fixture_count=("multiplier", "size"),
        fixture_multiplier=("multiplier", "sum"),
        mean_difficulty=("difficulty", "mean"),
    )[columns]


def add_outlook(
    players: pd.DataFrame,
    schedule: pd.DataFrame,
    from_gameweek: int,
    horizon: int,
    form_column: str = "form",
) -> pd.DataFrame:
    """Attach fixture outlook and an ``outlook_score`` to each player.

    ``outlook_score`` is ``form × summed fixture multiplier``: roughly "points
    this player might manage over the horizon if recent form holds and fixture
    difficulty means what FPL says it means". Both of those are assumptions,
    which is why this is a scouting aid and not a projection.

    Players whose team has no fixtures in the window score 0, not NaN -- a
    blank is a real, known answer.
    """
    outlook = team_outlook(schedule, from_gameweek, horizon)
    df = players.merge(outlook, on="team", how="left")

    df["fixture_count"] = df["fixture_count"].fillna(0).astype(int)
    df["fixture_multiplier"] = df["fixture_multiplier"].fillna(0.0)
    df["outlook_score"] = df[form_column].fillna(0) * df["fixture_multiplier"]
    return df
