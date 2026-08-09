"""Turning an expected-points vector into a season-opening squad, with reasons.

The split this module exists to hold: a *strategy* says what each player is
worth, the *constructor* says which fifteen to buy, and the *recommender* says
why — three questions with three different failure modes.

Keeping them apart is not tidiness. The optimiser is provably optimal given its
inputs, so every squad-level disappointment traces to the expected-points
vector rather than to the search. Folding the two together would make that
impossible to see, and this project has repeatedly found the inputs at fault
while the search was fine.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from fpl.optimise.squad import InfeasibleSquad, Squad, SquadConstraints, optimise_squad

# A player the model cannot see is excluded, not guessed at. Reported, because
# on real data it is a third of the priced list.
REQUIRED_FOR_SELECTION = ["price", "position", "team", "expected_points"]


@dataclass
class SquadRecommendation:
    """A squad, and everything needed to argue with it."""

    squad: Squad
    strategy: str
    considered: int
    excluded: int
    warnings: list[str] = field(default_factory=list)

    @property
    def players(self) -> pd.DataFrame:
        return self.squad.players

    @property
    def cost(self) -> float:
        return self.squad.cost

    def summary(self) -> str:
        return (
            f"{self.strategy}: {self.squad.formation}, £{self.squad.cost:.1f}m, "
            f"{self.squad.expected_points:.0f} expected points from {self.considered} "
            f"candidates ({self.excluded} unpriced or unmodelled)"
        )


def construct_squad(
    pool: pd.DataFrame,
    expected: pd.Series,
    constraints: SquadConstraints | None = None,
) -> Squad | None:
    """The fifteen the rules and the numbers imply. Optimal, or nothing.

    Returns ``None`` when no legal squad exists rather than a best-effort one:
    an illegal squad silently returned would be scored as though it could be
    fielded.
    """
    if pool.empty:
        return None

    candidates = pool.assign(expected_points=expected).dropna(subset=REQUIRED_FOR_SELECTION)
    if candidates.empty:
        return None

    try:
        return optimise_squad(candidates, constraints)
    except InfeasibleSquad:
        return None


def recommend_squad(
    pool: pd.DataFrame,
    expected: pd.Series,
    strategy_name: str,
    constraints: SquadConstraints | None = None,
    defensive_status: str = "",
) -> SquadRecommendation | None:
    """Construct a squad and report what the model could and could not see.

    The warnings are the point. A squad presented without them reads as a
    complete answer to a question the model only partly saw — a third of the
    pool is typically invisible to it, and in 2025-26 an entire scoring route
    was.
    """
    squad = construct_squad(pool, expected, constraints)
    if squad is None:
        return None

    usable = pool.assign(expected_points=expected).dropna(subset=REQUIRED_FOR_SELECTION)
    excluded = int(len(pool) - len(usable))

    warnings: list[str] = []
    if excluded:
        cheap = 0
        if "price" in pool.columns:
            missing = pool.index.difference(usable.index)
            cheap = int((pool.loc[missing, "price"] < 5.0).sum())
        warnings.append(
            f"{excluded} of {len(pool)} priced players had no usable history and were "
            f"excluded before optimising ({cheap} of them under £5.0m, where the bench is)"
        )
    if defensive_status == "blind":
        warnings.append(
            "this season scores defensive contributions but no prior season recorded "
            "the actions, so that route to points is invisible to the model"
        )

    return SquadRecommendation(
        squad=squad,
        strategy=strategy_name,
        considered=int(len(usable)),
        excluded=excluded,
        warnings=warnings,
    )


def explain_selection(pool: pd.DataFrame, expected: pd.Series, squad: Squad) -> pd.DataFrame:
    """Why each chosen player was chosen, next to what they cost.

    Points per million is the honest summary of an optimiser's reasoning: under
    a budget constraint it does not buy the best players, it buys the best value
    ones, and a recommendation that hides that invites the user to override it
    for the wrong reason.
    """
    if squad is None or pool.empty:
        return pd.DataFrame()

    valued = pool.assign(expected_points=expected)
    chosen = valued[valued["element"].isin(squad.players["element"])].copy()
    if chosen.empty:
        return pd.DataFrame()

    chosen["value"] = chosen["expected_points"] / chosen["price"].replace(0, pd.NA)
    starters = set(squad.starting["element"])
    chosen["role"] = chosen["element"].map(
        lambda element: "start" if element in starters else "bench"
    )

    columns = [
        column
        for column in (
            "player_name",
            "position",
            "team",
            "price",
            "expected_points",
            "value",
            "role",
        )
        if column in chosen.columns
    ]
    return chosen[columns].sort_values("expected_points", ascending=False).reset_index(drop=True)
