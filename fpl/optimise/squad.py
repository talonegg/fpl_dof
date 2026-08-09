"""Choosing the best legal squad for a set of expected points.

This is the half of the problem that has a right answer. Given what each player
is expected to score, "which fifteen maximise the total subject to the rules"
is an integer program, and the solver returns *the* optimum rather than a good
guess. That is worth insisting on: greedy value-per-million heuristics feel
sensible and are reliably beaten, because the budget constraint couples every
choice to every other one.

Nothing here knows how the expected points were produced. That separation is
the point -- swap the predictor and the optimiser needs no changes, and when a
recommendation is wrong you can tell which half was wrong.

The two-layer structure of an FPL squad is modelled explicitly: fifteen players
are *bought*, but only eleven *start* and one of those is captain. Optimising
the fifteen alone would happily spend the budget on bench players who never
score.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd
import pulp

from fpl.domain.rules import (
    BUDGET_MILLIONS,
    MAX_PLAYERS_PER_CLUB,
    SQUAD_COMPOSITION,
    SQUAD_SIZE,
    STARTING_XI_LIMITS,
    STARTING_XI_SIZE,
)

# Bench players do occasionally come on, but a squad optimised as though they
# never play still picks better than one that values them fully. This weight
# says "worth something, but not much" -- it breaks ties towards a useful bench
# without letting the bench compete with the starting eleven for budget.
DEFAULT_BENCH_WEIGHT = 0.1

REQUIRED_COLUMNS = ("element", "position", "team", "price", "expected_points")


@dataclass
class Squad:
    """A chosen squad, split into the parts FPL actually scores."""

    starting: pd.DataFrame
    bench: pd.DataFrame
    captain: int
    expected_points: float
    cost: float

    @property
    def players(self) -> pd.DataFrame:
        """All fifteen, starters first."""
        return pd.concat([self.starting, self.bench], ignore_index=True)

    @property
    def formation(self) -> str:
        """The starting XI's shape, e.g. ``"3-5-2"``."""
        counts = self.starting["position"].value_counts()
        return "-".join(
            str(int(counts.get(position, 0))) for position in ("Defender", "Midfielder", "Forward")
        )

    def summary(self) -> str:
        return f"{self.formation}, £{self.cost:.1f}m, {self.expected_points:.1f} expected points"


@dataclass
class SquadConstraints:
    """The rules the squad must satisfy.

    Defaults are the real FPL rules from :mod:`fpl.domain.rules`. They are
    parameters here only so tests can build small worlds -- never so that a
    caller can quietly play a different game.

    Note the starting XI's *shape* is not a parameter: ``STARTING_XI_LIMITS``
    lives in the rules module, so an XI always needs 1 goalkeeper, 3 defenders,
    2 midfielders and 1 forward at minimum. ``starting_size`` below 7 is
    therefore infeasible no matter what is passed here.
    """

    budget: float = BUDGET_MILLIONS
    squad_size: int = SQUAD_SIZE
    starting_size: int = STARTING_XI_SIZE
    max_per_club: int = MAX_PLAYERS_PER_CLUB
    composition: dict[str, int] = field(default_factory=lambda: dict(SQUAD_COMPOSITION))
    bench_weight: float = DEFAULT_BENCH_WEIGHT
    # Players who must be in the squad, by element id. Used by the transfer
    # planner to hold the rest of a squad still while it changes a few places.
    must_include: tuple[int, ...] = ()
    must_exclude: tuple[int, ...] = ()

    # Minimum spend. Unspent budget earns nothing, so a squad leaving money
    # behind has usually mispriced someone rather than found a bargain --
    # but forcing a spend can also buy a worse player, so this is optional and
    # its effect is measured rather than assumed.
    min_spend: float = 0.0


class InfeasibleSquad(ValueError):
    """No legal squad exists for these players and constraints."""


def _solver() -> pulp.LpSolver:
    """The CBC solver, preferring the one PuLP 4 will keep.

    ``COIN_CMD`` needs a separately installed binary (``pip install pulp[cbc]``)
    and is what survives into PuLP 4; ``PULP_CBC_CMD`` is the bundled solver
    that works out of the box today but is deprecated. Preferring the former
    when present means installing the extra is all that is ever needed.
    """
    coin = pulp.COIN_CMD(msg=False)
    if coin.available():
        return coin
    return pulp.PULP_CBC_CMD(msg=False)


def _validate(players: pd.DataFrame) -> None:
    missing = [column for column in REQUIRED_COLUMNS if column not in players.columns]
    if missing:
        raise ValueError(f"players is missing required columns: {missing}")


def optimise_squad(players: pd.DataFrame, constraints: SquadConstraints | None = None) -> Squad:
    """Return the highest-scoring legal squad, or raise if none exists.

    The objective counts a starter once, the captain twice (the armband doubles
    their score), and a bench player at ``bench_weight``.
    """
    _validate(players)
    constraints = constraints or SquadConstraints()

    pool = players[~players["element"].isin(constraints.must_exclude)]
    pool = pool.drop_duplicates(subset="element").reset_index(drop=True)
    if pool.empty:
        raise InfeasibleSquad("no players to choose from")

    elements = pool["element"].tolist()
    points = dict(zip(elements, pool["expected_points"], strict=True))
    price = dict(zip(elements, pool["price"], strict=True))
    position = dict(zip(elements, pool["position"], strict=True))
    club = dict(zip(elements, pool["team"], strict=True))

    problem = pulp.LpProblem("fpl_squad", pulp.LpMaximize)

    # Three nested decisions per player: bought, started, captained. Nesting is
    # enforced below -- you cannot start a player you did not buy.
    squad = problem.add_variable_dicts("squad", elements, cat="Binary")
    starting = problem.add_variable_dicts("starting", elements, cat="Binary")
    captain = problem.add_variable_dicts("captain", elements, cat="Binary")

    problem += pulp.lpSum(
        points[element] * starting[element]
        + points[element] * captain[element]
        + constraints.bench_weight * points[element] * (squad[element] - starting[element])
        for element in elements
    )

    problem += pulp.lpSum(squad[e] for e in elements) == constraints.squad_size
    problem += pulp.lpSum(starting[e] for e in elements) == constraints.starting_size
    problem += pulp.lpSum(captain[e] for e in elements) == 1
    problem += pulp.lpSum(price[e] * squad[e] for e in elements) <= constraints.budget
    if constraints.min_spend > 0:
        problem += pulp.lpSum(price[e] * squad[e] for e in elements) >= constraints.min_spend

    for element in elements:
        problem += starting[element] <= squad[element]
        problem += captain[element] <= starting[element]

    for role, count in constraints.composition.items():
        problem += pulp.lpSum(squad[e] for e in elements if position[e] == role) == count

    for role, limits in STARTING_XI_LIMITS.items():
        in_role = [starting[e] for e in elements if position[e] == role]
        problem += pulp.lpSum(in_role) >= limits.minimum
        problem += pulp.lpSum(in_role) <= limits.maximum

    for team in set(club.values()):
        problem += (
            pulp.lpSum(squad[e] for e in elements if club[e] == team) <= constraints.max_per_club
        )

    for element in constraints.must_include:
        if element not in points:
            raise InfeasibleSquad(f"required player {element} is not in the pool")
        problem += squad[element] == 1

    problem.solve(_solver())

    if pulp.LpStatus[problem.status] != "Optimal":
        raise InfeasibleSquad(
            "no legal squad exists for these players and constraints "
            f"(solver said: {pulp.LpStatus[problem.status]})"
        )

    chosen = {e for e in elements if squad[e].value() > 0.5}
    started = {e for e in elements if starting[e].value() > 0.5}
    captained = next(e for e in elements if captain[e].value() > 0.5)

    selected = pool[pool["element"].isin(chosen)]
    starters = selected[selected["element"].isin(started)]
    bench = selected[~selected["element"].isin(started)]

    return Squad(
        starting=starters.sort_values("expected_points", ascending=False).reset_index(drop=True),
        bench=bench.sort_values("expected_points", ascending=False).reset_index(drop=True),
        captain=int(captained),
        expected_points=float(starters["expected_points"].sum() + points[captained]),
        cost=float(selected["price"].sum()),
    )


def squad_value(squad_elements: list[int], players: pd.DataFrame) -> float:
    """Selling value of a set of players, at current prices."""
    owned = players[players["element"].isin(squad_elements)]
    return float(owned["price"].sum())
