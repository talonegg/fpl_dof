"""Deciding what to change about a squad you already own.

Buying a squad from nothing and improving one you have are different problems.
The second has a cost the first does not: every transfer beyond your free one
costs 4 points, which is roughly what a good player scores in a week. That
turns "is this player better?" into "is this player better *by more than the
price of getting them*?", and the answer is usually no.

So the planner's job is mostly to say **no**. It compares doing nothing,
using the free transfer, and taking hits, and returns whichever wins over the
horizon. Rolling a transfer is a real option and appears as such.

Horizon matters more here than anywhere else. A transfer judged on one gameweek
almost never pays; the same transfer over six gameweeks often does, because the
4 points are paid once and the gain accrues weekly.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import pandas as pd

from fpl.domain.rules import (
    FREE_TRANSFERS_PER_GAMEWEEK,
    MAX_ROLLED_FREE_TRANSFERS,
    TRANSFER_HIT_POINTS,
)
from fpl.optimise.squad import (
    InfeasibleSquad,
    SquadConstraints,
    optimise_squad,
)

DEFAULT_HORIZON = 5

# Trying every possible number of transfers is pointless: beyond a handful the
# hits swamp any plausible gain. The floor is the maximum number of free
# transfers that can be banked -- a manager holding five owes nothing for the
# fifth, so refusing to consider it would rule out a move that is free.
MAX_TRANSFERS_CONSIDERED = max(4, MAX_ROLLED_FREE_TRANSFERS)


@dataclass
class TransferPlan:
    """A recommended set of changes, and the arithmetic behind it."""

    transfers_out: list[int]
    transfers_in: list[int]
    transfer_count: int
    hits: int
    points_cost: int
    gross_gain: float
    net_gain: float
    horizon: int

    @property
    def is_worth_it(self) -> bool:
        """Whether acting beats doing nothing, after paying for the hits."""
        return self.net_gain > 0

    def describe(self, names: dict[int, str] | None = None) -> str:
        if self.transfer_count == 0:
            return "Roll the transfer — no move gains enough to be worth making."

        def label(element: int) -> str:
            return names.get(element, str(element)) if names else str(element)

        moves = ", ".join(
            f"{label(out)} → {label(into)}"
            for out, into in zip(self.transfers_out, self.transfers_in, strict=False)
        )
        cost = f" (−{self.points_cost} for {self.hits} hit(s))" if self.hits else ""
        return f"{moves}{cost}: +{self.net_gain:.1f} points over {self.horizon} gameweeks"


def _squad_expected_points(
    squad_elements: list[int], players: pd.DataFrame, constraints: SquadConstraints
) -> float:
    """Best achievable score from a fixed set of fifteen.

    The squad is given, but the starting eleven and captain are still choices,
    so this re-optimises those. Scoring a squad by its raw total would credit
    points to players sat on the bench.
    """
    owned = players[players["element"].isin(squad_elements)]
    if len(owned) < constraints.squad_size:
        raise InfeasibleSquad("current squad is not a full fifteen in this pool")

    held = optimise_squad(
        owned,
        # `replace` rather than a new SquadConstraints: constructing one fresh
        # silently reverts squad_size and starting_size to their defaults, so a
        # caller's custom sizes were being ignored here.
        replace(
            constraints,
            # The squad is fixed, so budget and club limits are already
            # satisfied by construction. Relax them -- an existing squad must
            # never be rejected by its own selling prices -- but keep the
            # budget finite, since a solver cannot take an infinite bound.
            budget=float(owned["price"].sum()),
            max_per_club=constraints.squad_size,
            must_include=(),
            must_exclude=(),
        ),
    )
    return held.expected_points


def plan_transfers(
    players: pd.DataFrame,
    current_squad: list[int],
    bank: float = 0.0,
    free_transfers: int = FREE_TRANSFERS_PER_GAMEWEEK,
    horizon: int = DEFAULT_HORIZON,
    max_transfers: int = MAX_TRANSFERS_CONSIDERED,
    constraints: SquadConstraints | None = None,
) -> TransferPlan:
    """Find the transfer plan with the best net gain over ``horizon`` gameweeks.

    ``players`` must carry ``expected_points`` for a single gameweek; the gain
    is scaled by ``horizon`` on the assumption that the edge persists. That is
    an approximation, and an optimistic one -- it ignores that form and
    fixtures both move -- so it is stated rather than hidden.

    ``bank`` is money not tied up in the squad. Selling prices are taken as
    current prices, which ignores FPL's sell-on rule for profit.
    """
    constraints = constraints or SquadConstraints()
    current_squad = list(current_squad)

    # Validate here rather than letting the search loop swallow it. Every
    # candidate would raise InfeasibleSquad, the loop would `continue` past
    # each one, and an impossible request would come back as a cheerful
    # "roll the transfer".
    available = set(players["element"])
    for required in constraints.must_include:
        if required not in available:
            raise InfeasibleSquad(f"required player {required} is not in the pool")

    baseline = _squad_expected_points(current_squad, players, constraints)
    budget = float(players[players["element"].isin(current_squad)]["price"].sum() + bank)

    best = TransferPlan(
        transfers_out=[],
        transfers_in=[],
        transfer_count=0,
        hits=0,
        points_cost=0,
        gross_gain=0.0,
        net_gain=0.0,
        horizon=horizon,
    )

    for count in range(1, max_transfers + 1):
        # Force the optimiser to keep all but `count` of the current squad, by
        # requiring at least (15 - count) of them. Expressed as a lower bound
        # on retained players rather than by enumerating which to sell, which
        # would be combinatorial.
        keep = constraints.squad_size - count
        try:
            candidate = _best_squad_keeping(players, current_squad, keep, budget, constraints)
        except InfeasibleSquad:
            continue

        actual_out = [e for e in current_squad if e not in set(candidate.players["element"])]
        actual_in = [int(e) for e in candidate.players["element"] if e not in set(current_squad)]
        moves = len(actual_in)
        if moves == 0:
            continue

        hits = max(0, moves - free_transfers)
        points_cost = hits * TRANSFER_HIT_POINTS
        gross = (candidate.expected_points - baseline) * horizon
        net = gross - points_cost

        if net > best.net_gain:
            best = TransferPlan(
                transfers_out=[int(e) for e in actual_out],
                transfers_in=actual_in,
                transfer_count=moves,
                hits=hits,
                points_cost=points_cost,
                gross_gain=float(gross),
                net_gain=float(net),
                horizon=horizon,
            )

    return best


def _best_squad_keeping(
    players: pd.DataFrame,
    current_squad: list[int],
    keep: int,
    budget: float,
    constraints: SquadConstraints,
):
    """Best squad that retains at least ``keep`` of the current players."""
    import pulp

    from fpl.optimise.squad import _solver

    # must_exclude has to be honoured here too. optimise_squad applies both
    # lists; if this path ignored them the same SquadConstraints would mean
    # different things depending on which entry point you called.
    pool = players[~players["element"].isin(constraints.must_exclude)]
    pool = pool.drop_duplicates(subset="element").reset_index(drop=True)
    if pool.empty:
        raise InfeasibleSquad("no players to choose from")
    elements = pool["element"].tolist()
    owned = set(current_squad)

    for required in constraints.must_include:
        if required not in elements:
            raise InfeasibleSquad(f"required player {required} is not in the pool")

    problem = pulp.LpProblem("fpl_transfers", pulp.LpMaximize)
    squad = problem.add_variable_dicts("squad", elements, cat="Binary")
    starting = problem.add_variable_dicts("starting", elements, cat="Binary")
    captain = problem.add_variable_dicts("captain", elements, cat="Binary")

    points = dict(zip(elements, pool["expected_points"], strict=True))
    price = dict(zip(elements, pool["price"], strict=True))
    position = dict(zip(elements, pool["position"], strict=True))
    club = dict(zip(elements, pool["team"], strict=True))

    problem += pulp.lpSum(
        points[e] * starting[e]
        + points[e] * captain[e]
        + constraints.bench_weight * points[e] * (squad[e] - starting[e])
        for e in elements
    )

    problem += pulp.lpSum(squad[e] for e in elements) == constraints.squad_size
    problem += pulp.lpSum(starting[e] for e in elements) == constraints.starting_size
    problem += pulp.lpSum(captain[e] for e in elements) == 1
    problem += pulp.lpSum(price[e] * squad[e] for e in elements) <= budget
    problem += pulp.lpSum(squad[e] for e in elements if e in owned) >= keep

    for required in constraints.must_include:
        problem += squad[required] == 1

    for element in elements:
        problem += starting[element] <= squad[element]
        problem += captain[element] <= starting[element]

    for role, count in constraints.composition.items():
        problem += pulp.lpSum(squad[e] for e in elements if position[e] == role) == count

    from fpl.domain.rules import STARTING_XI_LIMITS

    for role, limits in STARTING_XI_LIMITS.items():
        in_role = [starting[e] for e in elements if position[e] == role]
        problem += pulp.lpSum(in_role) >= limits.minimum
        problem += pulp.lpSum(in_role) <= limits.maximum

    for team in set(club.values()):
        problem += (
            pulp.lpSum(squad[e] for e in elements if club[e] == team) <= constraints.max_per_club
        )

    problem.solve(_solver())
    if pulp.LpStatus[problem.status] != "Optimal":
        raise InfeasibleSquad("no legal squad retains that many current players")

    from fpl.optimise.squad import Squad

    chosen = {e for e in elements if squad[e].value() > 0.5}
    started = {e for e in elements if starting[e].value() > 0.5}
    captained = next(e for e in elements if captain[e].value() > 0.5)

    selected = pool[pool["element"].isin(chosen)]
    starters = selected[selected["element"].isin(started)]

    return Squad(
        starting=starters.reset_index(drop=True),
        bench=selected[~selected["element"].isin(started)].reset_index(drop=True),
        captain=int(captained),
        expected_points=float(starters["expected_points"].sum() + points[captained]),
        cost=float(selected["price"].sum()),
    )
