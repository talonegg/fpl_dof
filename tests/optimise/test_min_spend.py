"""Tests for the minimum-spend constraint."""

from __future__ import annotations

import pandas as pd
import pytest

from fpl.optimise.squad import InfeasibleSquad, SquadConstraints, optimise_squad


def pool():
    """A pool with cheap and expensive options in every position."""
    rows = []
    element = 0
    plan = [("Goalkeeper", 4), ("Defender", 10), ("Midfielder", 10), ("Forward", 6)]
    for position, count in plan:
        for index in range(count):
            # Six clubs: three per club is the limit, so fewer cannot fill 15.
            for club in range(6):
                element += 1
                expensive = index % 2 == 0
                rows.append(
                    {
                        "element": element,
                        "position": position,
                        "team": f"Club{club}",
                        "price": 8.0 if expensive else 4.0,
                        # Cheap players score the same, so an unconstrained
                        # optimiser has no reason to spend.
                        "expected_points": 5.0,
                    }
                )
    return pd.DataFrame(rows)


def test_without_a_floor_the_optimiser_may_leave_money_unspent():
    squad = optimise_squad(pool(), SquadConstraints())

    assert squad.cost < 100.0


def test_a_minimum_spend_is_respected():
    squad = optimise_squad(pool(), SquadConstraints(min_spend=95.0))

    assert squad.cost >= 95.0


def test_the_budget_still_binds_above_the_floor():
    squad = optimise_squad(pool(), SquadConstraints(min_spend=95.0))

    assert squad.cost <= 100.0


def test_a_floor_above_the_budget_is_infeasible():
    with pytest.raises(InfeasibleSquad):
        optimise_squad(pool(), SquadConstraints(min_spend=101.0))


def test_a_floor_the_pool_cannot_reach_is_infeasible():
    cheap = pool()
    cheap["price"] = 4.0

    with pytest.raises(InfeasibleSquad):
        optimise_squad(cheap, SquadConstraints(min_spend=95.0))


def test_zero_is_the_default_and_changes_nothing():
    assert SquadConstraints().min_spend == 0.0
