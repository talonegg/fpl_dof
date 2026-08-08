"""Tests for transfer planning.

The planner's most important job is refusing to act. A 4-point hit is about a
week's return from a good player, so most attractive-looking transfers lose
money, and a planner that does not say "roll it" often is broken.
"""

from __future__ import annotations

import pandas as pd
import pytest

from fpl.domain.rules import TRANSFER_HIT_POINTS
from fpl.optimise.transfers import plan_transfers

POSITIONS = ["Goalkeeper"] * 3 + ["Defender"] * 7 + ["Midfielder"] * 7 + ["Forward"] * 5


def make_world(upgrade_points=None, upgrade_price=5.0):
    """A 22-player world: 15 owned, 7 alternatives.

    Everyone is worth 2 points and costs 5.0, so nothing is worth doing unless
    a test makes it so.
    """
    rows = []
    for index, position in enumerate(POSITIONS, start=1):
        rows.append(
            {
                "element": index,
                "web_name": f"P{index}",
                "position": position,
                "team": index % 10,
                "price": 5.0,
                "expected_points": 2.0,
            }
        )
    world = pd.DataFrame(rows)

    if upgrade_points is not None:
        # Element 22 is a forward who is better than the forward at element 21.
        world.loc[world["element"] == 22, "expected_points"] = upgrade_points
        world.loc[world["element"] == 22, "price"] = upgrade_price

    return world


# The owned squad: 2 GK, 5 DEF, 5 MID, 3 FWD taken from the front of each block.
OWNED = [1, 2] + [4, 5, 6, 7, 8] + [11, 12, 13, 14, 15] + [18, 19, 20]


def test_an_identical_alternative_is_not_worth_a_transfer():
    plan = plan_transfers(make_world(), OWNED, horizon=5)

    assert plan.transfer_count == 0
    assert not plan.is_worth_it


def test_doing_nothing_is_described_as_rolling_the_transfer():
    plan = plan_transfers(make_world(), OWNED, horizon=5)

    assert "Roll the transfer" in plan.describe()


def test_a_clear_upgrade_is_taken_with_the_free_transfer():
    world = make_world(upgrade_points=8.0)

    plan = plan_transfers(world, OWNED, horizon=5)

    assert plan.transfer_count == 1
    assert 22 in plan.transfers_in
    assert plan.hits == 0
    assert plan.points_cost == 0


def test_a_marginal_upgrade_is_refused_when_it_would_cost_a_hit():
    """Two transfers with one free: the second must clear 4 points to be worth it."""
    world = make_world(upgrade_points=2.5)
    # A second, equally marginal upgrade elsewhere.
    world.loc[world["element"] == 17, "expected_points"] = 2.5

    plan = plan_transfers(world, OWNED, free_transfers=1, horizon=1)

    assert plan.hits == 0, "a hit was taken for a gain smaller than the hit"


def test_a_big_upgrade_justifies_a_hit_over_a_long_horizon():
    """The 4 points are paid once; the gain accrues every week."""
    world = make_world(upgrade_points=9.0)
    world.loc[world["element"] == 17, "expected_points"] = 9.0

    plan = plan_transfers(world, OWNED, free_transfers=1, horizon=6)

    assert plan.transfer_count == 2
    assert plan.hits == 1
    assert plan.points_cost == TRANSFER_HIT_POINTS
    assert plan.is_worth_it


def test_the_same_upgrade_is_refused_over_a_single_gameweek():
    """Horizon is what decides most transfers, so it must actually bite."""
    world = make_world(upgrade_points=5.0)
    world.loc[world["element"] == 17, "expected_points"] = 5.0

    long_plan = plan_transfers(world, OWNED, free_transfers=1, horizon=6)
    short_plan = plan_transfers(world, OWNED, free_transfers=1, horizon=1)

    assert long_plan.transfer_count >= short_plan.transfer_count


def test_extra_free_transfers_remove_the_hit():
    world = make_world(upgrade_points=9.0)
    world.loc[world["element"] == 17, "expected_points"] = 9.0

    plan = plan_transfers(world, OWNED, free_transfers=2, horizon=6)

    assert plan.transfer_count == 2
    assert plan.hits == 0


def test_an_unaffordable_upgrade_is_not_recommended():
    world = make_world(upgrade_points=20.0, upgrade_price=60.0)

    plan = plan_transfers(world, OWNED, bank=0.0, horizon=5)

    assert 22 not in plan.transfers_in


def test_money_in_the_bank_makes_an_upgrade_affordable():
    world = make_world(upgrade_points=9.0, upgrade_price=12.0)

    without_bank = plan_transfers(world, OWNED, bank=0.0, horizon=5)
    with_bank = plan_transfers(world, OWNED, bank=10.0, horizon=5)

    assert 22 not in without_bank.transfers_in
    assert 22 in with_bank.transfers_in


def test_transfers_in_and_out_are_the_same_length():
    world = make_world(upgrade_points=9.0)

    plan = plan_transfers(world, OWNED, horizon=5)

    assert len(plan.transfers_in) == len(plan.transfers_out)


def test_a_player_is_never_transferred_in_when_already_owned():
    world = make_world(upgrade_points=9.0)

    plan = plan_transfers(world, OWNED, horizon=5)

    assert not set(plan.transfers_in) & set(OWNED)


def test_net_gain_is_gross_minus_the_hits():
    world = make_world(upgrade_points=9.0)
    world.loc[world["element"] == 17, "expected_points"] = 9.0

    plan = plan_transfers(world, OWNED, free_transfers=1, horizon=6)

    assert plan.net_gain == pytest.approx(plan.gross_gain - plan.points_cost)


def test_describe_names_players_when_given_a_lookup():
    world = make_world(upgrade_points=9.0)
    plan = plan_transfers(world, OWNED, horizon=5)

    described = plan.describe({21: "Old", 22: "New"})

    assert "New" in described


def test_a_squad_that_is_not_fifteen_players_is_rejected():
    from fpl.optimise.squad import InfeasibleSquad

    with pytest.raises(InfeasibleSquad, match="full fifteen"):
        plan_transfers(make_world(), OWNED[:10], horizon=5)
