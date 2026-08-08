"""Tests for squad optimisation.

An optimiser is testable in a way a model is not: for small worlds the right
answer can be worked out by hand, and every constraint has a violation that
must be impossible rather than merely unlikely. So these check three things —
that known optima are found, that no constraint can be broken, and that the
result is stable for a fixed input.
"""

from __future__ import annotations

import pandas as pd
import pytest

from fpl.domain.rules import (
    BUDGET_MILLIONS,
    MAX_PLAYERS_PER_CLUB,
    SQUAD_COMPOSITION,
    STARTING_XI_SIZE,
)
from fpl.optimise.squad import (
    InfeasibleSquad,
    SquadConstraints,
    optimise_squad,
    squad_value,
)


def make_pool(per_position=6, clubs=8, price=5.0, points=None):
    """A pool with enough players at every position to build a legal squad."""
    rows = []
    element = 1
    for position in ("Goalkeeper", "Defender", "Midfielder", "Forward"):
        for index in range(per_position):
            rows.append(
                {
                    "element": element,
                    "web_name": f"{position[:3]}{index}",
                    "position": position,
                    "team": element % clubs,
                    "price": price,
                    "expected_points": points(position, index) if points else 1.0,
                }
            )
            element += 1
    return pd.DataFrame(rows)


def test_a_squad_has_fifteen_players():
    squad = optimise_squad(make_pool())

    assert len(squad.players) == 15


def test_eleven_players_start_and_four_sit():
    squad = optimise_squad(make_pool())

    assert len(squad.starting) == STARTING_XI_SIZE
    assert len(squad.bench) == 4


def test_the_squad_composition_is_exactly_two_five_five_three():
    squad = optimise_squad(make_pool())

    counts = squad.players["position"].value_counts()
    for position, expected in SQUAD_COMPOSITION.items():
        assert counts[position] == expected


def test_the_budget_is_never_exceeded():
    """Expensive players are the best ones, so the budget has to bind."""
    pool = make_pool(per_position=8, price=4.0)
    # Half the pool is brilliant but costly: affordable only in part.
    pool.loc[pool.index % 2 == 0, ["price", "expected_points"]] = [11.0, 9.0]

    squad = optimise_squad(pool)

    assert squad.cost <= BUDGET_MILLIONS
    # Confirm the constraint actually bit, or this proves nothing.
    assert squad.cost > BUDGET_MILLIONS * 0.8


def test_no_more_than_three_players_from_one_club():
    """The constraint a greedy points-first approach quietly violates."""
    pool = make_pool(per_position=8, clubs=8)
    # Club 0's players are the best in the league at every position, so an
    # unconstrained optimiser would take far more than three of them.
    pool.loc[pool["team"] == 0, "expected_points"] = 20.0

    squad = optimise_squad(pool)

    counts = squad.players["team"].value_counts()
    assert counts.max() <= MAX_PLAYERS_PER_CLUB
    # It should still take its full allowance of the good club.
    assert counts.get(0, 0) == MAX_PLAYERS_PER_CLUB


def test_the_starting_eleven_is_a_legal_formation():
    squad = optimise_squad(make_pool())

    counts = squad.starting["position"].value_counts()
    assert counts.get("Goalkeeper", 0) == 1
    assert 3 <= counts.get("Defender", 0) <= 5
    assert 1 <= counts.get("Forward", 0) <= 3


def test_the_best_players_are_the_ones_who_start_within_their_position():
    """Compared across the whole squad this is false, and correctly so.

    A squad carries two goalkeepers but may start only one, so the reserve
    keeper sits despite possibly out-scoring an outfield starter. The real
    guarantee is per position.
    """
    squad = optimise_squad(make_pool(points=lambda p, i: 10.0 - i))

    for position in squad.bench["position"].unique():
        benched = squad.bench[squad.bench["position"] == position]
        starters = squad.starting[squad.starting["position"] == position]
        if starters.empty:
            continue
        assert starters["expected_points"].min() >= benched["expected_points"].max()


def test_the_captain_is_a_starter():
    squad = optimise_squad(make_pool(points=lambda p, i: 10.0 - i))

    assert squad.captain in set(squad.starting["element"])


def test_the_captain_is_the_highest_scoring_starter():
    """The armband doubles a score, so it belongs on the best player.

    Asserted on the points rather than the identity: several players can tie
    at the top, and any of them is an equally correct answer.
    """
    squad = optimise_squad(make_pool(points=lambda p, i: 10.0 - i))

    captain_points = squad.starting[squad.starting["element"] == squad.captain][
        "expected_points"
    ].iloc[0]
    assert captain_points == squad.starting["expected_points"].max()


def test_expected_points_counts_the_captain_twice():
    squad = optimise_squad(make_pool(points=lambda p, i: 10.0 - i))

    captain_points = squad.starting[squad.starting["element"] == squad.captain][
        "expected_points"
    ].iloc[0]
    assert squad.expected_points == pytest.approx(
        squad.starting["expected_points"].sum() + captain_points
    )


def test_a_cheap_high_scorer_is_preferred_to_an_expensive_one():
    """The whole reason to optimise rather than pick by points."""
    pool = make_pool(per_position=8, price=4.0)
    # One brilliant but ruinously expensive forward.
    pool.loc[pool["position"] == "Forward", "expected_points"] = 2.0
    expensive = pool[pool["position"] == "Forward"].index[0]
    pool.loc[expensive, ["price", "expected_points"]] = [60.0, 12.0]

    squad = optimise_squad(pool)

    # Taking them would leave nothing for the other fourteen.
    assert pool.loc[expensive, "element"] not in set(squad.players["element"])


def test_a_required_player_is_always_included():
    pool = make_pool(points=lambda p, i: 10.0 - i)
    # The worst player in the pool.
    unwanted = int(pool.sort_values("expected_points").iloc[0]["element"])

    squad = optimise_squad(pool, SquadConstraints(must_include=(unwanted,)))

    assert unwanted in set(squad.players["element"])


def test_an_excluded_player_is_never_included():
    pool = make_pool(points=lambda p, i: 10.0 - i)
    best = int(pool.sort_values("expected_points", ascending=False).iloc[0]["element"])

    squad = optimise_squad(pool, SquadConstraints(must_exclude=(best,)))

    assert best not in set(squad.players["element"])


def test_requiring_a_player_who_does_not_exist_is_an_error():
    with pytest.raises(InfeasibleSquad, match="not in the pool"):
        optimise_squad(make_pool(), SquadConstraints(must_include=(9999,)))


def test_too_few_players_is_infeasible_rather_than_a_bad_squad():
    """Better to refuse than to return something illegal."""
    with pytest.raises(InfeasibleSquad):
        optimise_squad(make_pool(per_position=2))


def test_an_unaffordable_pool_is_infeasible():
    with pytest.raises(InfeasibleSquad):
        optimise_squad(make_pool(price=50.0))


def test_an_empty_pool_is_infeasible():
    with pytest.raises(InfeasibleSquad, match="no players"):
        optimise_squad(make_pool().head(0))


def test_missing_columns_are_reported_clearly():
    with pytest.raises(ValueError, match="missing required columns"):
        optimise_squad(pd.DataFrame([{"element": 1}]))


def test_the_same_input_gives_the_same_squad():
    """A recommendation that changes between identical runs cannot be trusted."""
    pool = make_pool(points=lambda p, i: 10.0 - i)

    first = optimise_squad(pool)
    second = optimise_squad(pool)

    assert sorted(first.players["element"]) == sorted(second.players["element"])
    assert first.captain == second.captain


def test_formation_is_reported_readably():
    squad = optimise_squad(make_pool())

    defenders, midfielders, forwards = (int(n) for n in squad.formation.split("-"))
    assert defenders + midfielders + forwards == STARTING_XI_SIZE - 1


def test_summary_mentions_cost_and_points():
    summary = optimise_squad(make_pool()).summary()

    assert "£" in summary
    assert "expected points" in summary


def test_squad_value_sums_current_prices():
    pool = make_pool(price=5.0)

    assert squad_value([1, 2, 3], pool) == pytest.approx(15.0)


def test_a_duplicated_player_row_cannot_be_bought_twice():
    pool = make_pool(points=lambda p, i: 10.0 - i)
    duplicated = pd.concat([pool, pool.head(1)])

    squad = optimise_squad(duplicated)

    assert not squad.players["element"].duplicated().any()
