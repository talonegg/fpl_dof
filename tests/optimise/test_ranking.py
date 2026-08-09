"""Tests for enumerating the best N squads.

The property that matters is that this is a *ranking*, not a sample. Each squad
returned is the best one that is not already on the list, which is what makes
the twentieth entry meaningful rather than arbitrary.
"""

from __future__ import annotations

import pandas as pd
import pytest

from fpl.optimise.ranking import rank_squads, squad_differences
from fpl.optimise.squad import SquadConstraints

# The solver's objective carries float noise far below a meaningful point.
TOLERANCE = 1e-6

PLAN = [("Goalkeeper", 6), ("Defender", 14), ("Midfielder", 14), ("Forward", 8)]


def pool(distinct_points: bool = True):
    rows = []
    element = 0
    for position, count in PLAN:
        for _ in range(count):
            element += 1
            rows.append(
                {
                    "element": element,
                    "player_name": f"Player {element}",
                    "position": position,
                    "team": f"Club{element % 10}",
                    "price": 4.5,
                    "expected_points": (100 - element) if distinct_points else 10,
                }
            )
    return pd.DataFrame(rows)


def test_twenty_squads_are_returned():
    assert len(rank_squads(pool(), count=20)) == 20


def test_every_squad_is_distinct():
    """The whole point of a no-good cut: no combination repeats."""
    shortlist = rank_squads(pool(), count=20)

    combinations = {frozenset(entry.players["element"]) for entry in shortlist.squads}
    assert len(combinations) == 20


def test_each_squad_still_has_fifteen_players():
    shortlist = rank_squads(pool(), count=5)

    assert all(len(entry.players) == 15 for entry in shortlist.squads)


def test_the_ranking_never_improves_as_it_descends():
    """Ordered by the objective, which is what the optimiser actually maximises."""
    shortlist = rank_squads(pool(), count=20)

    scores = [entry.score for entry in shortlist.squads]
    # Tolerance because the solver returns objective values carrying float
    # noise well below any meaningful difference in points.
    assert all(a >= b - TOLERANCE for a, b in zip(scores, scores[1:], strict=False))


def test_the_ranking_is_by_the_objective_not_the_starting_eleven():
    """A stronger bench can win on the objective while showing fewer XI points.

    Ranking by ``expected_points`` would therefore produce a list that looks
    wrongly sorted. This is the bug that only real data exposed.
    """
    shortlist = rank_squads(pool(), count=20)

    assert all(entry.score >= entry.expected_points for entry in shortlist.squads)


def test_the_gap_is_never_negative():
    shortlist = rank_squads(pool(), count=20)

    assert all(entry.gap_to_best >= -TOLERANCE for entry in shortlist.squads)


def test_the_first_squad_is_the_unconstrained_optimum():
    """Ranking must not cost anything at the top of the list."""
    from fpl.optimise.squad import optimise_squad

    best = optimise_squad(pool())
    ranked = rank_squads(pool(), count=3).squads[0]

    assert ranked.score == pytest.approx(best.objective)


def test_the_gap_to_best_starts_at_zero_and_grows():
    shortlist = rank_squads(pool(), count=10)

    gaps = [entry.gap_to_best for entry in shortlist.squads]
    assert gaps[0] == 0
    assert all(a <= b + TOLERANCE for a, b in zip(gaps, gaps[1:], strict=False))


def test_the_second_squad_differs_from_the_first_by_as_little_as_possible():
    """Excluding players rather than combinations would change many at once."""
    shortlist = rank_squads(pool(), count=2)

    assert shortlist.squads[1].changes_from_best == 1


def test_the_spread_measures_top_against_bottom():
    shortlist = rank_squads(pool(), count=10)

    assert shortlist.spread == (shortlist.squads[0].score - shortlist.squads[-1].score)


def test_an_empty_pool_ranks_nothing():
    assert len(rank_squads(pd.DataFrame(), count=20)) == 0


def test_asking_for_no_squads_returns_none():
    assert len(rank_squads(pool(), count=0)) == 0


def test_a_pool_that_cannot_fill_a_squad_returns_what_it_can():
    """Shorter than asked for, rather than padded with illegal squads."""
    thin = pool().head(15)

    assert len(rank_squads(thin, count=20)) <= 1


def test_ranking_does_not_mutate_the_constraints_it_was_given():
    """A constraints object reused after a ranking run must behave the same."""
    constraints = SquadConstraints()
    rank_squads(pool(), count=3, constraints=constraints)

    assert constraints.forbidden_squads == ()


def test_the_table_has_one_row_per_squad():
    shortlist = rank_squads(pool(), count=7)

    assert len(shortlist.table()) == 7


def test_the_table_of_nothing_is_empty():
    assert rank_squads(pd.DataFrame(), count=5).table().empty


# -- What separates them --------------------------------------------------


def test_the_differences_name_who_comes_in_and_who_goes_out():
    differences = squad_differences(rank_squads(pool(), count=5))

    assert set(differences["change"]) == {"in", "out"}


def test_the_differences_skip_the_best_squad_itself():
    differences = squad_differences(rank_squads(pool(), count=5))

    assert 1 not in set(differences["rank"])


def test_a_single_squad_has_nothing_to_compare():
    assert squad_differences(rank_squads(pool(), count=1)).empty
