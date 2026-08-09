"""Tests for constructing and explaining a season-opening squad."""

from __future__ import annotations

import pandas as pd

from fpl.optimise.preseason import (
    construct_squad,
    explain_selection,
    recommend_squad,
)
from fpl.optimise.squad import SquadConstraints

PLAN = [("Goalkeeper", 4), ("Defender", 10), ("Midfielder", 10), ("Forward", 6)]


def pool(price=5.0):
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
                    "team": f"Club{element % 8}",
                    "price": price,
                }
            )
    return pd.DataFrame(rows)


def points(candidates):
    return pd.Series(range(len(candidates)), index=candidates.index, dtype="float64")


def test_a_constructed_squad_has_fifteen_players():
    squad = construct_squad(pool(), points(pool()))

    assert len(squad.players) == 15


def test_an_impossible_budget_returns_nothing_rather_than_an_illegal_squad():
    """A squad silently returned over budget would be scored as if fieldable."""
    assert construct_squad(pool(price=50.0), points(pool())) is None


def test_an_empty_pool_constructs_nothing():
    assert construct_squad(pd.DataFrame(), pd.Series(dtype="float64")) is None


def test_players_the_model_cannot_value_are_excluded_not_guessed():
    candidates = pool()
    expected = points(candidates)
    # Defenders, where the pool has slack -- nulling the four goalkeepers
    # would make the squad infeasible rather than merely poorer.
    expected.iloc[4:9] = None

    squad = construct_squad(candidates, expected)

    assert not set(squad.players["element"]) & set(candidates["element"].iloc[4:9])


# -- The recommendation ---------------------------------------------------


def test_a_recommendation_reports_how_many_players_it_could_see():
    candidates = pool()
    expected = points(candidates)
    expected.iloc[4:8] = None

    recommendation = recommend_squad(candidates, expected, "Components")

    assert recommendation.excluded == 4
    assert recommendation.considered == len(candidates) - 4


def test_excluded_players_produce_a_warning():
    candidates = pool()
    expected = points(candidates)
    expected.iloc[4:8] = None

    recommendation = recommend_squad(candidates, expected, "Components")

    assert any("no usable history" in warning for warning in recommendation.warnings)


def test_a_blind_defensive_season_is_warned_about():
    """2025-26: the points existed and the model could not see them."""
    candidates = pool()

    recommendation = recommend_squad(
        candidates, points(candidates), "Components", defensive_status="blind"
    )

    assert any("defensive contributions" in warning for warning in recommendation.warnings)


def test_a_forecastable_defensive_season_is_not_warned_about():
    candidates = pool()

    recommendation = recommend_squad(
        candidates, points(candidates), "Components", defensive_status="forecast"
    )

    assert not any("defensive contributions" in warning for warning in recommendation.warnings)


def test_a_clean_run_carries_no_warnings():
    candidates = pool()

    recommendation = recommend_squad(candidates, points(candidates), "Components")

    assert recommendation.warnings == []


def test_the_summary_names_the_strategy_behind_it():
    candidates = pool()

    recommendation = recommend_squad(candidates, points(candidates), "Components")

    assert "Components" in recommendation.summary()


def test_a_recommendation_that_cannot_be_built_is_none():
    assert recommend_squad(pool(price=50.0), points(pool()), "Components") is None


def test_a_minimum_spend_the_pool_cannot_reach_is_infeasible():
    cheap = pool(price=4.0)

    assert construct_squad(cheap, points(cheap), SquadConstraints(min_spend=95.0)) is None


# -- Explaining it --------------------------------------------------------


def test_the_explanation_covers_the_whole_squad():
    candidates = pool()
    expected = points(candidates)
    squad = construct_squad(candidates, expected)

    assert len(explain_selection(candidates, expected, squad)) == 15


def test_the_explanation_separates_starters_from_the_bench():
    candidates = pool()
    expected = points(candidates)
    squad = construct_squad(candidates, expected)

    roles = explain_selection(candidates, expected, squad)["role"]
    assert set(roles) == {"start", "bench"}


def test_the_explanation_reports_value_per_million():
    """Under a budget the optimiser buys value, and hiding that invites overrides."""
    candidates = pool()
    expected = points(candidates)
    squad = construct_squad(candidates, expected)

    assert "value" in explain_selection(candidates, expected, squad).columns


def test_explaining_nothing_is_empty():
    assert explain_selection(pd.DataFrame(), pd.Series(dtype="float64"), None).empty
