"""Tests for the catalogue of season-opening strategies.

The registry exists so that "which model is best" is a measurement rather than
an argument. These tests protect the properties that make that true: every
strategy is interchangeable, none can see the season it is about to be scored
on, and adding one to the catalogue is enough to get it compared.
"""

from __future__ import annotations

import pandas as pd
import pytest

from fpl.models.preseason_strategies import (
    Components,
    ComponentsWithFixtures,
    PreseasonContext,
    PreseasonStrategy,
    strategies,
    strategy_by_name,
)


def pool(rows=3):
    return pd.DataFrame(
        [
            {
                "element": index,
                "match_key": f"player {index}",
                "player_name": f"Player {index}",
                "position": "Midfielder",
                "team": "Arsenal",
                "price": 5.0,
                "career_minutes": 2000.0,
                "career_appearances": 30.0,
                "seasons_seen": 2,
                "total_points_per_90": 4.0,
            }
            for index in range(1, rows + 1)
        ]
    )


def context(target="2025-26", prior=None):
    return PreseasonContext(target=target, prior_seasons=prior or {}, horizon=7)


def test_every_registered_strategy_satisfies_the_protocol():
    assert all(isinstance(strategy, PreseasonStrategy) for strategy in strategies())


def test_every_strategy_has_a_distinct_name():
    names = [strategy.name for strategy in strategies()]

    assert len(names) == len(set(names))


def test_every_strategy_declares_what_it_uses():
    """A strategy that ignores minutes is a different claim, not a worse version."""
    assert all(hasattr(strategy, "uses") for strategy in strategies())


def test_every_strategy_returns_a_value_for_every_player():
    """Interchangeability: the constructor must be able to consume any of them."""
    candidates = pool()

    for strategy in strategies():
        expected = strategy.expected_points(candidates, context())
        assert len(expected) == len(candidates), strategy.name


def test_every_strategy_survives_an_empty_prior_history():
    """The first season of the archive has nothing behind it."""
    for strategy in strategies():
        assert strategy.expected_points(pool(), context(prior={})) is not None


def test_a_strategy_can_be_looked_up_by_name():
    assert strategy_by_name("Components").name == "Components"


def test_an_unknown_strategy_raises_rather_than_defaulting():
    """A silent fallback would report results under the wrong name."""
    with pytest.raises(KeyError):
        strategy_by_name("NoSuchModel")


def test_the_error_names_the_strategies_that_do_exist():
    with pytest.raises(KeyError, match="Components"):
        strategy_by_name("NoSuchModel")


# -- The context ----------------------------------------------------------


def test_the_context_knows_when_defensive_contributions_are_scored():
    assert context("2025-26").scores_defensive_contributions
    assert not context("2024-25").scores_defensive_contributions


def test_the_context_offers_no_route_to_the_target_season():
    """The point-in-time guarantee: a strategy cannot read its own answer."""
    fields = set(PreseasonContext.__dataclass_fields__)

    assert "season" not in fields
    assert "target_season_data" not in fields


def test_the_latest_prior_is_the_most_recent_one():
    prior = {"2022-23": pd.DataFrame([{"a": 1}]), "2024-25": pd.DataFrame([{"a": 2}])}

    assert context(prior=prior).latest_prior["a"].iloc[0] == 2


def test_team_defence_is_computed_once_and_reused():
    ctx = context(prior={})

    assert ctx.team_defence is ctx.team_defence


# -- Fixture usage --------------------------------------------------------


def test_the_fixture_variant_is_the_component_model_with_fixtures_on():
    assert ComponentsWithFixtures().use_fixtures
    assert not Components().use_fixtures


def test_the_fixture_variant_declares_that_it_uses_fixtures():
    assert "fixtures" in ComponentsWithFixtures().uses
