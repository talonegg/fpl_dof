"""Tests for penalty duty.

The taker shares are assumptions rather than measurements, so these tests
check the *structure* — ordering, scaling, bounds — rather than pinning
numbers that are expected to be replaced once daily captures accumulate.
"""

from __future__ import annotations

import pandas as pd
import pytest

from fpl.features.penalties import (
    PENALTIES_AWARDED_PER_TEAM_MATCH,
    PENALTY_CONVERSION,
    TAKER_SHARE,
    UNATTRIBUTED_SHARE,
    add_penalty_metrics,
    expected_penalty_goals,
    implied_attempts,
    taker_probability,
)

PLAYERS = pd.DataFrame(
    [
        {"web_name": "First", "penalties_order": 1},
        {"web_name": "Second", "penalties_order": 2},
        {"web_name": "Third", "penalties_order": 3},
        {"web_name": "Unranked", "penalties_order": None},
    ]
)


def test_the_designated_taker_is_most_likely_to_take_it():
    result = taker_probability(PLAYERS)

    assert result.iloc[0] > result.iloc[1] > result.iloc[2]


def test_the_first_choice_does_not_take_every_penalty():
    """They are sometimes off the pitch, or defer after a miss."""
    assert TAKER_SHARE[1] < 1.0


def test_an_unranked_player_is_credited_with_none():
    """The odd unranked taker exists, but spreading it over 509 players would
    invent five penalty takers a season out of nothing."""
    result = taker_probability(PLAYERS)

    assert result.iloc[3] == 0.0


def test_the_shares_leave_a_remainder_rather_than_summing_to_one():
    assert UNATTRIBUTED_SHARE > 0
    assert sum(TAKER_SHARE.values()) + UNATTRIBUTED_SHARE == pytest.approx(1.0)


def test_probabilities_are_bounded():
    result = taker_probability(PLAYERS)

    assert result.between(0, 1).all()


def test_a_frame_without_the_order_column_yields_zero():
    result = taker_probability(pd.DataFrame([{"web_name": "Nobody"}]))

    assert result.iloc[0] == 0.0


def test_expected_penalty_goals_scale_with_taker_rank():
    result = expected_penalty_goals(PLAYERS)

    assert result.iloc[0] > result.iloc[1] > result.iloc[2]


def test_expected_penalty_goals_are_a_plausible_size():
    """A first-choice taker is worth roughly four penalty goals a season."""
    result = expected_penalty_goals(PLAYERS)

    per_season = result.iloc[0] * 38
    assert 2 < per_season < 8


def test_a_bench_player_takes_fewer_penalties_than_an_ever_present():
    everyone = pd.Series([1.0, 1.0, 1.0, 1.0])
    benched = pd.Series([0.2, 0.2, 0.2, 0.2])

    full = expected_penalty_goals(PLAYERS, minutes_share=everyone)
    partial = expected_penalty_goals(PLAYERS, minutes_share=benched)

    assert partial.iloc[0] < full.iloc[0]


def test_minutes_share_is_clipped_to_a_fraction():
    absurd = pd.Series([5.0, 5.0, 5.0, 5.0])

    result = expected_penalty_goals(PLAYERS, minutes_share=absurd)

    assert result.iloc[0] == pytest.approx(expected_penalty_goals(PLAYERS).iloc[0])


def test_attempts_are_inferred_from_the_failures_we_can_see():
    """2025-26: 15 missed and 11 saved."""
    attempts = implied_attempts(missed=15, saved=11)

    assert 100 < attempts < 150


def test_a_higher_assumed_conversion_implies_far_more_attempts():
    """The estimate is sensitive, which is why it is reported as an estimate."""
    low = implied_attempts(15, 11, conversion=0.785)
    high = implied_attempts(15, 11, conversion=0.897)

    assert high > 2 * low


def test_no_failures_means_no_inferred_attempts():
    assert implied_attempts(0, 0) == 0.0


def test_the_base_rates_are_in_a_defensible_range():
    # Published: 74.8% to 89.7% by season, 81.9% over four seasons.
    assert 0.70 < PENALTY_CONVERSION < 0.92
    # Roughly 0.32 penalties per match, shared between two teams.
    assert 0.10 < PENALTIES_AWARDED_PER_TEAM_MATCH < 0.25


def test_add_penalty_metrics_attaches_both_columns():
    result = add_penalty_metrics(PLAYERS)

    assert "penalty_taker_probability" in result.columns
    assert "expected_penalty_goals" in result.columns


def test_add_penalty_metrics_does_not_mutate_the_input():
    add_penalty_metrics(PLAYERS)

    assert "expected_penalty_goals" not in PLAYERS.columns


def test_empty_input_is_safe():
    assert taker_probability(pd.DataFrame()).empty
    assert add_penalty_metrics(pd.DataFrame()).empty


def test_penalty_duty_on_the_real_snapshot(bootstrap):
    from fpl.domain.players import build_players_frame

    players = build_players_frame(bootstrap)

    result = add_penalty_metrics(players)

    assert result["penalty_taker_probability"].between(0, 1).all()
    # Every club has a designated taker, so some players must be ranked.
    assert (result["penalty_taker_probability"] > 0.5).any()
