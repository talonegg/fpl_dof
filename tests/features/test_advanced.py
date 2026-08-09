"""Tests for advanced signals taken from the official API.

Understat and FBref are excluded by their own terms, so these cover what the
FPL API publishes instead: set-piece duties and finishing over-performance.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fpl.features.advanced import (
    MIN_MINUTES_FOR_FINISHING,
    AdvancedDataUnavailable,
    add_advanced_metrics,
    finishing_delta,
    finishing_delta_per_90,
    has_set_piece_data,
    set_piece_duties,
    set_piece_takers,
)

PLAYERS = pd.DataFrame(
    [
        {
            "web_name": "Taker",
            "penalties_order": 1,
            "corners_and_indirect_freekicks_order": 1,
            "direct_freekicks_order": 1,
            "goals_scored": 10,
            "expected_goals": 6.0,
            "minutes": 900,
            "total_points": 100,
        },
        {
            "web_name": "Backup",
            "penalties_order": 2,
            "corners_and_indirect_freekicks_order": None,
            "direct_freekicks_order": None,
            "goals_scored": 3,
            "expected_goals": 5.0,
            "minutes": 900,
            "total_points": 40,
        },
        {
            "web_name": "Nobody",
            "penalties_order": None,
            "corners_and_indirect_freekicks_order": None,
            "direct_freekicks_order": None,
            "goals_scored": 1,
            "expected_goals": 1.0,
            "minutes": 200,
            "total_points": 10,
        },
    ]
)


def test_the_first_choice_taker_is_flagged():
    result = set_piece_duties(PLAYERS)

    assert result.loc[0, "takes_penalties"]
    assert result.loc[0, "takes_corners"]
    assert result.loc[0, "takes_free_kicks"]


def test_the_second_in_the_queue_is_not_flagged():
    """Second choice only takes them when the first is off; that is not a duty."""
    result = set_piece_duties(PLAYERS)

    assert not result.loc[1, "takes_penalties"]


def test_a_player_with_no_orders_is_flagged_for_nothing():
    result = set_piece_duties(PLAYERS)

    assert result.loc[2, "set_piece_duties"] == 0


def test_duties_are_counted():
    result = set_piece_duties(PLAYERS)

    assert result.loc[0, "set_piece_duties"] == 3


def test_set_piece_duties_does_not_mutate_the_input():
    set_piece_duties(PLAYERS)

    assert "takes_penalties" not in PLAYERS.columns


def test_finishing_delta_is_goals_minus_expected_goals():
    result = finishing_delta(PLAYERS)

    assert result.iloc[0] == pytest.approx(4.0)
    assert result.iloc[1] == pytest.approx(-2.0)


def test_a_short_sample_gives_no_finishing_rate():
    """One goal in 200 minutes is not a finishing signal."""
    result = finishing_delta_per_90(PLAYERS)

    assert pd.isna(result.iloc[2])
    assert MIN_MINUTES_FOR_FINISHING == 450


def test_a_long_enough_sample_gives_a_rate():
    result = finishing_delta_per_90(PLAYERS)

    assert result.iloc[0] == pytest.approx(4.0 * 90 / 900)


def test_finishing_delta_without_the_columns_is_nan_not_zero():
    bare = pd.DataFrame([{"web_name": "Bare", "minutes": 900}])

    assert finishing_delta(bare).isna().all()


def test_set_piece_takers_lists_the_designated_players_first():
    result = set_piece_takers(PLAYERS)

    assert result["web_name"].tolist() == ["Taker"]


def test_add_advanced_metrics_attaches_everything():
    result = add_advanced_metrics(PLAYERS)

    assert {"takes_penalties", "set_piece_duties", "finishing_delta"} <= set(result.columns)


# --- Live-only: the archive never recorded who takes the corners ---


def test_archive_shaped_data_is_recognised_as_carrying_no_set_pieces():
    archive_shaped = pd.DataFrame([{"element": 1, "gameweek": 5, "total_points": 6, "minutes": 90}])

    assert not has_set_piece_data(archive_shaped)


def test_asking_for_set_pieces_on_historical_data_is_refused():
    archive_shaped = pd.DataFrame([{"element": 1, "gameweek": 5, "total_points": 6}])

    with pytest.raises(AdvancedDataUnavailable, match="never recorded historically"):
        set_piece_duties(archive_shaped)


def test_add_advanced_metrics_leaves_historical_data_alone():
    """No columns is the honest answer; misleading ones would be worse."""
    archive_shaped = pd.DataFrame([{"element": 1, "gameweek": 5, "total_points": 6}])

    result = add_advanced_metrics(archive_shaped)

    assert "set_piece_duties" not in result.columns
    assert len(result) == 1


def test_the_real_archive_carries_no_set_piece_data(archive):
    assert not has_set_piece_data(archive)


# --- Against the real snapshot ---


def test_set_pieces_on_the_real_snapshot(bootstrap):
    from fpl.domain.players import build_players_frame

    players = build_players_frame(bootstrap)

    result = add_advanced_metrics(players)

    assert len(result) == len(players)
    assert result["set_piece_duties"].between(0, 3).all()


def test_the_real_snapshot_has_designated_penalty_takers(bootstrap):
    from fpl.domain.players import build_players_frame

    takers = set_piece_takers(build_players_frame(bootstrap))

    assert not takers.empty
    assert takers["takes_penalties"].any()


def test_finishing_delta_is_finite_on_the_real_snapshot(bootstrap):
    from fpl.domain.players import build_players_frame

    result = finishing_delta(build_players_frame(bootstrap))

    assert np.isfinite(result.dropna()).all()


# --- Regressions found by checking against live data ---


def test_first_choice_is_the_lowest_order_in_the_team_not_the_number_one():
    """Corner orders start at 2 on live data; testing for 1 flags nobody."""
    squad = pd.DataFrame(
        [
            {"web_name": "A", "team": 1, "corners_and_indirect_freekicks_order": 2},
            {"web_name": "B", "team": 1, "corners_and_indirect_freekicks_order": 5},
            {"web_name": "C", "team": 2, "corners_and_indirect_freekicks_order": 4},
        ]
    )

    result = set_piece_duties(squad)

    assert result["takes_corners"].tolist() == [True, False, True]


def test_first_choice_is_decided_per_team():
    """A team's best taker is first choice even if another club's rank is lower."""
    squad = pd.DataFrame(
        [
            {"web_name": "Elite", "team": 1, "penalties_order": 1},
            {"web_name": "OnlyTaker", "team": 2, "penalties_order": 3},
        ]
    )

    result = set_piece_duties(squad)

    assert result["takes_penalties"].tolist() == [True, True]


def test_a_player_with_no_order_is_never_first_choice():
    squad = pd.DataFrame(
        [
            {"web_name": "Taker", "team": 1, "penalties_order": 2},
            {"web_name": "None", "team": 1, "penalties_order": None},
        ]
    )

    result = set_piece_duties(squad)

    assert result["takes_penalties"].tolist() == [True, False]


def test_goals_without_recorded_expected_goals_give_no_finishing_delta():
    """11 goals against 0.00 xG is a missing column, not a finishing miracle."""
    odd = pd.DataFrame(
        [{"web_name": "Unrecorded", "goals_scored": 11, "expected_goals": 0.0, "minutes": 900}]
    )

    assert pd.isna(finishing_delta(odd).iloc[0])


def test_a_goalless_player_with_no_expected_goals_is_still_zero():
    """Genuinely no attacking threat is a real answer, not missing data."""
    quiet = pd.DataFrame(
        [{"web_name": "Defender", "goals_scored": 0, "expected_goals": 0.0, "minutes": 900}]
    )

    assert finishing_delta(quiet).iloc[0] == 0.0


def test_the_real_snapshot_finds_corner_takers(bootstrap):
    """The bug in full: this returned zero corner takers across the league."""
    from fpl.domain.players import build_players_frame

    result = add_advanced_metrics(build_players_frame(bootstrap))

    assert result["takes_corners"].sum() > 0
