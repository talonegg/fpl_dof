"""Tests for rate and value metrics.

Every expected value here is computed by hand in the test, so a change in
behaviour has to be argued with rather than blessed.
"""

from __future__ import annotations

import pandas as pd
import pytest

from fpl.features.rates import (
    LOW_MINUTES_THRESHOLD,
    add_per_90_columns,
    add_scouting_metrics,
    minutes_share,
    per_90,
    points_per_million,
    team_matches_played,
)

PLAYERS = pd.DataFrame(
    [
        # 180 minutes, 12 points -> 6.0 per 90. Price 6.0 -> 2.0 points per £m.
        {"element": 1, "team": 1, "minutes": 180, "total_points": 12, "price": 6.0},
        # 90 minutes, 3 points -> 3.0 per 90. Price 5.0 -> 0.6 points per £m.
        {"element": 2, "team": 2, "minutes": 90, "total_points": 3, "price": 5.0},
        # Never played. Rates are unknowable, not zero.
        {"element": 3, "team": 1, "minutes": 0, "total_points": 0, "price": 4.0},
    ]
)


def test_per_90_is_the_rate_per_ninety_minutes():
    result = per_90(PLAYERS, "total_points")

    assert result.tolist()[:2] == [6.0, 3.0]


def test_zero_minutes_gives_nan_not_zero():
    result = per_90(PLAYERS, "total_points")

    assert pd.isna(result.iloc[2]), "a player who has not played is unknown, not bad"


def test_add_per_90_columns_names_them_predictably():
    result = add_per_90_columns(PLAYERS, ["total_points"])

    assert result["total_points_per_90"].iloc[0] == 6.0


def test_add_per_90_columns_skips_columns_that_are_absent():
    result = add_per_90_columns(PLAYERS, ["goals_scored"])

    assert "goals_scored_per_90" not in result.columns


def test_add_per_90_columns_does_not_mutate_the_input():
    add_per_90_columns(PLAYERS, ["total_points"])

    assert "total_points_per_90" not in PLAYERS.columns


def test_points_per_million():
    result = points_per_million(PLAYERS)

    assert result.tolist() == [2.0, 0.6, 0.0]


def test_team_matches_played_counts_only_finished_fixtures():
    schedule = pd.DataFrame(
        [
            {"team": 1, "finished": True},
            {"team": 1, "finished": True},
            {"team": 1, "finished": False},
            {"team": 2, "finished": True},
        ]
    )

    played = team_matches_played(schedule)

    assert played[1] == 2
    assert played[2] == 1


def test_team_matches_played_of_an_empty_schedule_is_empty():
    assert team_matches_played(pd.DataFrame()).empty


def test_minutes_share_is_the_fraction_of_available_minutes():
    df = pd.DataFrame([{"minutes": 180}, {"minutes": 45}])
    played = pd.Series([2, 1])

    result = minutes_share(df, played)

    assert result.tolist() == [1.0, 0.5]


def test_minutes_share_is_nan_before_a_team_has_played():
    df = pd.DataFrame([{"minutes": 0}])

    assert pd.isna(minutes_share(df, pd.Series([0])).iloc[0])


def test_add_scouting_metrics_adds_the_expected_columns():
    schedule = pd.DataFrame(
        [
            {"team": 1, "finished": True},
            {"team": 1, "finished": True},
            {"team": 2, "finished": True},
        ]
    )

    result = add_scouting_metrics(PLAYERS, schedule)

    assert result["total_points_per_90"].iloc[0] == 6.0
    assert result["points_per_million"].iloc[0] == 2.0
    assert result["minutes_share"].iloc[0] == 1.0


def test_low_minutes_flags_rates_that_cannot_be_trusted_yet():
    result = add_scouting_metrics(PLAYERS, None)

    # 180 and 90 minutes are both below the threshold; nobody here is safe.
    assert result["low_minutes"].tolist() == [True, True, True]
    assert LOW_MINUTES_THRESHOLD == 270


def test_minutes_share_is_nan_when_no_schedule_is_supplied():
    result = add_scouting_metrics(PLAYERS, None)

    assert result["minutes_share"].isna().all()


def test_scouting_metrics_on_the_real_snapshot(bootstrap, schedule):
    from fpl.domain.players import build_players_frame

    players = build_players_frame(bootstrap)

    result = add_scouting_metrics(players, schedule)

    assert len(result) == len(players)
    assert "points_per_million" in result.columns


@pytest.mark.parametrize("minutes,points,expected", [(90, 6, 6.0), (45, 3, 6.0), (270, 9, 3.0)])
def test_per_90_scales_with_minutes(minutes, points, expected):
    df = pd.DataFrame([{"minutes": minutes, "total_points": points}])

    assert per_90(df, "total_points").iloc[0] == expected
