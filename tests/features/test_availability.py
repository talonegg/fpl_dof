"""Tests for player availability.

The trap these exist to guard: ``chance_of_playing_next_round`` is null for
most players, and null means "no news", not "fit". Reading it the wrong way
round marks long-term absentees as fully available — and on live data that is
59 players, several of them expensive and popular.
"""

from __future__ import annotations

import pandas as pd
import pytest

from fpl.features.availability import (
    SELECTABLE_THRESHOLD,
    AvailabilityUnavailable,
    add_availability,
    availability,
    discount_expected_points,
    flagged,
    has_availability_data,
    selectable,
)

PLAYERS = pd.DataFrame(
    [
        {"web_name": "Fit", "status": "a", "chance_of_playing_next_round": None},
        {"web_name": "Injured", "status": "i", "chance_of_playing_next_round": None},
        {"web_name": "Doubtful", "status": "d", "chance_of_playing_next_round": 25.0},
        {"web_name": "Likely", "status": "d", "chance_of_playing_next_round": 75.0},
        {"web_name": "Suspended", "status": "s", "chance_of_playing_next_round": 0.0},
        {"web_name": "Left", "status": "u", "chance_of_playing_next_round": None},
    ]
)


def test_a_fit_player_with_no_news_is_fully_available():
    result = availability(PLAYERS)

    assert result.iloc[0] == 1.0


def test_an_injured_player_with_no_published_chance_is_not_available():
    """The bug this guards: null means no news, not a clean bill of health."""
    result = availability(PLAYERS)

    assert result.iloc[1] == 0.0


def test_a_player_who_has_left_is_not_available():
    result = availability(PLAYERS)

    assert result.iloc[5] == 0.0


def test_a_published_percentage_is_used_when_present():
    result = availability(PLAYERS)

    assert result.iloc[2] == 0.25
    assert result.iloc[3] == 0.75


def test_an_unknown_status_is_treated_as_doubtful_not_fit():
    """A new status code must not quietly promote an absentee."""
    unknown = pd.DataFrame(
        [{"web_name": "Mystery", "status": "z", "chance_of_playing_next_round": None}]
    )

    assert availability(unknown).iloc[0] == 0.5


def test_availability_is_bounded_to_a_probability():
    odd = pd.DataFrame([{"web_name": "Odd", "status": "a", "chance_of_playing_next_round": 150.0}])

    assert availability(odd).iloc[0] == 1.0


def test_data_without_availability_fields_is_refused_not_assumed_fit():
    """Historical data has no status; "everyone fit" would be silently wrong."""
    with pytest.raises(AvailabilityUnavailable, match="live-only signal"):
        availability(pd.DataFrame([{"web_name": "Bare", "total_points": 5}]))


def test_an_archive_season_is_recognised_as_carrying_no_availability():
    archive_shaped = pd.DataFrame([{"element": 1, "gameweek": 5, "total_points": 6, "minutes": 90}])

    assert not has_availability_data(archive_shaped)


def test_a_live_frame_is_recognised_as_carrying_availability():
    assert has_availability_data(PLAYERS)


def test_availability_of_nothing_is_empty():
    assert availability(pd.DataFrame()).empty


def test_only_fit_players_are_selectable():
    result = selectable(PLAYERS)

    assert sorted(result["web_name"]) == ["Fit", "Likely"]


def test_the_threshold_matches_the_apis_own_expected_to_play_band():
    assert SELECTABLE_THRESHOLD == 0.75


def test_a_stricter_threshold_excludes_more():
    result = selectable(PLAYERS, threshold=1.0)

    assert result["web_name"].tolist() == ["Fit"]


def test_add_availability_labels_without_dropping_anyone():
    result = add_availability(PLAYERS)

    assert len(result) == len(PLAYERS)
    assert result["is_selectable"].tolist() == [True, False, False, True, False, False]


def test_add_availability_does_not_mutate_the_input():
    add_availability(PLAYERS)

    assert "availability" not in PLAYERS.columns


def test_discounting_scales_points_by_the_chance_of_playing():
    pool = PLAYERS.assign(expected_points=4.0)

    result = discount_expected_points(pool)

    assert result["expected_points"].iloc[0] == 4.0  # fit
    assert result["expected_points"].iloc[1] == 0.0  # injured
    assert result["expected_points"].iloc[2] == pytest.approx(1.0)  # 25%


def test_discounting_a_pool_without_points_is_harmless():
    result = discount_expected_points(PLAYERS)

    assert len(result) == len(PLAYERS)


def test_flagged_lists_everyone_with_news_worst_first():
    result = flagged(PLAYERS)

    assert "Fit" not in result["web_name"].tolist()
    assert result["web_name"].iloc[0] in {"Injured", "Left", "Suspended"}
    assert result["availability"].is_monotonic_increasing


def test_nobody_flagged_gives_an_empty_frame():
    fit_only = PLAYERS.head(1)

    assert flagged(fit_only).empty


def test_availability_on_the_real_snapshot(bootstrap):
    from fpl.domain.players import build_players_frame

    players = build_players_frame(bootstrap)

    result = availability(players)

    assert result.between(0, 1).all()
    assert len(result) == len(players)


def test_the_real_archive_carries_no_availability_data(archive):
    """The guarantee this whole guard exists for, checked on real archive data."""
    assert not has_availability_data(archive)


def test_applying_availability_to_the_real_archive_is_refused(archive):
    with pytest.raises(AvailabilityUnavailable):
        availability(archive)
