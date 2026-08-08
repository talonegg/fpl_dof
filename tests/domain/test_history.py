"""Tests for per-player history parsing."""

from __future__ import annotations

import pytest

from fpl.domain.history import build_gameweek_history, build_past_seasons

SUMMARY = {
    "history": [
        {
            "element": 1,
            "round": 2,
            "fixture": 12,
            "opponent_team": 5,
            "was_home": True,
            "minutes": 90,
            "total_points": 6,
            "value": 55,
            "influence": "12.4",
        },
        {
            "element": 1,
            "round": 1,
            "fixture": 3,
            "opponent_team": 9,
            "was_home": False,
            "minutes": 45,
            "total_points": 2,
            "value": 55,
            "influence": "3.2",
        },
    ],
    "history_past": [
        {"element_code": 154561, "season_name": "2024/25", "total_points": 120, "start_cost": 50},
        {"element_code": 154561, "season_name": "2021/22", "total_points": 95, "start_cost": 45},
    ],
}


def test_gameweek_history_is_sorted_by_gameweek():
    df = build_gameweek_history(SUMMARY)

    assert list(df["gameweek"]) == [1, 2]


def test_gameweek_history_converts_value_to_price():
    df = build_gameweek_history(SUMMARY)

    assert df["price"].tolist() == [5.5, 5.5]


def test_gameweek_history_coerces_string_numerics():
    df = build_gameweek_history(SUMMARY)

    assert df["influence"].sum() == pytest.approx(15.6)


def test_a_player_who_has_not_played_gives_an_empty_frame_not_an_error():
    df = build_gameweek_history({"history": [], "history_past": []})

    assert df.empty
    # Callers concatenate these, so the columns must still be there.
    assert {"gameweek", "total_points", "minutes"} <= set(df.columns)


def test_missing_history_key_is_treated_as_no_history():
    assert build_gameweek_history({}).empty


def test_past_seasons_are_sorted_oldest_first():
    df = build_past_seasons(SUMMARY)

    assert list(df["season_name"]) == ["2021/22", "2024/25"]


def test_past_seasons_convert_cost_to_price():
    df = build_past_seasons(SUMMARY)

    assert df["start_price"].tolist() == [4.5, 5.0]


def test_past_seasons_of_a_new_player_is_empty():
    assert build_past_seasons({"history_past": []}).empty
