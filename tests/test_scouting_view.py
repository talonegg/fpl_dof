"""Tests for the scouting view's data reshaping.

The charts are not tested; the numbers behind them are. These cover the cases
that broke against real archive data but that a tidy synthetic fixture never
produces.
"""

from __future__ import annotations

import pandas as pd

from app.scouting_view import _history_for

HISTORY = pd.DataFrame(
    [
        {"current_element": 1, "gameweek": 1, "total_points": 6, "minutes": 90, "price": 5.5},
        # Double gameweek: two rows for the same player in gameweek 2.
        {"current_element": 1, "gameweek": 2, "total_points": 2, "minutes": 90, "price": 5.6},
        {"current_element": 1, "gameweek": 2, "total_points": 9, "minutes": 85, "price": 5.6},
        {"current_element": 2, "gameweek": 1, "total_points": 1, "minutes": 12, "price": 4.5},
    ]
)


def test_only_the_requested_player_is_returned():
    result = _history_for(HISTORY, 2)

    assert result["total_points"].tolist() == [1]


def test_a_double_gameweek_collapses_to_one_row():
    result = _history_for(HISTORY, 1)

    assert result["gameweek"].tolist() == [1, 2]
    assert not result["gameweek"].duplicated().any()


def test_a_double_gameweeks_points_are_summed():
    result = _history_for(HISTORY, 1)

    assert result[result["gameweek"] == 2]["total_points"].iloc[0] == 11


def test_a_double_gameweeks_minutes_are_summed():
    result = _history_for(HISTORY, 1)

    assert result[result["gameweek"] == 2]["minutes"].iloc[0] == 175


def test_price_is_a_snapshot_not_a_sum():
    result = _history_for(HISTORY, 1)

    assert result[result["gameweek"] == 2]["price"].iloc[0] == 5.6


def test_rows_come_back_in_gameweek_order():
    shuffled = HISTORY.iloc[::-1]

    result = _history_for(shuffled, 1)

    assert result["gameweek"].is_monotonic_increasing


def test_a_player_with_no_history_gives_an_empty_frame():
    assert _history_for(HISTORY, 99).empty


def test_an_empty_history_gives_an_empty_frame():
    assert _history_for(pd.DataFrame(), 1).empty
