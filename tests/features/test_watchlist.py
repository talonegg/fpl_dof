"""Tests for the persisted watchlist."""

from __future__ import annotations

import pandas as pd

from fpl.features import watchlist

PLAYERS = pd.DataFrame(
    [
        {"element": 1, "code": 111, "web_name": "Saka"},
        {"element": 2, "code": 222, "web_name": "Isak"},
        {"element": 3, "code": 333, "web_name": "Palmer"},
    ]
)


def test_loading_a_watchlist_that_does_not_exist_yet_is_empty(tmp_path):
    assert watchlist.load(tmp_path / "nope.json") == []


def test_save_then_load_round_trips(tmp_path):
    path = tmp_path / "watchlist.json"

    watchlist.save(path, [222, 111])

    assert watchlist.load(path) == [111, 222]


def test_save_deduplicates(tmp_path):
    path = tmp_path / "watchlist.json"

    watchlist.save(path, [111, 111, 222])

    assert watchlist.load(path) == [111, 222]


def test_save_creates_missing_directories(tmp_path):
    path = tmp_path / "nested" / "watchlist.json"

    watchlist.save(path, [111])

    assert path.exists()


def test_a_corrupt_file_is_treated_as_empty_rather_than_fatal(tmp_path):
    path = tmp_path / "watchlist.json"
    path.write_text("{not json", encoding="utf-8")

    assert watchlist.load(path) == []


def test_a_file_holding_the_wrong_shape_is_treated_as_empty(tmp_path):
    path = tmp_path / "watchlist.json"
    path.write_text('{"codes": [1]}', encoding="utf-8")

    assert watchlist.load(path) == []


def test_toggle_adds_a_missing_code():
    assert watchlist.toggle([111], 222) == [111, 222]


def test_toggle_removes_a_present_code():
    assert watchlist.toggle([111, 222], 111) == [222]


def test_toggle_does_not_mutate_the_input():
    codes = [111]

    watchlist.toggle(codes, 222)

    assert codes == [111]


def test_filter_players_selects_by_code_not_element():
    result = watchlist.filter_players(PLAYERS, [222, 333])

    assert result["web_name"].tolist() == ["Isak", "Palmer"]


def test_filter_players_with_an_empty_watchlist_returns_no_rows():
    result = watchlist.filter_players(PLAYERS, [])

    assert result.empty
    # Columns must survive so the caller can still render a table.
    assert list(result.columns) == list(PLAYERS.columns)
