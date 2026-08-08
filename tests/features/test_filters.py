"""Tests for player filtering."""

from __future__ import annotations

import pandas as pd

from fpl.features.filters import (
    PlayerFilter,
    apply_filter,
    options,
    price_bounds,
    restrict_to_available,
)

PLAYERS = pd.DataFrame(
    [
        {
            "element": 1,
            "web_name": "Raya",
            "position": "Goalkeeper",
            "team_name": "Arsenal",
            "price": 5.5,
        },
        {
            "element": 2,
            "web_name": "Saka",
            "position": "Midfielder",
            "team_name": "Arsenal",
            "price": 10.0,
        },
        {
            "element": 3,
            "web_name": "Isak",
            "position": "Forward",
            "team_name": "Newcastle",
            "price": 8.5,
        },
        {
            "element": 4,
            "web_name": "Hall",
            "position": "Defender",
            "team_name": "Newcastle",
            "price": 4.5,
        },
    ]
)


def test_an_empty_filter_keeps_everyone():
    assert len(apply_filter(PLAYERS, PlayerFilter())) == 4


def test_filtering_by_position():
    result = apply_filter(PLAYERS, PlayerFilter(positions=("Midfielder",)))

    assert result["web_name"].tolist() == ["Saka"]


def test_filtering_by_club():
    result = apply_filter(PLAYERS, PlayerFilter(teams=("Newcastle",)))

    assert sorted(result["web_name"]) == ["Hall", "Isak"]


def test_filtering_by_price_range_is_inclusive():
    result = apply_filter(PLAYERS, PlayerFilter(price_range=(4.5, 8.5)))

    assert sorted(result["web_name"]) == ["Hall", "Isak", "Raya"]


def test_filters_combine():
    result = apply_filter(PLAYERS, PlayerFilter(teams=("Arsenal",), price_range=(0.0, 6.0)))

    assert result["web_name"].tolist() == ["Raya"]


def test_deselecting_everything_shows_nothing_rather_than_everything():
    """An empty selection is a real choice, and is not the same as no filter."""
    assert apply_filter(PLAYERS, PlayerFilter(positions=())).empty
    assert PlayerFilter(positions=()).is_empty


def test_no_constraint_is_distinct_from_an_empty_one():
    assert not PlayerFilter(positions=None).is_empty
    assert len(apply_filter(PLAYERS, PlayerFilter(positions=None))) == 4


def test_filtering_an_empty_frame_is_safe():
    assert apply_filter(PLAYERS.head(0), PlayerFilter(positions=("Forward",))).empty


def test_filtering_does_not_mutate_the_input():
    apply_filter(PLAYERS, PlayerFilter(positions=("Forward",)))

    assert len(PLAYERS) == 4


def test_a_missing_column_is_ignored_rather_than_raising():
    without_price = PLAYERS.drop(columns=["price"])

    result = apply_filter(without_price, PlayerFilter(price_range=(0.0, 1.0)))

    assert len(result) == 4


def test_price_bounds_are_the_extremes_present():
    assert price_bounds(PLAYERS) == (4.5, 10.0)


def test_price_bounds_of_an_empty_frame_is_none():
    assert price_bounds(PLAYERS.head(0)) is None


def test_options_are_sorted_and_unique():
    assert options(PLAYERS, "team_name") == ["Arsenal", "Newcastle"]


def test_options_of_a_missing_column_is_empty():
    assert options(PLAYERS, "nope") == []


def test_stale_selections_are_dropped():
    """A selection can outlive the data behind it; passing it to a widget raises."""
    assert restrict_to_available(["Arsenal", "Luton"], ["Arsenal", "Newcastle"]) == ["Arsenal"]


def test_describe_summarises_the_active_filters():
    described = PlayerFilter(
        positions=("Midfielder",), teams=("Arsenal",), price_range=(4.0, 9.0)
    ).describe()

    assert "1 position(s)" in described
    assert "£4.0m–£9.0m" in described


def test_describe_says_so_when_nothing_is_filtered():
    assert PlayerFilter().describe() == "no filters"
