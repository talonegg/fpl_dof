"""Tests for the historical archive source.

The reader is injected, so these never download the real CSV.
"""

from __future__ import annotations

import pandas as pd

from fpl.sources.archive import (
    fetch_season_gameweeks,
    points_per_gameweek,
    season_gameweeks_url,
)

RAW = pd.DataFrame(
    [
        {
            "name": "Bukayo Saka",
            "GW": 2,
            "element": 10,
            "team": "Arsenal",
            "position": "MID",
            "value": 100,
            "minutes": 90,
            "total_points": 9,
            "xP": 5.4,
        },
        {
            "name": "Alexander Isak",
            "GW": 1,
            "element": 20,
            "team": "Newcastle",
            "position": "FWD",
            "value": 85,
            "minutes": 78,
            "total_points": 2,
            "xP": 4.1,
        },
    ]
)


def reader(url):
    reader.urls.append(url)
    return RAW.copy()


reader.urls = []


def test_url_is_built_from_the_season():
    url = season_gameweeks_url("2025-26")

    assert url.endswith("/data/2025-26/gws/merged_gw.csv")


def test_archive_columns_are_renamed_to_our_vocabulary():
    df = fetch_season_gameweeks("2025-26", reader=reader)

    assert {"gameweek", "player_name", "team_name", "expected_points"} <= set(df.columns)
    assert "GW" not in df.columns


def test_season_column_is_added_so_frames_can_be_concatenated():
    df = fetch_season_gameweeks("2025-26", reader=reader)

    assert set(df["season"]) == {"2025-26"}


def test_value_is_converted_to_price():
    df = fetch_season_gameweeks("2025-26", reader=reader)

    assert sorted(df["price"]) == [8.5, 10.0]


def test_rows_are_sorted_by_gameweek():
    df = fetch_season_gameweeks("2025-26", reader=reader)

    assert list(df["gameweek"]) == [1, 2]


def test_points_per_gameweek_keeps_only_the_backtest_columns():
    df = points_per_gameweek(fetch_season_gameweeks("2025-26", reader=reader))

    assert set(df.columns) == {
        "season",
        "gameweek",
        "element",
        "player_name",
        "team_name",
        "position",
        "price",
        "minutes",
        "total_points",
    }
    # xP is the archive's own prediction; leaking it into a backtest would
    # flatter any model built on top.
    assert "expected_points" not in df.columns


def test_points_per_gameweek_tolerates_missing_columns():
    df = points_per_gameweek(pd.DataFrame([{"gameweek": 1, "total_points": 5}]))

    assert list(df.columns) == ["gameweek", "total_points"]
