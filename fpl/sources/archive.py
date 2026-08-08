"""Historical per-gameweek data for past seasons.

The FPL API only serves the current season, so backtesting needs an external
archive. This uses the community-maintained ``vaastav/Fantasy-Premier-League``
dataset, which publishes a ``merged_gw.csv`` per season: one row per player per
gameweek, with points, minutes, price, xG/xA and the fixture played.

Column names there follow the API's own per-gameweek shape, with a few
differences this module normalises away so the rest of the codebase sees one
vocabulary regardless of where a row came from.
"""

from __future__ import annotations

from collections.abc import Callable

import pandas as pd

ARCHIVE_BASE_URL = "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data"

# Archive naming -> our vocabulary. `element` is deliberately left alone: it is
# only meaningful within its own season (see fpl/domain/identity.py).
COLUMN_RENAMES = {
    "GW": "gameweek",
    "name": "player_name",
    "team": "team_name",
    "xP": "expected_points",
}

CsvReader = Callable[[str], pd.DataFrame]


def _read_csv(url: str) -> pd.DataFrame:
    return pd.read_csv(url)


def season_gameweeks_url(season: str) -> str:
    """URL of the merged per-gameweek CSV for a season, e.g. ``"2025-26"``."""
    return f"{ARCHIVE_BASE_URL}/{season}/gws/merged_gw.csv"


def fetch_season_gameweeks(season: str, reader: CsvReader = _read_csv) -> pd.DataFrame:
    """Load one season of per-gameweek player rows.

    Adds a ``season`` column so frames from several seasons can be concatenated
    and still be told apart, and a ``price`` column (``value`` is in integer
    tenths, as everywhere else in FPL).
    """
    df = reader(season_gameweeks_url(season))
    df = df.rename(columns=COLUMN_RENAMES)
    df["season"] = season

    if "value" in df.columns:
        df["price"] = df["value"] / 10
    if "gameweek" in df.columns:
        df["gameweek"] = df["gameweek"].astype(int)

    return df.sort_values(["gameweek", "player_name"]).reset_index(drop=True)


def points_per_gameweek(season_df: pd.DataFrame) -> pd.DataFrame:
    """Collapse to the minimum a backtest needs: who scored what, when.

    Keeping this narrow matters -- a backtest that carries the full 46-column
    frame around invites accidentally using a column that would not have been
    knowable at prediction time.
    """
    columns = [
        "season",
        "gameweek",
        "element",
        "player_name",
        "team_name",
        "position",
        "price",
        "minutes",
        "total_points",
    ]
    available = [column for column in columns if column in season_df.columns]
    return season_df[available].copy()
