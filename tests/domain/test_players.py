"""Tests for building the canonical player table."""

from __future__ import annotations

import pandas as pd
from pandas.api.types import is_numeric_dtype

from fpl.domain.players import build_players_frame, coerce_numeric_columns


def test_build_players_frame_has_readable_columns(bootstrap):
    df = build_players_frame(bootstrap)

    assert len(df) == len(bootstrap["elements"])
    assert set(df["position"]) <= {"Goalkeeper", "Defender", "Midfielder", "Forward"}
    assert df["team_name"].notna().all()
    assert (df["price"] > 0).all()


def test_numeric_columns_are_not_left_as_strings(bootstrap):
    """The API sends these as strings; sorting them as text gives wrong answers."""
    df = build_players_frame(bootstrap)

    for column in ("form", "points_per_game", "selected_by_percent", "ict_index"):
        assert is_numeric_dtype(df[column]), f"{column} should be numeric"


def test_string_numerics_sort_by_value_not_lexically():
    df = pd.DataFrame([{"ict_index": "9.5"}, {"ict_index": "12.0"}])

    result = coerce_numeric_columns(df).sort_values("ict_index", ascending=False)

    # Lexical ordering would put "9.5" first.
    assert result.iloc[0]["ict_index"] == 12.0


def test_unparseable_values_become_nan_rather_than_raising():
    df = pd.DataFrame([{"form": "3.5"}, {"form": ""}, {"form": None}])

    result = coerce_numeric_columns(df)

    assert result.loc[0, "form"] == 3.5
    assert result["form"].isna().sum() == 2


def test_coerce_ignores_columns_that_are_absent():
    df = pd.DataFrame([{"web_name": "Raya"}])

    result = coerce_numeric_columns(df)

    assert list(result.columns) == ["web_name"]
