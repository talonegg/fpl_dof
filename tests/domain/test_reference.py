"""Tests for the id -> readable name lookups."""

from __future__ import annotations

import pandas as pd

from fpl.domain.reference import (
    add_readable_columns,
    build_position_map,
    build_team_map,
)


def test_build_team_map(bootstrap):
    team_map = build_team_map(bootstrap["teams"])

    assert team_map[1] == "Arsenal"
    assert len(team_map) == len(bootstrap["teams"])


def test_build_position_map(bootstrap):
    position_map = build_position_map(bootstrap["element_types"])

    assert position_map[1] == "Goalkeeper"
    assert position_map[4] == "Forward"


def test_add_readable_columns_maps_ids_and_converts_price():
    df = pd.DataFrame([{"team": 1, "element_type": 2, "now_cost": 55}])

    result = add_readable_columns(
        df,
        teams=[{"id": 1, "name": "Arsenal"}],
        element_types=[{"id": 2, "singular_name": "Defender"}],
    )

    assert result.loc[0, "team_name"] == "Arsenal"
    assert result.loc[0, "position"] == "Defender"
    # now_cost is in integer tenths of a million.
    assert result.loc[0, "price"] == 5.5


def test_add_readable_columns_does_not_mutate_input():
    df = pd.DataFrame([{"team": 1, "element_type": 2, "now_cost": 55}])

    add_readable_columns(
        df,
        teams=[{"id": 1, "name": "Arsenal"}],
        element_types=[{"id": 2, "singular_name": "Defender"}],
    )

    assert list(df.columns) == ["team", "element_type", "now_cost"]
