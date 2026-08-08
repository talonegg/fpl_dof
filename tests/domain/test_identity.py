"""Tests for cross-season player matching."""

from __future__ import annotations

import numpy as np
import pandas as pd

from fpl.domain.identity import (
    add_match_key,
    match_rate,
    match_to_current_players,
    normalise_name,
    normalise_positions,
    player_match_keys,
    unmatched_names,
)

PLAYERS = pd.DataFrame(
    [
        {"element": 1, "first_name": "Bukayo", "second_name": "Saka"},
        {"element": 2, "first_name": "Gabriel", "second_name": "dos Santos Magalhães"},
        {"element": 3, "first_name": "Aarón", "second_name": "Anselmino"},
    ]
)


def test_accents_are_stripped():
    assert normalise_name("Aarón Anselmino") == "aaron anselmino"


def test_case_and_extra_whitespace_are_collapsed():
    assert normalise_name("  Bukayo   SAKA ") == "bukayo saka"


def test_punctuation_is_removed():
    assert normalise_name("N'Golo Kanté") == "n golo kante"


def test_non_strings_normalise_to_empty_rather_than_raising():
    assert normalise_name(np.nan) == ""
    assert normalise_name(None) == ""


def test_add_match_key_does_not_mutate_the_input():
    df = pd.DataFrame([{"player_name": "Bukayo Saka"}])

    add_match_key(df, "player_name")

    assert list(df.columns) == ["player_name"]


def test_player_match_keys_uses_the_full_name_not_web_name():
    keys = player_match_keys(PLAYERS)

    assert set(keys["match_key"]) == {
        "bukayo saka",
        "gabriel dos santos magalhaes",
        "aaron anselmino",
    }


def test_archive_rows_gain_the_current_element_id():
    archive = pd.DataFrame([{"player_name": "Bukayo Saka", "total_points": 9}])

    matched = match_to_current_players(archive, PLAYERS)

    assert matched.loc[0, "current_element"] == 1


def test_accented_archive_names_still_match():
    archive = pd.DataFrame([{"player_name": "Gabriel dos Santos Magalhaes"}])

    matched = match_to_current_players(archive, PLAYERS)

    assert matched.loc[0, "current_element"] == 2


def test_a_player_who_left_the_league_keeps_a_null_element():
    archive = pd.DataFrame([{"player_name": "Someone Departed"}])

    matched = match_to_current_players(archive, PLAYERS)

    assert pd.isna(matched.loc[0, "current_element"])


def test_unmatched_names_are_reported_not_dropped():
    archive = pd.DataFrame([{"player_name": "Bukayo Saka"}, {"player_name": "Someone Departed"}])

    matched = match_to_current_players(archive, PLAYERS)

    assert len(matched) == 2, "rows must survive so counts stay honest"
    assert unmatched_names(matched) == ["Someone Departed"]


def test_match_rate_is_the_fraction_matched():
    archive = pd.DataFrame([{"player_name": "Bukayo Saka"}, {"player_name": "Someone Departed"}])

    assert match_rate(match_to_current_players(archive, PLAYERS)) == 0.5


def test_match_rate_of_an_empty_frame_is_zero():
    assert match_rate(pd.DataFrame()) == 0.0


def test_a_duplicate_name_does_not_fan_one_row_into_several():
    players = pd.DataFrame(
        [
            {"element": 1, "first_name": "Danny", "second_name": "Ward"},
            {"element": 2, "first_name": "Danny", "second_name": "Ward"},
        ]
    )
    archive = pd.DataFrame([{"player_name": "Danny Ward"}])

    matched = match_to_current_players(archive, players)

    assert len(matched) == 1


def test_archive_position_codes_are_expanded():
    df = pd.DataFrame([{"position": "MID"}, {"position": "GK"}, {"position": "FWD"}])

    result = normalise_positions(df)

    assert list(result["position"]) == ["Midfielder", "Goalkeeper", "Forward"]


def test_unknown_position_codes_are_left_alone():
    df = pd.DataFrame([{"position": "AM"}])

    assert normalise_positions(df).loc[0, "position"] == "AM"
