"""Tests for cross-season player matching."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fpl.domain.identity import (
    add_match_key,
    ambiguous_names,
    match_across_seasons,
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


# --- Regressions from the second review pass ---


def test_a_name_collision_is_reported_rather_than_guessed_at():
    """Two Danny Wards cannot be told apart; picking one misattributes a season."""
    players = pd.DataFrame(
        [
            {"element": 1, "first_name": "Danny", "second_name": "Ward"},
            {"element": 2, "first_name": "Danny", "second_name": "Ward"},
        ]
    )

    assert ambiguous_names(players) == ["Danny Ward"]


def test_a_colliding_name_does_not_match_at_all():
    players = pd.DataFrame(
        [
            {"element": 1, "first_name": "Danny", "second_name": "Ward"},
            {"element": 2, "first_name": "Danny", "second_name": "Ward"},
        ]
    )
    archive = pd.DataFrame([{"player_name": "Danny Ward"}])

    matched = match_to_current_players(archive, players)

    assert pd.isna(matched.loc[0, "current_element"])
    assert match_rate(matched) == 0.0, "an ambiguous match must not count as matched"


def test_unique_names_are_unaffected_by_a_collision_elsewhere():
    players = pd.DataFrame(
        [
            {"element": 1, "first_name": "Danny", "second_name": "Ward"},
            {"element": 2, "first_name": "Danny", "second_name": "Ward"},
            {"element": 3, "first_name": "Bukayo", "second_name": "Saka"},
        ]
    )
    archive = pd.DataFrame([{"player_name": "Bukayo Saka"}])

    matched = match_to_current_players(archive, players)

    assert matched.loc[0, "current_element"] == 3


def test_players_with_no_usable_name_do_not_match_each_other():
    """Empty keys are equal, so nameless rows would join and count as matches."""
    players = pd.DataFrame([{"element": 1, "first_name": "", "second_name": ""}])
    archive = pd.DataFrame([{"player_name": None}])

    matched = match_to_current_players(archive, players)

    assert pd.isna(matched.loc[0, "current_element"])
    assert match_rate(matched) == 0.0


def test_no_collisions_means_nothing_to_report():
    assert ambiguous_names(PLAYERS) == []


# --- Matching between any two seasons, not just onto the current one ---

SEASON_A = pd.DataFrame(
    [
        {"player_name": "Bukayo Saka", "element": 10},
        {"player_name": "Aarón Anselmino", "element": 20},
        {"player_name": "Departed Player", "element": 30},
    ]
)
SEASON_B = pd.DataFrame(
    [
        {"player_name": "Bukayo Saka", "element": 400},
        {"player_name": "Aaron Anselmino", "element": 500},
        {"player_name": "New Signing", "element": 600},
    ]
)


def test_players_are_matched_between_two_seasons():
    result = match_across_seasons(SEASON_A, SEASON_B)

    assert result.matched == 2


def test_the_element_ids_differ_between_the_seasons():
    """The whole reason this function exists."""
    result = match_across_seasons(SEASON_A, SEASON_B)
    saka = result.pairs[result.pairs["name_left"] == "Bukayo Saka"].iloc[0]

    assert saka["left"] == 10
    assert saka["right"] == 400


def test_accents_do_not_prevent_a_match():
    result = match_across_seasons(SEASON_A, SEASON_B)

    assert "aaron anselmino" in result.pairs["match_key"].tolist()


def test_a_player_who_left_is_reported_not_dropped_silently():
    result = match_across_seasons(SEASON_A, SEASON_B)

    assert result.unmatched_left == ["Departed Player"]


def test_a_new_arrival_is_reported_too():
    result = match_across_seasons(SEASON_A, SEASON_B)

    assert result.unmatched_right == ["New Signing"]


def test_coverage_is_the_share_of_the_left_season_matched():
    result = match_across_seasons(SEASON_A, SEASON_B)

    assert result.coverage == pytest.approx(2 / 3)


def test_a_duplicated_name_is_excluded_and_reported():
    left = pd.DataFrame(
        [
            {"player_name": "Danny Ward", "element": 1},
            {"player_name": "Danny Ward", "element": 2},
        ]
    )
    right = pd.DataFrame([{"player_name": "Danny Ward", "element": 9}])

    result = match_across_seasons(left, right)

    assert result.matched == 0
    assert result.ambiguous == ["Danny Ward"]


def test_an_ambiguity_on_either_side_blocks_the_match():
    left = pd.DataFrame([{"player_name": "Danny Ward", "element": 1}])
    right = pd.DataFrame(
        [
            {"player_name": "Danny Ward", "element": 8},
            {"player_name": "Danny Ward", "element": 9},
        ]
    )

    result = match_across_seasons(left, right)

    assert result.matched == 0
    assert result.ambiguous == ["Danny Ward"]


def test_repeated_rows_for_one_player_do_not_look_ambiguous():
    """Archive frames have a row per gameweek; that is not two players."""
    left = pd.DataFrame([{"player_name": "Bukayo Saka", "element": 10}] * 38)
    right = pd.DataFrame([{"player_name": "Bukayo Saka", "element": 400}] * 38)

    result = match_across_seasons(left, right)

    assert result.matched == 1
    assert result.ambiguous == []


def test_matching_nothing_is_safe():
    empty = pd.DataFrame(columns=["player_name", "element"])

    result = match_across_seasons(empty, empty)

    assert result.matched == 0
    assert result.coverage == 0.0


def test_the_summary_reads_as_a_sentence():
    assert "matched" in match_across_seasons(SEASON_A, SEASON_B).summary()
