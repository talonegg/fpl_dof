"""Tests for team defensive strength.

The requirement these serve: a defender who transfers should be rated on their
new club's record, not the one they accumulated their own history at.
"""

from __future__ import annotations

import math

import pandas as pd
import pytest

from fpl.features.team_strength import (
    DEFAULT_PROMOTED_XGC,
    blend_team_defence,
    clean_sheet_outlook,
    clean_sheet_probability,
    estimate_promoted_prior,
    expected_concession,
    season_defence,
    team_match_defence,
)


def club_season(club, conceded, xgc, gameweeks=10, players=2):
    """A club's season, recorded on players who completed 60+ minutes."""
    return pd.DataFrame(
        [
            {
                "team_name": club,
                "gameweek": gameweek,
                "player_name": f"{club}{player}",
                "minutes": 90,
                "goals_conceded": conceded,
                "expected_goals_conceded": xgc,
                "clean_sheets": 1 if conceded == 0 else 0,
            }
            for gameweek in range(1, gameweeks + 1)
            for player in range(players)
        ]
    )


def test_a_team_match_is_one_row_per_club_per_gameweek():
    result = team_match_defence(club_season("Arsenal", 1, 1.0, gameweeks=5, players=3))

    assert len(result) == 5


def test_players_who_did_not_finish_the_match_are_excluded():
    """Goals conceded is per player; a substitute did not see them all."""
    data = club_season("Arsenal", 2, 1.5, gameweeks=1)
    partial = data.copy()
    partial["minutes"] = 20
    partial["goals_conceded"] = 0

    result = team_match_defence(pd.concat([data, partial]))

    assert result["goals_conceded"].iloc[0] == 2


def test_season_defence_averages_across_matches():
    result = season_defence(club_season("Arsenal", 1, 0.8, gameweeks=10))

    assert result["goals_conceded_per_match"].iloc[0] == pytest.approx(1.0)
    assert result["expected_goals_conceded_per_match"].iloc[0] == pytest.approx(0.8)


def test_a_mean_club_and_a_leaky_one_are_separated():
    data = pd.concat([club_season("Arsenal", 0, 0.7), club_season("Burnley", 2, 2.0)])

    result = season_defence(data).set_index("team_name")

    assert (
        result.loc["Arsenal", "expected_goals_conceded_per_match"]
        < (result.loc["Burnley", "expected_goals_conceded_per_match"])
    )


def test_blending_weights_the_recent_season_more():
    data = {
        "2024-25": club_season("Arsenal", 2, 2.0),
        "2025-26": club_season("Arsenal", 0, 0.5),
    }

    blended = blend_team_defence(data)

    # Recent 0.5 outweighs older 2.0, so below the midpoint of 1.25.
    assert blended["expected_goals_conceded_per_match"].iloc[0] < 1.25


def test_a_club_absent_from_a_season_is_not_penalised_for_it():
    """A newly promoted club is rated on its one season, not on its absence."""
    data = {
        "2024-25": club_season("Arsenal", 1, 1.0),
        "2025-26": pd.concat([club_season("Arsenal", 1, 1.0), club_season("Leeds", 1, 1.0)]),
    }

    blended = blend_team_defence(data).set_index("team_name")

    assert blended.loc["Leeds", "expected_goals_conceded_per_match"] == pytest.approx(1.0)
    assert blended.loc["Leeds", "seasons_seen"] == 1


def test_a_promoted_club_prior_is_estimated_from_actual_newcomers():
    data = {
        "2024-25": club_season("Arsenal", 1, 1.0),
        "2025-26": pd.concat([club_season("Arsenal", 1, 1.0), club_season("Burnley", 2, 1.9)]),
    }

    assert estimate_promoted_prior(data) == pytest.approx(1.9)


def test_the_promoted_prior_falls_back_when_there_is_nothing_to_estimate_from():
    assert estimate_promoted_prior({"2025-26": club_season("Arsenal", 1, 1.0)}) == (
        DEFAULT_PROMOTED_XGC
    )


def test_an_unknown_club_is_treated_as_promoted_not_average():
    """Over-rating a promoted side's cheap defenders is the error to avoid."""
    blended = blend_team_defence({"2025-26": club_season("Arsenal", 0, 0.7)})

    assert expected_concession("Sunderland", blended, promoted_prior=1.8) == 1.8


def test_a_known_club_uses_its_own_record():
    blended = blend_team_defence({"2025-26": club_season("Arsenal", 0, 0.7)})

    assert expected_concession("Arsenal", blended) == pytest.approx(0.7)


def test_clean_sheet_probability_falls_as_concession_rises():
    assert clean_sheet_probability(0.7) > clean_sheet_probability(2.0)


def test_clean_sheet_probability_matches_the_poisson():
    assert clean_sheet_probability(1.0) == pytest.approx(math.exp(-1.0))


def test_a_perfect_defence_always_keeps_a_clean_sheet():
    assert clean_sheet_probability(0.0) == 1.0


def test_the_transfer_case_moves_the_expectation():
    """The requirement: a player joining a leakier club loses clean sheets."""
    data = {
        "2025-26": pd.concat([club_season("Arsenal", 0, 0.76), club_season("Burnley", 2, 2.02)])
    }
    blended = blend_team_defence(data)

    outlook = clean_sheet_outlook(pd.Series(["Arsenal", "Burnley"]), blended).set_index("team_name")

    assert outlook.loc["Arsenal", "clean_sheet_probability"] > 0.4
    assert outlook.loc["Burnley", "clean_sheet_probability"] < 0.2


def test_the_outlook_flags_promoted_clubs():
    blended = blend_team_defence({"2025-26": club_season("Arsenal", 0, 0.7)})

    outlook = clean_sheet_outlook(pd.Series(["Arsenal", "Leeds"]), blended).set_index("team_name")

    assert not outlook.loc["Arsenal", "is_promoted"]
    assert outlook.loc["Leeds", "is_promoted"]


def test_empty_input_is_safe():
    assert team_match_defence(pd.DataFrame()).empty
    assert season_defence(pd.DataFrame()).empty
    assert blend_team_defence({}).empty
