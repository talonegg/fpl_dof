"""Tests for the fixture-aware scouting heuristic."""

from __future__ import annotations

import pandas as pd
import pytest

from fpl.domain.fixtures import build_team_schedule
from fpl.features.scouting import add_outlook, difficulty_multiplier, team_outlook

TEAMS = [
    {"id": 1, "name": "Arsenal"},
    {"id": 2, "name": "Aston Villa"},
    {"id": 3, "name": "Bournemouth"},
]

PLAYERS = pd.DataFrame(
    [
        {"element": 1, "team": 1, "web_name": "Saka", "form": 5.0},
        {"element": 2, "team": 2, "web_name": "Watkins", "form": 4.0},
    ]
)


def make_fixture(fixture_id, event, home, away, home_difficulty, away_difficulty):
    return {
        "id": fixture_id,
        "event": event,
        "team_h": home,
        "team_a": away,
        "team_h_difficulty": home_difficulty,
        "team_a_difficulty": away_difficulty,
        "kickoff_time": "2026-08-21T19:00:00Z",
        "finished": False,
    }


def test_a_neutral_fixture_does_not_move_the_number():
    assert difficulty_multiplier(3) == 1.0


def test_the_easiest_fixture_is_a_twenty_percent_uplift():
    assert difficulty_multiplier(1) == pytest.approx(1.2)


def test_the_hardest_fixture_is_a_twenty_percent_discount():
    assert difficulty_multiplier(5) == pytest.approx(0.8)


def test_team_outlook_sums_multipliers_over_the_window():
    # Arsenal: difficulty 2 (1.1) then difficulty 1 (1.2) -> 2.3
    fixtures = [
        make_fixture(1, 1, home=1, away=2, home_difficulty=2, away_difficulty=4),
        make_fixture(2, 2, home=1, away=3, home_difficulty=1, away_difficulty=4),
    ]
    schedule = build_team_schedule(fixtures, TEAMS)

    outlook = team_outlook(schedule, from_gameweek=1, horizon=2)
    arsenal = outlook[outlook["team"] == 1].iloc[0]

    assert arsenal["fixture_multiplier"] == pytest.approx(2.3)
    assert arsenal["fixture_count"] == 2


def test_a_double_gameweek_earns_a_bigger_multiplier_than_a_single():
    fixtures = [
        make_fixture(1, 1, home=1, away=2, home_difficulty=3, away_difficulty=3),
        make_fixture(2, 1, home=3, away=1, home_difficulty=3, away_difficulty=3),
    ]
    schedule = build_team_schedule(fixtures, TEAMS)

    outlook = team_outlook(schedule, from_gameweek=1, horizon=1)
    arsenal = outlook[outlook["team"] == 1].iloc[0]
    villa = outlook[outlook["team"] == 2].iloc[0]

    # Two neutral fixtures beat one, which is the whole point of summing.
    assert arsenal["fixture_multiplier"] == pytest.approx(2.0)
    assert villa["fixture_multiplier"] == pytest.approx(1.0)


def test_team_outlook_of_an_empty_window_is_empty():
    schedule = build_team_schedule([make_fixture(1, 1, 1, 2, 3, 3)], TEAMS)

    assert team_outlook(schedule, from_gameweek=10, horizon=3).empty


def test_outlook_score_is_form_times_the_summed_multiplier():
    fixtures = [make_fixture(1, 1, home=1, away=2, home_difficulty=1, away_difficulty=5)]
    schedule = build_team_schedule(fixtures, TEAMS)

    result = add_outlook(PLAYERS, schedule, from_gameweek=1, horizon=1)

    saka = result[result["web_name"] == "Saka"].iloc[0]
    watkins = result[result["web_name"] == "Watkins"].iloc[0]

    # Saka: form 5.0 x 1.2 (easiest fixture) = 6.0
    assert saka["outlook_score"] == pytest.approx(6.0)
    # Watkins: form 4.0 x 0.8 (hardest fixture) = 3.2
    assert watkins["outlook_score"] == pytest.approx(3.2)


def test_form_alone_can_be_beaten_by_a_kinder_fixture():
    """The whole reason the heuristic exists."""
    fixtures = [make_fixture(1, 1, home=1, away=2, home_difficulty=1, away_difficulty=5)]
    schedule = build_team_schedule(fixtures, TEAMS)
    players = pd.DataFrame(
        [
            {"element": 1, "team": 1, "web_name": "EasyRun", "form": 4.5},
            {"element": 2, "team": 2, "web_name": "HardRun", "form": 5.0},
        ]
    )

    result = add_outlook(players, schedule, from_gameweek=1, horizon=1)
    ranked = result.sort_values("outlook_score", ascending=False)

    assert ranked.iloc[0]["web_name"] == "EasyRun"


def test_a_blank_gameweek_scores_zero_not_nan():
    fixtures = [make_fixture(1, 1, home=2, away=3, home_difficulty=3, away_difficulty=3)]
    schedule = build_team_schedule(fixtures, TEAMS)

    result = add_outlook(PLAYERS, schedule, from_gameweek=1, horizon=1)
    arsenal_player = result[result["team"] == 1].iloc[0]

    assert arsenal_player["fixture_count"] == 0
    assert arsenal_player["outlook_score"] == 0.0


def test_missing_form_is_treated_as_zero():
    fixtures = [make_fixture(1, 1, home=1, away=2, home_difficulty=3, away_difficulty=3)]
    schedule = build_team_schedule(fixtures, TEAMS)
    players = pd.DataFrame([{"element": 1, "team": 1, "web_name": "New", "form": None}])

    result = add_outlook(players, schedule, from_gameweek=1, horizon=1)

    assert result.loc[0, "outlook_score"] == 0.0


def test_every_player_survives_the_merge(bootstrap, schedule):
    from fpl.domain.players import build_players_frame

    players = build_players_frame(bootstrap)

    result = add_outlook(players, schedule, from_gameweek=1, horizon=5)

    assert len(result) == len(players)
