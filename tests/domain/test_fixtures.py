"""Tests for the team-perspective fixture schedule.

Blank and double gameweeks are constructed by hand rather than taken from the
snapshot: the real early-season fixture list has none, and those are exactly
the cases that matter.
"""

from __future__ import annotations

import pandas as pd

from fpl.domain.fixtures import (
    blanks_and_doubles,
    build_team_schedule,
    difficulty_summary,
    next_gameweek,
    upcoming,
)

TEAMS = [
    {"id": 1, "name": "Arsenal"},
    {"id": 2, "name": "Aston Villa"},
    {"id": 3, "name": "Bournemouth"},
]


def make_fixture(fixture_id, event, home, away, home_difficulty=2, away_difficulty=4):
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


def test_each_match_becomes_two_rows(schedule, fixtures_snapshot):
    assert len(schedule) == 2 * len(fixtures_snapshot["fixtures"])


def test_home_and_away_rows_mirror_each_other():
    schedule = build_team_schedule([make_fixture(1, 1, home=1, away=2)], TEAMS)

    home = schedule[schedule["is_home"]].iloc[0]
    away = schedule[~schedule["is_home"]].iloc[0]

    assert home["team_name"] == "Arsenal"
    assert home["opponent_name"] == "Aston Villa"
    assert away["team_name"] == "Aston Villa"
    assert away["opponent_name"] == "Arsenal"
    # Difficulty is per side, not per match.
    assert home["difficulty"] == 2
    assert away["difficulty"] == 4


def test_every_team_plays_once_per_gameweek_in_a_normal_season(schedule):
    counts = schedule.groupby(["team", "gameweek"]).size()

    assert set(counts.unique()) == {1}


def test_fixtures_without_a_gameweek_are_dropped():
    fixtures = [make_fixture(1, 1, 1, 2), make_fixture(2, None, 1, 3)]

    schedule = build_team_schedule(fixtures, TEAMS)

    assert set(schedule["fixture_id"]) == {1}


def test_build_team_schedule_handles_no_fixtures():
    assert build_team_schedule([], TEAMS).empty


def test_upcoming_window_is_inclusive_of_both_ends(schedule):
    window = upcoming(schedule, from_gameweek=2, horizon=3)

    assert sorted(window["gameweek"].unique()) == [2, 3, 4]


def test_difficulty_summary_ranks_the_easiest_run_first():
    fixtures = [
        # Arsenal get two easy games, Villa two hard ones.
        make_fixture(1, 1, home=1, away=2, home_difficulty=2, away_difficulty=5),
        make_fixture(2, 2, home=1, away=3, home_difficulty=2, away_difficulty=3),
        make_fixture(3, 2, home=2, away=3, home_difficulty=5, away_difficulty=3),
    ]
    schedule = build_team_schedule(fixtures, TEAMS)

    summary = difficulty_summary(schedule, from_gameweek=1, horizon=2)

    assert summary.iloc[0]["team_name"] == "Arsenal"
    assert summary.iloc[0]["total_difficulty"] == 4
    assert summary.iloc[0]["fixture_count"] == 2


def test_a_double_gameweek_counts_both_fixtures():
    fixtures = [
        make_fixture(1, 1, home=1, away=2),
        make_fixture(2, 1, home=3, away=1),  # Arsenal again in the same gameweek
    ]
    schedule = build_team_schedule(fixtures, TEAMS)

    summary = difficulty_summary(schedule, from_gameweek=1, horizon=1)
    arsenal = summary[summary["team_name"] == "Arsenal"].iloc[0]

    assert arsenal["fixture_count"] == 2


def test_blanks_and_doubles_finds_both():
    fixtures = [
        make_fixture(1, 1, home=1, away=2),
        make_fixture(2, 1, home=3, away=1),  # Arsenal double in GW1
        make_fixture(3, 2, home=2, away=3),  # Arsenal blank in GW2
    ]
    schedule = build_team_schedule(fixtures, TEAMS)

    notable = blanks_and_doubles(schedule, from_gameweek=1, horizon=2)

    arsenal = notable[notable["team_name"] == "Arsenal"]
    assert list(arsenal["kind"]) == ["double", "blank"]
    assert list(arsenal["fixture_count"]) == [2, 0]


def test_a_normal_run_of_fixtures_has_nothing_notable(schedule):
    assert blanks_and_doubles(schedule, from_gameweek=1, horizon=6).empty


def test_next_gameweek_prefers_the_flagged_event():
    events = [
        {"id": 1, "finished": True, "is_next": False},
        {"id": 2, "finished": False, "is_next": True},
    ]

    assert next_gameweek(events) == 2


def test_next_gameweek_falls_back_before_the_season_opens():
    events = [{"id": 1, "finished": False}, {"id": 2, "finished": False}]

    assert next_gameweek(events) == 1


def test_next_gameweek_is_none_when_the_season_is_over():
    events = [{"id": 38, "finished": True, "is_next": False}]

    assert next_gameweek(events) is None


def test_next_gameweek_on_the_real_snapshot(bootstrap):
    assert next_gameweek(bootstrap["events"]) == 1


def test_difficulty_summary_of_an_empty_window_is_empty(schedule):
    summary = difficulty_summary(schedule, from_gameweek=30, horizon=5)

    assert summary.empty
    assert "mean_difficulty" in summary.columns


def test_schedule_columns_are_stable(schedule):
    assert isinstance(schedule, pd.DataFrame)
    assert {"team_name", "opponent_name", "gameweek", "is_home", "difficulty"} <= set(
        schedule.columns
    )
