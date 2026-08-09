"""Tests for opening-run fixture difficulty."""

from __future__ import annotations

import pandas as pd

from fpl.features.team_strength import opening_run_difficulty, opponent_names


def season(rows):
    return pd.DataFrame(rows)


def match(fixture, gameweek, home, away):
    return [
        {"fixture": fixture, "gameweek": gameweek, "team_name": home, "minutes": 90},
        {"fixture": fixture, "gameweek": gameweek, "team_name": away, "minutes": 90},
    ]


ATTACK = pd.DataFrame(
    [
        {"team_name": "Leaky", "expected_goals_conceded_per_match": 2.0},
        {"team_name": "Solid", "expected_goals_conceded_per_match": 1.0},
        {"team_name": "Alpha", "expected_goals_conceded_per_match": 1.5},
        {"team_name": "Beta", "expected_goals_conceded_per_match": 1.5},
    ]
)


def test_opponents_resolve_through_the_shared_fixture_id():
    """The archive gives no name for opponent_team; both sides share a fixture."""
    schedule = opponent_names(season(match(1, 1, "Alpha", "Beta")))

    alpha = schedule[schedule["team_name"] == "Alpha"]
    assert alpha["opponent_name"].iloc[0] == "Beta"


def test_both_sides_of_a_match_appear():
    schedule = opponent_names(season(match(1, 1, "Alpha", "Beta")))

    assert set(schedule["team_name"]) == {"Alpha", "Beta"}


def test_a_run_of_leaky_opponents_rates_above_one():
    rows = match(1, 1, "Alpha", "Leaky") + match(2, 2, "Alpha", "Leaky")
    difficulty = opening_run_difficulty(season(rows), ATTACK)

    alpha = difficulty[difficulty["team_name"] == "Alpha"]["opening_difficulty"].iloc[0]
    assert alpha > 1.0


def test_a_run_of_solid_opponents_rates_below_one():
    rows = match(1, 1, "Alpha", "Solid") + match(2, 2, "Alpha", "Solid")
    difficulty = opening_run_difficulty(season(rows), ATTACK)

    alpha = difficulty[difficulty["team_name"] == "Alpha"]["opening_difficulty"].iloc[0]
    assert alpha < 1.0


def test_the_early_fixture_dominates_the_late_one():
    """Gameweek 1 must outweigh gameweek 10, per the requirement."""
    early = season(match(1, 1, "Alpha", "Leaky") + match(2, 10, "Alpha", "Solid"))
    late = season(match(1, 1, "Beta", "Solid") + match(2, 10, "Beta", "Leaky"))

    early_rating = opening_run_difficulty(early, ATTACK)
    late_rating = opening_run_difficulty(late, ATTACK)

    alpha = early_rating[early_rating["team_name"] == "Alpha"]["opening_difficulty"].iloc[0]
    beta = late_rating[late_rating["team_name"] == "Beta"]["opening_difficulty"].iloc[0]
    assert alpha > beta


def test_fixtures_beyond_the_horizon_are_ignored():
    inside = season(match(1, 1, "Alpha", "Solid"))
    plus_outside = season(match(1, 1, "Alpha", "Solid") + match(2, 30, "Alpha", "Leaky"))

    a = opening_run_difficulty(inside, ATTACK)
    a = a[a["team_name"] == "Alpha"]["opening_difficulty"].iloc[0]
    b = opening_run_difficulty(plus_outside, ATTACK)
    b = b[b["team_name"] == "Alpha"]["opening_difficulty"].iloc[0]

    assert a == b


def test_an_empty_season_gives_no_ratings():
    assert opening_run_difficulty(pd.DataFrame(), ATTACK).empty


def test_no_attack_table_gives_no_ratings():
    rows = match(1, 1, "Alpha", "Beta")
    assert opening_run_difficulty(season(rows), pd.DataFrame()).empty
