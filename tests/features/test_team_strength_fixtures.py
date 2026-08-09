"""Tests for opening-run fixture difficulty."""

from __future__ import annotations

import pandas as pd

from fpl.features.team_strength import (
    fixture_weights,
    opening_run_difficulty,
    opponent_names,
)


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
    """Gameweek 1 must outweigh gameweek 7, per the requirement."""
    early = season(match(1, 1, "Alpha", "Leaky") + match(2, 7, "Alpha", "Solid"))
    late = season(match(1, 1, "Beta", "Solid") + match(2, 7, "Beta", "Leaky"))

    early_rating = opening_run_difficulty(early, ATTACK)
    late_rating = opening_run_difficulty(late, ATTACK)

    alpha = early_rating[early_rating["team_name"] == "Alpha"]["opening_difficulty"].iloc[0]
    beta = late_rating[late_rating["team_name"] == "Beta"]["opening_difficulty"].iloc[0]
    assert alpha > beta


def test_fixtures_beyond_the_horizon_are_ignored():
    inside = season(match(1, 1, "Alpha", "Solid"))
    plus_outside = season(match(1, 1, "Alpha", "Solid") + match(2, 12, "Alpha", "Leaky"))

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


# -- The weighting curve --------------------------------------------------


def test_the_opening_three_gameweeks_share_full_weight():
    """The squad is certainly held for these, so none is discounted."""
    weights = fixture_weights()

    assert weights[0] == weights[1] == weights[2] == 1.0


def test_weight_diminishes_across_gameweeks_four_to_seven():
    weights = fixture_weights()

    assert weights[3] > weights[4] > weights[5] > weights[6]


def test_the_fourth_gameweek_counts_less_than_the_third():
    weights = fixture_weights()

    assert weights[3] < weights[2]


def test_the_run_stops_at_seven():
    assert len(fixture_weights()) == 7


def test_the_seventh_gameweek_still_carries_real_weight():
    """Diminishing, not switched off -- gameweek 7 is a quarter of gameweek 1."""
    weights = fixture_weights()

    assert 0.15 < weights[6] < 0.35


def test_the_plateau_is_adjustable():
    assert fixture_weights(horizon=7, plateau=1)[1] < 1.0


def test_a_leaky_opponent_in_the_plateau_beats_one_in_the_tail():
    """The shape must reach selection, which it only does through difficulty."""
    early = season(match(1, 1, "Alpha", "Leaky") + match(2, 7, "Alpha", "Solid"))
    late = season(match(1, 1, "Beta", "Solid") + match(2, 7, "Beta", "Leaky"))

    early_rating = opening_run_difficulty(early, ATTACK)
    late_rating = opening_run_difficulty(late, ATTACK)

    alpha = early_rating[early_rating["team_name"] == "Alpha"]["opening_difficulty"].iloc[0]
    beta = late_rating[late_rating["team_name"] == "Beta"]["opening_difficulty"].iloc[0]
    assert alpha > beta


def test_the_first_three_gameweeks_are_interchangeable_to_difficulty():
    """Flat weighting means gameweek 1 and gameweek 3 rate a club identically."""
    first = season(match(1, 1, "Alpha", "Leaky") + match(2, 2, "Alpha", "Solid"))
    third = season(match(1, 3, "Alpha", "Leaky") + match(2, 2, "Alpha", "Solid"))

    a = opening_run_difficulty(first, ATTACK)
    a = a[a["team_name"] == "Alpha"]["opening_difficulty"].iloc[0]
    b = opening_run_difficulty(third, ATTACK)
    b = b[b["team_name"] == "Alpha"]["opening_difficulty"].iloc[0]

    assert a == b
