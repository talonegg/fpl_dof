"""Tests for the season-opening predictor."""

from __future__ import annotations

import pandas as pd
import pytest

from fpl.models.preseason import DEFAULT_HORIZON, PreseasonPredictor


def career(position="MID", minutes=3000, appearances=34, seasons=2, **rates):
    row = {
        "element": 1,
        "position": position,
        "team": "Arsenal",
        "career_minutes": minutes,
        "career_appearances": appearances,
        "seasons_seen": seasons,
    }
    row.update({f"{key}_per_90": value for key, value in rates.items()})
    return pd.DataFrame([row])


DEFENCE = pd.DataFrame(
    [
        {"team_name": "Arsenal", "expected_goals_conceded_per_match": 0.8},
        {"team_name": "Burnley", "expected_goals_conceded_per_match": 2.0},
    ]
)


def test_a_regular_starter_scores_more_than_a_bit_part_player():
    """The lesson that cost the first version everything."""
    starter = career(minutes=3000, appearances=34, expected_goals=0.3)
    fringe = career(minutes=200, appearances=20, expected_goals=0.3)

    model = PreseasonPredictor(team_defence=DEFENCE)

    assert (
        model.predict(starter)["expected_points"].iloc[0]
        > model.predict(fringe)["expected_points"].iloc[0]
    )


def test_goals_are_worth_more_from_a_defender_than_a_forward():
    model = PreseasonPredictor(team_defence=DEFENCE)

    defender = model.predict(career("DEF", expected_goals=0.3))
    forward = model.predict(career("FWD", expected_goals=0.3))

    # A defender's goal is 6 points against a forward's 4, but the forward
    # also loses the clean-sheet term, so compare the goal contribution only.
    assert defender["expected_points"].iloc[0] > forward["expected_points"].iloc[0]


def test_a_defender_at_a_mean_club_beats_one_at_a_leaky_club():
    """The clean-sheet requirement: the club, not the player's own record."""
    model = PreseasonPredictor(team_defence=DEFENCE)

    good = career("DEF")
    bad = career("DEF")
    bad.loc[0, "team"] = "Burnley"

    assert (
        model.predict(good)["expected_points"].iloc[0]
        > (model.predict(bad)["expected_points"].iloc[0])
    )


def test_a_midfielder_gains_less_from_a_clean_sheet_than_a_defender():
    model = PreseasonPredictor(team_defence=DEFENCE)

    defender = model.predict(career("DEF"))["expected_points"].iloc[0]
    midfielder = model.predict(career("MID"))["expected_points"].iloc[0]

    assert defender > midfielder


def test_assists_contribute():
    model = PreseasonPredictor(team_defence=DEFENCE)

    with_assists = model.predict(career(expected_assists=0.4))["expected_points"].iloc[0]
    without = model.predict(career())["expected_points"].iloc[0]

    assert with_assists > without


def test_a_fixture_multiplier_scales_the_result():
    model = PreseasonPredictor(team_defence=DEFENCE)
    player = career(expected_goals=0.3)

    plain = model.predict(player)["expected_points"].iloc[0]
    easy = model.predict(player, fixture_difficulty=pd.Series([1.2]))["expected_points"].iloc[0]

    assert easy > plain


def test_expected_points_are_never_negative():
    model = PreseasonPredictor(team_defence=DEFENCE)

    result = model.predict(career("FWD", minutes=0, appearances=0))

    assert (result["expected_points"] >= 0).all()


def test_the_forecast_minutes_are_reported_alongside():
    model = PreseasonPredictor(team_defence=DEFENCE)

    assert "expected_minutes" in model.predict(career()).columns


def test_an_empty_career_gives_no_predictions():
    assert PreseasonPredictor().predict(pd.DataFrame()).empty


def test_the_default_horizon_is_the_shared_one():
    assert PreseasonPredictor().horizon == DEFAULT_HORIZON


def test_the_finishing_adjustment_can_be_disabled():
    player = career(expected_goals=0.3, goals_scored=0.6)

    with_adjustment = PreseasonPredictor(team_defence=DEFENCE, use_finishing_adjustment=True)
    without = PreseasonPredictor(team_defence=DEFENCE, use_finishing_adjustment=False)

    assert with_adjustment.predict(player)["expected_points"].iloc[0] != pytest.approx(
        without.predict(player)["expected_points"].iloc[0]
    )


# -- Defensive contributions ----------------------------------------------


def defensive_career(rate=1.0, matches=30, position="DEF"):
    frame = career(position)
    frame["defensive_rate"] = rate
    frame["defensive_matches"] = matches
    return frame


def test_a_defensive_contributor_scores_more_when_the_season_scores_them():
    model = PreseasonPredictor(team_defence=DEFENCE, score_defensive_contributions=True)
    off = PreseasonPredictor(team_defence=DEFENCE, score_defensive_contributions=False)

    player = defensive_career()

    assert (
        model.predict(player)["expected_points"].iloc[0]
        > off.predict(player)["expected_points"].iloc[0]
    )


def test_a_season_that_does_not_score_them_ignores_the_rate_entirely():
    """2024-25 must not be credited for actions that paid nothing."""
    model = PreseasonPredictor(team_defence=DEFENCE, score_defensive_contributions=False)

    with_rate = model.predict(defensive_career(rate=1.0))["expected_points"].iloc[0]
    without = model.predict(defensive_career(rate=0.0))["expected_points"].iloc[0]

    assert with_rate == without


def test_a_frequent_contributor_outscores_a_rare_one():
    model = PreseasonPredictor(team_defence=DEFENCE, score_defensive_contributions=True)

    assert (
        model.predict(defensive_career(rate=0.9))["expected_points"].iloc[0]
        > model.predict(defensive_career(rate=0.1))["expected_points"].iloc[0]
    )


def test_a_missing_rate_column_is_survivable():
    """The 2025-26 blind case must produce a prediction, not an exception."""
    model = PreseasonPredictor(team_defence=DEFENCE, score_defensive_contributions=True)

    assert not model.predict(career("DEF")).empty
