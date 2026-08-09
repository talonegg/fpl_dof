"""Tests for the component model.

Each component is checked against the FPL scoring rules by hand, in isolation,
by building a history where only that component is non-zero. The rules are
exact, so these expectations are exact.
"""

from __future__ import annotations

import pandas as pd
import pytest

from fpl.models.base import PREDICTION_COLUMNS, Predictor
from fpl.models.components import ComponentPredictor

BLANK = {
    "expected_goals": 0.0,
    "expected_assists": 0.0,
    "goals_scored": 0,
    "assists": 0,
    "saves": 0,
    "bonus": 0,
    "goals_conceded": 0,
    "yellow_cards": 0,
    "red_cards": 0,
    "clean_sheets": 0,
    "defensive_contribution": 0,
    "opponent_team": 10,
    "was_home": True,
}


def make_history(position, gameweeks=4, minutes=90, **overrides):
    """A player who plays every week, with only the named components non-zero."""
    return pd.DataFrame(
        [
            {
                **BLANK,
                "element": 1,
                "gameweek": gameweek,
                "position": position,
                "minutes": minutes,
                "total_points": 0,
                **overrides,
            }
            for gameweek in range(1, gameweeks + 1)
        ]
    )


def predict_one(history, fixtures=None):
    result = ComponentPredictor().predict(history, gameweek=99, fixtures=fixtures)
    return result["expected_points"].iloc[0]


def test_satisfies_the_predictor_protocol():
    assert isinstance(ComponentPredictor(), Predictor)


def test_returns_the_expected_columns():
    result = ComponentPredictor().predict(make_history("MID"), 5)

    assert set(PREDICTION_COLUMNS) <= set(result.columns)


def test_an_empty_history_gives_an_empty_frame():
    assert ComponentPredictor().predict(pd.DataFrame(), 1).empty


def test_history_without_positions_cannot_be_used():
    history = make_history("MID").drop(columns=["position"])

    assert ComponentPredictor().predict(history, 5).empty


def test_an_ever_present_player_earns_two_appearance_points():
    assert predict_one(make_history("MID")) == pytest.approx(2.0)


def test_a_substitute_who_never_reaches_an_hour_earns_one():
    assert predict_one(make_history("MID", minutes=45)) == pytest.approx(1.0)


def test_a_player_who_never_plays_earns_nothing():
    assert predict_one(make_history("MID", minutes=0)) == pytest.approx(0.0)


@pytest.mark.parametrize("position,points", [("GK", 6), ("DEF", 6), ("MID", 5), ("FWD", 4)])
def test_goals_are_worth_the_right_amount_per_position(position, points):
    # One xG per match, playing full matches: one goal's worth of points.
    history = make_history(position, expected_goals=1.0)

    # Appearance (2) plus one goal at the position's rate.
    assert predict_one(history) == pytest.approx(2.0 + points)


def test_assists_are_worth_three():
    history = make_history("MID", expected_assists=1.0)

    assert predict_one(history) == pytest.approx(2.0 + 3.0)


def test_goals_come_from_expected_goals_not_actual_goals():
    """The central premise of the model: finishing is noisy, xG is not."""
    lucky = make_history("FWD", expected_goals=0.1, goals_scored=2)

    from_xg = ComponentPredictor(use_expected_goals=True).predict(lucky, 5)
    from_actuals = ComponentPredictor(use_expected_goals=False).predict(lucky, 5)

    assert from_xg["expected_points"].iloc[0] < from_actuals["expected_points"].iloc[0]


@pytest.mark.parametrize("position,points", [("GK", 4), ("DEF", 4), ("MID", 1), ("FWD", 0)])
def test_clean_sheets_are_worth_the_right_amount_per_position(position, points):
    history = make_history(position, clean_sheets=1)

    assert predict_one(history) == pytest.approx(2.0 + points)


def test_a_clean_sheet_does_not_pay_a_player_who_never_lasts_an_hour():
    """FPL requires 60 minutes for the clean-sheet points."""
    history = make_history("DEF", minutes=45, clean_sheets=1)

    # One appearance point, no clean-sheet points.
    assert predict_one(history) == pytest.approx(1.0)


def test_saves_are_worth_a_point_per_three():
    history = make_history("GK", saves=3)

    assert predict_one(history) == pytest.approx(2.0 + 1.0)


def test_defenders_lose_a_point_per_two_goals_conceded():
    history = make_history("DEF", goals_conceded=2)

    assert predict_one(history) == pytest.approx(2.0 - 1.0)


def test_midfielders_do_not_lose_points_for_goals_conceded():
    history = make_history("MID", goals_conceded=2)

    assert predict_one(history) == pytest.approx(2.0)


def test_a_yellow_card_costs_a_point():
    history = make_history("MID", yellow_cards=1)

    assert predict_one(history) == pytest.approx(2.0 - 1.0)


def test_a_red_card_costs_three():
    history = make_history("MID", red_cards=1)

    assert predict_one(history) == pytest.approx(2.0 - 3.0)


def test_bonus_is_carried_through():
    history = make_history("MID", bonus=3)

    assert predict_one(history) == pytest.approx(2.0 + 3.0)


def test_defenders_clearing_the_defensive_threshold_earn_two():
    history = make_history("DEF", defensive_contribution=12)

    assert predict_one(history) == pytest.approx(2.0 + 2.0)


def test_defenders_below_the_threshold_earn_nothing_for_it():
    history = make_history("DEF", defensive_contribution=9)

    assert predict_one(history) == pytest.approx(2.0)


def test_midfielders_have_a_higher_defensive_threshold_than_defenders():
    """10 actions pays a defender but not a midfielder."""
    defender = make_history("DEF", defensive_contribution=10)
    midfielder = make_history("MID", defensive_contribution=10)

    assert predict_one(defender) == pytest.approx(4.0)
    assert predict_one(midfielder) == pytest.approx(2.0)


def test_goalkeepers_are_not_eligible_for_defensive_contributions():
    history = make_history("GK", defensive_contribution=29)

    assert predict_one(history) == pytest.approx(2.0)


def test_a_generous_opponent_raises_the_clean_sheet_expectation():
    # Team 10 concedes clean sheets freely; team 20 does not.
    history = pd.concat(
        [
            make_history("DEF", clean_sheets=1).assign(opponent_team=10),
            make_history("DEF", clean_sheets=0).assign(element=2, opponent_team=20),
        ]
    )
    easy = pd.DataFrame([{"element": 1, "opponent_team": 10, "was_home": True}])
    hard = pd.DataFrame([{"element": 1, "opponent_team": 20, "was_home": True}])

    against_easy = ComponentPredictor().predict(history, 5, easy)
    against_hard = ComponentPredictor().predict(history, 5, hard)

    easy_points = against_easy[against_easy["element"] == 1]["expected_points"].iloc[0]
    hard_points = against_hard[against_hard["element"] == 1]["expected_points"].iloc[0]
    assert easy_points > hard_points


def test_components_add_up_rather_than_replacing_each_other():
    history = make_history("MID", expected_goals=1.0, expected_assists=1.0, bonus=3, clean_sheets=1)

    # 2 appearance + 5 goal + 3 assist + 3 bonus + 1 clean sheet
    assert predict_one(history) == pytest.approx(14.0)


def test_predictions_are_never_nan():
    history = make_history("MID", minutes=0)

    result = ComponentPredictor().predict(history, 5)

    assert result["expected_points"].notna().all()


def test_one_row_per_player():
    history = pd.concat([make_history("MID"), make_history("DEF").assign(element=2)])

    result = ComponentPredictor().predict(history, 5)

    assert not result["element"].duplicated().any()


def test_the_model_name_records_its_configuration():
    assert ComponentPredictor(4).name == "Component(4)"
    assert ComponentPredictor(4, use_expected_goals=False).name == "Component(4, actuals)"


# --- Regressions from the second review pass ---


def test_the_sixty_minute_test_is_not_applied_twice():
    """`clean_sheets` already means "played 60+ and kept one"."""
    # Started 2 of 4 matches, kept a clean sheet in both starts.
    history = pd.concat(
        [
            make_history("DEF", gameweeks=2, minutes=90, clean_sheets=1),
            make_history("DEF", gameweeks=2, minutes=20, clean_sheets=0).assign(gameweek=[3, 4]),
        ]
    )

    points = predict_one(history)

    # Clean sheet in every start, starting half the time: 4 x 0.5 = 2.0 for the
    # clean-sheet term. Halving it again would give 1.0.
    appearance = 0.5 * 2 + 0.5 * 1
    assert points == pytest.approx(appearance + 2.0, abs=0.01)


@pytest.mark.parametrize("spelling", ["GK", "GKP", "Goalkeeper"])
def test_goalkeepers_score_the_same_however_the_position_is_spelled(spelling):
    """The archive uses both GK and GKP; an unknown spelling scored zero."""
    history = make_history(spelling, clean_sheets=1, saves=3)

    # 2 appearance + 4 clean sheet + 1 saves
    assert predict_one(history) == pytest.approx(7.0)


def test_an_unrecognised_position_does_not_silently_drop_the_conceded_penalty():
    defender = make_history("DEF", goals_conceded=2)
    spelled_out = make_history("Defender", goals_conceded=2)

    assert predict_one(defender) == pytest.approx(predict_one(spelled_out))


def test_defensive_contributions_are_scaled_by_expected_minutes():
    """A benched defender kept his contribution points without playing."""
    starter = make_history("DEF", minutes=90, defensive_contribution=12)
    benched = make_history("DEF", minutes=5, defensive_contribution=12)

    starter_points = predict_one(starter)
    benched_points = predict_one(benched)

    assert starter_points > benched_points
    # The bench player should keep only a small fraction of the DC points.
    assert benched_points < 1.5
