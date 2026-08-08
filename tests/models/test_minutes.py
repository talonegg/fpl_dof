"""Tests for the minutes-based models, on hand-computed examples."""

from __future__ import annotations

import pandas as pd
import pytest

from fpl.models.base import PREDICTION_COLUMNS, Predictor
from fpl.models.minutes import MinutesAdjustedPredictor, OpponentAdjustedPredictor


def history_rows(element, points_and_minutes, opponent=10):
    return [
        {
            "element": element,
            "gameweek": gameweek,
            "total_points": points,
            "minutes": minutes,
            "opponent_team": opponent,
            "was_home": True,
        }
        for gameweek, (points, minutes) in enumerate(points_and_minutes, start=1)
    ]


# Element 1: a starter. 4 games, 90 minutes each, 6 points each.
# Element 2: a bench player with the same per-90 rate but a quarter of the
# minutes -- identical rate, very different expected return.
HISTORY = pd.DataFrame(history_rows(1, [(6, 90)] * 4) + history_rows(2, [(1.5, 22.5)] * 4))


@pytest.mark.parametrize("predictor", [MinutesAdjustedPredictor(), OpponentAdjustedPredictor()])
def test_satisfies_the_predictor_protocol(predictor):
    assert isinstance(predictor, Predictor)


@pytest.mark.parametrize("predictor", [MinutesAdjustedPredictor(), OpponentAdjustedPredictor()])
def test_returns_the_expected_columns(predictor):
    result = predictor.predict(HISTORY, gameweek=5)

    assert set(PREDICTION_COLUMNS) <= set(result.columns)


@pytest.mark.parametrize("predictor", [MinutesAdjustedPredictor(), OpponentAdjustedPredictor()])
def test_handles_an_empty_history(predictor):
    assert predictor.predict(HISTORY.head(0), gameweek=1).empty


def test_a_full_time_starter_gets_their_per_90_rate():
    # 24 points from 360 minutes is 6.0 per 90; expected minutes are 90.
    result = MinutesAdjustedPredictor().predict(HISTORY, gameweek=5)

    starter = result[result["element"] == 1]["expected_points"].iloc[0]
    assert starter == pytest.approx(6.0)


def test_the_same_rate_on_fewer_minutes_predicts_fewer_points():
    """The whole point of the model: rate and playing time are different things."""
    result = MinutesAdjustedPredictor().predict(HISTORY, gameweek=5)

    starter = result[result["element"] == 1]["expected_points"].iloc[0]
    bench = result[result["element"] == 2]["expected_points"].iloc[0]

    assert bench < starter
    # A quarter of the minutes at the same rate is a quarter of the points.
    assert bench == pytest.approx(starter / 4)


def test_a_player_who_has_stopped_playing_is_marked_down():
    """Recent minutes, not season minutes, drive the expectation."""
    dropped = pd.DataFrame(history_rows(1, [(6, 90), (6, 90), (6, 90), (6, 90), (0, 0), (0, 0)]))

    result = MinutesAdjustedPredictor(minutes_window=2).predict(dropped, gameweek=7)

    assert result["expected_points"].iloc[0] == pytest.approx(0.0)


def test_a_thin_minutes_sample_falls_back_instead_of_extrapolating():
    """One 5-minute cameo must not imply a spectacular per-90 rate."""
    cameo = pd.DataFrame(history_rows(1, [(2, 5)]))

    result = MinutesAdjustedPredictor().predict(cameo, gameweek=2)

    # Extrapolating would give 2 * 90 / 5 = 36 points. The fallback is the
    # plain average: 2.
    assert result["expected_points"].iloc[0] == pytest.approx(2.0)


def test_predictions_are_never_nan():
    zero_minutes = pd.DataFrame(history_rows(1, [(0, 0), (0, 0)]))

    result = MinutesAdjustedPredictor().predict(zero_minutes, gameweek=3)

    assert result["expected_points"].notna().all()


def test_one_row_per_player():
    result = MinutesAdjustedPredictor().predict(HISTORY, gameweek=5)

    assert not result["element"].duplicated().any()


# --- Opponent adjustment ---

# Team 10 concedes freely, team 20 does not.
GENEROUS_HISTORY = pd.DataFrame(
    history_rows(1, [(10, 90)] * 3, opponent=10) + history_rows(2, [(2, 90)] * 3, opponent=20)
)


def test_a_generous_opponent_raises_the_prediction():
    fixtures = pd.DataFrame([{"element": 1, "opponent_team": 10, "was_home": True}])

    base = MinutesAdjustedPredictor().predict(GENEROUS_HISTORY, 4)
    adjusted = OpponentAdjustedPredictor().predict(GENEROUS_HISTORY, 4, fixtures)

    base_value = base[base["element"] == 1]["expected_points"].iloc[0]
    adjusted_value = adjusted[adjusted["element"] == 1]["expected_points"].iloc[0]

    assert adjusted_value > base_value


def test_a_mean_opponent_leaves_the_prediction_alone():
    flat = pd.DataFrame(
        history_rows(1, [(5, 90)] * 3, opponent=10) + history_rows(2, [(5, 90)] * 3, opponent=20)
    )
    fixtures = pd.DataFrame([{"element": 1, "opponent_team": 20, "was_home": True}])

    base = MinutesAdjustedPredictor().predict(flat, 4)
    adjusted = OpponentAdjustedPredictor().predict(flat, 4, fixtures)

    assert adjusted[adjusted["element"] == 1]["expected_points"].iloc[0] == pytest.approx(
        base[base["element"] == 1]["expected_points"].iloc[0]
    )


def test_the_adjustment_is_clamped_against_noisy_early_data():
    """One 30-point haul against a team must not triple every prediction."""
    extreme = pd.DataFrame(
        history_rows(1, [(60, 90)], opponent=10) + history_rows(2, [(1, 90)] * 10, opponent=20)
    )
    fixtures = pd.DataFrame([{"element": 2, "opponent_team": 10, "was_home": True}])

    predictor = OpponentAdjustedPredictor(max_adjustment=0.35)
    base = MinutesAdjustedPredictor().predict(extreme, 12)
    adjusted = predictor.predict(extreme, 12, fixtures)

    base_value = base[base["element"] == 2]["expected_points"].iloc[0]
    adjusted_value = adjusted[adjusted["element"] == 2]["expected_points"].iloc[0]

    assert adjusted_value <= base_value * 1.35 + 1e-9


def test_without_fixtures_it_degrades_to_the_unadjusted_model():
    base = MinutesAdjustedPredictor().predict(GENEROUS_HISTORY, 4)
    adjusted = OpponentAdjustedPredictor().predict(GENEROUS_HISTORY, 4, None)

    pd.testing.assert_frame_equal(base.reset_index(drop=True), adjusted.reset_index(drop=True))


def test_an_unknown_opponent_is_left_unadjusted():
    fixtures = pd.DataFrame([{"element": 1, "opponent_team": 999, "was_home": True}])

    base = MinutesAdjustedPredictor().predict(GENEROUS_HISTORY, 4)
    adjusted = OpponentAdjustedPredictor().predict(GENEROUS_HISTORY, 4, fixtures)

    assert adjusted[adjusted["element"] == 1]["expected_points"].iloc[0] == pytest.approx(
        base[base["element"] == 1]["expected_points"].iloc[0]
    )
