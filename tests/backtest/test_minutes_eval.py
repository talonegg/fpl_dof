"""Tests for minutes evaluation.

Minutes are a calibration problem rather than a ranking one, so the metrics
are error and calibration and these check them against hand-worked cases.
"""

from __future__ import annotations

import pandas as pd
import pytest

from fpl.backtest.minutes import (
    compare_forecasters,
    evaluate_minutes,
    minutes_mae,
    replay_minutes,
    start_accuracy,
    start_brier,
    zero_recall,
)
from fpl.models.minutes_forecast import RecentMinutes, SeasonMinutes

SEASON = pd.DataFrame(
    [
        {
            "element": element,
            "gameweek": gameweek,
            "minutes": 90 if element == 1 else 0,
            "total_points": 2,
        }
        for element in (1, 2)
        for gameweek in range(1, 11)
    ]
)


def test_a_perfect_forecast_has_no_error():
    joined = pd.DataFrame([{"expected_minutes": 90, "start_probability": 1.0, "minutes": 90}])

    assert minutes_mae(joined) == 0.0
    assert start_brier(joined) == 0.0


def test_mae_is_the_average_miss_in_minutes():
    joined = pd.DataFrame(
        [
            {"expected_minutes": 90, "start_probability": 1.0, "minutes": 60},
            {"expected_minutes": 0, "start_probability": 0.0, "minutes": 20},
        ]
    )

    assert minutes_mae(joined) == 25.0


def test_a_confident_wrong_forecast_scores_worse_than_a_hedged_one():
    """Why Brier rather than accuracy: calibration is the point."""
    confident = pd.DataFrame([{"expected_minutes": 90, "start_probability": 1.0, "minutes": 0}])
    hedged = pd.DataFrame([{"expected_minutes": 45, "start_probability": 0.5, "minutes": 0}])

    assert start_brier(confident) > start_brier(hedged)


def test_a_coin_toss_scores_a_quarter():
    joined = pd.DataFrame(
        [
            {"expected_minutes": 45, "start_probability": 0.5, "minutes": 90},
            {"expected_minutes": 45, "start_probability": 0.5, "minutes": 0},
        ]
    )

    assert start_brier(joined) == pytest.approx(0.25)


def test_accuracy_asks_only_which_side_of_a_half():
    joined = pd.DataFrame(
        [
            {"expected_minutes": 90, "start_probability": 0.9, "minutes": 90},
            {"expected_minutes": 10, "start_probability": 0.1, "minutes": 0},
        ]
    )

    assert start_accuracy(joined) == 1.0


def test_zero_recall_catches_the_most_expensive_error():
    """Fielding someone who does not play is the worst thing a model can do."""
    joined = pd.DataFrame(
        [
            {"expected_minutes": 5, "start_probability": 0.0, "minutes": 0},
            {"expected_minutes": 85, "start_probability": 0.9, "minutes": 0},
        ]
    )

    assert zero_recall(joined) == 0.5


def test_zero_recall_is_undefined_when_everyone_played():
    joined = pd.DataFrame([{"expected_minutes": 90, "start_probability": 1.0, "minutes": 90}])

    assert pd.isna(zero_recall(joined))


def test_metrics_of_nothing_are_nan_not_zero():
    empty = pd.DataFrame(columns=["expected_minutes", "start_probability", "minutes"])

    assert pd.isna(minutes_mae(empty))
    assert pd.isna(start_brier(empty))


def test_replay_scores_every_gameweek_from_the_start_point():
    result = replay_minutes(SEASON, SeasonMinutes(), first_gameweek=4)

    assert result.gameweeks == [4, 5, 6, 7, 8, 9, 10]


def test_replay_never_shows_the_forecaster_the_target_gameweek():
    """The same point-in-time rule as the points harness."""

    class Spy:
        name = "Spy"

        def __init__(self):
            self.seen = []

        def forecast(self, history, gameweek):
            self.seen.append((gameweek, history["gameweek"].max()))
            return pd.DataFrame(columns=["element", "expected_minutes", "start_probability"])

    spy = Spy()
    replay_minutes(SEASON, spy, first_gameweek=3)

    for target, highest in spy.seen:
        assert highest < target


def test_evaluation_reports_every_metric():
    result = replay_minutes(SEASON, SeasonMinutes(), first_gameweek=4)

    summary = evaluate_minutes(result.forecasts)

    assert set(summary) == {
        "rows",
        "minutes_mae",
        "start_brier",
        "start_accuracy",
        "zero_recall",
    }


def test_comparison_ranks_by_calibration():
    table = compare_forecasters(SEASON, [SeasonMinutes(), RecentMinutes(3)], first_gameweek=4)

    assert len(table) == 2
    assert table["start_brier"].is_monotonic_increasing


def test_an_empty_season_produces_nothing():
    empty = pd.DataFrame(columns=["element", "gameweek", "total_points", "minutes"])

    assert replay_minutes(empty, SeasonMinutes()).forecasts.empty
