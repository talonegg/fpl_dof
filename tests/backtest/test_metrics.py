"""Tests for the evaluation metrics, on hand-computed examples."""

from __future__ import annotations

import pandas as pd
import pytest

from fpl.backtest import metrics

PERFECT = pd.DataFrame(
    [
        {"element": 1, "expected_points": 10.0, "total_points": 10, "gameweek": 1},
        {"element": 2, "expected_points": 5.0, "total_points": 5, "gameweek": 1},
        {"element": 3, "expected_points": 2.0, "total_points": 2, "gameweek": 1},
    ]
)

BACKWARDS = pd.DataFrame(
    [
        {"element": 1, "expected_points": 2.0, "total_points": 10, "gameweek": 1},
        {"element": 2, "expected_points": 5.0, "total_points": 5, "gameweek": 1},
        {"element": 3, "expected_points": 10.0, "total_points": 2, "gameweek": 1},
    ]
)


def test_perfect_predictions_have_no_error():
    assert metrics.mean_absolute_error(PERFECT) == 0.0
    assert metrics.root_mean_squared_error(PERFECT) == 0.0


def test_mean_absolute_error_is_the_average_miss():
    # misses of 8, 0 and 8 -> 16/3
    assert metrics.mean_absolute_error(BACKWARDS) == pytest.approx(16 / 3)


def test_rmse_punishes_big_misses_harder_than_mae():
    assert metrics.root_mean_squared_error(BACKWARDS) > metrics.mean_absolute_error(BACKWARDS)


def test_perfect_ordering_correlates_at_one():
    assert metrics.rank_correlation(PERFECT) == pytest.approx(1.0)


def test_reversed_ordering_correlates_at_minus_one():
    assert metrics.rank_correlation(BACKWARDS) == pytest.approx(-1.0)


def test_a_constant_prediction_has_no_rank_correlation():
    """Predicting the same value for everyone is not 'zero correlation', it is no signal."""
    flat = PERFECT.copy()
    flat["expected_points"] = 4.0

    assert pd.isna(metrics.rank_correlation(flat))


def test_rank_correlation_of_a_single_row_is_undefined():
    assert pd.isna(metrics.rank_correlation(PERFECT.head(1)))


def test_top_n_precision_is_one_when_the_right_players_are_picked():
    assert metrics.top_n_precision(PERFECT, n=2) == 1.0


def test_top_n_precision_falls_when_the_wrong_players_are_picked():
    # Predicted top 2 is {3, 2}; actual top 2 is {1, 2}. One of two right.
    assert metrics.top_n_precision(BACKWARDS, n=2) == 0.5


def test_top_n_larger_than_the_field_uses_the_whole_field():
    assert metrics.top_n_precision(PERFECT, n=99) == 1.0


def test_mean_actual_of_top_n_is_what_you_would_have_scored():
    # Picking the predicted best 2 gets the players who scored 10 and 5.
    assert metrics.mean_actual_of_top_n(PERFECT, n=2) == pytest.approx(7.5)


def test_a_bad_model_underperforms_the_field_average():
    # Predicted top 2 actually scored 2 and 5 -> 3.5, against a field mean of 17/3.
    assert metrics.mean_actual_of_top_n(BACKWARDS, n=2) == pytest.approx(3.5)
    assert metrics.mean_actual_overall(BACKWARDS) == pytest.approx(17 / 3)
    assert metrics.mean_actual_of_top_n(BACKWARDS, n=2) < metrics.mean_actual_overall(BACKWARDS)


def test_metrics_of_an_empty_frame_are_nan_not_zero():
    empty = PERFECT.head(0)

    assert pd.isna(metrics.mean_absolute_error(empty))
    assert pd.isna(metrics.top_n_precision(empty))
    assert pd.isna(metrics.mean_actual_of_top_n(empty))


def test_evaluate_reports_every_metric():
    result = metrics.evaluate(PERFECT, n=2)

    assert set(result) == {
        "rows",
        "mae",
        "rmse",
        "rank_correlation",
        "top_2_precision",
        "top_2_mean_actual",
        "field_mean_actual",
    }


def test_metrics_are_computed_per_gameweek_not_pooled():
    """Pooling would let a good week paper over a bad one."""
    two_weeks = pd.concat([PERFECT, BACKWARDS.assign(gameweek=2)])

    per_gameweek = metrics.evaluate_by_gameweek(two_weeks, n=2)

    assert list(per_gameweek.index) == [1, 2]
    assert per_gameweek.loc[1, "rank_correlation"] == pytest.approx(1.0)
    assert per_gameweek.loc[2, "rank_correlation"] == pytest.approx(-1.0)


def test_summarise_averages_the_per_gameweek_metrics():
    two_weeks = pd.concat([PERFECT, BACKWARDS.assign(gameweek=2)])

    summary = metrics.summarise(two_weeks, n=2)

    assert summary["gameweeks"] == 2
    assert summary["rank_correlation"] == pytest.approx(0.0)


def test_summarise_of_nothing_is_empty():
    assert metrics.summarise(PERFECT.head(0)) == {}
