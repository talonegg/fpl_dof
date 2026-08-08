"""Tests for the noise check.

This is the guard against the results table becoming a record of which model
got the luckier season, so it needs to be right in both directions: it must
detect a real difference, and it must refuse to endorse a tiny one.
"""

from __future__ import annotations

import pandas as pd
import pytest

from fpl.backtest.significance import (
    SIGNIFICANCE_THRESHOLD,
    Comparison,
    compare_to_benchmark,
    comparison_table,
)
from fpl.models.base import empty_predictions
from fpl.models.naive import NaiveFormPredictor, SeasonMeanPredictor

SEASON = pd.DataFrame(
    [
        {
            "element": element,
            "gameweek": gameweek,
            "total_points": element * gameweek % 7,
            "minutes": 90,
        }
        for element in range(1, 25)
        for gameweek in range(1, 21)
    ]
)


class ConstantOffsetPredictor:
    """Predicts a fixed ranking, so its per-gameweek score is stable."""

    def __init__(self, name, offset):
        self.name = name
        self.offset = offset

    def predict(self, history, gameweek, fixtures=None):
        if history.empty:
            return empty_predictions()
        elements = history["element"].drop_duplicates().sort_values()
        return pd.DataFrame({"element": elements, "expected_points": elements * self.offset})


def test_a_model_compared_with_itself_shows_no_difference():
    predictor = NaiveFormPredictor(window=5)

    result = compare_to_benchmark(SEASON, predictor, predictor)

    assert result.mean_difference == 0.0
    assert not result.is_distinguishable
    assert result.verdict == "indistinguishable from the benchmark"


def test_a_reversed_ranking_is_detected_as_worse():
    # A season where the ranking genuinely means something: higher element,
    # more points. Reversing the order must therefore pick worse players.
    ranked_season = pd.DataFrame(
        [
            {"element": element, "gameweek": gameweek, "total_points": element, "minutes": 90}
            for element in range(1, 41)
            for gameweek in range(1, 21)
        ]
    )
    good = ConstantOffsetPredictor("Good", 1)
    bad = ConstantOffsetPredictor("Bad", -1)

    result = compare_to_benchmark(ranked_season, bad, good)

    assert result.mean_difference < 0
    assert result.verdict == "worse than the benchmark"


def test_the_comparison_is_paired_gameweek_by_gameweek():
    result = compare_to_benchmark(SEASON, NaiveFormPredictor(), SeasonMeanPredictor())

    assert result.gameweeks > 10
    assert 0 <= result.wins <= result.gameweeks


def test_a_difference_within_noise_is_not_endorsed():
    small = Comparison(
        "A", "B", "metric", gameweeks=33, wins=19, mean_difference=0.172, t_statistic=0.92
    )

    assert not small.is_distinguishable
    assert small.verdict == "indistinguishable from the benchmark"


def test_a_difference_beyond_the_threshold_is_endorsed():
    large = Comparison(
        "A", "B", "metric", gameweeks=33, wins=28, mean_difference=1.4, t_statistic=3.5
    )

    assert large.is_distinguishable
    assert large.verdict == "better than the benchmark"


def test_a_large_negative_difference_is_called_worse():
    poor = Comparison(
        "A", "B", "metric", gameweeks=33, wins=5, mean_difference=-1.4, t_statistic=-3.5
    )

    assert poor.verdict == "worse than the benchmark"


@pytest.mark.parametrize("t", [-SIGNIFICANCE_THRESHOLD, SIGNIFICANCE_THRESHOLD])
def test_the_threshold_itself_counts_as_distinguishable(t):
    borderline = Comparison("A", "B", "m", 33, 20, 0.1, t)

    assert borderline.is_distinguishable


def test_comparison_table_excludes_the_benchmark_from_its_own_table():
    benchmark = SeasonMeanPredictor()

    table = comparison_table(SEASON, [benchmark, NaiveFormPredictor()], benchmark)

    assert benchmark.name not in table.index


def test_comparison_table_reports_wins_as_a_fraction():
    benchmark = SeasonMeanPredictor()

    table = comparison_table(SEASON, [NaiveFormPredictor()], benchmark)

    assert "/" in table.iloc[0]["wins"]


def test_comparison_table_of_nothing_is_empty():
    assert comparison_table(SEASON, [], SeasonMeanPredictor()).empty


def test_an_empty_season_gives_no_gameweeks_rather_than_raising():
    empty = pd.DataFrame(columns=["element", "gameweek", "total_points"])

    result = compare_to_benchmark(empty, NaiveFormPredictor(), SeasonMeanPredictor())

    assert result.gameweeks == 0


def test_a_perfectly_consistent_gap_is_conclusive_not_unmeasurable():
    """Zero variance means maximum confidence, not missing data."""
    consistent = Comparison(
        "A", "B", "m", gameweeks=33, wins=33, mean_difference=1.0, t_statistic=float("inf")
    )

    assert consistent.is_distinguishable
    assert consistent.verdict == "better than the benchmark"
