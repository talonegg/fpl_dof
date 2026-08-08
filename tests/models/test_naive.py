"""Tests for the baseline predictors."""

from __future__ import annotations

import pandas as pd
import pytest

from fpl.models.base import PREDICTION_COLUMNS, Predictor
from fpl.models.naive import NaiveFormPredictor, SeasonMeanPredictor, ZeroPredictor

HISTORY = pd.DataFrame(
    [
        {"element": 1, "gameweek": 1, "total_points": 2},
        {"element": 1, "gameweek": 2, "total_points": 2},
        {"element": 1, "gameweek": 3, "total_points": 8},
        {"element": 1, "gameweek": 4, "total_points": 8},
        {"element": 2, "gameweek": 1, "total_points": 6},
        {"element": 2, "gameweek": 2, "total_points": 6},
    ]
)


@pytest.mark.parametrize(
    "predictor", [NaiveFormPredictor(), SeasonMeanPredictor(), ZeroPredictor()]
)
def test_every_baseline_satisfies_the_predictor_protocol(predictor):
    assert isinstance(predictor, Predictor)


@pytest.mark.parametrize(
    "predictor", [NaiveFormPredictor(), SeasonMeanPredictor(), ZeroPredictor()]
)
def test_every_baseline_returns_the_expected_columns(predictor):
    result = predictor.predict(HISTORY, gameweek=5)

    assert set(PREDICTION_COLUMNS) <= set(result.columns)


@pytest.mark.parametrize(
    "predictor", [NaiveFormPredictor(), SeasonMeanPredictor(), ZeroPredictor()]
)
def test_every_baseline_handles_an_empty_history(predictor):
    result = predictor.predict(HISTORY.head(0), gameweek=1)

    assert result.empty
    assert list(result.columns) == PREDICTION_COLUMNS


def test_naive_form_averages_only_the_recent_window():
    # Element 1's last two gameweeks are 8 and 8.
    result = NaiveFormPredictor(window=2).predict(HISTORY, gameweek=5)

    assert result[result["element"] == 1]["expected_points"].iloc[0] == 8.0


def test_a_wider_window_reaches_further_back():
    # All four of element 1's gameweeks: 2, 2, 8, 8 -> 5.0
    result = NaiveFormPredictor(window=4).predict(HISTORY, gameweek=5)

    assert result[result["element"] == 1]["expected_points"].iloc[0] == 5.0


def test_a_window_longer_than_the_history_uses_what_there_is():
    result = NaiveFormPredictor(window=99).predict(HISTORY, gameweek=5)

    assert result[result["element"] == 2]["expected_points"].iloc[0] == 6.0


def test_season_mean_ignores_recency():
    result = SeasonMeanPredictor().predict(HISTORY, gameweek=5)

    assert result[result["element"] == 1]["expected_points"].iloc[0] == 5.0


def test_zero_predictor_predicts_zero_for_every_known_player():
    result = ZeroPredictor().predict(HISTORY, gameweek=5)

    assert set(result["element"]) == {1, 2}
    assert (result["expected_points"] == 0.0).all()


def test_one_row_per_player():
    result = NaiveFormPredictor().predict(HISTORY, gameweek=5)

    assert not result["element"].duplicated().any()


def test_the_window_appears_in_the_model_name():
    assert NaiveFormPredictor(window=3).name == "NaiveForm(3)"
