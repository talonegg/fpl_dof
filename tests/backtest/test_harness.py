"""Tests for the replay harness."""

from __future__ import annotations

import pandas as pd
import pytest

from fpl.backtest.harness import compare, prepare_season, replay
from fpl.models.naive import NaiveFormPredictor, SeasonMeanPredictor

SEASON = pd.DataFrame(
    [
        {"element": element, "gameweek": gameweek, "total_points": element}
        for element in (1, 2)
        for gameweek in range(1, 9)
    ]
)


def test_prepare_season_rejects_data_missing_required_columns():
    with pytest.raises(ValueError, match="missing required columns"):
        prepare_season(pd.DataFrame([{"element": 1}]))


def test_prepare_season_collapses_a_double_gameweek():
    doubled = pd.DataFrame(
        [
            {"element": 1, "gameweek": 1, "total_points": 5, "minutes": 90},
            {"element": 1, "gameweek": 1, "total_points": 7, "minutes": 80},
        ]
    )

    result = prepare_season(doubled)

    assert len(result) == 1
    assert result.loc[0, "total_points"] == 12
    assert result.loc[0, "minutes"] == 170


def test_replay_covers_every_gameweek_from_the_start_point():
    result = replay(SEASON, NaiveFormPredictor(), first_gameweek=4)

    assert result.gameweeks == [4, 5, 6, 7, 8]


def test_replay_respects_an_explicit_last_gameweek():
    result = replay(SEASON, NaiveFormPredictor(), first_gameweek=4, last_gameweek=6)

    assert result.gameweeks == [4, 5, 6]


def test_replay_records_which_model_produced_it():
    result = replay(SEASON, NaiveFormPredictor(window=3), first_gameweek=4)

    assert result.model == "NaiveForm(3)"


def test_predictions_are_joined_to_what_actually_happened():
    result = replay(SEASON, NaiveFormPredictor(), first_gameweek=4)

    assert {"element", "expected_points", "total_points", "gameweek"} <= set(
        result.predictions.columns
    )


def test_a_player_who_did_not_feature_is_dropped_rather_than_scored():
    """No actual means no comparison; keeping it would reward predicting for absentees."""
    season = pd.DataFrame(
        [
            {"element": 1, "gameweek": 1, "total_points": 5},
            {"element": 2, "gameweek": 1, "total_points": 5},
            # Only element 1 plays in gameweek 2.
            {"element": 1, "gameweek": 2, "total_points": 6},
        ]
    )

    result = replay(season, SeasonMeanPredictor(), first_gameweek=2)

    assert result.predictions["element"].tolist() == [1]


def test_an_empty_season_produces_an_empty_result():
    result = replay(
        pd.DataFrame(columns=["element", "gameweek", "total_points"]), NaiveFormPredictor()
    )

    assert result.predictions.empty
    assert result.gameweeks == []


def test_a_gameweek_with_no_prior_history_is_skipped():
    result = replay(SEASON, NaiveFormPredictor(), first_gameweek=1)

    assert 1 not in result.gameweeks


def test_compare_returns_a_row_per_model():
    table = compare(SEASON, [NaiveFormPredictor(), SeasonMeanPredictor()], first_gameweek=4)

    assert set(table.index) == {"NaiveForm(5)", "SeasonMean"}


def test_compare_of_no_models_is_empty():
    assert compare(SEASON, [], first_gameweek=4).empty
