"""Tests for horizon scoring.

The leakage risk is larger here than in the one-week harness: a prediction is
scored against a *window* of future gameweeks, so a model that saw any of them
would look excellent. The spy and canary are repeated for that reason.
"""

from __future__ import annotations

import pandas as pd
import pytest

from fpl.backtest.horizon import (
    compare_horizons,
    evaluate_horizon,
    horizon_actuals,
    replay_horizon,
    squad_turnover,
)
from fpl.models.base import empty_predictions
from fpl.models.naive import NaiveFormPredictor, SeasonMeanPredictor

SEASON = pd.DataFrame(
    [
        {"element": element, "gameweek": gameweek, "total_points": element, "minutes": 90}
        for element in (1, 2, 3)
        for gameweek in range(1, 13)
    ]
)


class SpyPredictor:
    """Records the highest gameweek it was shown."""

    name = "Spy"

    def __init__(self):
        self.seen = []

    def predict(self, history, gameweek, fixtures=None):
        self.seen.append((gameweek, history["gameweek"].max()))
        return empty_predictions()


def test_the_window_sums_points_across_the_horizon():
    actual = horizon_actuals(SEASON, start=3, horizon=4)

    # Element 1 scores 1 a week for four weeks.
    assert actual[actual["element"] == 1]["total_points"].iloc[0] == 4


def test_the_window_reports_how_many_gameweeks_were_played():
    actual = horizon_actuals(SEASON, start=3, horizon=4)

    assert actual["gameweeks_played"].iloc[0] == 4


def test_a_player_absent_from_the_window_is_dropped_not_scored_zero():
    """The model answered a different question; scoring it zero is unfair."""
    season = pd.concat(
        [SEASON, pd.DataFrame([{"element": 9, "gameweek": 1, "total_points": 5, "minutes": 90}])]
    )

    actual = horizon_actuals(season, start=5, horizon=3)

    assert 9 not in set(actual["element"])


def test_an_empty_window_is_empty():
    assert horizon_actuals(SEASON, start=99, horizon=3).empty


def test_the_predictor_never_sees_the_window_it_is_scored_on():
    """The leakage rule, which matters more here: the whole window is future."""
    spy = SpyPredictor()

    replay_horizon(SEASON, spy, horizon=4, first_gameweek=3)

    for target, highest_seen in spy.seen:
        assert highest_seen < target


def test_corrupting_the_window_does_not_change_the_prediction():
    """The canary, applied to the horizon replay."""
    predictor = NaiveFormPredictor(window=3)
    honest = replay_horizon(SEASON, predictor, horizon=3, first_gameweek=3)

    corrupted_season = SEASON.copy()
    corrupted_season.loc[corrupted_season["gameweek"] >= 6, "total_points"] = 9999
    corrupted = replay_horizon(corrupted_season, predictor, horizon=3, first_gameweek=3)

    early = honest.predictions[honest.predictions["gameweek"] < 6]
    corrupted_early = corrupted.predictions[corrupted.predictions["gameweek"] < 6]

    pd.testing.assert_series_equal(
        early["expected_points"].reset_index(drop=True),
        corrupted_early["expected_points"].reset_index(drop=True),
    )


def test_windows_that_run_off_the_end_of_the_season_are_skipped():
    """Judging a six-week horizon on three weeks would look artificially volatile."""
    result = replay_horizon(SEASON, SeasonMeanPredictor(), horizon=4, first_gameweek=3)

    # Season ends at 12, so the last valid start is 9.
    assert max(result.gameweeks) == 12 - 4 + 1


def test_a_horizon_of_one_reproduces_the_single_gameweek_replay():
    """The control: any difference at longer horizons is the horizon, not the metric."""
    from fpl.backtest.harness import replay

    single = replay(SEASON, SeasonMeanPredictor(), first_gameweek=3, last_gameweek=12)
    horizon = replay_horizon(SEASON, SeasonMeanPredictor(), horizon=1, first_gameweek=3)

    assert horizon.gameweeks == single.gameweeks
    assert len(horizon.predictions) == len(single.predictions)


def test_the_result_records_its_horizon():
    result = replay_horizon(SEASON, SeasonMeanPredictor(), horizon=5, first_gameweek=3)

    assert result.horizon == 5


def test_an_empty_season_produces_nothing():
    empty = pd.DataFrame(columns=["element", "gameweek", "total_points"])

    assert replay_horizon(empty, SeasonMeanPredictor()).predictions.empty


def test_a_stable_model_has_no_turnover():
    """The season mean barely moves, so its top fifteen should barely move."""
    result = replay_horizon(SEASON, SeasonMeanPredictor(), horizon=3, first_gameweek=4)

    assert squad_turnover(result, n=2) == 0.0


def test_turnover_is_reported_as_a_share_of_the_squad():
    result = replay_horizon(SEASON, NaiveFormPredictor(window=1), horizon=3, first_gameweek=4)

    turnover = squad_turnover(result, n=2)
    assert 0.0 <= turnover <= 1.0


def test_turnover_needs_at_least_two_starts():
    result = replay_horizon(SEASON, SeasonMeanPredictor(), horizon=9, first_gameweek=4)

    assert pd.isna(squad_turnover(result, n=2))


def test_scores_are_reported_per_gameweek_not_per_window():
    """So the number is comparable with the existing one-week figure."""
    one = evaluate_horizon(
        replay_horizon(SEASON, SeasonMeanPredictor(), horizon=1, first_gameweek=4), n=2
    )
    six = evaluate_horizon(
        replay_horizon(SEASON, SeasonMeanPredictor(), horizon=6, first_gameweek=4), n=2
    )

    # Same players, same weekly scoring: the per-gameweek rate should match.
    assert one["top_2_mean_actual"] == pytest.approx(six["top_2_mean_actual"])


def test_evaluation_reports_the_horizon_and_the_number_of_starts():
    summary = evaluate_horizon(
        replay_horizon(SEASON, SeasonMeanPredictor(), horizon=3, first_gameweek=4), n=2
    )

    assert summary["horizon"] == 3
    assert summary["starts"] > 0


def test_evaluating_nothing_is_empty():
    empty = pd.DataFrame(columns=["element", "gameweek", "total_points"])

    assert evaluate_horizon(replay_horizon(empty, SeasonMeanPredictor())) == {}


def test_comparison_covers_every_model_and_horizon():
    table = compare_horizons(
        SEASON,
        [SeasonMeanPredictor(), NaiveFormPredictor(window=3)],
        horizons=(1, 3),
        first_gameweek=4,
        n=2,
    )

    assert len(table) == 4
    assert set(table.index.get_level_values("horizon")) == {1, 3}
