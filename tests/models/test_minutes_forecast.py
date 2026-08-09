"""Tests for minutes forecasting.

Minutes are the one input with complete data, so the forecasters are checked
on constructed cases where the right answer is arithmetic.
"""

from __future__ import annotations

import pandas as pd
import pytest

from fpl.models.minutes_forecast import (
    FORECAST_COLUMNS,
    MinutesForecaster,
    RecentMinutes,
    SeasonMinutes,
    StartWeightedMinutes,
    forecasters,
)


def history(minutes_by_gameweek, element=1):
    return pd.DataFrame(
        [
            {"element": element, "gameweek": gameweek, "minutes": minutes}
            for gameweek, minutes in enumerate(minutes_by_gameweek, start=1)
        ]
    )


@pytest.mark.parametrize("forecaster", forecasters())
def test_every_forecaster_satisfies_the_protocol(forecaster):
    assert isinstance(forecaster, MinutesForecaster)


@pytest.mark.parametrize("forecaster", forecasters())
def test_every_forecaster_returns_the_expected_columns(forecaster):
    result = forecaster.forecast(history([90, 90, 90]), gameweek=4)

    assert list(result.columns) == FORECAST_COLUMNS


@pytest.mark.parametrize("forecaster", forecasters())
def test_every_forecaster_handles_an_empty_history(forecaster):
    assert forecaster.forecast(pd.DataFrame(), gameweek=1).empty


@pytest.mark.parametrize("forecaster", forecasters())
def test_an_ever_present_starter_is_forecast_to_start(forecaster):
    result = forecaster.forecast(history([90, 90, 90, 90, 90]), gameweek=6)

    assert result["expected_minutes"].iloc[0] == pytest.approx(90.0)
    assert result["start_probability"].iloc[0] == pytest.approx(1.0)


@pytest.mark.parametrize("forecaster", forecasters())
def test_a_player_who_never_plays_is_forecast_not_to(forecaster):
    result = forecaster.forecast(history([0, 0, 0, 0, 0]), gameweek=6)

    assert result["expected_minutes"].iloc[0] == pytest.approx(0.0)
    assert result["start_probability"].iloc[0] == pytest.approx(0.0)


@pytest.mark.parametrize("forecaster", forecasters())
def test_probabilities_stay_within_bounds(forecaster):
    result = forecaster.forecast(history([90, 0, 45, 90, 20]), gameweek=6)

    assert 0.0 <= result["start_probability"].iloc[0] <= 1.0
    assert 0.0 <= result["expected_minutes"].iloc[0] <= 90.0


def test_recent_minutes_averages_only_its_window():
    result = RecentMinutes(window=2).forecast(history([0, 0, 90, 90]), gameweek=5)

    assert result["expected_minutes"].iloc[0] == pytest.approx(90.0)


def test_season_minutes_uses_everything():
    result = SeasonMinutes().forecast(history([0, 0, 90, 90]), gameweek=5)

    assert result["expected_minutes"].iloc[0] == pytest.approx(45.0)


def test_the_start_weighted_forecaster_leans_on_recent_evidence():
    """A player newly promoted to the starting eleven."""
    promoted = history([0, 0, 0, 90, 90])

    recent = RecentMinutes(window=5).forecast(promoted, gameweek=6)
    weighted = StartWeightedMinutes(window=5).forecast(promoted, gameweek=6)

    assert weighted["expected_minutes"].iloc[0] > recent["expected_minutes"].iloc[0]


def test_a_rotation_pattern_is_not_reported_as_a_half_start():
    """Averaging 90 and 0 gives 45, a number the player never actually plays."""
    rotating = history([90, 0, 90, 0, 90, 0])

    result = StartWeightedMinutes(window=6).forecast(rotating, gameweek=7)

    assert 0.0 < result["start_probability"].iloc[0] < 1.0


def test_a_substitute_is_distinguished_from_a_substituted_starter():
    """Both average about 45 minutes and are not the same player."""
    starter_subbed = history([60, 60, 60, 60])
    late_substitute = history([0, 90, 0, 90])

    starter = StartWeightedMinutes(window=4).forecast(starter_subbed, gameweek=5)
    substitute = StartWeightedMinutes(window=4).forecast(late_substitute, gameweek=5)

    assert starter["start_probability"].iloc[0] > substitute["start_probability"].iloc[0]


def test_forecasts_cover_every_player():
    two = pd.concat([history([90, 90]), history([0, 45], element=2)])

    result = StartWeightedMinutes(window=2).forecast(two, gameweek=3)

    assert set(result["element"]) == {1, 2}
