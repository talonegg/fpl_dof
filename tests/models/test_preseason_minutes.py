"""Tests for the pre-season minutes forecaster."""

from __future__ import annotations

import pandas as pd

from fpl.models.minutes_forecast import PreseasonMinutes


def careers(rows):
    return pd.DataFrame(rows)


def player(element, minutes, appearances, seasons=1):
    return {
        "element": element,
        "career_minutes": minutes,
        "career_appearances": appearances,
        "seasons_seen": seasons,
    }


def test_an_ever_present_is_forecast_near_a_full_match():
    forecast = PreseasonMinutes().forecast(careers([player(1, 3420, 38)]))

    assert forecast["expected_minutes"].iloc[0] > 80


def test_a_regular_starter_outranks_a_substitute():
    forecast = PreseasonMinutes().forecast(careers([player(1, 3000, 34), player(2, 400, 30)]))

    assert (
        forecast.set_index("element")["expected_minutes"].loc[1]
        > (forecast.set_index("element")["expected_minutes"].loc[2])
    )


def test_a_thin_history_is_regressed_towards_the_population():
    """The failure that cost the naive version everything: three minutes, full trust."""
    rows = careers([player(1, 3420, 38), player(2, 3420, 38), player(3, 90, 1)])

    forecast = PreseasonMinutes().forecast(rows).set_index("element")

    # The one-appearance player played a full match, but must not be forecast
    # as a certain starter on that basis.
    assert forecast["start_probability"].loc[3] < forecast["start_probability"].loc[1]


def test_minutes_never_exceed_a_match():
    forecast = PreseasonMinutes().forecast(careers([player(1, 99_999, 38)]))

    assert forecast["expected_minutes"].iloc[0] <= 90


def test_start_probability_stays_a_probability():
    forecast = PreseasonMinutes().forecast(
        careers([player(1, 3420, 38), player(2, 0, 0), player(3, 500, 20)])
    )

    assert forecast["start_probability"].between(0, 1).all()


def test_a_player_who_never_played_forecasts_low():
    forecast = PreseasonMinutes().forecast(
        careers([player(1, 3420, 38), player(2, 3420, 38), player(3, 0, 0)])
    )

    assert forecast.set_index("element")["expected_minutes"].loc[3] < 90


def test_missing_career_columns_give_an_empty_forecast():
    """Better to return nothing than to invent minutes from absent data."""
    assert PreseasonMinutes().forecast(pd.DataFrame([{"element": 1}])).empty


def test_an_empty_history_gives_an_empty_forecast():
    assert PreseasonMinutes().forecast(pd.DataFrame()).empty
