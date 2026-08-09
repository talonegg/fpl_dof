"""Tests for player availability.

The trap these exist to guard: ``chance_of_playing_next_round`` is null for
most players, and null means "no news", not "fit". Reading it the wrong way
round marks long-term absentees as fully available — and on live data that is
59 players, several of them expensive and popular.
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from fpl.features.availability import (
    AVAILABILITY_BANDS,
    RETURN_DEPARTED,
    RETURN_KNOWN,
    RETURN_NOT_APPLICABLE,
    RETURN_UNKNOWN,
    SELECTABLE_THRESHOLD,
    AvailabilityUnavailable,
    add_availability,
    add_return_dates,
    availability,
    availability_band,
    discount_expected_points,
    flagged,
    has_availability_data,
    parse_return_date,
    return_status,
    selectable,
    unavailability_reason,
)

PLAYERS = pd.DataFrame(
    [
        {"web_name": "Fit", "status": "a", "chance_of_playing_next_round": None},
        {"web_name": "Injured", "status": "i", "chance_of_playing_next_round": None},
        {"web_name": "Doubtful", "status": "d", "chance_of_playing_next_round": 25.0},
        {"web_name": "Likely", "status": "d", "chance_of_playing_next_round": 75.0},
        {"web_name": "Suspended", "status": "s", "chance_of_playing_next_round": 0.0},
        {"web_name": "Left", "status": "u", "chance_of_playing_next_round": None},
    ]
)


def test_a_fit_player_with_no_news_is_fully_available():
    result = availability(PLAYERS)

    assert result.iloc[0] == 1.0


def test_an_injured_player_with_no_published_chance_is_not_available():
    """The bug this guards: null means no news, not a clean bill of health."""
    result = availability(PLAYERS)

    assert result.iloc[1] == 0.0


def test_a_player_who_has_left_is_not_available():
    result = availability(PLAYERS)

    assert result.iloc[5] == 0.0


def test_a_published_percentage_is_used_when_present():
    result = availability(PLAYERS)

    assert result.iloc[2] == 0.25
    assert result.iloc[3] == 0.75


def test_an_unknown_status_is_treated_as_doubtful_not_fit():
    """A new status code must not quietly promote an absentee."""
    unknown = pd.DataFrame(
        [{"web_name": "Mystery", "status": "z", "chance_of_playing_next_round": None}]
    )

    assert availability(unknown).iloc[0] == 0.5


def test_availability_is_bounded_to_a_probability():
    odd = pd.DataFrame([{"web_name": "Odd", "status": "a", "chance_of_playing_next_round": 150.0}])

    assert availability(odd).iloc[0] == 1.0


def test_data_without_availability_fields_is_refused_not_assumed_fit():
    """Historical data has no status; "everyone fit" would be silently wrong."""
    with pytest.raises(AvailabilityUnavailable, match="live-only signal"):
        availability(pd.DataFrame([{"web_name": "Bare", "total_points": 5}]))


def test_an_archive_season_is_recognised_as_carrying_no_availability():
    archive_shaped = pd.DataFrame([{"element": 1, "gameweek": 5, "total_points": 6, "minutes": 90}])

    assert not has_availability_data(archive_shaped)


def test_a_live_frame_is_recognised_as_carrying_availability():
    assert has_availability_data(PLAYERS)


def test_availability_of_nothing_is_empty():
    assert availability(pd.DataFrame()).empty


def test_only_fit_players_are_selectable():
    result = selectable(PLAYERS)

    assert sorted(result["web_name"]) == ["Fit", "Likely"]


def test_the_threshold_matches_the_apis_own_expected_to_play_band():
    assert SELECTABLE_THRESHOLD == 0.75


def test_a_stricter_threshold_excludes_more():
    result = selectable(PLAYERS, threshold=1.0)

    assert result["web_name"].tolist() == ["Fit"]


def test_add_availability_labels_without_dropping_anyone():
    result = add_availability(PLAYERS)

    assert len(result) == len(PLAYERS)
    assert result["is_selectable"].tolist() == [True, False, False, True, False, False]


def test_add_availability_does_not_mutate_the_input():
    add_availability(PLAYERS)

    assert "availability" not in PLAYERS.columns


def test_discounting_scales_points_by_the_chance_of_playing():
    pool = PLAYERS.assign(expected_points=4.0)

    result = discount_expected_points(pool)

    assert result["expected_points"].iloc[0] == 4.0  # fit
    assert result["expected_points"].iloc[1] == 0.0  # injured
    assert result["expected_points"].iloc[2] == pytest.approx(1.0)  # 25%


def test_discounting_a_pool_without_points_is_harmless():
    result = discount_expected_points(PLAYERS)

    assert len(result) == len(PLAYERS)


def test_flagged_lists_everyone_with_news_worst_first():
    result = flagged(PLAYERS)

    assert "Fit" not in result["web_name"].tolist()
    assert result["web_name"].iloc[0] in {"Injured", "Left", "Suspended"}
    assert result["availability"].is_monotonic_increasing


def test_nobody_flagged_gives_an_empty_frame():
    fit_only = PLAYERS.head(1)

    assert flagged(fit_only).empty


def test_availability_on_the_real_snapshot(bootstrap):
    from fpl.domain.players import build_players_frame

    players = build_players_frame(bootstrap)

    result = availability(players)

    assert result.between(0, 1).all()
    assert len(result) == len(players)


def test_the_real_archive_carries_no_availability_data(archive):
    """The guarantee this whole guard exists for, checked on real archive data."""
    assert not has_availability_data(archive)


def test_applying_availability_to_the_real_archive_is_refused(archive):
    with pytest.raises(AvailabilityUnavailable):
        availability(archive)


# --- Return dates, parsed out of the news prose ---

NEWS = pd.DataFrame(
    [
        {"web_name": "Dated", "status": "i", "news": "Groin injury - Expected back 21 Aug"},
        {"web_name": "Unknown", "status": "i", "news": "Knee injury - Unknown return date"},
        {"web_name": "Banned", "status": "s", "news": "Suspended until 29 Aug"},
        {"web_name": "Chance", "status": "d", "news": "Knee injury - 75% chance of playing"},
        {"web_name": "Loaned", "status": "u", "news": "Has joined Leicester City on loan"},
        {"web_name": "Gone", "status": "u", "news": "has departed the club as a free agent."},
        {"web_name": "Fine", "status": "a", "news": ""},
    ]
)


def test_an_expected_back_date_is_extracted():
    assert parse_return_date("Groin injury - Expected back 21 Aug", date(2026, 8, 9)) == date(
        2026, 8, 21
    )


def test_a_suspension_end_date_is_extracted():
    assert parse_return_date("Suspended until 29 Aug", date(2026, 8, 9)) == date(2026, 8, 29)


def test_a_date_earlier_in_the_year_than_the_news_rolls_forward():
    """The API publishes no year; a January date in December means next year."""
    assert parse_return_date("Expected back 6 Jan", date(2026, 12, 20)) == date(2027, 1, 6)


def test_an_unknown_return_date_yields_no_date():
    assert parse_return_date("Knee injury - Unknown return date", date(2026, 8, 9)) is None


def test_a_percentage_chance_is_not_a_date():
    assert parse_return_date("Knee injury - 75% chance of playing", date(2026, 8, 9)) is None


def test_news_without_a_date_yields_nothing():
    assert parse_return_date("", date(2026, 8, 9)) is None
    assert parse_return_date(None, date(2026, 8, 9)) is None


def test_an_impossible_date_is_rejected_rather_than_raising():
    assert parse_return_date("Expected back 30 Feb", date(2026, 8, 9)) is None


def test_return_status_distinguishes_unknown_from_departed():
    """Out indefinitely and gone for good are not the same problem."""
    assert return_status("Knee injury - Unknown return date") == RETURN_UNKNOWN
    assert return_status("Has joined Leicester City on loan") == RETURN_DEPARTED
    assert return_status("Groin injury - Expected back 21 Aug") == RETURN_KNOWN


def test_a_player_with_no_news_needs_no_return():
    assert return_status("") == RETURN_NOT_APPLICABLE


def test_add_return_dates_labels_every_row():
    result = add_return_dates(NEWS)

    assert result["return_status"].tolist() == [
        RETURN_KNOWN,
        RETURN_UNKNOWN,
        RETURN_KNOWN,
        "No date given",
        RETURN_DEPARTED,
        RETURN_DEPARTED,
        RETURN_NOT_APPLICABLE,
    ]


def test_unknown_return_dates_are_identifiable_as_missing():
    result = add_return_dates(NEWS)

    assert pd.isna(result.loc[1, "return_date"])
    assert result.loc[1, "return_status"] == RETURN_UNKNOWN


def test_a_frame_without_news_still_gets_the_columns():
    result = add_return_dates(pd.DataFrame([{"web_name": "Bare", "status": "a"}]))

    assert "return_date" in result.columns
    assert result.loc[0, "return_status"] == RETURN_NOT_APPLICABLE


# --- Bands, for the sidebar filter ---


def test_bands_split_available_doubtful_and_unavailable():
    result = availability_band(PLAYERS)

    assert result.tolist() == [
        "Available",
        "Unavailable",
        "Doubtful",
        "Available",
        "Unavailable",
        "Unavailable",
    ]


def test_the_bands_are_offered_best_first():
    assert AVAILABILITY_BANDS[0] == "Available"
    assert AVAILABILITY_BANDS[-1] == "Unavailable"


def test_return_dates_on_the_real_snapshot(bootstrap):
    from fpl.domain.players import build_players_frame

    players = build_players_frame(bootstrap)

    result = add_return_dates(flagged(players))

    assert result["return_status"].notna().all()
    assert (result["return_status"] != "").all()


# --- The reason for unavailability, as distinct from its timing ---


def test_an_injury_reason_is_the_text_before_the_separator():
    assert unavailability_reason("Hamstring injury - Expected back 23 Aug") == "Hamstring injury"
    assert unavailability_reason("Knee injury - Unknown return date") == "Knee injury"


def test_a_non_injury_reason_is_kept_verbatim():
    assert (
        unavailability_reason("Lack of match fitness - 25% chance of playing")
        == "Lack of match fitness"
    )


def test_a_suspension_is_named_as_one():
    """It has no separator, so the generic split would return the whole sentence."""
    assert unavailability_reason("Suspended until 29 Aug") == "Suspension"


def test_a_loan_names_the_club():
    assert (
        unavailability_reason("Has joined Leicester City on loan for the rest of the season")
        == "On loan at Leicester City"
    )


def test_a_permanent_move_names_the_club():
    assert (
        unavailability_reason("Has joined New England Revolution permanently")
        == "Transferred to New England Revolution"
    )


def test_a_return_to_a_parent_club_names_it():
    assert unavailability_reason("has returned to Getafe CF") == "Returned to Getafe CF"


def test_a_free_agent_departure_is_named():
    assert unavailability_reason("has departed the club as a free agent.") == "Left the club"


def test_no_news_means_no_reason():
    assert unavailability_reason("") == ""
    assert unavailability_reason(None) == ""


def test_an_unrecognised_phrasing_is_shown_rather_than_swallowed():
    """A new wording should look odd in the UI, not silently blank."""
    assert unavailability_reason("gone fishing") == "Gone fishing"


def test_add_return_dates_attaches_the_reason():
    result = add_return_dates(NEWS)

    assert result["reason"].tolist() == [
        "Groin injury",
        "Knee injury",
        "Suspension",
        "Knee injury",
        "On loan at Leicester City",
        "Left the club",
        "",
    ]


def test_the_reason_distinguishes_unknown_return_from_departure():
    """Both show no return date; only the reason says which is which."""
    result = add_return_dates(NEWS)

    unknown = result[result["web_name"] == "Unknown"].iloc[0]
    gone = result[result["web_name"] == "Gone"].iloc[0]

    assert pd.isna(unknown["return_date"]) and pd.isna(gone["return_date"])
    assert unknown["reason"] != gone["reason"]


def test_every_reason_on_the_real_snapshot_is_populated(bootstrap):
    from fpl.domain.players import build_players_frame

    result = add_return_dates(flagged(build_players_frame(bootstrap)))

    assert (result["reason"].str.len() > 0).all()
