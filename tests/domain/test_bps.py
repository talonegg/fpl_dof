"""Tests for the bonus points system table and reconstruction."""

from __future__ import annotations

import pandas as pd
import pytest

from fpl.domain.bps import (
    BPS_ACTIONS,
    action_table,
    observable_actions,
    penalty_correction,
    reconstruct,
    reconstruction_gap,
    unobservable_actions,
    unobservable_weight,
)


def appearance(position="MID", **stats):
    row = {"position": position, "minutes": 90}
    row.update(stats)
    return row


def test_the_table_records_both_what_we_can_and_cannot_see():
    """The shape of the gap is the useful part, so unobservable actions stay in."""
    assert len(observable_actions()) > 0
    assert len(unobservable_actions()) > 0
    assert len(BPS_ACTIONS) == len(observable_actions()) + len(unobservable_actions())


def test_key_passes_and_big_chances_are_known_to_be_unobservable():
    names = {action.name for action in unobservable_actions()}

    assert "Key pass" in names
    assert "Big chance created" in names


def test_a_full_appearance_scores_more_than_a_cameo():
    full = pd.DataFrame([appearance(minutes=90)])
    cameo = pd.DataFrame([appearance(minutes=30)])

    assert reconstruct(full).iloc[0] == 6
    assert reconstruct(cameo).iloc[0] == 3


def test_an_unused_substitute_scores_nothing():
    assert reconstruct(pd.DataFrame([appearance(minutes=0)])).iloc[0] == 0


@pytest.mark.parametrize("position,value", [("GK", 12), ("DEF", 12), ("MID", 18), ("FWD", 24)])
def test_goals_are_worth_more_from_further_forward(position, value):
    frame = pd.DataFrame([appearance(position, goals_scored=1)])

    assert reconstruct(frame).iloc[0] == 6 + value


def test_an_assist_is_worth_nine():
    frame = pd.DataFrame([appearance(assists=1)])

    assert reconstruct(frame).iloc[0] == 6 + 9


def test_a_clean_sheet_counts_only_for_defenders_and_keepers():
    defender = pd.DataFrame([appearance("DEF", clean_sheets=1)])
    midfielder = pd.DataFrame([appearance("MID", clean_sheets=1)])

    assert reconstruct(defender).iloc[0] == 6 + 12
    assert reconstruct(midfielder).iloc[0] == 6


def test_clearances_score_one_point_per_two():
    """Per completed pair, so three actions are still only one point."""
    three = pd.DataFrame([appearance(clearances_blocks_interceptions=3)])
    four = pd.DataFrame([appearance(clearances_blocks_interceptions=4)])

    assert reconstruct(three).iloc[0] == 6 + 1
    assert reconstruct(four).iloc[0] == 6 + 2


def test_recoveries_score_one_point_per_three():
    frame = pd.DataFrame([appearance(recoveries=7)])

    assert reconstruct(frame).iloc[0] == 6 + 2


def test_a_tackle_is_worth_two():
    frame = pd.DataFrame([appearance(tackles=3)])

    assert reconstruct(frame).iloc[0] == 6 + 6


def test_cards_and_own_goals_subtract():
    frame = pd.DataFrame([appearance(yellow_cards=1, own_goals=1)])

    assert reconstruct(frame).iloc[0] == 6 - 3 - 6


def test_goals_conceded_only_hurt_defenders_and_keepers():
    defender = pd.DataFrame([appearance("DEF", goals_conceded=2)])
    forward = pd.DataFrame([appearance("FWD", goals_conceded=2)])

    assert reconstruct(defender).iloc[0] == 6 - 8
    assert reconstruct(forward).iloc[0] == 6


def test_missing_columns_are_treated_as_zero_not_an_error():
    frame = pd.DataFrame([{"position": "MID", "minutes": 90}])

    assert reconstruct(frame).iloc[0] == 6


def test_the_gap_is_published_minus_reconstructed():
    frame = pd.DataFrame([appearance(bps=20)])

    assert reconstruction_gap(frame).iloc[0] == 20 - 6


def test_the_gap_is_empty_without_a_published_figure():
    assert reconstruction_gap(pd.DataFrame([appearance()])).empty


def test_empty_input_is_safe():
    assert reconstruct(pd.DataFrame()).empty


@pytest.mark.backtest
def test_the_reconstruction_tracks_published_bps_on_a_real_season():
    """Unfitted: the official coefficients applied to the inputs we have."""
    from fpl.sources.archive import fetch_season_gameweeks

    played = fetch_season_gameweeks("2025-26")
    played = played[played["minutes"] > 0]

    reconstructed = reconstruct(played)
    published = pd.to_numeric(played["bps"])

    assert reconstructed.corr(published) > 0.85
    # Always an underestimate in aggregate: the unobservable actions are
    # overwhelmingly positive ones.
    assert reconstructed.sum() < published.sum()
    assert reconstructed.sum() / published.sum() > 0.80


# --- The table as queryable data ---


def test_the_action_table_has_a_row_per_action():
    table = action_table()

    assert len(table) == len(BPS_ACTIONS)
    assert set(table.columns) == {"action", "bps", "observable", "note"}


def test_the_table_can_be_queried_for_what_we_cannot_see():
    """The question this exists to answer."""
    table = action_table()
    invisible = table[~table["observable"]]

    assert len(invisible) == 22
    assert "Key pass" in invisible["action"].tolist()


def test_the_biggest_invisible_action_is_flagged_with_a_reason():
    table = action_table()
    penalty_goal = table[table["action"].str.contains("penalty, any position")].iloc[0]

    assert not penalty_goal["observable"]
    assert "split" in penalty_goal["note"]


def test_unobservable_weight_is_the_absolute_bps_we_cannot_see():
    assert unobservable_weight() > 0
    assert unobservable_weight() == sum(abs(a.value) for a in unobservable_actions())


# --- Penalty goals score 12 whatever the position ---


def test_a_midfielders_penalty_is_credited_six_bps_less_than_an_open_play_goal():
    frame = pd.DataFrame([appearance("MID", goals_scored=1)])
    penalty = pd.Series([1.0])

    open_play = reconstruct(frame).iloc[0]
    from_penalty = reconstruct(frame, penalty_goals=penalty).iloc[0]

    assert open_play - from_penalty == 6  # 18 -> 12


def test_a_forwards_penalty_is_credited_twelve_bps_less():
    frame = pd.DataFrame([appearance("FWD", goals_scored=1)])

    difference = (
        reconstruct(frame).iloc[0] - reconstruct(frame, penalty_goals=pd.Series([1.0])).iloc[0]
    )

    assert difference == 12  # 24 -> 12


def test_a_defenders_penalty_needs_no_correction():
    """Their goals are already worth 12."""
    frame = pd.DataFrame([appearance("DEF", goals_scored=1)])

    assert reconstruct(frame).iloc[0] == reconstruct(frame, penalty_goals=pd.Series([1.0])).iloc[0]


def test_a_fractional_penalty_share_scales_the_correction():
    """Callers supply expected penalty goals, which are rarely whole numbers."""
    frame = pd.DataFrame([appearance("FWD", goals_scored=1)])

    difference = (
        reconstruct(frame).iloc[0] - reconstruct(frame, penalty_goals=pd.Series([0.5])).iloc[0]
    )

    assert difference == 6


def test_no_penalty_goals_leaves_the_reconstruction_untouched():
    frame = pd.DataFrame([appearance("FWD", goals_scored=2)])

    assert reconstruct(frame).iloc[0] == reconstruct(frame, penalty_goals=pd.Series([0.0])).iloc[0]


def test_the_correction_is_never_negative():
    """It only ever removes over-credit; it cannot invent BPS."""
    frame = pd.DataFrame([appearance("GK", goals_scored=1), appearance("DEF", goals_scored=1)])

    assert (penalty_correction(frame, pd.Series([1.0, 1.0])) >= 0).all()


def test_correcting_an_empty_frame_is_safe():
    assert penalty_correction(pd.DataFrame(), pd.Series(dtype="float64")).empty
