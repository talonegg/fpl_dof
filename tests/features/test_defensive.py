"""Tests for defensive contributions.

The formula is checkable rather than assumed: the API publishes both the
component actions and the total they sum to, so these assert the identity
holds instead of trusting it.
"""

from __future__ import annotations

import pandas as pd
import pytest

from fpl.features.defensive import (
    DEFENSIVE_THRESHOLD,
    add_defensive_metrics,
    clears_threshold,
    defensive_actions,
    defensive_points,
    formula_agreement,
    threshold_for,
)


def appearance(position, cbi=0, tackles=0, recoveries=0, published=None):
    row = {
        "position": position,
        "clearances_blocks_interceptions": cbi,
        "tackles": tackles,
        "recoveries": recoveries,
    }
    if published is not None:
        row["defensive_contribution"] = published
    return row


def test_a_defender_is_scored_on_clearances_blocks_interceptions_and_tackles():
    frame = pd.DataFrame([appearance("DEF", cbi=7, tackles=3, recoveries=5)])

    # Recoveries must not count for a defender.
    assert defensive_actions(frame).iloc[0] == 10


def test_a_midfielder_is_scored_on_the_same_plus_recoveries():
    frame = pd.DataFrame([appearance("MID", cbi=7, tackles=3, recoveries=5)])

    assert defensive_actions(frame).iloc[0] == 15


def test_a_forward_is_scored_like_a_midfielder():
    frame = pd.DataFrame([appearance("FWD", cbi=4, tackles=2, recoveries=6)])

    assert defensive_actions(frame).iloc[0] == 12


def test_a_goalkeeper_is_ineligible_rather_than_scoring_zero():
    """Zero would read as "did nothing"; they simply cannot earn these."""
    frame = pd.DataFrame([appearance("GK", cbi=20, tackles=5)])

    assert pd.isna(defensive_actions(frame).iloc[0])
    assert not clears_threshold(frame).iloc[0]
    assert defensive_points(frame).iloc[0] == 0


def test_defenders_have_the_lower_threshold():
    """Recoveries are far more common in midfield, hence the higher bar there."""
    assert DEFENSIVE_THRESHOLD["DEF"] == 10
    assert DEFENSIVE_THRESHOLD["MID"] == 12
    assert DEFENSIVE_THRESHOLD["FWD"] == 12


def test_exactly_meeting_the_threshold_earns_the_points():
    frame = pd.DataFrame([appearance("DEF", cbi=10)])

    assert clears_threshold(frame).iloc[0]
    assert defensive_points(frame).iloc[0] == 2


def test_one_short_of_the_threshold_earns_nothing():
    frame = pd.DataFrame([appearance("DEF", cbi=9)])

    assert not clears_threshold(frame).iloc[0]
    assert defensive_points(frame).iloc[0] == 0


def test_ten_actions_pays_a_defender_but_not_a_midfielder():
    frame = pd.DataFrame([appearance("DEF", cbi=10), appearance("MID", cbi=10)])

    assert clears_threshold(frame).tolist() == [True, False]


def test_thresholds_are_reported_per_position():
    frame = pd.DataFrame([appearance("DEF"), appearance("MID"), appearance("GK")])

    thresholds = threshold_for(frame)

    assert thresholds.iloc[0] == 10
    assert thresholds.iloc[1] == 12
    assert pd.isna(thresholds.iloc[2])


def test_positions_spelled_out_are_understood():
    """The archive says DEF; the API says Defender."""
    frame = pd.DataFrame([appearance("Defender", cbi=10)])

    assert clears_threshold(frame).iloc[0]


def test_the_computed_total_matches_the_published_one():
    frame = pd.DataFrame(
        [
            appearance("DEF", cbi=7, tackles=3, recoveries=4, published=10),
            appearance("MID", cbi=5, tackles=2, recoveries=6, published=13),
        ]
    )

    assert formula_agreement(frame) == 1.0


def test_a_disagreement_is_visible_rather_than_silent():
    """If the rule changes, this is what says so."""
    frame = pd.DataFrame([appearance("DEF", cbi=7, tackles=3, published=99)])

    assert formula_agreement(frame) == 0.0


def test_agreement_ignores_goalkeepers():
    frame = pd.DataFrame([appearance("GK", cbi=5, published=5)])

    assert pd.isna(formula_agreement(frame))


def test_add_defensive_metrics_attaches_every_column():
    frame = pd.DataFrame([appearance("DEF", cbi=12)])

    result = add_defensive_metrics(frame)

    assert result["defensive_actions"].iloc[0] == 12
    assert result["defensive_threshold"].iloc[0] == 10
    assert result["cleared_defensive_threshold"].iloc[0]
    assert result["defensive_points"].iloc[0] == 2


def test_missing_component_columns_do_not_raise():
    frame = pd.DataFrame([{"position": "DEF"}])

    assert defensive_actions(frame).iloc[0] == 0


def test_empty_input_is_safe():
    assert defensive_actions(pd.DataFrame()).empty
    assert add_defensive_metrics(pd.DataFrame()).empty


# --- Against the real archive ---


@pytest.mark.backtest
def test_the_formula_holds_exactly_on_a_real_season():
    from fpl.sources.archive import fetch_season_gameweeks

    season = fetch_season_gameweeks("2025-26")
    played = season[season["minutes"] > 0]

    assert formula_agreement(played) == 1.0


@pytest.mark.backtest
def test_defensive_points_are_a_plausible_share_of_a_real_season():
    from fpl.sources.archive import fetch_season_gameweeks

    season = fetch_season_gameweeks("2025-26")
    result = add_defensive_metrics(season[season["minutes"] > 0])

    share = result["cleared_defensive_threshold"].mean()
    assert 0.05 < share < 0.30, "roughly one appearance in eight should qualify"
