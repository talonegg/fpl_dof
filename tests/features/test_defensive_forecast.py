"""Tests for forecasting defensive contributions before a season starts.

The rule arrived in 2025-26. The three seasons before it had no such route to
points, and — critically — no record of the underlying actions either. So the
question "can this model see defensive contributions" has three answers, not
two, and the tests here exist to keep them apart.
"""

from __future__ import annotations

import pandas as pd

from fpl.domain.rules import season_scores_defensive_contributions
from fpl.features.defensive import (
    defensive_contribution_rate,
    expected_defensive_points,
)


def appearances(rows):
    return pd.DataFrame(rows)


def player(name, position, cbi=0, tackles=0, recoveries=0, minutes=90):
    return {
        "player_name": name,
        "position": position,
        "minutes": minutes,
        "clearances_blocks_interceptions": cbi,
        "tackles": tackles,
        "recoveries": recoveries,
    }


# -- Which seasons score them ---------------------------------------------


def test_the_rule_applies_from_2025_26():
    assert season_scores_defensive_contributions("2025-26")


def test_the_rule_continues_into_2026_27():
    assert season_scores_defensive_contributions("2026-27")


def test_the_rule_did_not_exist_in_2024_25():
    assert not season_scores_defensive_contributions("2024-25")


def test_the_rule_did_not_exist_in_2023_24():
    assert not season_scores_defensive_contributions("2023-24")


# -- Rates from seasons that recorded the actions -------------------------


def test_a_defender_clearing_ten_actions_counts():
    data = {"2025-26": appearances([player("A", "DEF", cbi=10)])}

    assert defensive_contribution_rate(data)["defensive_rate"].iloc[0] == 1.0


def test_a_defender_below_the_threshold_does_not():
    data = {"2025-26": appearances([player("A", "DEF", cbi=9)])}

    assert defensive_contribution_rate(data)["defensive_rate"].iloc[0] == 0.0


def test_a_midfielder_needs_twelve_and_recoveries_count():
    """CBIRT for midfielders: recoveries are included where they are not for defenders."""
    data = {"2025-26": appearances([player("A", "MID", cbi=6, tackles=2, recoveries=4)])}

    assert defensive_contribution_rate(data)["defensive_rate"].iloc[0] == 1.0


def test_the_rate_is_the_share_of_matches_cleared():
    data = {"2025-26": appearances([player("A", "DEF", cbi=10), player("A", "DEF", cbi=2)])}

    assert defensive_contribution_rate(data)["defensive_rate"].iloc[0] == 0.5


def test_a_season_without_the_action_counts_contributes_nothing():
    """Not zero -- nothing. A player is not a poor defender because nobody counted."""
    data = {"2024-25": pd.DataFrame([{"player_name": "A", "position": "DEF", "minutes": 90}])}

    assert defensive_contribution_rate(data).empty


def test_a_season_with_counts_but_no_positions_is_skipped():
    """2018-19 is exactly this, and the threshold is positional."""
    rows = pd.DataFrame(
        [{"player_name": "A", "minutes": 90, "clearances_blocks_interceptions": 20}]
    )

    assert defensive_contribution_rate({"2018-19": rows}).empty


def test_only_full_appearances_count_towards_the_rate():
    """The threshold is unreachable in a cameo, so a cameo is not evidence against."""
    data = {"2025-26": appearances([player("A", "DEF", cbi=10), player("A", "DEF", minutes=10)])}

    assert defensive_contribution_rate(data)["defensive_matches"].iloc[0] == 1


# -- Turning a rate into points -------------------------------------------


def test_a_certain_contributor_who_always_starts_earns_two_points():
    points = expected_defensive_points(pd.Series([1.0]), pd.Series([1.0]))

    assert points.iloc[0] == 2.0


def test_a_player_who_rarely_starts_earns_less():
    always = expected_defensive_points(pd.Series([1.0]), pd.Series([1.0])).iloc[0]
    rarely = expected_defensive_points(pd.Series([1.0]), pd.Series([0.25])).iloc[0]

    assert rarely < always


def test_a_thin_sample_is_regressed_towards_the_population():
    """One match at a 100% rate is not a certainty."""
    rate = pd.Series([1.0, 0.2, 0.2, 0.2])
    starts = pd.Series([1.0, 1.0, 1.0, 1.0])
    matches = pd.Series([1, 30, 30, 30])

    regressed = expected_defensive_points(rate, starts, matches)

    assert regressed.iloc[0] < 2.0


def test_a_deep_sample_keeps_its_own_rate():
    rate = pd.Series([1.0, 0.2])
    starts = pd.Series([1.0, 1.0])
    matches = pd.Series([40, 40])

    assert expected_defensive_points(rate, starts, matches).iloc[0] == 2.0


def test_no_rates_gives_no_points():
    assert expected_defensive_points(pd.Series(dtype="float64"), pd.Series(dtype="float64")).empty
