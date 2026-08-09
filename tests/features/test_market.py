"""Tests for converting bookmaker prices into probabilities.

The arithmetic here is checkable by hand, and the failure mode it guards
against is silent: raw implied probabilities look entirely reasonable and are
systematically about 5% too high.
"""

from __future__ import annotations

import math

import pandas as pd
import pytest

from fpl.features.market import (
    DEFAULT_TOTAL_GOALS,
    devig,
    implied_probability,
    match_probabilities,
    overround,
    team_expectations,
    total_goals_line,
)
from fpl.sources.odds import flatten_odds


@pytest.fixture
def odds(odds_payload):
    return flatten_odds(odds_payload)


def test_implied_probability_is_the_reciprocal_price():
    assert implied_probability(pd.Series([2.0, 4.0])).tolist() == [0.5, 0.25]


def test_a_real_book_sums_to_more_than_one():
    """If it does not, the prices have been misparsed."""
    prices = pd.Series([1.75, 3.9, 4.8])

    assert overround(prices) > 1.0


def test_devigged_probabilities_sum_to_one():
    result = devig(pd.Series([1.75, 3.9, 4.8]))

    assert result.sum() == pytest.approx(1.0)


def test_devigging_lowers_every_probability():
    """The margin inflates all of them, so removing it deflates all of them."""
    prices = pd.Series([1.75, 3.9, 4.8])

    assert (devig(prices) < implied_probability(prices)).all()


def test_devigging_preserves_the_ordering():
    prices = pd.Series([1.75, 3.9, 4.8])

    result = devig(prices)

    assert result.iloc[0] > result.iloc[1] > result.iloc[2]


def test_a_fair_book_is_unchanged_by_devigging():
    # Two outcomes at evens: already sums to 1.
    prices = pd.Series([2.0, 2.0])

    assert devig(prices).tolist() == [0.5, 0.5]


def test_match_probabilities_sum_to_one(odds):
    result = match_probabilities(odds)

    totals = result[["home_win", "draw", "away_win"]].sum(axis=1)
    assert totals.round(6).eq(1.0).all()


def test_the_favourite_has_the_highest_probability(odds):
    result = match_probabilities(odds)
    arsenal = result[result["home_team"] == "Arsenal"].iloc[0]

    assert arsenal["home_win"] > arsenal["away_win"]
    assert arsenal["home_win"] > arsenal["draw"]


def test_a_strong_away_favourite_is_reflected(odds):
    result = match_probabilities(odds)
    burnley = result[result["home_team"] == "Burnley"].iloc[0]

    assert burnley["away_win"] > burnley["home_win"]


def test_the_number_of_bookmakers_is_reported(odds):
    """A one-bookmaker consensus is not a consensus, and callers should see that."""
    result = match_probabilities(odds)

    arsenal = result[result["home_team"] == "Arsenal"].iloc[0]
    burnley = result[result["home_team"] == "Burnley"].iloc[0]
    assert arsenal["bookmakers"] == 2
    assert burnley["bookmakers"] == 1


def test_match_probabilities_of_nothing_is_empty():
    assert match_probabilities(pd.DataFrame()).empty


def test_expected_total_goals_is_plausible(odds):
    result = total_goals_line(odds)

    assert (result["expected_total_goals"] > 1.5).all()
    assert (result["expected_total_goals"] < 5.0).all()


def test_a_higher_over_price_implies_fewer_goals(odds):
    """Burnley-Liverpool is priced over-heavy; Arsenal's is closer to even."""
    result = total_goals_line(odds).set_index("match_id")

    assert (
        result.loc["match-burnley-liverpool", "expected_total_goals"]
        > result.loc["match-arsenal-manutd", "expected_total_goals"]
    )


def test_team_expectations_give_two_rows_per_match(odds):
    result = team_expectations(odds)

    assert len(result) == 2 * len(match_probabilities(odds))


def test_the_favourite_is_expected_to_score_more(odds):
    result = team_expectations(odds)
    liverpool = result[result["team"] == "Liverpool"].iloc[0]
    burnley = result[result["team"] == "Burnley"].iloc[0]

    assert liverpool["expected_goals_for"] > burnley["expected_goals_for"]


def test_a_team_facing_a_weak_attack_has_the_better_clean_sheet_chance(odds):
    result = team_expectations(odds)
    liverpool = result[result["team"] == "Liverpool"].iloc[0]
    burnley = result[result["team"] == "Burnley"].iloc[0]

    assert liverpool["clean_sheet"] > burnley["clean_sheet"]


def test_clean_sheet_is_the_chance_the_opponent_fails_to_score(odds):
    result = team_expectations(odds)
    row = result.iloc[0]

    assert row["clean_sheet"] == pytest.approx(math.exp(-row["expected_goals_against"]))


def test_goals_for_and_against_mirror_across_the_two_rows(odds):
    result = team_expectations(odds)
    match = result[result["match_id"] == "match-arsenal-manutd"]
    home = match[match["is_home"]].iloc[0]
    away = match[~match["is_home"]].iloc[0]

    assert home["expected_goals_for"] == pytest.approx(away["expected_goals_against"])


def test_probabilities_are_all_between_zero_and_one(odds):
    result = team_expectations(odds)

    assert result["clean_sheet"].between(0, 1).all()
    assert result["win"].between(0, 1).all()


def test_a_missing_totals_market_falls_back_to_the_league_average(odds):
    without_totals = odds[odds["market"] != "totals"]

    result = team_expectations(without_totals)

    match = result[result["match_id"] == "match-arsenal-manutd"]
    total = match["expected_goals_for"].sum()
    assert total == pytest.approx(DEFAULT_TOTAL_GOALS)


def test_team_expectations_of_nothing_is_empty():
    assert team_expectations(pd.DataFrame()).empty
