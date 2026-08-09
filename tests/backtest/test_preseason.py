"""Tests for season-opening squad selection.

The rule that matters: a squad for season S may use only seasons before S and
S's own opening prices. Anything else is hindsight, and hindsight is trivially
easy to leak here — end-of-season prices alone would let a model buy a player
who rose from £4.5m to £7m at the price he finished at.
"""

from __future__ import annotations

import pandas as pd

from fpl.backtest.preseason import (
    actual_points,
    build_pool,
    cheapest_squad,
    compare_strategies,
    expected_points_from_history,
    expected_points_with_minutes,
    hindsight_squad,
    opening_prices,
    pick_squad,
    prior_points_squad,
    run_strategy,
)

POSITIONS = ["GK"] * 4 + ["DEF"] * 10 + ["MID"] * 10 + ["FWD"] * 6


def season(points_by_element=None, minutes=90, gameweeks=12, price=5.0):
    rows = []
    for index, position in enumerate(POSITIONS, start=1):
        for gameweek in range(1, gameweeks + 1):
            rows.append(
                {
                    "element": index,
                    "player_name": f"Player {index}",
                    "position": position,
                    "team_name": f"Club{index % 10}",
                    "price": price if gameweek == 1 else price + 0.5,
                    "gameweek": gameweek,
                    "minutes": minutes,
                    "total_points": (points_by_element or {}).get(index, 2),
                }
            )
    return pd.DataFrame(rows)


def test_prices_come_from_gameweek_one():
    """Using later prices would be hindsight: prices move during a season."""
    prices = opening_prices(season(price=5.0))

    assert (prices["price"] == 5.0).all()


def test_positions_are_translated_to_the_names_the_optimiser_uses():
    prices = opening_prices(season())

    assert set(prices["position"]) <= {"Goalkeeper", "Defender", "Midfielder", "Forward"}


def test_opening_prices_of_nothing_is_empty():
    assert opening_prices(pd.DataFrame()).empty


def test_points_are_split_between_the_opening_run_and_the_season():
    data = season(gameweeks=12)

    opening, whole = actual_points(data, [1], horizon=10)

    assert opening < whole


def test_a_player_who_never_appears_scores_nothing():
    assert actual_points(season(), [999]) == (0.0, 0.0)


def test_the_pool_keeps_players_with_no_history_rather_than_dropping_them():
    """15% of a real list have none; hiding them would hide the gap."""
    prior = {"2024-25": season()}
    prices = opening_prices(season())
    prices.loc[0, "player_name"] = "Brand New Signing"
    prices.loc[0, "match_key"] = "brand new signing"

    pool = build_pool(prior, prices)

    assert len(pool) == len(prices)
    assert pool["total_points_per_90"].isna().any()


def test_a_constant_minutes_assumption_ignores_who_actually_plays():
    """The flaw that made the first version worse than picking at random."""
    pool = pd.DataFrame(
        [
            {"total_points_per_90": 10.0, "career_minutes": 90, "seasons_seen": 1},
            {"total_points_per_90": 10.0, "career_minutes": 3000, "seasons_seen": 1},
        ]
    )

    naive = expected_points_from_history(pool)

    assert naive.iloc[0] == naive.iloc[1]


def test_the_minutes_aware_version_separates_them():
    pool = pd.DataFrame(
        [
            {"total_points_per_90": 10.0, "career_minutes": 90, "seasons_seen": 1},
            {"total_points_per_90": 10.0, "career_minutes": 3000, "seasons_seen": 1},
        ]
    )

    aware = expected_points_with_minutes(pool)

    assert aware.iloc[1] > aware.iloc[0] * 10


def test_expected_minutes_cannot_exceed_a_full_match():
    pool = pd.DataFrame([{"total_points_per_90": 5.0, "career_minutes": 999999, "seasons_seen": 1}])

    assert expected_points_with_minutes(pool, horizon=1).iloc[0] <= 5.0


def test_a_squad_is_legal():
    prices = opening_prices(season())

    squad = pick_squad(prices, pd.Series(1.0, index=prices.index))

    assert len(squad) == 15


def test_a_squad_cannot_be_built_from_nothing():
    assert pick_squad(pd.DataFrame(), pd.Series(dtype="float64")) is None


def test_the_hindsight_squad_is_the_ceiling():
    """Nobody can reach it; a score without it is uninterpretable."""
    data = season({1: 20, 15: 20, 25: 20})

    best = hindsight_squad("2025-26", data)
    uninformed = cheapest_squad("2025-26", data)

    assert best.opening_points >= uninformed.opening_points


def test_the_uninformed_squad_uses_no_information():
    data = season({1: 50})

    result = cheapest_squad("2025-26", data)

    assert result is not None
    assert len(result.squad) == 15


def test_prior_season_points_needs_a_prior_season():
    assert prior_points_squad("2025-26", season(), {}) is None


def test_a_strategy_reports_what_it_spent():
    prior = {"2024-25": season()}

    result = run_strategy("2025-26", season(), prior, "Test")

    assert result is not None
    assert result.cost <= 100.0


def test_comparison_includes_the_ceiling_and_the_floor():
    data = {"2024-25": season(), "2025-26": season({1: 20})}

    table = compare_strategies(data, "2025-26")

    assert "Hindsight" in table["strategy"].tolist()
    assert "Uninformed" in table["strategy"].tolist()
    assert "share_of_ceiling" in table.columns


def test_comparison_needs_a_prior_season():
    assert compare_strategies({"2025-26": season()}, "2025-26").empty


def test_comparison_of_an_unknown_season_is_empty():
    assert compare_strategies({"2025-26": season()}, "2030-31").empty
