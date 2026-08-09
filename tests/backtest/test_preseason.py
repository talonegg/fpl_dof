"""Tests for replaying season-opening squad selection.

The rule that matters: a squad for season S may use only seasons before S and
S's own opening prices. Anything else is hindsight, and hindsight is trivially
easy to leak here — end-of-season prices alone would let a model buy a player
who rose from £4.5m to £7m at the price he finished at.
"""

from __future__ import annotations

import pandas as pd

from fpl.backtest.preseason import (
    SCORING_HORIZONS,
    actual_points,
    compare_horizons,
    compare_strategies,
    defensive_forecast_status,
    hindsight_squad,
    horizon_table,
    run_strategy,
)
from fpl.features.preseason_pool import build_pool, opening_prices
from fpl.models.preseason_strategies import (
    BlendedRates,
    BlendedRatesWithMinutes,
    PriorSeasonPoints,
    Uninformed,
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


def defensive_season(cbi=20):
    """A season carrying the action counts, as 2025-26 does."""
    data = season()
    data["clearances_blocks_interceptions"] = cbi
    data["tackles"] = 0
    data["recoveries"] = 0
    return data


# -- Prices and scoring ---------------------------------------------------


def test_prices_come_from_gameweek_one():
    prices = opening_prices(season(price=5.0))

    assert (prices["price"] == 5.0).all()


def test_positions_are_translated_to_the_names_the_optimiser_uses():
    prices = opening_prices(season())

    assert "Goalkeeper" in set(prices["position"])


def test_opening_prices_of_nothing_is_empty():
    assert opening_prices(pd.DataFrame()).empty


def test_points_are_split_between_the_opening_run_and_the_season():
    opening, whole = actual_points(season(gameweeks=12), [1], horizon=3)

    assert opening == 6
    assert whole == 24


def test_a_player_who_never_appears_scores_nothing():
    assert actual_points(season(), [999]) == (0.0, 0.0)


# -- The pool -------------------------------------------------------------


def test_the_pool_keeps_players_with_no_history_rather_than_dropping_them():
    """They are excluded at selection, not silently removed from the count."""
    prior = season().head(12 * 3)  # only the first three players have history

    pool = build_pool({"2024-25": prior}, opening_prices(season()))

    assert len(pool) == len(POSITIONS)
    assert pool["career_minutes"].isna().any()


# -- Strategies -----------------------------------------------------------


def context_for(target, prior):
    from fpl.models.preseason_strategies import PreseasonContext

    return PreseasonContext(target=target, prior_seasons=prior, horizon=7)


def test_a_constant_minutes_assumption_prefers_the_player_who_does_not_play():
    """The failure that cost the first version everything, in one assertion.

    Both players score the same points per match. One plays 90 minutes, the
    other 10 — so the substitute's *per-90 rate* is nine times higher, and a
    model that multiplies it by a constant buys him. This is why that strategy
    scored below a randomly chosen legal squad.
    """
    prior = pd.concat([season(minutes=90).head(12), season(minutes=10).iloc[12:24]])
    pool = build_pool({"2024-25": prior}, opening_prices(season()))

    expected = BlendedRates().expected_points(pool, context_for("2025-26", {"2024-25": prior}))

    starter = expected[pool["element"] == 1].iloc[0]
    substitute = expected[pool["element"] == 2].iloc[0]
    assert substitute > starter


def test_the_minutes_aware_version_separates_them():
    prior = pd.concat([season(minutes=90).head(12), season(minutes=10).iloc[12:24]])
    pool = build_pool({"2024-25": prior}, opening_prices(season()))

    expected = BlendedRatesWithMinutes().expected_points(
        pool, context_for("2025-26", {"2024-25": prior})
    )

    starter = expected[pool["element"] == 1].iloc[0]
    substitute = expected[pool["element"] == 2].iloc[0]
    assert starter > substitute


def test_the_uninformed_strategy_uses_no_information():
    pool = build_pool({"2024-25": season()}, opening_prices(season()))

    expected = Uninformed().expected_points(pool, context_for("2025-26", {}))

    assert expected.nunique() == 1


def test_prior_season_points_needs_a_prior_season():
    pool = build_pool({"2024-25": season()}, opening_prices(season()))

    expected = PriorSeasonPoints().expected_points(pool, context_for("2025-26", {}))

    assert (expected == 0).all()


# -- Running and benchmarking ---------------------------------------------


def test_a_squad_is_legal():
    prior = {"2023-24": season()}
    result = run_strategy("2024-25", season(), prior, BlendedRatesWithMinutes(), horizon=3)

    assert len(result.squad) == 15
    assert result.cost <= 100.0


def test_a_strategy_reports_what_it_spent():
    prior = {"2023-24": season()}
    result = run_strategy("2024-25", season(), prior, BlendedRatesWithMinutes(), horizon=3)

    assert result.cost > 0


def test_the_hindsight_squad_is_the_ceiling():
    """Nobody can reach it, which is what makes it the yardstick."""
    generous = {index: 100 for index in range(1, 16)}
    data = season(points_by_element=generous)

    ceiling = hindsight_squad("2024-25", data, horizon=3)
    honest = run_strategy(
        "2024-25", data, {"2023-24": season()}, BlendedRatesWithMinutes(), horizon=3
    )

    assert ceiling.opening_points >= honest.opening_points


def test_a_strategy_cannot_run_without_a_pool():
    assert run_strategy("2024-25", pd.DataFrame(), {}, Uninformed()) is None


# -- Comparison -----------------------------------------------------------


def test_comparison_includes_the_ceiling_and_the_floor():
    data = {"2023-24": season(), "2024-25": season()}

    frame = compare_strategies(data, "2024-25", horizon=3)

    assert "Hindsight" in set(frame["strategy"])
    assert "Uninformed" in set(frame["strategy"])


def test_comparison_covers_every_registered_strategy():
    """A strategy added to the registry must appear without touching the harness."""
    from fpl.models.preseason_strategies import strategies

    data = {"2023-24": season(), "2024-25": season()}
    frame = compare_strategies(data, "2024-25", horizon=3)

    assert {strategy.name for strategy in strategies()} <= set(frame["strategy"])


def test_comparison_needs_a_prior_season():
    assert compare_strategies({"2024-25": season()}, "2024-25").empty


def test_comparison_of_an_unknown_season_is_empty():
    assert compare_strategies({"2023-24": season()}, "2030-31").empty


# -- Horizons -------------------------------------------------------------


def test_every_horizon_and_season_appears():
    data = {"2023-24": season(), "2024-25": season(), "2025-26": season()}

    comparison = compare_horizons(data, targets=("2024-25", "2025-26"), horizons=(3, 5))

    assert set(comparison["horizon"]) == {3, 5}
    assert set(comparison["season"]) == {"2024-25", "2025-26"}


def test_a_shorter_horizon_scores_fewer_points():
    """Sanity: three gameweeks cannot out-score seven of the same squad."""
    data = {"2023-24": season(), "2024-25": season()}

    comparison = compare_horizons(data, targets=("2024-25",), horizons=(3, 7))

    short = comparison[comparison["horizon"] == 3]["opening_points"].max()
    long = comparison[comparison["horizon"] == 7]["opening_points"].max()
    assert short < long


def test_the_table_puts_horizons_across_and_strategies_down():
    data = {"2023-24": season(), "2024-25": season()}

    table = horizon_table(compare_horizons(data, targets=("2024-25",), horizons=(3, 5, 7)))

    assert list(table.columns) == [3, 5, 7]
    assert "Hindsight" in table.index


def test_the_ceiling_is_the_ceiling_at_every_horizon():
    data = {"2023-24": season(), "2024-25": season()}

    table = horizon_table(compare_horizons(data, targets=("2024-25",), horizons=(3, 5, 7)))

    assert (table.loc["Hindsight"] == 1.0).all()


def test_an_empty_comparison_makes_an_empty_table():
    assert horizon_table(pd.DataFrame()).empty


def test_horizons_default_to_three_five_and_seven():
    assert SCORING_HORIZONS == (3, 5, 7)


# -- Defensive contributions: three states, not two -----------------------


def test_a_season_before_the_rule_is_not_blind_it_is_correct():
    """2024-25 scoring no defensive contributions is right, not a gap."""
    pool = build_pool({"2023-24": season()}, opening_prices(season()))

    assert defensive_forecast_status("2024-25", pool) == "not scored"


def test_the_rule_applying_without_the_data_is_reported_as_blind():
    """The 2025-26 case: the points existed, the model could not see them."""
    pool = build_pool({"2024-25": season()}, opening_prices(season()))

    assert defensive_forecast_status("2025-26", pool) == "blind"


def test_the_rule_applying_with_the_data_is_a_forecast():
    """The 2026-27 case: 2025-26 recorded the actions."""
    pool = build_pool({"2025-26": defensive_season()}, opening_prices(season()))

    assert defensive_forecast_status("2026-27", pool) == "forecast"


def test_the_status_travels_with_the_result():
    """A score cannot be read without seeing whether it was blind."""
    data = {"2023-24": season(), "2024-25": season()}

    frame = compare_strategies(data, "2024-25", horizon=3)

    assert set(frame["defensive"]) == {"not scored"}


def test_defensive_rates_reach_the_pool_when_a_prior_season_has_them():
    pool = build_pool({"2025-26": defensive_season()}, opening_prices(season()))

    assert pool["defensive_rate"].notna().any()


def test_no_defensive_column_appears_when_no_season_supplies_one():
    """Absent, not zero -- the distinction the whole status rests on."""
    pool = build_pool({"2024-25": season()}, opening_prices(season()))

    assert "defensive_rate" not in pool.columns
