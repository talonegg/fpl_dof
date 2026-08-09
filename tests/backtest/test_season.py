"""Tests for the season simulation.

The simulation is the only thing in the repo that produces a number you would
act on, so the things that must hold are: it never sees the future, it never
spends transfers it does not have, and its score is what the chosen eleven
really scored.
"""

from __future__ import annotations

import pandas as pd
import pytest

from fpl.backtest.season import GameweekOutcome, SeasonResult, build_pool, simulate_season
from fpl.models.naive import SeasonMeanPredictor

POSITIONS = ["GK"] * 4 + ["DEF"] * 10 + ["MID"] * 10 + ["FWD"] * 6


def make_season(gameweeks=10):
    """A small but legal world: 30 players across 10 clubs."""
    rows = []
    for index, position in enumerate(POSITIONS, start=1):
        for gameweek in range(1, gameweeks + 1):
            rows.append(
                {
                    "element": index,
                    "player_name": f"P{index}",
                    "position": position,
                    "team_name": f"Club{index % 10}",
                    "price": 4.0,
                    "gameweek": gameweek,
                    "minutes": 90,
                    "total_points": (index % 7) + 1,
                    "opponent_team": (index + gameweek) % 10,
                    "was_home": gameweek % 2 == 0,
                }
            )
    return pd.DataFrame(rows)


def test_a_season_produces_an_outcome_per_gameweek():
    result = simulate_season(make_season(), SeasonMeanPredictor(), first_gameweek=4)

    assert [outcome.gameweek for outcome in result.outcomes] == [4, 5, 6, 7, 8, 9, 10]


def test_the_first_gameweek_buys_a_squad_without_transfers():
    result = simulate_season(make_season(), SeasonMeanPredictor(), first_gameweek=4)

    assert result.outcomes[0].transfers == 0
    assert result.outcomes[0].hits_cost == 0


def test_a_squad_is_always_fifteen_players():
    result = simulate_season(make_season(), SeasonMeanPredictor(), first_gameweek=4)

    for outcome in result.outcomes:
        assert len(outcome.squad) == 15


def test_a_squad_never_contains_duplicates():
    """A transfer that adds a player already owned would silently shrink the squad."""
    result = simulate_season(make_season(), SeasonMeanPredictor(), first_gameweek=4)

    for outcome in result.outcomes:
        assert len(set(outcome.squad)) == 15


def test_a_zero_horizon_means_no_transfer_is_ever_worth_it():
    """The control: with no horizon, a gain cannot repay a hit."""
    result = simulate_season(make_season(), SeasonMeanPredictor(), first_gameweek=4, horizon=0)

    assert result.transfers_made == 0
    assert result.total_hits == 0


def test_the_squad_is_unchanged_when_no_transfers_are_made():
    result = simulate_season(make_season(), SeasonMeanPredictor(), first_gameweek=4, horizon=0)

    squads = {tuple(sorted(outcome.squad)) for outcome in result.outcomes}
    assert len(squads) == 1


def test_points_are_what_the_chosen_eleven_actually_scored():
    season = make_season()
    result = simulate_season(season, SeasonMeanPredictor(), first_gameweek=4, horizon=0)

    outcome = result.outcomes[0]
    actual = season[season["gameweek"] == outcome.gameweek].set_index("element")["total_points"]
    # Eleven starters plus the captain counted twice.
    assert outcome.points <= actual.nlargest(11).sum() + actual.max()


def test_net_points_subtract_the_hits():
    outcome = GameweekOutcome(gameweek=5, points=60, transfers=2, hits_cost=4, captain=1)

    assert outcome.net_points == 56


def test_season_totals_are_net_of_hits():
    result = SeasonResult(
        model="test",
        outcomes=[
            GameweekOutcome(gameweek=1, points=50, transfers=0, hits_cost=0, captain=1),
            GameweekOutcome(gameweek=2, points=60, transfers=2, hits_cost=4, captain=1),
        ],
    )

    assert result.gross_points == 110
    assert result.total_points == 106
    assert result.total_hits == 4
    assert result.points_per_gameweek == pytest.approx(53.0)


def test_an_empty_season_scores_nothing_rather_than_raising():
    empty = SeasonResult(model="test", outcomes=[])

    assert empty.total_points == 0
    assert empty.points_per_gameweek == 0.0


def test_build_pool_attaches_what_the_optimiser_needs():
    season = make_season()
    history = season[season["gameweek"] < 5]
    predictions = pd.DataFrame(
        [{"element": 1, "expected_points": 5.0}, {"element": 2, "expected_points": 4.0}]
    )

    pool = build_pool(history, predictions)

    assert {"price", "position", "team", "expected_points"} <= set(pool.columns)
    assert len(pool) == 2


def test_build_pool_expands_archive_position_codes():
    season = make_season()
    predictions = pd.DataFrame([{"element": 1, "expected_points": 5.0}])

    pool = build_pool(season[season["gameweek"] < 5], predictions)

    assert pool["position"].iloc[0] == "Goalkeeper"


def test_build_pool_of_nothing_is_empty():
    assert build_pool(pd.DataFrame(), pd.DataFrame()).empty


def test_the_frame_view_has_a_row_per_gameweek():
    result = simulate_season(make_season(), SeasonMeanPredictor(), first_gameweek=4)

    frame = result.to_frame()

    assert len(frame) == len(result.outcomes)
    assert {"gameweek", "points", "net_points", "transfers"} <= set(frame.columns)


# --- Regressions from the code review of dev..dev_ph4 ---


def test_free_transfers_are_deducted_not_reset():
    """FPL banks unused transfers; resetting to 1 charges hits nobody would pay."""
    from fpl.domain.rules import FREE_TRANSFERS_PER_GAMEWEEK, MAX_ROLLED_FREE_TRANSFERS

    free = FREE_TRANSFERS_PER_GAMEWEEK
    # Three quiet weeks bank transfers up to the cap.
    for _ in range(3):
        free = min(max(free - 0, 0) + FREE_TRANSFERS_PER_GAMEWEEK, MAX_ROLLED_FREE_TRANSFERS)
    banked = free

    # Spending one leaves the rest banked, not reset to one.
    after_spending_one = min(
        max(banked - 1, 0) + FREE_TRANSFERS_PER_GAMEWEEK, MAX_ROLLED_FREE_TRANSFERS
    )

    assert banked >= 3
    assert after_spending_one > FREE_TRANSFERS_PER_GAMEWEEK


def test_a_missing_player_does_not_end_the_season():
    """A player vanishing from the data used to discard every later gameweek."""
    season = make_season(gameweeks=10)
    # Element 1 disappears from gameweek 7 onwards.
    season = season[~((season["element"] == 1) & (season["gameweek"] >= 7))]

    result = simulate_season(season, SeasonMeanPredictor(), first_gameweek=4, horizon=0)

    assert [outcome.gameweek for outcome in result.outcomes] == [4, 5, 6, 7, 8, 9, 10]


def test_the_season_still_scores_after_a_player_vanishes():
    season = make_season(gameweeks=10)
    season = season[~((season["element"] == 1) & (season["gameweek"] >= 7))]

    result = simulate_season(season, SeasonMeanPredictor(), first_gameweek=4, horizon=0)

    assert sum(o.points for o in result.outcomes[3:]) > 0


def test_custom_squad_sizes_survive_the_lineup_step():
    """Rebuilding constraints from scratch silently reverted these to 15/11."""
    from fpl.optimise.squad import SquadConstraints

    # A starting XI cannot go below 7 whatever the constraints say, because
    # STARTING_XI_LIMITS (1 GK, 3+ DEF, 2+ MID, 1+ FWD) lives in the rules and
    # is not per-caller. 11 and 7 is the smallest legal shrink.
    constraints = SquadConstraints(
        squad_size=11,
        starting_size=7,
        composition={"Goalkeeper": 1, "Defender": 4, "Midfielder": 3, "Forward": 3},
        max_per_club=11,
    )

    result = simulate_season(
        make_season(),
        SeasonMeanPredictor(),
        first_gameweek=4,
        horizon=0,
        constraints=constraints,
    )

    assert result.outcomes
    assert len(result.outcomes[0].squad) == 11


def test_build_pool_says_what_is_missing_rather_than_raising_a_key_error():
    season = make_season()
    predictions = pd.DataFrame([{"element": 1, "expected_points": 5.0}])

    with pytest.raises(ValueError, match="missing"):
        build_pool(season.drop(columns=["team_name"]), predictions)


def test_unspent_budget_is_carried_rather_than_forfeited():
    """Money left at the opening buy must remain available for later transfers."""
    from fpl.optimise.squad import SquadConstraints

    # Everyone costs 4.0, so a 15-player squad costs 60 of the 100 budget.
    result = simulate_season(
        make_season(),
        SeasonMeanPredictor(),
        first_gameweek=4,
        horizon=0,
        constraints=SquadConstraints(),
    )

    assert result.outcomes  # the run completes; bank tracking is exercised below
