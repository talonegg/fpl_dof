"""The real thing: baselines replayed over a full past season.

Marked ``backtest`` because it downloads a season of data and takes seconds
rather than milliseconds. CI runs ``-m "not backtest"``; run it deliberately
with ``pytest -m backtest``.

These assertions are deliberately loose. The point is not to pin exact numbers
-- archive data gets revised -- but to catch the failures that would mean the
harness itself is broken: a model that ranks worse than chance, or a "top 15"
that does not beat picking at random.
"""

from __future__ import annotations

import pytest

from fpl.backtest.baselines import baseline_predictors
from fpl.backtest.harness import compare, replay
from fpl.backtest.metrics import summarise
from fpl.models.naive import NaiveFormPredictor, ZeroPredictor
from fpl.sources.archive import fetch_season_gameweeks

SEASON = "2025-26"

pytestmark = pytest.mark.backtest


@pytest.fixture(scope="module")
def season():
    return fetch_season_gameweeks(SEASON)


def test_the_season_has_the_gameweeks_we_expect(season):
    assert season["gameweek"].min() == 1
    assert season["gameweek"].max() == 38


def test_naive_form_ranks_better_than_chance(season):
    result = replay(season, NaiveFormPredictor(window=5))
    summary = summarise(result.predictions)

    assert summary["rank_correlation"] > 0.1, (
        "recent form carries no ranking information at all — suspect the harness"
    )


def test_naive_forms_picks_beat_the_field_average(season):
    """If the top 15 does not beat picking at random, the model is worthless."""
    result = replay(season, NaiveFormPredictor(window=5))
    summary = summarise(result.predictions)

    assert summary["top_15_mean_actual"] > summary["field_mean_actual"]


def test_the_zero_model_has_no_ranking_information(season):
    """A sanity check on the metrics: a constant prediction must not look good."""
    result = replay(season, ZeroPredictor())
    summary = summarise(result.predictions)

    import pandas as pd

    assert pd.isna(summary["rank_correlation"])


def test_every_baseline_produces_a_comparable_row(season):
    table = compare(season, baseline_predictors())

    assert len(table) == len(baseline_predictors())
    assert table["gameweeks"].min() > 30


# --- The whole system, playing a real season ---


def test_the_optimiser_produces_a_legal_squad_from_real_data(season):
    """The constraints must survive contact with 700 real players."""
    from fpl.backtest.season import build_pool, simulate_season
    from fpl.domain.rules import MAX_PLAYERS_PER_CLUB, SQUAD_COMPOSITION
    from fpl.models.components import ComponentPredictor

    result = simulate_season(
        season, ComponentPredictor(4), first_gameweek=10, last_gameweek=12, horizon=0
    )

    assert result.outcomes
    assert len(result.outcomes[0].squad) == 15

    # Re-derive the squad's shape from the pool to check it is legal.
    from fpl.backtest.harness import known_fixtures, prepare_season

    prepared = prepare_season(season)
    history = prepared[prepared["gameweek"] < 10]
    pool = build_pool(
        history,
        ComponentPredictor(4).predict(history, 10, known_fixtures(prepared, 10)),
    )
    chosen = pool[pool["element"].isin(result.outcomes[0].squad)]

    counts = chosen["position"].value_counts()
    for position, expected in SQUAD_COMPOSITION.items():
        assert counts.get(position, 0) == expected
    assert chosen["team"].value_counts().max() <= MAX_PLAYERS_PER_CLUB
    assert chosen["price"].sum() <= 100.0


def test_a_simulated_season_scores_plausibly(season):
    """A sanity band, not a target: real FPL managers average roughly 40-70."""
    from fpl.backtest.season import simulate_season
    from fpl.models.components import ComponentPredictor

    result = simulate_season(
        season, ComponentPredictor(4), first_gameweek=10, last_gameweek=16, horizon=0
    )

    assert 25 < result.points_per_gameweek < 90


def test_transferring_costs_more_than_it_gains(season):
    """Documents the finding, so a change that fixes it fails loudly here.

    The transfer planner scales a single gameweek's edge by the horizon, which
    assumes the edge persists. It does not — predictions move week to week —
    so the planner churns the squad and pays hits for noise. Until that is
    fixed, holding beats transferring.
    """
    from fpl.backtest.season import simulate_season
    from fpl.models.components import ComponentPredictor

    holding = simulate_season(
        season, ComponentPredictor(4), first_gameweek=10, last_gameweek=20, horizon=0
    )
    transferring = simulate_season(
        season, ComponentPredictor(4), first_gameweek=10, last_gameweek=20, horizon=5
    )

    assert transferring.transfers_made > 0, "the planner should be making transfers"
    assert holding.total_points > transferring.total_points, (
        "transfers now help — the churn problem may be fixed; update this test "
        "and docs/model-results.md"
    )
