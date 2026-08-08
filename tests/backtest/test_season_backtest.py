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


def test_transferring_on_a_volatile_predictor_still_costs_more_than_it_gains(season):
    """Documents the finding, so a change that fixes it fails loudly here.

    The transfer planner scales a single gameweek's edge by the horizon, which
    assumes the edge persists. It does not — predictions move week to week — so
    with a responsive predictor the planner churns and pays hits for noise.

    Note this is *predictor-specific*: the same planner beats holding when fed
    the stable season mean. The earlier blanket claim that transferring always
    loses was partly an artefact of the free-transfer accounting bug.
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
        "transfers now help even the volatile predictor — the churn problem may "
        "be fixed; update this test, docs/optimiser-results.md and CLAUDE.md"
    )


def test_transferring_on_a_stable_predictor_pays(season):
    """The other half of the finding, which the earlier claim missed."""
    from fpl.backtest.season import simulate_season
    from fpl.models.naive import SeasonMeanPredictor

    holding = simulate_season(
        season, SeasonMeanPredictor(), first_gameweek=6, last_gameweek=20, horizon=0
    )
    transferring = simulate_season(
        season, SeasonMeanPredictor(), first_gameweek=6, last_gameweek=20, horizon=5
    )

    assert transferring.total_points > holding.total_points
    assert transferring.total_hits <= 8, "should be paying very few hits"


# --- Across seasons: the evaluation that settles what one season could not ---


@pytest.fixture(scope="module")
def multi_season():
    from fpl.backtest.seasons import load_seasons, season_capabilities

    load = load_seasons(("2022-23", "2023-24", "2024-25", "2025-26"))
    if load.failures:
        # A silently dropped season would weaken every assertion below without
        # failing any of them.
        pytest.skip(f"could not load seasons: {load.failures}")
    capabilities = season_capabilities(load.seasons)
    usable = {c.season for c in capabilities if c.supports_components}
    return {season: frame for season, frame in load.items() if season in usable}


def test_four_seasons_are_usable_for_the_component_model(multi_season):
    assert len(multi_season) == 4


def test_older_seasons_are_readable_despite_not_being_utf8():
    """2016-17 onwards are latin-1; a naive read raises UnicodeDecodeError."""
    from fpl.sources.archive import fetch_season_gameweeks

    season = fetch_season_gameweeks("2016-17")

    assert len(season) > 10000


def test_pooling_seasons_multiplies_the_paired_observations(multi_season):
    """The whole reason for multi-season evaluation."""
    from fpl.backtest.seasons import compare_many
    from fpl.backtest.significance import compare_across_seasons
    from fpl.models.components import ComponentPredictor
    from fpl.models.naive import SeasonMeanPredictor

    per_gameweek = compare_many(multi_season, [SeasonMeanPredictor(), ComponentPredictor(4)])
    comparison = compare_across_seasons(per_gameweek, "Component(4)", "SeasonMean")

    assert comparison.gameweeks > 100, "one season gives 33; four should give ~130"


def test_the_season_mean_still_picks_the_best_top_fifteen(multi_season):
    """Pins the finding. If a model finally beats it, this fails and says so."""
    from fpl.backtest.baselines import all_predictors
    from fpl.backtest.seasons import compare_many, summarise_many

    per_gameweek = compare_many(multi_season, all_predictors())
    summary = summarise_many(per_gameweek, "top_15_mean_actual")

    assert summary.index[0] == "SeasonMean", (
        f"{summary.index[0]} now out-picks the benchmark — update "
        "docs/multi-season-results.md, CLAUDE.md and fpl/backtest/baselines.py"
    )


def test_ranking_skill_is_inverted_against_selection_skill(multi_season):
    """The central finding: the worst ranker is the best selector."""
    from fpl.backtest.baselines import all_predictors
    from fpl.backtest.seasons import compare_many, summarise_many

    per_gameweek = compare_many(multi_season, all_predictors())
    ranking = summarise_many(per_gameweek, "rank_correlation")
    selection = summarise_many(per_gameweek, "top_15_mean_actual")

    # Best selector, worst ranker (Zero has no ranking, so exclude it).
    assert selection.index[0] == "SeasonMean"
    assert ranking.dropna().index[-1] == "SeasonMean"
