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
