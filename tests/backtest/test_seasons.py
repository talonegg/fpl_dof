"""Tests for multi-season evaluation.

The two things that must not go wrong: season-scoped element ids must never be
pooled together, and a season missing the columns a model needs must be
reported rather than quietly producing worse numbers.
"""

from __future__ import annotations

import pandas as pd
import pytest

from fpl.backtest.seasons import (
    compare_many,
    replay_many,
    season_capabilities,
    summarise_many,
)
from fpl.backtest.significance import compare_across_seasons, season_comparison_table
from fpl.models.naive import NaiveFormPredictor, SeasonMeanPredictor


def make_season(offset=0, gameweeks=12, with_position=True, with_expected=True):
    rows = []
    for element in range(1, 21):
        for gameweek in range(1, gameweeks + 1):
            row = {
                "element": element,
                "gameweek": gameweek,
                "minutes": 90,
                "total_points": (element + offset) % 9,
                "opponent_team": element % 5,
            }
            if with_position:
                row["position"] = ["GK", "DEF", "MID", "FWD"][element % 4]
            if with_expected:
                row["expected_goals"] = 0.1
                row["expected_assists"] = 0.1
            rows.append(row)
    return pd.DataFrame(rows)


SEASONS = {"2023-24": make_season(offset=0), "2024-25": make_season(offset=3)}


def test_capabilities_report_a_fully_equipped_season():
    capability = season_capabilities({"2025-26": make_season()})[0]

    assert capability.supports_basic
    assert capability.supports_components
    assert capability.supports_expected_goals


def test_a_season_without_positions_cannot_support_the_component_model():
    """Older archive seasons lack this, and the model would silently degrade."""
    capability = season_capabilities({"2019-20": make_season(with_position=False)})[0]

    assert capability.supports_basic
    assert not capability.supports_components
    assert "position" in capability.missing


def test_a_season_without_expected_goals_is_flagged():
    capability = season_capabilities({"2021-22": make_season(with_expected=False)})[0]

    assert not capability.supports_expected_goals
    assert "expected_goals" in capability.missing


def test_capabilities_report_size():
    capability = season_capabilities({"2024-25": make_season(gameweeks=12)})[0]

    assert capability.gameweeks == 12
    assert capability.rows == 240


def test_replay_many_returns_a_row_per_season_and_gameweek():
    result = replay_many(SEASONS, SeasonMeanPredictor(), first_gameweek=6)

    assert set(result["season"]) == {"2023-24", "2024-25"}
    assert not result.duplicated(subset=["season", "gameweek"]).any()


def test_replay_many_labels_the_model():
    result = replay_many(SEASONS, NaiveFormPredictor(window=3), first_gameweek=6)

    assert set(result["model"]) == {"NaiveForm(3)"}


def test_replay_many_of_no_seasons_is_empty():
    assert replay_many({}, SeasonMeanPredictor()).empty


def test_compare_many_stacks_every_model():
    result = compare_many(SEASONS, [SeasonMeanPredictor(), NaiveFormPredictor()], first_gameweek=6)

    assert set(result["model"]) == {"SeasonMean", "NaiveForm(5)"}


def test_summarise_reports_spread_across_seasons():
    per_gameweek = compare_many(SEASONS, [SeasonMeanPredictor()], first_gameweek=6)

    summary = summarise_many(per_gameweek, "top_15_mean_actual")

    assert summary.loc["SeasonMean", "seasons"] == 2
    assert "std_across_seasons" in summary.columns


def test_summarise_of_an_unknown_metric_is_empty():
    per_gameweek = compare_many(SEASONS, [SeasonMeanPredictor()], first_gameweek=6)

    assert summarise_many(per_gameweek, "not_a_metric").empty


def test_comparing_a_model_with_itself_shows_no_difference():
    per_gameweek = compare_many(
        SEASONS, [SeasonMeanPredictor(), NaiveFormPredictor()], first_gameweek=6
    )

    result = compare_across_seasons(per_gameweek, "SeasonMean", "SeasonMean")

    assert result.mean_difference == 0.0


def test_pairing_is_within_a_season_not_across_seasons():
    """Pairing a 2023 gameweek against a 2024 one would defeat the whole point."""
    per_gameweek = compare_many(
        SEASONS, [SeasonMeanPredictor(), NaiveFormPredictor()], first_gameweek=6
    )

    result = compare_across_seasons(per_gameweek, "NaiveForm(5)", "SeasonMean")

    per_season = per_gameweek[per_gameweek["model"] == "SeasonMean"]
    assert result.gameweeks == len(per_season)


def test_more_seasons_means_more_paired_observations():
    """The entire reason for this module."""
    one = compare_many(
        {"2023-24": SEASONS["2023-24"]},
        [SeasonMeanPredictor(), NaiveFormPredictor()],
        first_gameweek=6,
    )
    two = compare_many(SEASONS, [SeasonMeanPredictor(), NaiveFormPredictor()], first_gameweek=6)

    single = compare_across_seasons(one, "NaiveForm(5)", "SeasonMean")
    both = compare_across_seasons(two, "NaiveForm(5)", "SeasonMean")

    assert both.gameweeks == 2 * single.gameweeks


def test_the_benchmark_is_excluded_from_its_own_table():
    per_gameweek = compare_many(
        SEASONS, [SeasonMeanPredictor(), NaiveFormPredictor()], first_gameweek=6
    )

    table = season_comparison_table(per_gameweek, "SeasonMean")

    assert "SeasonMean" not in table.index


def test_a_season_table_of_nothing_is_empty():
    assert season_comparison_table(pd.DataFrame(), "SeasonMean").empty


def test_comparing_across_an_empty_frame_is_safe():
    result = compare_across_seasons(pd.DataFrame(), "A", "B")

    assert result.gameweeks == 0
    assert pd.isna(result.t_statistic)


@pytest.mark.parametrize("metric", ["mae", "rank_correlation"])
def test_any_metric_can_be_compared(metric):
    per_gameweek = compare_many(
        SEASONS, [SeasonMeanPredictor(), NaiveFormPredictor()], first_gameweek=6
    )

    result = compare_across_seasons(per_gameweek, "NaiveForm(5)", "SeasonMean", metric)

    assert result.metric == metric
    assert result.gameweeks > 0


# --- Scoring rules changed, which makes seasons more than "missing a column" ---


def make_season_with_defcon(**kwargs):
    season = make_season(**kwargs)
    season["defensive_contribution"] = 8
    return season


def test_a_season_with_defensive_contributions_matches_current_rules():
    """Defcons arrived in 2025-26 and continue in 2026-27."""
    capability = season_capabilities({"2025-26": make_season_with_defcon()})[0]

    assert capability.supports_defensive_contributions
    assert capability.matches_current_rules


def test_an_older_season_does_not_match_current_rules():
    """Not merely a missing column — that route to points did not exist."""
    capability = season_capabilities({"2023-24": make_season()})[0]

    assert not capability.supports_defensive_contributions
    assert not capability.matches_current_rules


def test_a_season_can_support_the_component_model_yet_predate_the_rules():
    """The distinction that matters: usable for evaluation, but not current."""
    capability = season_capabilities({"2023-24": make_season()})[0]

    assert capability.supports_components
    assert capability.supports_expected_goals
    assert not capability.matches_current_rules


# --- A season that fails to load must be reported, never silently dropped ---


def test_a_successful_load_reports_no_failures(monkeypatch):
    from fpl.backtest import seasons as seasons_module

    monkeypatch.setattr(seasons_module, "fetch_season_gameweeks", lambda season: make_season())

    load = seasons_module.load_seasons(("2023-24", "2024-25"))

    assert len(load) == 2
    assert load.failures == {}


def test_a_failed_season_is_recorded_rather_than_swallowed(monkeypatch):
    """A dropped season shrinks the evidence base without changing the numbers."""
    from fpl.backtest import seasons as seasons_module

    def flaky(season):
        if season == "2024-25":
            raise OSError("connection reset")
        return make_season()

    monkeypatch.setattr(seasons_module, "fetch_season_gameweeks", flaky)

    load = seasons_module.load_seasons(("2023-24", "2024-25"))

    assert set(load.seasons) == {"2023-24"}
    assert "2024-25" in load.failures
    assert "connection reset" in load.failures["2024-25"]


def test_one_bad_season_does_not_stop_the_others(monkeypatch):
    from fpl.backtest import seasons as seasons_module

    def flaky(season):
        if season == "2022-23":
            raise ValueError("bad file")
        return make_season()

    monkeypatch.setattr(seasons_module, "fetch_season_gameweeks", flaky)

    load = seasons_module.load_seasons(("2022-23", "2023-24", "2024-25"))

    assert len(load) == 2


def test_the_load_behaves_like_the_mapping_callers_expect(monkeypatch):
    from fpl.backtest import seasons as seasons_module

    monkeypatch.setattr(seasons_module, "fetch_season_gameweeks", lambda season: make_season())

    load = seasons_module.load_seasons(("2023-24",))

    assert list(load) == ["2023-24"]
    assert len(load["2023-24"]) > 0
    assert dict(load.items())


def test_error_metrics_are_sorted_best_first(monkeypatch):
    """Sorting MAE descending puts the least accurate model top and reads as a ranking."""
    from fpl.backtest import seasons as seasons_module

    monkeypatch.setattr(seasons_module, "fetch_season_gameweeks", lambda season: make_season())
    per_gameweek = compare_many(
        SEASONS, [SeasonMeanPredictor(), NaiveFormPredictor()], first_gameweek=6
    )

    by_error = summarise_many(per_gameweek, "mae")
    by_selection = summarise_many(per_gameweek, "top_15_mean_actual")

    assert by_error["mae"].is_monotonic_increasing, "lower error must come first"
    assert by_selection["top_15_mean_actual"].is_monotonic_decreasing
