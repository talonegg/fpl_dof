"""Evaluating across several seasons.

One season is 33 scored gameweeks, and every interesting difference between
models in ``docs/model-results.md`` sat inside the noise band at that sample
size. More seasons is the only way past that -- not because the models change,
but because the evidence about them stops being ambiguous.

Two things make this more than a loop:

**Seasons are not interchangeable.** The archive gained expected goals in
2022-23 and defensive contributions in 2025-26, and the older files are not
even UTF-8. A model asked to run on a season lacking its inputs does not fail,
it silently gets worse -- so :func:`season_capabilities` reports what each
season can support, and callers select rather than assume.

**Element ids are season-scoped.** Player 233 in 2022-23 is not player 233 in
2025-26. So each season is replayed independently and only the *metrics* are
pooled, never the rows.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from fpl.backtest.harness import DEFAULT_FIRST_GAMEWEEK, replay
from fpl.backtest.metrics import DEFAULT_TOP_N, evaluate_by_gameweek
from fpl.models.base import Predictor
from fpl.sources.archive import fetch_season_gameweeks

# Seasons the archive publishes with per-gameweek data.
ALL_SEASONS = (
    "2016-17",
    "2017-18",
    "2018-19",
    "2019-20",
    "2020-21",
    "2021-22",
    "2022-23",
    "2023-24",
    "2024-25",
    "2025-26",
)

# What a model needs present to be worth running at all.
REQUIRED_FOR_ANY_MODEL = ("element", "gameweek", "total_points", "minutes")
REQUIRED_FOR_COMPONENTS = ("position",)
REQUIRED_FOR_EXPECTED_GOALS = ("expected_goals", "expected_assists")


@dataclass(frozen=True)
class SeasonCapability:
    """What a season's data can and cannot support."""

    season: str
    rows: int
    gameweeks: int
    supports_basic: bool
    supports_components: bool
    supports_expected_goals: bool
    missing: tuple[str, ...]


def season_capabilities(
    season_data: dict[str, pd.DataFrame],
) -> list[SeasonCapability]:
    """Report what each loaded season supports."""
    capabilities = []
    for season, frame in season_data.items():
        columns = set(frame.columns)
        missing = tuple(
            column
            for group in (
                REQUIRED_FOR_ANY_MODEL,
                REQUIRED_FOR_COMPONENTS,
                REQUIRED_FOR_EXPECTED_GOALS,
            )
            for column in group
            if column not in columns
        )
        capabilities.append(
            SeasonCapability(
                season=season,
                rows=len(frame),
                gameweeks=int(frame["gameweek"].nunique()) if "gameweek" in columns else 0,
                supports_basic=all(c in columns for c in REQUIRED_FOR_ANY_MODEL),
                supports_components=all(
                    c in columns for c in (*REQUIRED_FOR_ANY_MODEL, *REQUIRED_FOR_COMPONENTS)
                ),
                supports_expected_goals=all(c in columns for c in REQUIRED_FOR_EXPECTED_GOALS),
                missing=missing,
            )
        )
    return capabilities


def load_seasons(seasons: tuple[str, ...] = ALL_SEASONS) -> dict[str, pd.DataFrame]:
    """Fetch several seasons, skipping any that cannot be read.

    A season that fails to download or parse is omitted with its name recorded
    in the result's absence, rather than taking the whole evaluation down.
    """
    loaded = {}
    for season in seasons:
        try:
            loaded[season] = fetch_season_gameweeks(season)
        except Exception:  # noqa: BLE001 - one bad season must not stop the rest
            continue
    return loaded


def replay_many(
    season_data: dict[str, pd.DataFrame],
    predictor: Predictor,
    first_gameweek: int = DEFAULT_FIRST_GAMEWEEK,
    top_n: int = DEFAULT_TOP_N,
) -> pd.DataFrame:
    """Per-gameweek metrics for one predictor across seasons.

    Returns one row per (season, gameweek). Pooling at this level rather than
    concatenating raw rows is what keeps season-scoped element ids from
    colliding.
    """
    frames = []
    for season, data in season_data.items():
        result = replay(data, predictor, first_gameweek=first_gameweek)
        if result.predictions.empty:
            continue
        per_gameweek = evaluate_by_gameweek(result.predictions, top_n)
        if per_gameweek.empty:
            continue
        per_gameweek = per_gameweek.reset_index()
        per_gameweek["season"] = season
        per_gameweek["model"] = predictor.name
        frames.append(per_gameweek)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def compare_many(
    season_data: dict[str, pd.DataFrame],
    predictors: list[Predictor],
    first_gameweek: int = DEFAULT_FIRST_GAMEWEEK,
    top_n: int = DEFAULT_TOP_N,
) -> pd.DataFrame:
    """Every predictor's per-(season, gameweek) metrics, stacked."""
    frames = [
        replay_many(season_data, predictor, first_gameweek, top_n) for predictor in predictors
    ]
    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def summarise_many(per_gameweek: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Model-level summary of a metric, with its per-season spread.

    ``std_across_seasons`` is the column to read second: a model that wins on
    the mean while swinging wildly between seasons has not been shown to be
    better, only luckier somewhere.
    """
    if per_gameweek.empty or metric not in per_gameweek.columns:
        return pd.DataFrame()

    by_season = per_gameweek.groupby(["model", "season"])[metric].mean()
    return (
        by_season.groupby("model")
        .agg(["mean", "std", "min", "max", "count"])
        .rename(
            columns={
                "mean": metric,
                "std": "std_across_seasons",
                "min": "worst_season",
                "max": "best_season",
                "count": "seasons",
            }
        )
        .sort_values(metric, ascending=False)
    )
