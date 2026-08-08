"""Evaluating across several seasons.

One season is 33 scored gameweeks, and every interesting difference between
models in ``docs/model-results.md`` sat inside the noise band at that sample
size. More seasons is the only way past that -- not because the models change,
but because the evidence about them stops being ambiguous.

Two things make this more than a loop:

**Seasons are not interchangeable, and the deeper problem is not missing
columns but changed rules.** The archive gained expected goals in 2022-23 and
the older files are not even UTF-8, which is merely inconvenient. Defensive
contributions are different: they were introduced in 2025-26 and continue into
2026-27, so seasons before that were *played under different scoring*. A model
evaluated on them is being judged at a game that is no longer the one being
played. :func:`season_capabilities` reports this per season, and
``matches_current_rules`` marks the ones that still count in full.

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

# Seasons worth downloading for an evaluation. Expected goals first appear in
# 2022-23, and every model in fpl/models/ that is not a pure baseline needs
# them, so the six earlier seasons cost minutes of download and contribute
# nothing. Widen this deliberately if a model appears that can use them.
EVALUATION_SEASONS = ("2022-23", "2023-24", "2024-25", "2025-26")

# What a model needs present to be worth running at all.
REQUIRED_FOR_ANY_MODEL = ("element", "gameweek", "total_points", "minutes")
REQUIRED_FOR_COMPONENTS = ("position",)
REQUIRED_FOR_EXPECTED_GOALS = ("expected_goals", "expected_assists")
REQUIRED_FOR_DEFENSIVE_CONTRIBUTIONS = ("defensive_contribution",)

# Defensive contribution points were introduced in 2025-26 and continue in
# 2026-27. Earlier seasons were played under rules with no such route to
# points at all -- the column is not merely missing from the archive, the
# scoring did not exist.
DEFENSIVE_CONTRIBUTIONS_FROM = "2025-26"

# The season whose rules match the one currently being played. Evidence from
# it is worth more than evidence from seasons scored under superseded rules.
CURRENT_RULES_SEASON = "2025-26"


@dataclass(frozen=True)
class SeasonCapability:
    """What a season's data can and cannot support."""

    season: str
    rows: int
    gameweeks: int
    supports_basic: bool
    supports_components: bool
    supports_expected_goals: bool
    supports_defensive_contributions: bool
    missing: tuple[str, ...]

    @property
    def matches_current_rules(self) -> bool:
        """Whether this season was scored under the rules now in force.

        Defensive contributions are the difference. A season without them is
        not just missing a column -- it was played under scoring where that
        route to points did not exist, so a model's performance on it says
        less about the season being played now.
        """
        return self.supports_defensive_contributions


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
                supports_defensive_contributions=all(
                    c in columns for c in REQUIRED_FOR_DEFENSIVE_CONTRIBUTIONS
                ),
                missing=missing,
            )
        )
    return capabilities


@dataclass
class SeasonLoad:
    """Seasons that loaded, and the ones that did not.

    Failures are carried rather than swallowed. A transient download error
    silently dropping a season shrinks the evidence base without changing
    anything visible in the numbers -- which is how a four-season conclusion
    quietly becomes a three-season one.
    """

    seasons: dict[str, pd.DataFrame]
    failures: dict[str, str]

    def __iter__(self):
        return iter(self.seasons)

    def __len__(self) -> int:
        return len(self.seasons)

    def __getitem__(self, season: str) -> pd.DataFrame:
        return self.seasons[season]

    def items(self):
        return self.seasons.items()


def load_seasons(seasons: tuple[str, ...] = ALL_SEASONS) -> SeasonLoad:
    """Fetch several seasons, recording rather than hiding any that fail.

    One unreadable season must not stop the rest -- older files have their own
    problems -- but the caller has to be told, because the alternative is an
    evaluation that reports fewer seasons than it claims.
    """
    loaded: dict[str, pd.DataFrame] = {}
    failures: dict[str, str] = {}
    for season in seasons:
        try:
            loaded[season] = fetch_season_gameweeks(season)
        except Exception as error:  # noqa: BLE001 - one bad season must not stop the rest
            failures[season] = f"{type(error).__name__}: {error}"
    return SeasonLoad(seasons=loaded, failures=failures)


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
