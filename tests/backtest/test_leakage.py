"""Proof that the harness cannot leak the future into a prediction.

These are the most important tests in the repository. Lookahead does not
crash and does not look wrong -- it just makes a model appear excellent in
testing and useless in August. Every other result depends on these holding.

Two independent checks, because one could pass while the other fails:

*The spy* watches what the predictor is actually handed.
*The canary* corrupts the future and asserts the predictions do not move.
"""

from __future__ import annotations

import pandas as pd

from fpl.backtest.harness import replay
from fpl.models.base import empty_predictions
from fpl.models.naive import NaiveFormPredictor

SEASON = pd.DataFrame(
    [
        {"element": element, "gameweek": gameweek, "total_points": gameweek + element}
        for element in (1, 2, 3)
        for gameweek in range(1, 11)
    ]
)


class SpyPredictor:
    """Records the highest gameweek it was ever shown."""

    name = "Spy"

    def __init__(self):
        self.seen = []

    def predict(self, history: pd.DataFrame, gameweek: int, fixtures=None) -> pd.DataFrame:
        self.seen.append((gameweek, history["gameweek"].max()))
        return empty_predictions()


def test_the_predictor_never_sees_the_gameweek_it_is_predicting():
    spy = SpyPredictor()

    replay(SEASON, spy, first_gameweek=3)

    for target, highest_seen in spy.seen:
        assert highest_seen < target, (
            f"predicting GW{target} but was shown data up to GW{highest_seen}"
        )


def test_the_predictor_sees_everything_before_the_target():
    """The flip side: withholding usable history would understate a model."""
    spy = SpyPredictor()

    replay(SEASON, spy, first_gameweek=3)

    for target, highest_seen in spy.seen:
        assert highest_seen == target - 1


def test_corrupting_the_future_does_not_change_any_prediction():
    """The canary. If a model can see ahead, this moves the numbers."""
    predictor = NaiveFormPredictor(window=5)
    honest = replay(SEASON, predictor, first_gameweek=3)

    corrupted_season = SEASON.copy()
    # Make gameweeks 6+ absurd. Anything that peeks will notice.
    future = corrupted_season["gameweek"] >= 6
    corrupted_season.loc[future, "total_points"] = 9999

    corrupted = replay(corrupted_season, predictor, first_gameweek=3)

    honest_early = honest.predictions[honest.predictions["gameweek"] < 6]
    corrupted_early = corrupted.predictions[corrupted.predictions["gameweek"] < 6]

    pd.testing.assert_series_equal(
        honest_early["expected_points"].reset_index(drop=True),
        corrupted_early["expected_points"].reset_index(drop=True),
    )


def test_the_canary_would_actually_catch_a_cheating_model():
    """A test that never fails proves nothing -- so verify it can fail."""

    class CheatingPredictor:
        name = "Cheat"

        def __init__(self, season):
            self.season = season

        def predict(self, history: pd.DataFrame, gameweek: int, fixtures=None) -> pd.DataFrame:
            # Reads the answer straight from the full season.
            actual = self.season[self.season["gameweek"] == gameweek]
            return actual[["element", "total_points"]].rename(
                columns={"total_points": "expected_points"}
            )

    honest = replay(SEASON, CheatingPredictor(SEASON), first_gameweek=3)

    corrupted_season = SEASON.copy()
    corrupted_season.loc[corrupted_season["gameweek"] >= 6, "total_points"] = 9999
    corrupted = replay(corrupted_season, CheatingPredictor(corrupted_season), first_gameweek=3)

    honest_points = honest.predictions["expected_points"].tolist()
    corrupted_points = corrupted.predictions["expected_points"].tolist()

    assert honest_points != corrupted_points, (
        "the canary failed to detect a model that reads the answer"
    )


# --- The fixture frame: the one thing a model is told about the target week ---

SEASON_WITH_FIXTURES = pd.DataFrame(
    [
        {
            "element": element,
            "gameweek": gameweek,
            "total_points": gameweek + element,
            "minutes": 90,
            "opponent_team": 20 - element,
            "was_home": gameweek % 2 == 0,
        }
        for element in (1, 2, 3)
        for gameweek in range(1, 11)
    ]
)


def test_the_fixture_frame_exposes_no_outcomes():
    """Who you play is knowable at the deadline. What you scored is not."""
    from fpl.backtest.harness import known_fixtures, prepare_season

    fixtures = known_fixtures(prepare_season(SEASON_WITH_FIXTURES), gameweek=5)

    assert set(fixtures.columns) == {"element", "opponent_team", "was_home"}
    for outcome in ("total_points", "minutes", "goals_scored", "bonus"):
        assert outcome not in fixtures.columns


def test_the_fixture_frame_is_an_allow_list_not_a_deny_list():
    """A new archive column must not silently become available to models."""
    from fpl.backtest.harness import known_fixtures, prepare_season

    season = SEASON_WITH_FIXTURES.copy()
    season["some_new_outcome_column"] = 999

    fixtures = known_fixtures(prepare_season(season), gameweek=5)

    assert "some_new_outcome_column" not in fixtures.columns


def test_the_fixture_frame_describes_the_target_gameweek():
    from fpl.backtest.harness import known_fixtures, prepare_season

    season = prepare_season(SEASON_WITH_FIXTURES)
    fixtures = known_fixtures(season, gameweek=5)

    assert sorted(fixtures["element"]) == [1, 2, 3]


class FixtureSpy:
    """Records every column it was offered for the target gameweek."""

    name = "FixtureSpy"

    def __init__(self):
        self.columns_seen = set()

    def predict(self, history, gameweek, fixtures=None):
        if fixtures is not None:
            self.columns_seen |= set(fixtures.columns)
        return empty_predictions()


def test_a_model_is_never_offered_an_outcome_column_for_the_target_week():
    spy = FixtureSpy()

    replay(SEASON_WITH_FIXTURES, spy, first_gameweek=3)

    assert spy.columns_seen == {"element", "opponent_team", "was_home"}


def test_opponent_adjustment_does_not_react_to_a_corrupted_future():
    """The canary again, now for the fixture-aware model."""
    from fpl.models.minutes import OpponentAdjustedPredictor

    predictor = OpponentAdjustedPredictor()
    honest = replay(SEASON_WITH_FIXTURES, predictor, first_gameweek=3)

    corrupted = SEASON_WITH_FIXTURES.copy()
    corrupted.loc[corrupted["gameweek"] >= 6, "total_points"] = 9999

    tainted = replay(corrupted, predictor, first_gameweek=3)

    honest_early = honest.predictions[honest.predictions["gameweek"] < 6]
    tainted_early = tainted.predictions[tainted.predictions["gameweek"] < 6]

    pd.testing.assert_series_equal(
        honest_early["expected_points"].reset_index(drop=True),
        tainted_early["expected_points"].reset_index(drop=True),
    )
