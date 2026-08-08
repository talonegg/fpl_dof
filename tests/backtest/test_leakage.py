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

    def predict(self, history: pd.DataFrame, gameweek: int) -> pd.DataFrame:
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

        def predict(self, history: pd.DataFrame, gameweek: int) -> pd.DataFrame:
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
