"""The set of models every new predictor is measured against.

Adding a model to ``fpl/models/`` is not enough to make it usable. It has to
appear in a comparison table beside these, and beat ``NaiveFormPredictor`` on
held-out gameweeks, before it gets wired into the UI.
"""

from __future__ import annotations

from fpl.models.base import Predictor
from fpl.models.naive import NaiveFormPredictor, SeasonMeanPredictor, ZeroPredictor


def baseline_predictors() -> list[Predictor]:
    """The standard comparison set, cheapest first."""
    return [
        ZeroPredictor(),
        SeasonMeanPredictor(),
        NaiveFormPredictor(window=3),
        NaiveFormPredictor(window=5),
        NaiveFormPredictor(window=10),
    ]


# The bar. A model that does not beat this is not ready for the UI.
#
# The roadmap assumed this would be NaiveFormPredictor -- "recent form
# continues" being what a human does by eye. Backtesting 2025-26 says
# otherwise: SeasonMean picks a better top 15 than any form window tried
# (4.25 points per pick against 3.85 for a 5-gameweek window), beating it in
# 20 of 33 gameweeks. So the season mean is the bar.
#
# Treat that as suggestive rather than settled -- it is one season, and 20 of
# 33 is a modest edge. It is a strong enough hint to raise the bar, not strong
# enough to conclude that recency is worthless.
BENCHMARK = SeasonMeanPredictor()
