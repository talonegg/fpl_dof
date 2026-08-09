"""Expected points for a season that has not started yet.

Composes the pieces the design calls for: blended career rates, a minutes
forecast, clean sheets from the club a player will actually play *for*, and
the opening run's fixtures weighted so the near ones matter most.

Structured like :mod:`fpl.models.components` — score each way of earning points
separately and add them up — but with every input drawn from prior seasons
rather than the current one, because there is no current one.

**Minutes come first and dominate.** Measured: a version of this using per-90
rates with a constant minutes assumption scored below a randomly chosen legal
squad, because it bought high-rate players who do not play. Every other
component here is a refinement on top of getting minutes approximately right.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from fpl.domain.positions import canonical_position
from fpl.features.career import finishing_multiplier
from fpl.features.team_strength import clean_sheet_probability, expected_concession
from fpl.models.minutes_forecast import PreseasonMinutes

MINUTES_PER_MATCH = 90
FULL_APPEARANCE_MINUTES = 60

# FPL scoring. Not parameters -- the game's rules.
GOAL_POINTS = {"GK": 6, "DEF": 6, "MID": 5, "FWD": 4}
CLEAN_SHEET_POINTS = {"GK": 4, "DEF": 4, "MID": 1, "FWD": 0}
ASSIST_POINTS = 3
APPEARANCE_FULL = 2
APPEARANCE_PARTIAL = 1

# Fixture weighting across the opening run. 0.78 puts gameweek 1 at 1.00 and
# gameweek 10 at 0.11 -- roughly a ninth -- which satisfies "immaterial after
# 10" without a cliff that would make the squad flip on a single fixture.
DEFAULT_FIXTURE_DECAY = 0.78
DEFAULT_HORIZON = 10

# A league-average opponent, used when a fixture list is not supplied.
NEUTRAL_OPPONENT_XG = 1.4


def fixture_weights(
    horizon: int = DEFAULT_HORIZON, decay: float = DEFAULT_FIXTURE_DECAY
) -> list[float]:
    """Weight per gameweek of the opening run, near fixtures highest.

    Truncated hard after ``horizon`` because the requirement is explicit that
    later gameweeks should not influence the opening squad. The decay has
    already done most of the work by then, so the truncation is not a cliff.
    """
    return [decay**index for index in range(horizon)]


@dataclass
class PreseasonPredictor:
    """Expected points over the opening run, from prior seasons alone.

    ``team_defence`` is the blended per-club concession table from
    :mod:`fpl.features.team_strength`. It is looked up by the club a player
    will play for *next* season, which for a transfer is not where their own
    record was earned.
    """

    horizon: int = DEFAULT_HORIZON
    fixture_decay: float = DEFAULT_FIXTURE_DECAY
    minutes: PreseasonMinutes = field(default_factory=PreseasonMinutes)
    team_defence: pd.DataFrame = field(default_factory=pd.DataFrame)
    promoted_prior: float = 1.75
    use_finishing_adjustment: bool = True

    @property
    def name(self) -> str:
        return f"Preseason({self.horizon})"

    def _clean_sheet_rate(self, clubs: pd.Series) -> pd.Series:
        """Per-club clean-sheet probability for the season ahead."""
        if self.team_defence.empty:
            return pd.Series(0.0, index=clubs.index)

        lookup = {
            club: clean_sheet_probability(
                expected_concession(club, self.team_defence, self.promoted_prior)
            )
            for club in clubs.dropna().unique()
        }
        return clubs.map(lookup).fillna(clean_sheet_probability(self.promoted_prior))

    def predict(
        self, career: pd.DataFrame, fixture_difficulty: pd.Series | None = None
    ) -> pd.DataFrame:
        """Expected points over the opening run for each player.

        ``career`` is the output of ``blend_career_rates`` joined to the
        current price list, so it carries both the historical rates and the
        club the player will play for.

        ``fixture_difficulty`` optionally supplies a per-player multiplier for
        the opening run — a club facing an easy start scores above its
        baseline.

        **Measured, and it does not point the same way twice.** See §10 of
        ``docs/season-opening-squad.md`` for the per-season table. Opening-run
        difficulty spans a narrow range across clubs, so it moves the ranking
        by about as much as it adds noise, and three seasons cannot separate
        those. Off by default, and the caller opts in.
        """
        if career.empty:
            return pd.DataFrame(columns=["element", "expected_points"])

        df = career.copy()
        positions = df["position"].map(canonical_position)

        forecast = self.minutes.forecast(df)
        if forecast.empty:
            return pd.DataFrame(columns=["element", "expected_points"])

        expected_minutes = forecast["expected_minutes"].to_numpy()
        start_probability = forecast["start_probability"].to_numpy()
        share = pd.Series(expected_minutes, index=df.index) / MINUTES_PER_MATCH
        starts = pd.Series(start_probability, index=df.index)

        def rate(column: str) -> pd.Series:
            values = df.get(f"{column}_per_90")
            return values.fillna(0.0) if values is not None else pd.Series(0.0, index=df.index)

        # Appearance points: two for a full appearance, one for a cameo.
        appearance = starts * APPEARANCE_FULL + (share - starts).clip(lower=0) * (
            APPEARANCE_PARTIAL
        )

        goal_value = positions.map(GOAL_POINTS).fillna(0)
        clean_sheet_value = positions.map(CLEAN_SHEET_POINTS).fillna(0)

        goals = rate("expected_goals") * share
        if self.use_finishing_adjustment:
            goals = goals * finishing_multiplier(df).fillna(1.0)

        assists = rate("expected_assists") * share * ASSIST_POINTS
        bonus = rate("bonus") * share

        clean_sheets = pd.Series(0.0, index=df.index)
        if "team" in df.columns:
            # Clean sheets pay only to a player who completes 60 minutes, and
            # they follow the new club rather than the player's own record.
            clean_sheets = self._clean_sheet_rate(df["team"]) * starts * clean_sheet_value

        per_match = appearance + goals * goal_value + assists + clean_sheets + bonus

        weights = fixture_weights(self.horizon, self.fixture_decay)
        weighted_matches = sum(weights)

        expected = per_match * weighted_matches
        if fixture_difficulty is not None:
            expected = expected * fixture_difficulty.reindex(df.index).fillna(1.0)

        return pd.DataFrame(
            {
                "element": df.get("element", pd.Series(range(len(df)), index=df.index)),
                "expected_points": expected.clip(lower=0),
                "expected_minutes": expected_minutes,
            }
        )
