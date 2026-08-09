"""Model each way of scoring separately, then add them up.

The baselines and the minutes models all predict *points*, which is a single
noisy number produced by half a dozen unrelated processes. A defender's points
come from turning up, keeping a clean sheet, making defensive actions, and
occasionally scoring. Those have completely different dynamics, and averaging
them together throws away most of what is knowable.

So this predicts each component from the statistic that actually drives it:

- **Goals** from expected goals, not goals. A player's finishing is noisy;
  their shot volume and quality are much less so. This is the single biggest
  reason to prefer a component model.
- **Assists** from expected assists, for the same reason.
- **Clean sheets** from how often the opponent has conceded one, blended with
  the player's own team's record.
- **Appearance, saves, defensive contributions, bonus, cards, goals conceded**
  from their own historical rates.

Every rate is computed from the history the harness supplies, so the whole
thing stays point-in-time. Nothing here is fitted; the coefficients are the
FPL scoring rules, which are known exactly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from fpl.domain.positions import canonical_position
from fpl.models.base import empty_predictions

MINUTES_PER_MATCH = 90
DEFAULT_MINUTES_WINDOW = 4

# The FPL scoring rules. These are not parameters to tune -- they are the game.
GOAL_POINTS = {"GK": 6, "DEF": 6, "MID": 5, "FWD": 4}
CLEAN_SHEET_POINTS = {"GK": 4, "DEF": 4, "MID": 1, "FWD": 0}
ASSIST_POINTS = 3
APPEARANCE_POINTS_FULL = 2  # 60 minutes or more
APPEARANCE_POINTS_PARTIAL = 1
SAVES_PER_POINT = 3
GOALS_CONCEDED_PER_PENALTY = 2  # GK and DEF lose a point per two conceded
DEFENSIVE_CONTRIBUTION_POINTS = 2
YELLOW_CARD_POINTS = -1
RED_CARD_POINTS = -3

# Defensive actions needed for the contribution point. Defenders have a lower
# bar because clearances, blocks and interceptions come more naturally to them.
DEFENSIVE_CONTRIBUTION_THRESHOLD = {"DEF": 10, "MID": 12, "FWD": 12}

FULL_APPEARANCE_MINUTES = 60

# Position spelling varies by source; the vocabulary lives in the domain layer.


# Opponent adjustment is noisy off a handful of matches, so cap how far it can
# swing a clean-sheet probability.
MAX_OPPONENT_ADJUSTMENT = 0.5


def _rate_per_90(totals: pd.Series, minutes: pd.Series) -> pd.Series:
    """Per-90 rate, with no minutes giving 0 rather than infinity."""
    safe_minutes = minutes.replace(0, np.nan)
    return (totals * MINUTES_PER_MATCH / safe_minutes).fillna(0.0)


@dataclass
class ComponentPredictor:
    """Sum of separately-modelled FPL scoring components.

    ``use_expected_goals`` exists so the backtest can answer the question this
    whole model rests on: does predicting from xG actually beat predicting from
    goals? Setting it False swaps in actual goals and assists and changes
    nothing else, which makes the comparison clean.
    """

    minutes_window: int = DEFAULT_MINUTES_WINDOW
    use_expected_goals: bool = True
    # Optional: a dedicated minutes forecaster. Minutes are the one input with
    # complete data, so they are worth predicting properly rather than with a
    # trailing mean buried in here. None keeps the original behaviour.
    minutes_forecaster: object | None = None

    @property
    def name(self) -> str:
        suffix = "" if self.use_expected_goals else ", actuals"
        if self.minutes_forecaster is not None:
            suffix += f", {self.minutes_forecaster.name}"
        return f"Component({self.minutes_window}{suffix})"

    # -- component rates, all derived from history only ---------------------

    def _player_rates(self, history: pd.DataFrame) -> pd.DataFrame:
        """Per-90 rates for every scoring component, per player."""
        history = history.assign(
            _full_appearance=(history["minutes"] >= FULL_APPEARANCE_MINUTES).astype(int),
            position=history["position"].map(canonical_position),
        )
        aggregations = {
            "minutes": ("minutes", "sum"),
            "appearances": ("minutes", "size"),
            "full_appearances": ("_full_appearance", "sum"),
            "position": ("position", "last"),
        }
        for column in (
            "expected_goals",
            "expected_assists",
            "goals_scored",
            "assists",
            "saves",
            "bonus",
            "goals_conceded",
            "yellow_cards",
            "red_cards",
            "clean_sheets",
        ):
            if column in history.columns:
                aggregations[column] = (column, "sum")

        totals = history.groupby("element", as_index=False).agg(**aggregations)

        goal_source = "expected_goals" if self.use_expected_goals else "goals_scored"
        assist_source = "expected_assists" if self.use_expected_goals else "assists"

        rates = pd.DataFrame({"element": totals["element"]})
        rates["position"] = totals["position"]
        for name, source in (
            ("goal_rate", goal_source),
            ("assist_rate", assist_source),
            ("saves_rate", "saves"),
            ("bonus_rate", "bonus"),
            ("conceded_rate", "goals_conceded"),
            ("yellow_rate", "yellow_cards"),
            ("red_rate", "red_cards"),
        ):
            rates[name] = (
                _rate_per_90(totals[source], totals["minutes"]) if source in totals.columns else 0.0
            )

        # Conditioned on the matches the player actually completed, because the
        # archive only credits a clean sheet to someone who played 60 minutes.
        # Dividing by every appearance and then multiplying by the full-
        # appearance rate downstream would apply that 60-minute test twice and
        # halve the term for anyone rotated.
        if "clean_sheets" in totals.columns and "full_appearances" in totals.columns:
            rates["clean_sheet_rate"] = (
                totals["clean_sheets"] / totals["full_appearances"].replace(0, np.nan)
            ).fillna(0.0)
        else:
            rates["clean_sheet_rate"] = 0.0
        return rates

    def _expected_minutes(self, history: pd.DataFrame) -> pd.DataFrame:
        """Recent minutes, plus how often the player lasted a full appearance."""
        if self.minutes_forecaster is not None:
            forecast = self.minutes_forecaster.forecast(history, gameweek=0)
            if not forecast.empty:
                return forecast.rename(
                    columns={
                        "expected_minutes": "expected_minutes",
                        "start_probability": "full_appearance_rate",
                    }
                ).assign(
                    any_appearance_rate=lambda frame: frame["full_appearance_rate"].clip(
                        lower=0.0, upper=1.0
                    )
                )[["element", "expected_minutes", "full_appearance_rate", "any_appearance_rate"]]

        recent = (
            history.sort_values("gameweek")
            .groupby("element", as_index=False)
            .tail(self.minutes_window)
        )
        return recent.groupby("element", as_index=False).agg(
            expected_minutes=("minutes", "mean"),
            full_appearance_rate=(
                "minutes",
                lambda values: (values >= FULL_APPEARANCE_MINUTES).mean(),
            ),
            any_appearance_rate=("minutes", lambda values: (values > 0).mean()),
        )

    def _defensive_contribution_rate(self, history: pd.DataFrame) -> pd.Series:
        """How often each player clears their defensive-actions threshold.

        The archive stores the raw action count, not the points, so the
        threshold has to be applied per match and then averaged.
        """
        if "defensive_contribution" not in history.columns:
            return pd.Series(dtype="float64")

        positions = history["position"].map(canonical_position)
        thresholds = positions.map(DEFENSIVE_CONTRIBUTION_THRESHOLD)
        # Goalkeepers are not eligible; an unmapped position yields NaN, which
        # compares False and so contributes nothing.
        cleared = history["defensive_contribution"] >= thresholds
        return cleared.groupby(history["element"]).mean()

    def _opponent_clean_sheet_factor(self, history: pd.DataFrame) -> pd.Series:
        """How much easier or harder each opponent is to keep a clean sheet against.

        Measured as the clean-sheet rate achieved *against* that team relative
        to the league average, so it is derived entirely from history.
        """
        if "clean_sheets" not in history.columns or "opponent_team" not in history.columns:
            return pd.Series(dtype="float64")

        league_rate = history["clean_sheets"].mean()
        if not league_rate:
            return pd.Series(dtype="float64")

        by_opponent = history.groupby("opponent_team")["clean_sheets"].mean()
        return (by_opponent / league_rate).clip(
            1 - MAX_OPPONENT_ADJUSTMENT, 1 + MAX_OPPONENT_ADJUSTMENT
        )

    # -- assembly -----------------------------------------------------------

    def predict(
        self,
        history: pd.DataFrame,
        gameweek: int,
        fixtures: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        if history.empty or "position" not in history.columns:
            return empty_predictions()

        rates = self._player_rates(history)
        minutes = self._expected_minutes(history)
        combined = rates.merge(minutes, on="element", how="left").fillna(
            {"expected_minutes": 0.0, "full_appearance_rate": 0.0, "any_appearance_rate": 0.0}
        )

        contribution = self._defensive_contribution_rate(history)
        combined["contribution_rate"] = (
            combined["element"].map(contribution).fillna(0.0) if not contribution.empty else 0.0
        )

        clean_sheet_probability = combined["clean_sheet_rate"].clip(0, 1)
        if fixtures is not None and not fixtures.empty and "opponent_team" in fixtures.columns:
            factor = self._opponent_clean_sheet_factor(history)
            if not factor.empty:
                opponents = fixtures[["element", "opponent_team"]].drop_duplicates("element")
                combined = combined.merge(opponents, on="element", how="left")
                clean_sheet_probability = (
                    clean_sheet_probability * combined["opponent_team"].map(factor).fillna(1.0)
                ).clip(0, 1)

        share_of_match = (combined["expected_minutes"] / MINUTES_PER_MATCH).clip(0, 1)
        goal_points = combined["position"].map(GOAL_POINTS).fillna(0)
        clean_sheet_points = combined["position"].map(CLEAN_SHEET_POINTS).fillna(0)
        is_defensive = combined["position"].isin(["GK", "DEF"])

        appearance = (
            combined["full_appearance_rate"] * APPEARANCE_POINTS_FULL
            + (combined["any_appearance_rate"] - combined["full_appearance_rate"]).clip(0)
            * APPEARANCE_POINTS_PARTIAL
        )
        goals = combined["goal_rate"] * share_of_match * goal_points
        assists = combined["assist_rate"] * share_of_match * ASSIST_POINTS
        # A clean sheet only pays out for a player who lasted 60 minutes.
        clean_sheets = (
            clean_sheet_probability * combined["full_appearance_rate"] * clean_sheet_points
        )
        saves = combined["saves_rate"] * share_of_match / SAVES_PER_POINT
        # Scaled by expected minutes like every other component. Left unscaled,
        # a defender dropped to the bench kept his historical contribution rate
        # and so kept most of his predicted points -- inflating exactly the
        # players a minutes-aware model exists to demote, on the newest and
        # most heavily weighted scoring rule.
        contributions = (
            combined["contribution_rate"] * share_of_match * DEFENSIVE_CONTRIBUTION_POINTS
        )
        bonus = combined["bonus_rate"] * share_of_match
        cards = (
            combined["yellow_rate"] * YELLOW_CARD_POINTS + combined["red_rate"] * RED_CARD_POINTS
        ) * share_of_match
        conceded = np.where(
            is_defensive,
            -(combined["conceded_rate"] * share_of_match) / GOALS_CONCEDED_PER_PENALTY,
            0.0,
        )

        combined["expected_points"] = (
            appearance
            + goals
            + assists
            + clean_sheets
            + saves
            + contributions
            + bonus
            + cards
            + conceded
        ).fillna(0.0)

        return combined[["element", "expected_points"]]
