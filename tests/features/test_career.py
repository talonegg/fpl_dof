"""Tests for cross-season blending.

The failure this guards against is a player's whole projection being set by
one fragment of a season, or by one huge season three years ago.
"""

from __future__ import annotations

import pandas as pd
import pytest

from fpl.features.career import (
    blend_career_rates,
    confidence,
    finishing_multiplier,
    season_totals,
    season_weights,
    shrink_towards_prior,
)


def season(name, minutes, points, goals=0, xg=0.0, club="Arsenal", gameweeks=10):
    """A player-season split evenly across gameweeks."""
    return pd.DataFrame(
        [
            {
                "player_name": name,
                "gameweek": gameweek,
                "minutes": minutes / gameweeks,
                "total_points": points / gameweeks,
                "goals_scored": goals / gameweeks,
                "expected_goals": xg / gameweeks,
                "team_name": club,
            }
            for gameweek in range(1, gameweeks + 1)
        ]
    )


def test_the_most_recent_season_carries_the_most_weight():
    weights = season_weights(["2022-23", "2023-24", "2024-25", "2025-26"])

    assert weights["2025-26"] > weights["2024-25"] > weights["2023-24"] > weights["2022-23"]


def test_weights_sum_to_one():
    weights = season_weights(["2023-24", "2024-25", "2025-26"])

    assert sum(weights.values()) == pytest.approx(1.0)


def test_the_recent_season_is_worth_about_half():
    weights = season_weights(["2022-23", "2023-24", "2024-25", "2025-26"])

    assert 0.45 < weights["2025-26"] < 0.55


def test_no_seasons_gives_no_weights():
    assert season_weights([]) == {}


def test_totals_are_summed_per_player_per_season():
    data = {"2025-26": season("A", minutes=900, points=90)}

    totals = season_totals(data)

    assert totals["minutes"].iloc[0] == pytest.approx(900)
    assert totals["total_points"].iloc[0] == pytest.approx(90)


def test_a_single_season_blends_to_its_own_rate():
    data = {"2025-26": season("A", minutes=900, points=90)}

    career = blend_career_rates(data)

    # 90 points in 900 minutes is 9 per 90.
    assert career["total_points_per_90"].iloc[0] == pytest.approx(9.0)


def test_a_recent_season_outweighs_an_older_one_at_equal_minutes():
    data = {
        "2024-25": season("A", minutes=900, points=90),  # 9 per 90
        "2025-26": season("A", minutes=900, points=180),  # 18 per 90
    }

    career = blend_career_rates(data)

    # Weighted towards the recent 18, so above the midpoint of 13.5.
    assert career["total_points_per_90"].iloc[0] > 13.5


def test_a_thin_recent_season_does_not_dominate_a_heavy_old_one():
    """The failure this weighting exists to prevent."""
    data = {
        "2024-25": season("A", minutes=3000, points=300),  # 9 per 90, big sample
        "2025-26": season("A", minutes=180, points=90),  # 45 per 90, tiny sample
    }

    career = blend_career_rates(data)

    # Recency alone would give ~45; minutes weighting must pull it far down.
    assert career["total_points_per_90"].iloc[0] < 20


def test_seasons_seen_is_reported():
    data = {
        "2023-24": season("A", 900, 90),
        "2024-25": season("A", 900, 90),
        "2025-26": season("A", 900, 90),
    }

    assert blend_career_rates(data)["seasons_seen"].iloc[0] == 3


def test_career_minutes_accumulate():
    data = {"2024-25": season("A", 900, 90), "2025-26": season("A", 600, 60)}

    assert blend_career_rates(data)["career_minutes"].iloc[0] == pytest.approx(1500)


def test_a_player_in_no_season_is_absent_rather_than_invented():
    data = {"2025-26": season("A", 900, 90)}

    career = blend_career_rates(data)

    assert "B" not in career["player_name"].tolist()


def test_the_most_recent_club_is_carried_for_transfer_detection():
    data = {
        "2024-25": season("A", 900, 90, club="Southampton"),
        "2025-26": season("A", 900, 90, club="Newcastle"),
    }

    assert blend_career_rates(data)["last_club"].iloc[0] == "Newcastle"


def test_accented_names_blend_as_one_player():
    """Identity is by normalised name, since element ids are season-scoped."""
    data = {
        "2024-25": season("Aarón Anselmino", 900, 90),
        "2025-26": season("Aaron Anselmino", 900, 90),
    }

    career = blend_career_rates(data)

    assert len(career) == 1
    assert career["seasons_seen"].iloc[0] == 2


def test_confidence_rises_with_minutes():
    thin = pd.DataFrame([{"career_minutes": 200, "seasons_seen": 1}])
    thick = pd.DataFrame([{"career_minutes": 3000, "seasons_seen": 1}])

    assert confidence(thick).iloc[0] > confidence(thin).iloc[0]


def test_confidence_rises_with_seasons_at_equal_minutes():
    one = pd.DataFrame([{"career_minutes": 900, "seasons_seen": 1}])
    four = pd.DataFrame([{"career_minutes": 900, "seasons_seen": 4}])

    assert confidence(four).iloc[0] > confidence(one).iloc[0]


def test_confidence_is_bounded():
    extreme = pd.DataFrame([{"career_minutes": 99999, "seasons_seen": 10}])

    assert 0.0 <= confidence(extreme).iloc[0] <= 1.0


def test_an_overperformer_is_credited_only_a_fraction_of_the_excess():
    """40% above expected goals earns about 10%, not 40%."""
    career = pd.DataFrame([{"goals_scored_per_90": 0.7, "expected_goals_per_90": 0.5}])

    multiplier = finishing_multiplier(career).iloc[0]

    assert 1.0 < multiplier < 1.15


def test_an_underperformer_is_credited_upwards():
    career = pd.DataFrame([{"goals_scored_per_90": 0.3, "expected_goals_per_90": 0.5}])

    assert finishing_multiplier(career).iloc[0] < 1.0


def test_the_multiplier_is_bounded_against_extremes():
    career = pd.DataFrame([{"goals_scored_per_90": 5.0, "expected_goals_per_90": 0.1}])

    assert finishing_multiplier(career).iloc[0] <= 1.15


def test_a_player_with_no_expected_goals_is_left_alone():
    career = pd.DataFrame([{"goals_scored_per_90": 0.5, "expected_goals_per_90": 0.0}])

    assert finishing_multiplier(career).iloc[0] == pytest.approx(1.0)


def test_blending_nothing_is_safe():
    assert blend_career_rates({}).empty
    assert season_totals({}).empty


# --- Shrinkage: stopping three-minute players rating 90 points per 90 ---


def test_an_unreliable_rate_is_pulled_towards_the_population():
    career = pd.DataFrame(
        [
            {"career_minutes": 3, "seasons_seen": 1, "total_points_per_90": 90.0},
            {"career_minutes": 3000, "seasons_seen": 4, "total_points_per_90": 5.0},
        ]
    )
    career["confidence"] = confidence(career)

    shrunk = shrink_towards_prior(career)

    assert shrunk["total_points_per_90"].iloc[0] < 20


def test_a_reliable_rate_is_left_almost_alone():
    career = pd.DataFrame(
        [
            {"career_minutes": 3000, "seasons_seen": 4, "total_points_per_90": 5.0},
            {"career_minutes": 3000, "seasons_seen": 4, "total_points_per_90": 7.0},
        ]
    )
    career["confidence"] = confidence(career)

    shrunk = shrink_towards_prior(career)

    assert shrunk["total_points_per_90"].iloc[0] == pytest.approx(5.0, abs=0.5)


def test_shrinkage_preserves_ordering_among_equally_reliable_players():
    career = pd.DataFrame(
        [
            {"career_minutes": 2000, "seasons_seen": 3, "total_points_per_90": 4.0},
            {"career_minutes": 2000, "seasons_seen": 3, "total_points_per_90": 8.0},
        ]
    )
    career["confidence"] = confidence(career)

    shrunk = shrink_towards_prior(career)

    assert shrunk["total_points_per_90"].iloc[1] > shrunk["total_points_per_90"].iloc[0]


def test_shrinking_nothing_is_safe():
    assert shrink_towards_prior(pd.DataFrame()).empty
