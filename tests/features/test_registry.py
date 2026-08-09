"""Tests for the derivation catalogue.

The behaviour that matters: a live frame gets everything, a historical frame
gets only what its data supports, and the difference is reported rather than
silent.
"""

from __future__ import annotations

import pandas as pd

from fpl.features.registry import DERIVATIONS, by_name, enrich, provider_of

LIVE = pd.DataFrame(
    [
        {
            "element": 1,
            "team": 1,
            "position": "MID",
            "price": 7.5,
            "minutes": 900,
            "total_points": 60,
            "goals_scored": 5,
            "expected_goals": 4.0,
            "status": "a",
            "chance_of_playing_next_round": None,
            "penalties_order": 1,
            "corners_and_indirect_freekicks_order": None,
            "direct_freekicks_order": None,
        }
    ]
)

ARCHIVE = pd.DataFrame(
    [{"element": 1, "gameweek": 5, "minutes": 90, "total_points": 6, "price": 7.5}]
)


def test_every_derivation_is_uniquely_named():
    assert len(by_name()) == len(DERIVATIONS)


def test_a_live_frame_gets_every_derivation():
    result = enrich(LIVE)

    assert set(result.applied) == {"rates", "availability", "advanced", "penalties"}
    assert not result.skipped


def test_the_promised_columns_actually_appear():
    """A derivation that lies about what it provides is worse than none."""
    result = enrich(LIVE)

    for derivation in DERIVATIONS:
        if derivation.name not in result.applied:
            continue
        for column in derivation.provides:
            assert column in result.frame.columns, f"{derivation.name} did not provide {column}"


def test_a_historical_frame_skips_the_live_only_derivations():
    """Applying them would invent injury status and set-piece duty."""
    result = enrich(ARCHIVE)

    assert "rates" in result.applied
    assert "availability" in result.skipped
    assert "advanced" in result.skipped
    assert "penalties" in result.skipped


def test_skipping_is_reported_with_a_reason():
    result = enrich(ARCHIVE)

    assert all(reason for reason in result.skipped.values())


def test_skipping_does_not_raise_or_invent_columns():
    result = enrich(ARCHIVE)

    assert "availability" not in result.frame.columns
    assert len(result.frame) == len(ARCHIVE)


def test_a_frame_missing_required_columns_skips_with_the_names():
    bare = pd.DataFrame([{"element": 1}])

    result = enrich(bare)

    assert "rates" in result.skipped
    assert "minutes" in result.skipped["rates"]


def test_the_catalogue_answers_where_a_column_came_from():
    """The question a hand-chained pipeline could not answer."""
    assert provider_of("expected_penalty_goals").name == "penalties"
    assert provider_of("points_per_million").name == "rates"
    assert provider_of("set_piece_duties").name == "advanced"


def test_an_unknown_column_has_no_provider():
    assert provider_of("not_a_real_column") is None


def test_the_summary_reads_as_a_sentence():
    assert "applied" in enrich(LIVE).summary()
    assert "skipped" in enrich(ARCHIVE).summary()


def test_enrichment_does_not_mutate_the_input():
    enrich(LIVE)

    assert "availability" not in LIVE.columns


def test_derivations_can_be_applied_selectively():
    only_rates = tuple(d for d in DERIVATIONS if d.name == "rates")

    result = enrich(LIVE, derivations=only_rates)

    assert result.applied == ["rates"]
    assert "availability" not in result.frame.columns


def test_the_real_snapshot_gets_every_derivation(bootstrap, schedule):
    from fpl.domain.players import build_players_frame

    players = build_players_frame(bootstrap)

    result = enrich(players, rates={"schedule": schedule})

    assert not result.skipped
    assert len(result.frame) == len(players)
