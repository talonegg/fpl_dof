"""Tests for matching team names across sources.

Checked against the real 20-team list in the fixtures snapshot, because the
names that disagree are exactly the big clubs whose odds matter most.
"""

from __future__ import annotations

import pandas as pd

from fpl.domain.teams import (
    build_team_lookup,
    match_teams,
    team_key,
    unmatched_teams,
)

# How bookmakers name the clubs whose names FPL abbreviates. Only teams in the
# snapshot's season: a club that is not in the league *should* fail to match.
BOOKMAKER_NAMES = [
    "Manchester United",
    "Manchester City",
    "Tottenham Hotspur",
    "Nottingham Forest",
    "Newcastle United",
    "AFC Bournemouth",
    "Ipswich Town",
    "Hull City",
    "Coventry City",
    "Leeds United",
    "Arsenal",
    "Liverpool",
    "Chelsea",
    "Everton",
    "Fulham",
    "Brentford",
]


def test_an_identical_name_matches_itself():
    assert team_key("Arsenal") == "arsenal"


def test_the_long_form_resolves_to_the_fpl_short_form():
    assert team_key("Manchester United") == team_key("Man Utd")
    assert team_key("Tottenham Hotspur") == team_key("Spurs")


def test_punctuation_differences_do_not_matter():
    """FPL writes Nott'm Forest; bookmakers write Nottingham Forest."""
    assert team_key("Nottingham Forest") == team_key("Nott'm Forest")


def test_case_and_spacing_do_not_matter():
    assert team_key("  MANCHESTER   CITY ") == team_key("Man City")


def test_the_lookup_registers_both_name_forms(fixtures_snapshot):
    lookup = build_team_lookup(fixtures_snapshot["teams"])

    arsenal = next(t for t in fixtures_snapshot["teams"] if t["name"] == "Arsenal")
    assert lookup[team_key("Arsenal")] == arsenal["id"]
    assert lookup[team_key(arsenal["short_name"])] == arsenal["id"]


def test_every_bookmaker_name_resolves_to_a_real_team(fixtures_snapshot):
    """The check that matters: no fixture's odds silently disappear."""
    unmatched = unmatched_teams(pd.Series(BOOKMAKER_NAMES), fixtures_snapshot["teams"])

    assert unmatched == []


def test_matching_returns_the_team_ids(fixtures_snapshot):
    ids = match_teams(pd.Series(["Arsenal", "Manchester United"]), fixtures_snapshot["teams"])

    assert ids.notna().all()


def test_an_unknown_team_is_reported_rather_than_dropped(fixtures_snapshot):
    unmatched = unmatched_teams(pd.Series(["Real Madrid"]), fixtures_snapshot["teams"])

    assert unmatched == ["Real Madrid"]


def test_an_unknown_team_resolves_to_nothing(fixtures_snapshot):
    ids = match_teams(pd.Series(["Real Madrid"]), fixtures_snapshot["teams"])

    assert ids.isna().all()


def test_every_fpl_team_matches_its_own_name(fixtures_snapshot):
    names = pd.Series([team["name"] for team in fixtures_snapshot["teams"]])

    assert unmatched_teams(names, fixtures_snapshot["teams"]) == []


def test_a_relegated_club_correctly_fails_to_match(fixtures_snapshot):
    """West Ham and Wolves are not in this snapshot's league, and should not be.

    An alias exists for both, so this is checking that the lookup is bounded by
    the season's actual teams rather than by the alias table.
    """
    unmatched = unmatched_teams(
        pd.Series(["West Ham United", "Wolverhampton Wanderers"]),
        fixtures_snapshot["teams"],
    )

    assert unmatched == ["West Ham United", "Wolverhampton Wanderers"]
