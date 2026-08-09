"""Where every data element comes from.

The model in ``docs/data-model.md`` describes 91 elements across 15 entities.
This records, for each one, which source supplies it and under what name — so
"can we actually get this?" is a question with a checkable answer rather than
an assumption.

Kept in code rather than only in the document for the same reason the BPS
table is: a mapping that lives only in prose drifts from the code silently.
``tests/test_lineage.py`` fetches the real sources and asserts every element
declared ``SOURCED`` is genuinely present.

The ``DERIVED`` and ``UNSOURCED`` origins matter as much as the sourced ones.
A model that quietly lists an element nobody publishes is how a project ends
up building on a column that will always be empty.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Origin(Enum):
    """How an element comes to exist."""

    SOURCED = "sourced"
    """Read directly from a source field."""

    DERIVED = "derived"
    """Computed from other elements by a named function."""

    UNSOURCED = "unsourced"
    """Nothing available publishes it. Recorded so the gap stays visible."""


# The feeds, by the key used throughout this module.
BOOTSTRAP_ELEMENTS = "bootstrap.elements"
BOOTSTRAP_TEAMS = "bootstrap.teams"
BOOTSTRAP_TYPES = "bootstrap.element_types"
BOOTSTRAP_EVENTS = "bootstrap.events"
FIXTURES = "fixtures"
ARCHIVE = "archive"
ODDS = "odds"

SOURCE_DESCRIPTIONS = {
    BOOTSTRAP_ELEMENTS: "FPL API /bootstrap-static/ -> elements",
    BOOTSTRAP_TEAMS: "FPL API /bootstrap-static/ -> teams",
    BOOTSTRAP_TYPES: "FPL API /bootstrap-static/ -> element_types",
    BOOTSTRAP_EVENTS: "FPL API /bootstrap-static/ -> events",
    FIXTURES: "FPL API /fixtures/",
    ARCHIVE: "vaastav/Fantasy-Premier-League merged_gw.csv",
    ODDS: "The Odds API /v4/sports/soccer_epl/odds",
}


@dataclass(frozen=True)
class Element:
    """One field in the data model, and where it comes from."""

    entity: str
    field: str
    origin: Origin
    source: str = ""
    source_field: str = ""
    derived_by: str = ""
    note: str = ""

    @property
    def is_sourced(self) -> bool:
        return self.origin is Origin.SOURCED


def sourced(entity, field, source, source_field=None, note=""):
    return Element(
        entity=entity,
        field=field,
        origin=Origin.SOURCED,
        source=source,
        source_field=source_field or field,
        note=note,
    )


def derived(entity, field, derived_by, note=""):
    return Element(
        entity=entity, field=field, origin=Origin.DERIVED, derived_by=derived_by, note=note
    )


def unsourced(entity, field, note):
    return Element(entity=entity, field=field, origin=Origin.UNSOURCED, note=note)


ELEMENTS: tuple[Element, ...] = (
    # -- PLAYER: the human, stable across seasons ---------------------------
    sourced("PLAYER", "code", BOOTSTRAP_ELEMENTS, "code", "stable across seasons"),
    derived("PLAYER", "full_name", "domain.identity.player_match_keys", "first + second name"),
    derived("PLAYER", "match_key", "domain.identity.normalise_name"),
    # -- PLAYER_SEASON: the registration, scoped to one season --------------
    derived("PLAYER_SEASON", "season", "sources.archive.fetch_season_gameweeks", "added on load"),
    sourced("PLAYER_SEASON", "element", BOOTSTRAP_ELEMENTS, "id", "reassigned each season"),
    sourced("PLAYER_SEASON", "code", BOOTSTRAP_ELEMENTS, "code", "absent from the archive"),
    sourced("PLAYER_SEASON", "club_id", BOOTSTRAP_ELEMENTS, "team"),
    sourced("PLAYER_SEASON", "position_id", BOOTSTRAP_ELEMENTS, "element_type"),
    # -- CLUB ---------------------------------------------------------------
    sourced("CLUB", "id", BOOTSTRAP_TEAMS, "id"),
    sourced("CLUB", "name", BOOTSTRAP_TEAMS, "name"),
    sourced("CLUB", "short_name", BOOTSTRAP_TEAMS, "short_name"),
    derived("CLUB", "team_key", "domain.teams.team_key", "normalised, for odds joins"),
    # -- POSITION -----------------------------------------------------------
    sourced("POSITION", "id", BOOTSTRAP_TYPES, "id"),
    sourced("POSITION", "singular_name", BOOTSTRAP_TYPES, "singular_name"),
    sourced("POSITION", "squad_min", BOOTSTRAP_TYPES, "squad_min_play"),
    sourced("POSITION", "squad_max", BOOTSTRAP_TYPES, "squad_max_play"),
    # -- SEASON: capability flags, not just a label -------------------------
    derived("SEASON", "season", "backtest.seasons.ALL_SEASONS"),
    derived("SEASON", "has_defensive_contributions", "backtest.seasons.season_capabilities"),
    derived("SEASON", "has_expected_goals", "backtest.seasons.season_capabilities"),
    # -- GAMEWEEK -----------------------------------------------------------
    derived("GAMEWEEK", "season", "backtest.seasons", "context, not a field"),
    sourced("GAMEWEEK", "gameweek", BOOTSTRAP_EVENTS, "id"),
    sourced("GAMEWEEK", "deadline_time", BOOTSTRAP_EVENTS, "deadline_time"),
    sourced("GAMEWEEK", "finished", BOOTSTRAP_EVENTS, "finished"),
    # -- FIXTURE ------------------------------------------------------------
    derived("FIXTURE", "season", "context"),
    sourced("FIXTURE", "fixture_id", FIXTURES, "id"),
    sourced("FIXTURE", "gameweek", FIXTURES, "event"),
    sourced("FIXTURE", "home_club", FIXTURES, "team_h"),
    sourced("FIXTURE", "away_club", FIXTURES, "team_a"),
    sourced("FIXTURE", "home_difficulty", FIXTURES, "team_h_difficulty"),
    sourced("FIXTURE", "away_difficulty", FIXTURES, "team_a_difficulty"),
    sourced("FIXTURE", "kickoff_time", FIXTURES, "kickoff_time"),
    # -- APPEARANCE: the modelling substrate --------------------------------
    derived("APPEARANCE", "season", "sources.archive.fetch_season_gameweeks"),
    sourced("APPEARANCE", "element", ARCHIVE, "element"),
    sourced("APPEARANCE", "fixture_id", ARCHIVE, "fixture", "the true grain"),
    sourced("APPEARANCE", "minutes", ARCHIVE, "minutes"),
    sourced("APPEARANCE", "total_points", ARCHIVE, "total_points"),
    sourced("APPEARANCE", "expected_goals", ARCHIVE, "expected_goals", "2022-23 onwards"),
    sourced("APPEARANCE", "expected_assists", ARCHIVE, "expected_assists", "2022-23 onwards"),
    sourced(
        "APPEARANCE",
        "clearances_blocks_interceptions",
        ARCHIVE,
        "clearances_blocks_interceptions",
    ),
    sourced("APPEARANCE", "tackles", ARCHIVE, "tackles"),
    sourced("APPEARANCE", "recoveries", ARCHIVE, "recoveries"),
    sourced(
        "APPEARANCE",
        "defensive_contribution",
        ARCHIVE,
        "defensive_contribution",
        "2025-26 onwards; equals CBIT or CBIRT by position",
    ),
    sourced("APPEARANCE", "bps", ARCHIVE, "bps", "total only; components not published"),
    sourced("APPEARANCE", "bonus", ARCHIVE, "bonus"),
    sourced("APPEARANCE", "saves", ARCHIVE, "saves"),
    sourced("APPEARANCE", "goals_conceded", ARCHIVE, "goals_conceded"),
    sourced("APPEARANCE", "own_goals", ARCHIVE, "own_goals"),
    sourced("APPEARANCE", "penalties_saved", ARCHIVE, "penalties_saved"),
    sourced("APPEARANCE", "penalties_missed", ARCHIVE, "penalties_missed"),
    sourced("APPEARANCE", "yellow_cards", ARCHIVE, "yellow_cards"),
    sourced("APPEARANCE", "red_cards", ARCHIVE, "red_cards"),
    # -- DAILY_SIGNAL: live-only, the reason the capture exists -------------
    derived("DAILY_SIGNAL", "captured_on", "store.snapshot.write_daily_signals"),
    sourced("DAILY_SIGNAL", "element", BOOTSTRAP_ELEMENTS, "id"),
    sourced("DAILY_SIGNAL", "code", BOOTSTRAP_ELEMENTS, "code"),
    sourced("DAILY_SIGNAL", "status", BOOTSTRAP_ELEMENTS, "status", "live only"),
    sourced(
        "DAILY_SIGNAL",
        "chance_of_playing",
        BOOTSTRAP_ELEMENTS,
        "chance_of_playing_next_round",
        "null means no news, not fit",
    ),
    sourced("DAILY_SIGNAL", "news", BOOTSTRAP_ELEMENTS, "news"),
    derived("DAILY_SIGNAL", "return_date", "features.availability.parse_return_date"),
    derived("DAILY_SIGNAL", "reason", "features.availability.unavailability_reason"),
    sourced("DAILY_SIGNAL", "penalties_order", BOOTSTRAP_ELEMENTS, "penalties_order"),
    sourced(
        "DAILY_SIGNAL",
        "corners_order",
        BOOTSTRAP_ELEMENTS,
        "corners_and_indirect_freekicks_order",
    ),
    sourced("DAILY_SIGNAL", "free_kicks_order", BOOTSTRAP_ELEMENTS, "direct_freekicks_order"),
    sourced("DAILY_SIGNAL", "now_cost", BOOTSTRAP_ELEMENTS, "now_cost", "integer tenths"),
    sourced("DAILY_SIGNAL", "selected_by_percent", BOOTSTRAP_ELEMENTS, "selected_by_percent"),
    # -- MARKET_PRICE -------------------------------------------------------
    sourced("MARKET_PRICE", "match_id", ODDS, "id"),
    sourced("MARKET_PRICE", "bookmaker", ODDS, "bookmakers[].key"),
    sourced("MARKET_PRICE", "market", ODDS, "bookmakers[].markets[].key"),
    sourced("MARKET_PRICE", "outcome", ODDS, "…outcomes[].name"),
    sourced("MARKET_PRICE", "price", ODDS, "…outcomes[].price"),
    derived("MARKET_PRICE", "captured_at", "sources.base.guarded"),
    # -- TEAM_EXPECTATION: all derived from the market ----------------------
    derived("TEAM_EXPECTATION", "match_id", "features.market.team_expectations"),
    derived("TEAM_EXPECTATION", "club_id", "domain.teams.match_teams"),
    derived("TEAM_EXPECTATION", "expected_goals_for", "features.market.team_expectations"),
    derived("TEAM_EXPECTATION", "clean_sheet_probability", "features.market.team_expectations"),
    derived("TEAM_EXPECTATION", "win_probability", "features.market.match_probabilities"),
    # -- BPS_ACTION: reference data, hand-curated from the published rules ---
    derived("BPS_ACTION", "action", "domain.bps.BPS_ACTIONS", "official Premier League table"),
    derived("BPS_ACTION", "bps", "domain.bps.BPS_ACTIONS"),
    derived("BPS_ACTION", "observable", "domain.bps.BPS_ACTIONS", "16 of 38"),
    derived("BPS_ACTION", "note", "domain.bps.BPS_ACTIONS"),
    # -- BPS_CONTRIBUTION: derived, deliberately unpersisted ----------------
    derived("BPS_CONTRIBUTION", "season", "context"),
    derived("BPS_CONTRIBUTION", "element", "context"),
    derived("BPS_CONTRIBUTION", "fixture_id", "context"),
    derived("BPS_CONTRIBUTION", "action", "domain.bps.BPS_ACTIONS"),
    derived("BPS_CONTRIBUTION", "bps", "domain.bps.reconstruct", "observable actions only"),
    # -- MODEL and PREDICTION -----------------------------------------------
    derived("MODEL", "name", "models.base.Predictor.name"),
    derived("MODEL", "config", "models.*", "the dataclass fields of the predictor"),
    derived("PREDICTION", "season", "context"),
    derived("PREDICTION", "gameweek", "backtest.harness.replay"),
    derived("PREDICTION", "element", "backtest.harness.replay"),
    derived("PREDICTION", "model", "models.base.Predictor.name"),
    derived("PREDICTION", "expected_points", "models.base.Predictor.predict"),
)


def by_entity(entity: str) -> tuple[Element, ...]:
    return tuple(element for element in ELEMENTS if element.entity == entity)


def by_origin(origin: Origin) -> tuple[Element, ...]:
    return tuple(element for element in ELEMENTS if element.origin is origin)


def entities() -> tuple[str, ...]:
    seen: list[str] = []
    for element in ELEMENTS:
        if element.entity not in seen:
            seen.append(element.entity)
    return tuple(seen)


def sources_used() -> tuple[str, ...]:
    return tuple(sorted({e.source for e in ELEMENTS if e.source}))


def coverage() -> dict[str, int]:
    """How many elements come from where."""
    return {origin.value: len(by_origin(origin)) for origin in Origin}
