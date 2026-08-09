"""Every element in the model can actually be sourced.

A data model that lists a field nobody publishes is worse than one that omits
it: you build on the column, and it is empty forever. These tests check the
mapping against the frozen fixtures, and — behind the ``backtest`` marker —
against the live feeds.
"""

from __future__ import annotations

import pytest

from fpl.domain.lineage import (
    ARCHIVE,
    BOOTSTRAP_ELEMENTS,
    BOOTSTRAP_EVENTS,
    BOOTSTRAP_TEAMS,
    BOOTSTRAP_TYPES,
    ELEMENTS,
    FIXTURES,
    ODDS,
    SOURCE_DESCRIPTIONS,
    Origin,
    by_origin,
    coverage,
    entities,
    sources_used,
)


def available_fields(
    bootstrap, fixtures_snapshot, archive_schema, odds_payload
) -> dict[str, set[str]]:
    """What each source actually offers, from the frozen fixtures."""
    first_odds = odds_payload[0]
    bookmaker = first_odds["bookmakers"][0]
    market = bookmaker["markets"][0]
    outcome = market["outcomes"][0]

    return {
        BOOTSTRAP_ELEMENTS: set(bootstrap["elements"][0]),
        BOOTSTRAP_TEAMS: set(bootstrap["teams"][0]),
        BOOTSTRAP_TYPES: set(bootstrap["element_types"][0]),
        BOOTSTRAP_EVENTS: set(bootstrap["events"][0]),
        FIXTURES: set(fixtures_snapshot["fixtures"][0]),
        ARCHIVE: set(archive_schema["2025-26"]),
        ODDS: (
            set(first_odds)
            | {f"bookmakers[].{k}" for k in bookmaker}
            | {f"bookmakers[].markets[].{k}" for k in market}
            | {f"…outcomes[].{k}" for k in outcome}
        ),
    }


def test_every_sourced_element_exists_in_its_source(
    bootstrap, fixtures_snapshot, archive_schema, odds_payload
):
    """The check this module exists for."""
    fields = available_fields(bootstrap, fixtures_snapshot, archive_schema, odds_payload)
    missing = []

    for element in by_origin(Origin.SOURCED):
        if element.source_field not in fields[element.source]:
            missing.append(
                f"{element.entity}.{element.field} <- {element.source}.{element.source_field}"
            )

    assert not missing, "elements claimed to be sourced but absent:\n  " + "\n  ".join(missing)


def test_every_element_declares_an_origin():
    for element in ELEMENTS:
        assert element.origin in Origin


def test_a_sourced_element_names_its_source_and_field():
    for element in by_origin(Origin.SOURCED):
        assert element.source, f"{element.entity}.{element.field} has no source"
        assert element.source_field, f"{element.entity}.{element.field} has no source field"


def test_a_derived_element_names_the_function_that_builds_it():
    """ "Derived" without a function name is just a shrug."""
    for element in by_origin(Origin.DERIVED):
        assert element.derived_by, f"{element.entity}.{element.field} says derived by nothing"


def test_an_unsourced_element_explains_itself():
    for element in by_origin(Origin.UNSOURCED):
        assert element.note, f"{element.entity}.{element.field} is unsourced with no reason"


def test_no_element_is_declared_twice():
    keys = [(element.entity, element.field) for element in ELEMENTS]

    assert len(keys) == len(set(keys))


def test_every_named_source_is_described():
    for source in sources_used():
        assert source in SOURCE_DESCRIPTIONS, f"{source} has no description"


def test_the_mapping_covers_the_documented_model():
    """The entities here and in docs/data-model.md must not drift apart."""
    import pathlib
    import re

    document = (
        pathlib.Path(__file__).resolve().parent.parent / "docs" / "data-model.md"
    ).read_text(encoding="utf-8")
    diagram = re.search(r"```mermaid\n(erDiagram.*?)\n```", document, re.S).group(1)
    documented = set(re.findall(r"^    ([A-Z_]+) \{", diagram, re.M))

    assert set(entities()) == documented


def test_every_documented_field_is_mapped():
    """Not just the entities — each field inside them."""
    import pathlib
    import re

    document = (
        pathlib.Path(__file__).resolve().parent.parent / "docs" / "data-model.md"
    ).read_text(encoding="utf-8")
    diagram = re.search(r"```mermaid\n(erDiagram.*?)\n```", document, re.S).group(1)

    mapped = {(element.entity, element.field) for element in ELEMENTS}
    missing = []
    for entity, body in re.findall(r"^    ([A-Z_]+) \{(.*?)^    \}", diagram, re.S | re.M):
        for line in body.strip().splitlines():
            parts = line.strip().split()
            if len(parts) >= 2 and (entity, parts[1]) not in mapped:
                missing.append(f"{entity}.{parts[1]}")

    assert not missing, "documented but unmapped:\n  " + "\n  ".join(missing)


def test_most_of_the_model_is_sourced_rather_than_invented():
    counts = coverage()

    assert counts["sourced"] > counts["unsourced"]
    assert counts["sourced"] + counts["derived"] + counts["unsourced"] == len(ELEMENTS)


@pytest.mark.backtest
def test_every_sourced_element_exists_in_the_live_feeds():
    """The fixtures are trimmed; this checks the real thing."""
    from fpl.sources.archive import fetch_season_gameweeks
    from fpl.sources.fpl_api import fetch_bootstrap, fetch_fixtures

    bootstrap = fetch_bootstrap()
    fixtures = {"fixtures": fetch_fixtures()}
    archive = fetch_season_gameweeks("2025-26")

    fields = {
        BOOTSTRAP_ELEMENTS: set(bootstrap["elements"][0]),
        BOOTSTRAP_TEAMS: set(bootstrap["teams"][0]),
        BOOTSTRAP_TYPES: set(bootstrap["element_types"][0]),
        BOOTSTRAP_EVENTS: set(bootstrap["events"][0]),
        FIXTURES: set(fixtures["fixtures"][0]),
        ARCHIVE: set(archive.columns),
    }

    missing = [
        f"{e.entity}.{e.field} <- {e.source}.{e.source_field}"
        for e in by_origin(Origin.SOURCED)
        if e.source in fields and e.source_field not in fields[e.source]
    ]

    assert not missing, "live feeds do not supply:\n  " + "\n  ".join(missing)
