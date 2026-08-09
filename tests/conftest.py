"""Shared test fixtures.

Tests must never hit the network. Everything here is served from the frozen
snapshots in ``data/fixtures/``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

FIXTURE_DIR = Path(__file__).resolve().parent.parent / "data" / "fixtures"


def load_fixture(name: str) -> Any:
    """Load a frozen JSON snapshot by filename."""
    return json.loads((FIXTURE_DIR / name).read_text(encoding="utf-8"))


@pytest.fixture
def bootstrap() -> dict[str, Any]:
    """A trimmed real ``bootstrap-static`` payload: 4 teams, 12 players."""
    return load_fixture("bootstrap_static_sample.json")


@pytest.fixture
def fixtures_snapshot() -> dict[str, Any]:
    """Real fixtures for gameweeks 1-6, plus all 20 teams."""
    return load_fixture("fixtures_sample.json")


@pytest.fixture
def schedule(fixtures_snapshot):
    """The team-perspective schedule built from the real fixtures snapshot."""
    from fpl.domain.fixtures import build_team_schedule

    return build_team_schedule(fixtures_snapshot["fixtures"], fixtures_snapshot["teams"])


@pytest.fixture
def archive(bootstrap) -> Any:
    """A small stand-in for a season of archive rows.

    Built from the bootstrap sample's own players so that cross-season name
    matching actually resolves, which is what the scouting views depend on.
    """
    import pandas as pd

    rows = []
    for element in bootstrap["elements"][:4]:
        name = f"{element['first_name']} {element['second_name']}"
        for gameweek in (1, 2, 3):
            rows.append(
                {
                    "player_name": name,
                    "gameweek": gameweek,
                    "element": element["id"],
                    "team_name": "Arsenal",
                    "position": "MID",
                    "minutes": 90,
                    "total_points": gameweek + 1,
                    "price": 5.5,
                    "season": "2025-26",
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture
def fake_fetcher(bootstrap, fixtures_snapshot):
    """A :data:`fpl.sources.fpl_api.Fetcher` that serves the frozen snapshots.

    Anything it does not recognise raises, so a test that reaches for the real
    network fails loudly instead of silently going online.
    """

    def fetch(url: str) -> Any:
        if url.endswith("/bootstrap-static/"):
            return bootstrap
        if url.endswith("/fixtures/"):
            return fixtures_snapshot["fixtures"]
        raise AssertionError(f"unexpected URL requested in a test: {url}")

    return fetch


@pytest.fixture
def odds_payload() -> Any:
    """A hand-built odds payload matching The Odds API's documented schema.

    Not a live capture: the free tier needs a key, and this project has none
    configured. It is shaped like the real thing and exercises every branch,
    but it has never been checked against a real response.
    """
    return load_fixture("odds_sample.json")


@pytest.fixture
def archive_schema() -> dict[str, list[str]]:
    """The real column list of each usable archive season.

    A committed snapshot rather than a download, so schema checks stay offline.
    It doubles as a contract test: the archive changed from 43 to 51 columns
    between seasons once already, and a silent change is how a model starts
    reading a column that is no longer there.
    """
    return load_fixture("archive_schema.json")
