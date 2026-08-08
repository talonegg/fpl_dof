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
