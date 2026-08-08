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
def fake_fetcher(bootstrap):
    """A :data:`fpl.sources.fpl_api.Fetcher` that serves the frozen snapshot."""

    def fetch(url: str) -> Any:
        if url.endswith("/bootstrap-static/"):
            return bootstrap
        raise AssertionError(f"unexpected URL requested in a test: {url}")

    return fetch
