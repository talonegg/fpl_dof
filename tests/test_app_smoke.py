"""End-to-end smoke test of the Streamlit app, offline.

Uses Streamlit's own ``AppTest`` harness so the real render path is exercised.
The data fetch is patched out, so this stays a unit test: no network.
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st
from streamlit.testing.v1 import AppTest

APP_TIMEOUT_SECONDS = 60

# AppTest resolves relative paths against the calling file, so be explicit.
APP_PATH = str(Path(__file__).resolve().parent.parent / "streamlit_app.py")


def _run_app(monkeypatch, bootstrap, fixtures_snapshot) -> AppTest:
    """Run the whole app offline.

    Both fetches must be patched: the fixtures tab reaches for a second
    endpoint, and an unpatched one would silently hit the live API.
    """
    # The fixtures snapshot carries all 20 teams; the bootstrap sample carries
    # 4. Use the full list so opponents resolve to names.
    full_bootstrap = {**bootstrap, "teams": fixtures_snapshot["teams"]}
    monkeypatch.setattr("app.data.fetch_bootstrap", lambda: full_bootstrap)
    monkeypatch.setattr("app.data.fetch_fixtures", lambda: fixtures_snapshot["fixtures"])
    # The loaders are cached, so a previous run's data would otherwise leak in.
    st.cache_data.clear()
    return AppTest.from_file(APP_PATH, default_timeout=APP_TIMEOUT_SECONDS).run()


def test_app_renders_without_exception(monkeypatch, bootstrap, fixtures_snapshot):
    at = _run_app(monkeypatch, bootstrap, fixtures_snapshot)

    assert not at.exception
    assert at.title[0].value == "FPL Data Explorer"


def test_app_shows_every_player_from_the_source(monkeypatch, bootstrap, fixtures_snapshot):
    at = _run_app(monkeypatch, bootstrap, fixtures_snapshot)

    expected = len(bootstrap["elements"])
    captions = [caption.value for caption in at.caption]
    assert f"Showing {expected} of {expected} players." in captions


def test_filtering_to_one_position_narrows_the_table(monkeypatch, bootstrap, fixtures_snapshot):
    at = _run_app(monkeypatch, bootstrap, fixtures_snapshot)

    at.sidebar.multiselect[0].set_value(["Goalkeeper"]).run()

    assert not at.exception
    expected = sum(e["element_type"] == 1 for e in bootstrap["elements"])
    captions = [caption.value for caption in at.caption]
    assert any(caption.startswith(f"Showing {expected} ") for caption in captions)


def test_empty_filter_selection_warns_instead_of_crashing(
    monkeypatch, bootstrap, fixtures_snapshot
):
    at = _run_app(monkeypatch, bootstrap, fixtures_snapshot)

    at.sidebar.multiselect[0].set_value([]).run()

    assert not at.exception
    assert "No players match" in at.warning[0].value


def test_fixture_ticker_renders_with_a_row_per_team(monkeypatch, bootstrap, fixtures_snapshot):
    at = _run_app(monkeypatch, bootstrap, fixtures_snapshot)

    assert not at.exception
    assert "Fixture ticker" in [header.value for header in at.subheader]
    captions = " ".join(caption.value for caption in at.caption)
    assert "Kindest run:" in captions


def test_fixture_ticker_reports_no_blanks_in_a_normal_window(
    monkeypatch, bootstrap, fixtures_snapshot
):
    at = _run_app(monkeypatch, bootstrap, fixtures_snapshot)

    captions = " ".join(caption.value for caption in at.caption)
    assert "No blank or double gameweeks" in captions
