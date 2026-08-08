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


def _run_app(monkeypatch, bootstrap) -> AppTest:
    monkeypatch.setattr("app.data.fetch_bootstrap", lambda: bootstrap)
    # The loader is cached, so a previous run's data would otherwise leak in.
    st.cache_data.clear()
    return AppTest.from_file(APP_PATH, default_timeout=APP_TIMEOUT_SECONDS).run()


def test_app_renders_without_exception(monkeypatch, bootstrap):
    at = _run_app(monkeypatch, bootstrap)

    assert not at.exception
    assert at.title[0].value == "FPL Data Explorer"
    assert len(at.dataframe) == 1


def test_app_shows_every_player_from_the_source(monkeypatch, bootstrap):
    at = _run_app(monkeypatch, bootstrap)

    expected = len(bootstrap["elements"])
    assert at.caption[0].value == f"Showing {expected} of {expected} players."


def test_filtering_to_one_position_narrows_the_table(monkeypatch, bootstrap):
    at = _run_app(monkeypatch, bootstrap)

    at.sidebar.multiselect[0].set_value(["Goalkeeper"]).run()

    assert not at.exception
    expected = sum(e["element_type"] == 1 for e in bootstrap["elements"])
    assert at.caption[0].value.startswith(f"Showing {expected} ")


def test_empty_filter_selection_warns_instead_of_crashing(monkeypatch, bootstrap):
    at = _run_app(monkeypatch, bootstrap)

    at.sidebar.multiselect[0].set_value([]).run()

    assert not at.exception
    assert "No players match" in at.warning[0].value
