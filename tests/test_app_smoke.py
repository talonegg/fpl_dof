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


def _run_app(monkeypatch, bootstrap, fixtures_snapshot, tmp_path, archive) -> AppTest:
    """Run the whole app offline.

    Every fetch must be patched. Each tab that reaches for a new source is a
    fresh chance to silently go online in what is supposed to be a unit test,
    so this list grows with the app.
    """
    # The fixtures snapshot carries all 20 teams; the bootstrap sample carries
    # 4. Use the full list so opponents resolve to names.
    full_bootstrap = {**bootstrap, "teams": fixtures_snapshot["teams"]}
    monkeypatch.setattr("app.data.fetch_bootstrap", lambda: full_bootstrap)
    monkeypatch.setattr("app.data.fetch_fixtures", lambda: fixtures_snapshot["fixtures"])
    monkeypatch.setattr("app.data.fetch_season_gameweeks", lambda season: archive)
    # Keep the watchlist out of the real data directory. This goes through the
    # environment because AppTest re-executes the script, so a module-level
    # constant patched here would just be redefined.
    monkeypatch.setenv("FPL_DOF_DATA_DIR", str(tmp_path))
    # The loaders are cached, so a previous run's data would otherwise leak in.
    st.cache_data.clear()
    return AppTest.from_file(APP_PATH, default_timeout=APP_TIMEOUT_SECONDS).run()


def test_app_renders_without_exception(
    monkeypatch, bootstrap, fixtures_snapshot, tmp_path, archive
):
    at = _run_app(monkeypatch, bootstrap, fixtures_snapshot, tmp_path, archive)

    assert not at.exception
    assert at.title[0].value == "FPL Data Explorer"


def test_app_shows_every_player_from_the_source(
    monkeypatch, bootstrap, fixtures_snapshot, tmp_path, archive
):
    at = _run_app(monkeypatch, bootstrap, fixtures_snapshot, tmp_path, archive)

    expected = len(bootstrap["elements"])
    captions = [caption.value for caption in at.caption]
    assert f"Showing {expected} of {expected} players." in captions


def test_filtering_to_one_position_narrows_the_table(
    monkeypatch, bootstrap, fixtures_snapshot, tmp_path, archive
):
    at = _run_app(monkeypatch, bootstrap, fixtures_snapshot, tmp_path, archive)

    at.sidebar.multiselect[0].set_value(["Goalkeeper"]).run()

    assert not at.exception
    expected = sum(e["element_type"] == 1 for e in bootstrap["elements"])
    captions = [caption.value for caption in at.caption]
    assert any(caption.startswith(f"Showing {expected} ") for caption in captions)


def test_empty_filter_selection_warns_instead_of_crashing(
    monkeypatch, bootstrap, fixtures_snapshot, tmp_path, archive
):
    at = _run_app(monkeypatch, bootstrap, fixtures_snapshot, tmp_path, archive)

    at.sidebar.multiselect[0].set_value([]).run()

    assert not at.exception
    assert "No players match" in at.warning[0].value


def test_fixture_ticker_renders_with_a_row_per_team(
    monkeypatch, bootstrap, fixtures_snapshot, tmp_path, archive
):
    at = _run_app(monkeypatch, bootstrap, fixtures_snapshot, tmp_path, archive)

    assert not at.exception
    assert "Fixture ticker" in [header.value for header in at.subheader]
    captions = " ".join(caption.value for caption in at.caption)
    assert "Kindest run:" in captions


def test_fixture_ticker_reports_no_blanks_in_a_normal_window(
    monkeypatch, bootstrap, fixtures_snapshot, tmp_path, archive
):
    at = _run_app(monkeypatch, bootstrap, fixtures_snapshot, tmp_path, archive)

    captions = " ".join(caption.value for caption in at.caption)
    assert "No blank or double gameweeks" in captions


def test_scouting_tab_renders_player_detail(
    monkeypatch, bootstrap, fixtures_snapshot, tmp_path, archive
):
    at = _run_app(monkeypatch, bootstrap, fixtures_snapshot, tmp_path, archive)

    assert not at.exception
    headers = [header.value for header in at.subheader]
    assert "Player detail" in headers
    assert "Compare players" in headers


def test_comparison_asks_for_two_players_before_charting(
    monkeypatch, bootstrap, fixtures_snapshot, tmp_path, archive
):
    at = _run_app(monkeypatch, bootstrap, fixtures_snapshot, tmp_path, archive)

    messages = [info.value for info in at.info]
    assert any("at least two players" in message for message in messages)


def test_watchlist_starts_empty_and_persists_a_selection(
    monkeypatch, bootstrap, fixtures_snapshot, tmp_path, archive
):
    at = _run_app(monkeypatch, bootstrap, fixtures_snapshot, tmp_path, archive)

    captions = [caption.value for caption in at.sidebar.caption]
    assert "Nothing watched yet." in captions

    watchlist_widget = at.session_state["watchlist_select"]
    assert watchlist_widget == []


# --- The sidebar filters must reach every tab, not just the one they sit next to ---


def _positions_offered_in_scouting(at):
    """Positions visible in the scouting player selector.

    AppTest reports options already formatted, so these are label strings of
    the form "Raya (Arsenal, Goalkeeper, £6.0m)".
    """
    return {label.split(", ")[1] for label in at.selectbox(key="detail_player").options}


def test_the_position_filter_reaches_the_scouting_tab(
    monkeypatch, bootstrap, fixtures_snapshot, tmp_path, archive
):
    at = _run_app(monkeypatch, bootstrap, fixtures_snapshot, tmp_path, archive)

    at.sidebar.multiselect(key="filter_positions").set_value(["Goalkeeper"]).run()

    assert not at.exception
    assert _positions_offered_in_scouting(at) == {"Goalkeeper"}


def test_the_position_filter_reaches_the_comparison_selector(
    monkeypatch, bootstrap, fixtures_snapshot, tmp_path, archive
):
    at = _run_app(monkeypatch, bootstrap, fixtures_snapshot, tmp_path, archive)

    at.sidebar.multiselect(key="filter_positions").set_value(["Goalkeeper"]).run()

    expected = sum(e["element_type"] == 1 for e in bootstrap["elements"])
    assert len(at.multiselect(key="comparison_players").options) == expected


def test_the_club_filter_reaches_the_fixtures_tab(
    monkeypatch, bootstrap, fixtures_snapshot, tmp_path, archive
):
    """The club filter is the only one that means anything for fixtures."""
    at = _run_app(monkeypatch, bootstrap, fixtures_snapshot, tmp_path, archive)

    at.sidebar.multiselect(key="filter_teams").set_value(["Arsenal"]).run()

    assert not at.exception
    captions = " ".join(caption.value for caption in at.caption)
    # Only one club left, so it is trivially the kindest run.
    assert "Kindest run: Arsenal." in captions


def test_the_club_filter_reaches_the_players_tab(
    monkeypatch, bootstrap, fixtures_snapshot, tmp_path, archive
):
    at = _run_app(monkeypatch, bootstrap, fixtures_snapshot, tmp_path, archive)

    at.sidebar.multiselect(key="filter_teams").set_value(["Arsenal"]).run()

    expected = sum(e["team"] == 1 for e in bootstrap["elements"])
    captions = [caption.value for caption in at.caption]
    assert any(caption.startswith(f"Showing {expected} of") for caption in captions)


def test_deselecting_every_club_warns_on_the_fixtures_tab(
    monkeypatch, bootstrap, fixtures_snapshot, tmp_path, archive
):
    at = _run_app(monkeypatch, bootstrap, fixtures_snapshot, tmp_path, archive)

    at.sidebar.multiselect(key="filter_teams").set_value([]).run()

    assert not at.exception
    warnings = [warning.value for warning in at.warning]
    assert any("No fixtures for the selected clubs" in warning for warning in warnings)


def test_a_selection_removed_by_a_filter_does_not_break_the_app(
    monkeypatch, bootstrap, fixtures_snapshot, tmp_path, archive
):
    """Pick a player, then filter them out. Streamlit would otherwise raise."""
    at = _run_app(monkeypatch, bootstrap, fixtures_snapshot, tmp_path, archive)

    forwards = [element["id"] for element in bootstrap["elements"] if element["element_type"] == 4]
    at.multiselect(key="comparison_players").set_value(forwards[:2]).run()
    assert not at.exception

    at.sidebar.multiselect(key="filter_positions").set_value(["Goalkeeper"]).run()

    assert not at.exception
    assert at.multiselect(key="comparison_players").value == []


# --- Regressions from the second review pass ---


def test_the_watchlist_file_is_not_rewritten_on_a_first_render(
    monkeypatch, bootstrap, fixtures_snapshot, tmp_path, archive
):
    """The widget returns element order; the stored list is sorted by code."""
    from fpl.features import watchlist

    path = tmp_path / "watchlist.json"
    codes = [element["code"] for element in bootstrap["elements"][:3]]
    watchlist.save(path, codes)
    before = path.stat().st_mtime_ns

    _run_app(monkeypatch, bootstrap, fixtures_snapshot, tmp_path, archive)

    assert path.stat().st_mtime_ns == before, "watchlist rewritten with no user action"


def test_comparison_series_keep_their_colour_when_one_is_removed(
    monkeypatch, bootstrap, fixtures_snapshot, tmp_path, archive
):
    """Colour is assigned by slot, so column order must follow selection order."""
    at = _run_app(monkeypatch, bootstrap, fixtures_snapshot, tmp_path, archive)

    elements = [element["id"] for element in bootstrap["elements"][:3]]
    # Select in an order that is not alphabetical by name.
    at.multiselect(key="comparison_players").set_value(elements).run()
    assert not at.exception

    at.multiselect(key="comparison_players").set_value(elements[:2]).run()
    assert not at.exception
