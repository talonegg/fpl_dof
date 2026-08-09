"""Tests for point-in-time snapshots."""

from __future__ import annotations

import json
from datetime import UTC, date, datetime

import pytest

from fpl.sources.snapshot import (
    available_gameweeks,
    captured_dates,
    read_daily_signals,
    read_snapshot,
    snapshot_directory,
    write_daily_signals,
    write_snapshot,
)

CAPTURED_AT = datetime(2026, 8, 8, 6, 0, tzinfo=UTC)


def test_snapshot_directory_is_zero_padded(tmp_path):
    assert snapshot_directory(tmp_path, 7).name == "gw07"
    assert snapshot_directory(tmp_path, 38).name == "gw38"


def test_write_snapshot_writes_all_three_files(tmp_path, bootstrap, fixtures_snapshot):
    paths = write_snapshot(
        bootstrap, fixtures_snapshot["fixtures"], root=tmp_path, captured_at=CAPTURED_AT
    )

    assert set(paths) == {"players", "schedule", "meta"}
    for path in paths.values():
        assert path.exists()


def test_snapshot_is_filed_under_the_next_gameweek(tmp_path, bootstrap, fixtures_snapshot):
    write_snapshot(bootstrap, fixtures_snapshot["fixtures"], root=tmp_path)

    assert (tmp_path / "gw01").is_dir()


def test_meta_records_what_was_captured_and_when(tmp_path, bootstrap, fixtures_snapshot):
    paths = write_snapshot(
        bootstrap, fixtures_snapshot["fixtures"], root=tmp_path, captured_at=CAPTURED_AT
    )

    meta = json.loads(paths["meta"].read_text(encoding="utf-8"))

    assert meta["gameweek"] == 1
    assert meta["captured_at"] == CAPTURED_AT.isoformat()
    assert meta["player_count"] == len(bootstrap["elements"])
    assert meta["fixture_count"] == len(fixtures_snapshot["fixtures"])


def test_snapshot_round_trips(tmp_path, bootstrap, fixtures_snapshot):
    write_snapshot(bootstrap, fixtures_snapshot["fixtures"], root=tmp_path)

    loaded = read_snapshot(tmp_path, 1)

    assert loaded is not None
    assert len(loaded["players"]) == len(bootstrap["elements"])
    assert len(loaded["schedule"]) == 2 * len(fixtures_snapshot["fixtures"])


def test_reading_a_gameweek_that_was_never_captured_is_none(tmp_path):
    assert read_snapshot(tmp_path, 12) is None


def test_rerunning_overwrites_rather_than_duplicating(tmp_path, bootstrap, fixtures_snapshot):
    write_snapshot(bootstrap, fixtures_snapshot["fixtures"], root=tmp_path)
    write_snapshot(bootstrap, fixtures_snapshot["fixtures"], root=tmp_path)

    assert available_gameweeks(tmp_path) == [1]


def test_a_finished_season_raises_rather_than_writing_a_bad_snapshot(
    tmp_path, bootstrap, fixtures_snapshot
):
    finished = {**bootstrap, "events": [{"id": 38, "finished": True, "is_next": False}]}

    with pytest.raises(ValueError, match="season is over"):
        write_snapshot(finished, fixtures_snapshot["fixtures"], root=tmp_path)


def test_available_gameweeks_is_empty_when_nothing_captured(tmp_path):
    assert available_gameweeks(tmp_path / "nope") == []


def test_available_gameweeks_ignores_unrelated_directories(tmp_path, bootstrap, fixtures_snapshot):
    write_snapshot(bootstrap, fixtures_snapshot["fixtures"], root=tmp_path)
    (tmp_path / "notes").mkdir()
    (tmp_path / "gwXX").mkdir()

    assert available_gameweeks(tmp_path) == [1]


# --- Daily capture: the only record live-only signals will ever have ---


def test_a_daily_capture_is_written(tmp_path, bootstrap):
    path = write_daily_signals(bootstrap, root=tmp_path, captured_on=date(2026, 8, 9))

    assert path is not None
    assert path.exists()
    assert path.name == "2026-08-09.parquet"


def test_the_capture_keeps_the_live_only_signals(tmp_path, bootstrap):
    write_daily_signals(bootstrap, root=tmp_path, captured_on=date(2026, 8, 9))

    daily = read_daily_signals(tmp_path)

    for column in ("status", "chance_of_playing_next_round", "penalties_order", "now_cost"):
        assert column in daily.columns


def test_the_capture_is_stamped_with_its_date(tmp_path, bootstrap):
    write_daily_signals(bootstrap, root=tmp_path, captured_on=date(2026, 8, 9))

    daily = read_daily_signals(tmp_path)

    assert set(daily["captured_on"]) == {"2026-08-09"}


def test_a_second_run_on_the_same_day_does_not_overwrite(tmp_path, bootstrap):
    """A re-run would replace the morning's injury news with the evening's."""
    write_daily_signals(bootstrap, root=tmp_path, captured_on=date(2026, 8, 9))

    again = write_daily_signals(bootstrap, root=tmp_path, captured_on=date(2026, 8, 9))

    assert again is None


def test_overwriting_is_possible_but_must_be_asked_for(tmp_path, bootstrap):
    write_daily_signals(bootstrap, root=tmp_path, captured_on=date(2026, 8, 9))

    again = write_daily_signals(
        bootstrap, root=tmp_path, captured_on=date(2026, 8, 9), overwrite=True
    )

    assert again is not None


def test_captures_accumulate_across_days(tmp_path, bootstrap):
    for day in (7, 8, 9):
        write_daily_signals(bootstrap, root=tmp_path, captured_on=date(2026, 8, day))

    assert captured_dates(tmp_path) == ["2026-08-07", "2026-08-08", "2026-08-09"]
    assert len(read_daily_signals(tmp_path)) == 3 * len(bootstrap["elements"])


def test_reading_before_anything_is_captured_is_empty(tmp_path):
    assert read_daily_signals(tmp_path).empty
    assert captured_dates(tmp_path) == []


def test_the_daily_capture_is_far_smaller_than_the_full_table(tmp_path, bootstrap):
    """The reason for a curated column list: 109 columns, most of them static."""
    from fpl.domain.players import build_players_frame

    daily = write_daily_signals(bootstrap, root=tmp_path, captured_on=date(2026, 8, 9))
    full = tmp_path / "full.parquet"
    build_players_frame(bootstrap).to_parquet(full, index=False)

    assert daily.stat().st_size < full.stat().st_size


def test_the_daily_capture_carries_the_stable_cross_season_code(tmp_path, bootstrap):
    """`element` is reassigned each season; `code` is what joins across them."""
    write_daily_signals(bootstrap, root=tmp_path, captured_on=date(2026, 8, 9))

    assert "code" in read_daily_signals(tmp_path).columns


def test_the_daily_capture_carries_every_observable_bps_input(tmp_path, bootstrap):
    """Differencing these recovers per-gameweek stats without the archive."""
    write_daily_signals(bootstrap, root=tmp_path, captured_on=date(2026, 8, 9))

    daily = read_daily_signals(tmp_path)

    for column in (
        "goals_scored",
        "assists",
        "clean_sheets",
        "goals_conceded",
        "own_goals",
        "penalties_saved",
        "penalties_missed",
        "yellow_cards",
        "red_cards",
        "saves",
        "tackles",
        "clearances_blocks_interceptions",
        "recoveries",
        "minutes",
        "bps",
    ):
        assert column in daily.columns, f"{column} missing from the daily capture"
