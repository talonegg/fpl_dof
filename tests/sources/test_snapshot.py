"""Tests for point-in-time snapshots."""

from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest

from fpl.sources.snapshot import (
    available_gameweeks,
    read_snapshot,
    snapshot_directory,
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
