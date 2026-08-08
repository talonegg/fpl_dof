"""Tests for the parquet cache.

``tmp_path`` keeps every test out of the real ``data/cache/`` directory.
"""

from __future__ import annotations

import os
import time

import pandas as pd
import pytest

from fpl.sources import cache

DF = pd.DataFrame([{"player": "Saka", "points": 9}, {"player": "Isak", "points": 2}])


def test_read_of_an_uncached_key_is_none(tmp_path):
    assert cache.read("missing", cache_dir=tmp_path) is None


def test_age_of_an_uncached_key_is_none(tmp_path):
    assert cache.age_seconds("missing", cache_dir=tmp_path) is None


def test_write_then_read_round_trips(tmp_path):
    cache.write("players", DF, cache_dir=tmp_path)

    pd.testing.assert_frame_equal(cache.read("players", cache_dir=tmp_path), DF)


def test_write_creates_the_cache_directory(tmp_path):
    nested = tmp_path / "does" / "not" / "exist"

    path = cache.write("players", DF, cache_dir=nested)

    assert path.exists()


def test_load_builds_and_caches_on_a_miss(tmp_path):
    calls = []

    def build():
        calls.append(1)
        return DF

    first = cache.load("players", build, max_age_seconds=60, cache_dir=tmp_path)
    second = cache.load("players", build, max_age_seconds=60, cache_dir=tmp_path)

    assert len(calls) == 1, "the second call should have been served from disk"
    pd.testing.assert_frame_equal(first, second)


def test_load_rebuilds_when_the_cache_is_stale(tmp_path):
    calls = []

    def build():
        calls.append(1)
        return DF

    cache.load("players", build, max_age_seconds=60, cache_dir=tmp_path)
    # Backdate the file rather than sleeping.
    path = cache.cache_path("players", tmp_path)
    old = time.time() - 3600
    os.utime(path, (old, old))

    cache.load("players", build, max_age_seconds=60, cache_dir=tmp_path)

    assert len(calls) == 2


def test_never_stale_keeps_archive_data_indefinitely(tmp_path):
    calls = []

    def build():
        calls.append(1)
        return DF

    cache.load("season", build, max_age_seconds=cache.NEVER_STALE, cache_dir=tmp_path)
    path = cache.cache_path("season", tmp_path)
    ancient = time.time() - 86400 * 365
    os.utime(path, (ancient, ancient))

    cache.load("season", build, max_age_seconds=cache.NEVER_STALE, cache_dir=tmp_path)

    assert len(calls) == 1


def test_build_is_not_called_when_the_cache_is_warm(tmp_path):
    cache.write("players", DF, cache_dir=tmp_path)

    def build():
        raise AssertionError("build should not run against a warm cache")

    result = cache.load("players", build, max_age_seconds=60, cache_dir=tmp_path)

    assert len(result) == 2


@pytest.mark.parametrize("key", ["players", "fixtures", "archive_2025-26"])
def test_cache_path_is_a_parquet_file_named_for_the_key(key, tmp_path):
    assert cache.cache_path(key, tmp_path) == tmp_path / f"{key}.parquet"
