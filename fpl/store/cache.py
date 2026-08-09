"""On-disk parquet cache for fetched data.

Deliberately dumb: a key maps to a file, a file has an age, and callers decide
what age is too old. There is no background refresh and no eviction -- the
whole dataset is a few megabytes, and predictability matters more than
cleverness when the thing being cached feeds a model.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path

import pandas as pd

DEFAULT_CACHE_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "cache"

# Past seasons never change, so archive data can be cached indefinitely.
NEVER_STALE = float("inf")


def cache_path(key: str, cache_dir: Path | None = None) -> Path:
    """Path of the parquet file backing ``key``."""
    directory = cache_dir or DEFAULT_CACHE_DIR
    return directory / f"{key}.parquet"


def age_seconds(key: str, cache_dir: Path | None = None) -> float | None:
    """Seconds since ``key`` was written, or ``None`` if it is not cached."""
    path = cache_path(key, cache_dir)
    if not path.exists():
        return None
    return time.time() - path.stat().st_mtime


def write(key: str, df: pd.DataFrame, cache_dir: Path | None = None) -> Path:
    """Write ``df`` to the cache under ``key`` and return the path written."""
    path = cache_path(key, cache_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return path


def read(key: str, cache_dir: Path | None = None) -> pd.DataFrame | None:
    """Read ``key`` from the cache, or ``None`` if it is not cached."""
    path = cache_path(key, cache_dir)
    if not path.exists():
        return None
    return pd.read_parquet(path)


def load(
    key: str,
    build: Callable[[], pd.DataFrame],
    max_age_seconds: float,
    cache_dir: Path | None = None,
) -> pd.DataFrame:
    """Return cached data for ``key``, calling ``build`` if it is missing or stale.

    ``build`` is only invoked when needed, so passing a function that hits the
    network is safe as long as the cache is warm.
    """
    age = age_seconds(key, cache_dir)
    if age is not None and age <= max_age_seconds:
        cached = read(key, cache_dir)
        if cached is not None:
            return cached

    fresh = build()
    write(key, fresh, cache_dir)
    return fresh
