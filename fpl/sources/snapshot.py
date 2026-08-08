"""Point-in-time snapshots of the live FPL state.

The API only ever serves *now*. Once a gameweek's deadline passes, the prices,
ownership and form that were true beforehand are gone for good. A model
backtested against data reconstructed after the fact will look better than it
could ever have been in practice, because it quietly knows things it could not
have known.

Snapshotting on a schedule is the cheap fix: capture the state, stamp it with
the gameweek it belongs to, and keep it. The files are the raw material for
honest point-in-time backtests later.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from fpl.domain.fixtures import build_team_schedule, next_gameweek
from fpl.domain.players import build_players_frame


def snapshot_directory(root: Path, gameweek: int) -> Path:
    """Directory holding the snapshot for a gameweek, e.g. ``snapshots/gw07``."""
    return Path(root) / f"gw{gameweek:02d}"


def write_snapshot(
    bootstrap: dict[str, Any],
    fixtures: list[dict[str, Any]],
    root: Path,
    captured_at: datetime | None = None,
) -> dict[str, Path]:
    """Write a snapshot of the current state and return the paths written.

    Re-running for the same gameweek overwrites it, so a workflow that fires
    daily converges on "the state just before this gameweek's deadline" rather
    than accumulating near-duplicates.
    """
    gameweek = next_gameweek(bootstrap["events"])
    if gameweek is None:
        raise ValueError("no upcoming gameweek: the season is over")

    directory = snapshot_directory(root, gameweek)
    directory.mkdir(parents=True, exist_ok=True)

    players = build_players_frame(bootstrap)
    schedule = build_team_schedule(fixtures, bootstrap["teams"])

    paths = {
        "players": directory / "players.parquet",
        "schedule": directory / "schedule.parquet",
        "meta": directory / "meta.json",
    }
    players.to_parquet(paths["players"], index=False)
    schedule.to_parquet(paths["schedule"], index=False)

    stamp = captured_at or datetime.now(UTC)
    paths["meta"].write_text(
        json.dumps(
            {
                "gameweek": gameweek,
                "captured_at": stamp.isoformat(),
                "player_count": len(players),
                "fixture_count": len(fixtures),
            },
            indent=1,
        )
        + "\n",
        encoding="utf-8",
    )
    return paths


def read_snapshot(root: Path, gameweek: int) -> dict[str, Any] | None:
    """Read back a snapshot, or ``None`` if that gameweek was never captured."""
    directory = snapshot_directory(root, gameweek)
    meta_path = directory / "meta.json"
    if not meta_path.exists():
        return None

    return {
        "meta": json.loads(meta_path.read_text(encoding="utf-8")),
        "players": pd.read_parquet(directory / "players.parquet"),
        "schedule": pd.read_parquet(directory / "schedule.parquet"),
    }


def available_gameweeks(root: Path) -> list[int]:
    """Gameweeks that have a snapshot on disk, ascending."""
    root = Path(root)
    if not root.exists():
        return []
    gameweeks = [
        int(path.name[2:])
        for path in root.iterdir()
        if path.is_dir() and path.name.startswith("gw") and path.name[2:].isdigit()
    ]
    return sorted(gameweeks)
