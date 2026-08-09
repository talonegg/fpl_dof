"""Point-in-time snapshots of the live FPL state.

The API only ever serves *now*. Once a gameweek's deadline passes, the prices,
ownership and form that were true beforehand are gone for good. A model
backtested against data reconstructed after the fact will look better than it
could ever have been in practice, because it quietly knows things it could not
have known.

Snapshotting on a schedule is the cheap fix: capture the state, stamp it with
the gameweek it belongs to, and keep it. The files are the raw material for
honest point-in-time backtests later.

Two different captures, on purpose:

**Per gameweek** (:func:`write_snapshot`) overwrites, converging on the state
just before each deadline. That is what a point-in-time backtest replays.

**Per day** (:func:`write_daily_signals`) appends, one file per date, and never
rewrites an existing one. This is the only record that will ever exist of the
live-only signals — injury status, set-piece duty, price, ownership — because
none of them appear in the historical archive. A daily file is 28KB, about
10MB a season; the full player table would be seven times that for columns
that do not change.
"""

from __future__ import annotations

import json
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from fpl.domain.fixtures import build_team_schedule, next_gameweek
from fpl.domain.players import build_players_frame

# The fields worth keeping every day: the ones that change, and that no
# historical source records. Everything omitted is either static (name, team)
# or recoverable from the archive after the season (goals, minutes).
DAILY_SIGNAL_COLUMNS = (
    # Identity. `code` is the stable cross-season id; `element` is not.
    "element",
    "code",
    "web_name",
    "team",
    "element_type",
    # Availability — the whole reason this capture exists.
    "status",
    "chance_of_playing_this_round",
    "chance_of_playing_next_round",
    "news",
    "news_added",
    # Set-piece duty, which changes during a season and is never archived.
    "penalties_order",
    "corners_and_indirect_freekicks_order",
    "direct_freekicks_order",
    # Price and market state. Prices move daily and the archive keeps only the
    # value at the moment a gameweek was played.
    "now_cost",
    "cost_change_event",
    "selected_by_percent",
    "transfers_in_event",
    "transfers_out_event",
    # Cheap context for interpreting the above.
    "form",
    # Defensive contribution inputs. Kept alongside the published total so the
    # threshold stays recomputable if the rule changes, and so the identity
    # between them can be checked rather than assumed.
    "clearances_blocks_interceptions",
    "tackles",
    "recoveries",
    "defensive_contribution",
    "bps",
    "total_points",
    "minutes",
)

DAILY_DIRECTORY = "daily"


def daily_path(root: Path, captured_on: date) -> Path:
    """Path of the append-only capture for a single day."""
    return Path(root) / DAILY_DIRECTORY / f"{captured_on.isoformat()}.parquet"


def write_daily_signals(
    bootstrap: dict[str, Any],
    root: Path,
    captured_on: date | None = None,
    overwrite: bool = False,
) -> Path | None:
    """Append today's live-only signals, or return ``None`` if already captured.

    Refuses to overwrite an existing day by default. These files are the only
    record of what was true on a given date, and a re-run later in the day
    would quietly replace the morning's injury news with the evening's —
    destroying exactly the point-in-time property the capture exists for.
    """
    captured_on = captured_on or datetime.now(UTC).date()
    path = daily_path(root, captured_on)

    if path.exists() and not overwrite:
        return None

    players = build_players_frame(bootstrap)
    columns = [column for column in DAILY_SIGNAL_COLUMNS if column in players.columns]
    daily = players[columns].copy()
    daily["captured_on"] = captured_on.isoformat()
    daily["gameweek"] = next_gameweek(bootstrap.get("events", []))

    path.parent.mkdir(parents=True, exist_ok=True)
    daily.to_parquet(path, index=False)
    return path


def read_daily_signals(root: Path) -> pd.DataFrame:
    """Every daily capture, concatenated, oldest first.

    The frame a future evaluation of any live-only signal will start from.
    """
    directory = Path(root) / DAILY_DIRECTORY
    if not directory.exists():
        return pd.DataFrame()

    frames = [pd.read_parquet(path) for path in sorted(directory.glob("*.parquet"))]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def captured_dates(root: Path) -> list[str]:
    """Dates that have a daily capture, ascending."""
    directory = Path(root) / DAILY_DIRECTORY
    if not directory.exists():
        return []
    return sorted(path.stem for path in directory.glob("*.parquet"))


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
