"""A persisted list of players to keep an eye on.

Stored as player *codes* rather than ``element`` ids: codes survive the season
rollover, so a watchlist built in May still means something in August. Ids do
not.

The file format is a plain JSON list so it can be read, edited or thrown away
by hand without the app running.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def load(path: Path) -> list[int]:
    """Read the watchlist, returning an empty list when there is not one yet.

    A corrupt file is treated as empty rather than fatal: losing a watchlist is
    an annoyance, but refusing to start the app over it is worse.
    """
    path = Path(path)
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return []
    if not isinstance(data, list):
        return []
    return [int(code) for code in data if isinstance(code, int | str) and str(code).isdigit()]


def save(path: Path, codes: list[int]) -> Path:
    """Write the watchlist, de-duplicated and in a stable order."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    unique = sorted(set(int(code) for code in codes))
    path.write_text(json.dumps(unique, indent=1) + "\n", encoding="utf-8")
    return path


def toggle(codes: list[int], code: int) -> list[int]:
    """Add ``code`` if absent, remove it if present. Returns a new list."""
    code = int(code)
    if code in codes:
        return [existing for existing in codes if existing != code]
    return [*codes, code]


def filter_players(players: pd.DataFrame, codes: list[int]) -> pd.DataFrame:
    """Rows of ``players`` that are on the watchlist, in the frame's own order."""
    if not codes:
        return players.iloc[0:0]
    return players[players["code"].isin(codes)]
