"""Capture a point-in-time snapshot of the live FPL state.

Run by ``.github/workflows/snapshot.yml``; also runnable by hand:

    python scripts/snapshot.py --root snapshots
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running as a plain script from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fpl.sources.fpl_api import fetch_bootstrap, fetch_fixtures  # noqa: E402
from fpl.store.snapshot import (  # noqa: E402
    captured_dates,
    write_daily_signals,
    write_snapshot,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        default="snapshots",
        type=Path,
        help="directory to write snapshots into (default: snapshots)",
    )
    args = parser.parse_args()

    bootstrap = fetch_bootstrap()
    fixtures = fetch_fixtures()

    try:
        paths = write_snapshot(bootstrap, fixtures, root=args.root)
    except ValueError as error:
        # The season being over is a normal end state, not a failure.
        print(f"nothing to snapshot: {error}")
        return 0

    for name, path in paths.items():
        print(f"wrote {name}: {path}")

    # The append-only daily capture. Separate from the gameweek snapshot
    # because it must never overwrite an earlier day.
    daily = write_daily_signals(bootstrap, root=args.root)
    if daily is None:
        print("daily signals already captured for today; left untouched")
    else:
        print(f"wrote daily signals: {daily}")
    print(f"daily captures held: {len(captured_dates(args.root))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
