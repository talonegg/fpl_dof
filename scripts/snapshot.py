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
from fpl.sources.snapshot import write_snapshot  # noqa: E402


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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
