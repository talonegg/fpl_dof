# Refresh schedule and orchestration

How each dataset stays current, what runs it, and why that cadence.

The policy lives in `fpl/store/refresh.py` and `tests/store/test_refresh.py`
checks it against the actual cron lines and cache TTLs — a schedule document
that disagrees with the code is worse than none, because it gets believed.

## The schedule

| Time (UTC) | What runs | Writes |
|---|---|---|
| **06:00 daily** | `scripts/snapshot.py` | daily signals (append-only) + gameweek snapshot |
| **11:30 daily** | `scripts/snapshot.py` | gameweek snapshot only (overwrites) |
| on push / PR | CI | nothing; tests only |
| on demand | the app | nothing; reads through caches |

Both scheduled runs are `.github/workflows/snapshot.yml`, which also accepts
`workflow_dispatch` — the first run ever had to be triggered by hand, and that
remains the way to test a change without waiting.

## Why two runs

**Deadlines are early afternoon.** Measured across all 38 gameweeks of 2026-27:

| | |
|---|---|
| Earliest deadline | **12:30 UTC** |
| 29 of 38 fall between | 12:00 and 14:00 UTC |
| Latest | 18:30 UTC |
| Most common day | Saturday (27 of 38) |

A capture at 06:00 is therefore **6.5 hours stale** by the time the earliest
deadline passes — and the hours immediately before a deadline are exactly when
team news lands. The 11:30 run exists to close that gap.

The two runs do different jobs, and the existing code already enforces the
split:

- **06:00** writes the append-only daily file. This is the only record that
  will ever exist of injury status, set-piece duty, price and ownership.
- **11:30** re-runs and finds the daily file already written, so leaves it
  alone (`write_daily_signals` refuses to overwrite a day). It overwrites the
  *gameweek* snapshot, which is meant to converge on the pre-deadline state.

That asymmetry is deliberate. Overwriting the daily file at 11:30 would
replace the morning's injury news with the afternoon's and destroy the
point-in-time property the capture exists for.

## Per-dataset policy

| Dataset | Cadence | Max age | Orchestrated by |
|---|---|---|---|
| bootstrap (players, teams, events) | on demand | 1 hour | `app/data.py` `@st.cache_data` |
| fixtures | on demand | 1 hour | `app/data.py` `@st.cache_data` |
| odds | on demand | 6 hours | `fpl/config.py` `ODDS_CACHE_SECONDS` |
| archive (past seasons) | **immutable** | never expires | `fpl/store/cache.py` `NEVER_STALE` |
| daily signals | daily | 1 day | workflow, 06:00 |
| gameweek snapshot | pre-deadline | 12 hours | workflow, 06:00 and 11:30 |

Three of these deserve their reasoning stated:

**Odds are cached for six hours, not one.** The free tier is 500 requests a
month — about 16 a day across a season. Odds barely move outside the hours
before kick-off, and losing the source to quota exhaustion is worse than
slightly stale prices.

**The archive never expires.** A finished season cannot change. It is the only
dataset with no expiry, and re-fetching it is pure waste.

**The app caches for an hour.** Long enough that widget interaction does not
re-hit the API — the bug fixed in Phase 0 — and short enough that a price
change or injury shows up the same session.

## What is not automated

| Thing | Why not |
|---|---|
| Backtests | Minutes to run and they download several seasons; run deliberately with `pytest -m backtest` |
| `docs/model-results.md` | Regenerate with `scripts/backtest.py` when a model changes |
| `docs/multi-season-results.md` | Regenerate with `scripts/backtest_seasons.py` |
| `docs/data-sources.md` | Regenerate with `scripts/data_sources.py` when the model changes |
| Odds capture | No `ODDS_API_KEY` is configured, so nothing fetches them yet |

## Known gaps

**Late deadlines are still under-served.** Five gameweeks have deadlines at
17:00 or 18:30 UTC, so the 11:30 capture is 5.5 to 7 hours early for those. A
third run would fix it; it has not been added because the gain is small and
every extra run is another commit to the `data` branch.

**Nothing captures the state *after* a deadline but before kick-off.** Lineups
are announced an hour before kick-off, and that information is never captured.
It is also not currently used by anything.

**Odds are not on any schedule.** They are fetched on demand by the app only.
Once a key exists, a pre-deadline odds capture would be the natural addition —
prices are most informative closest to kick-off, which is also when the quota
is best spent.

**A missed run is not detected.** If the workflow fails, nothing notices until
someone looks. The daily files are named by date, so a gap is visible after the
fact via `captured_dates()`, but there is no alert.
