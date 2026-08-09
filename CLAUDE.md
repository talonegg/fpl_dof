# FPL DOF — Fantasy Premier League Director of Football

A personal FPL decision-support web app: player scouting, expected-points modelling,
squad optimisation, and transfer recommendations.

## Prime directive: separate the brain from the screen

**`fpl/` must never import `streamlit`.** All data loading, feature engineering,
modelling, and optimisation lives in `fpl/` as pure, testable functions. The
Streamlit app in `app/` is a thin rendering layer that calls into `fpl/`.

If you catch yourself putting a calculation inside a `.py` file under `app/`,
move it to `fpl/` and write a test for it.

## Layout

```
streamlit_app.py        Entry point. Composes only -- it does not decide. Must stay
                        at the repo root: Streamlit puts the entry script's
                        directory on sys.path, which is what makes `import fpl`
                        work on Streamlit Cloud with no install.
app/                    Streamlit UI only. Tabs are registered in app/registry.py;
                        each declares which filters it honours.
fpl/
  sources/              Fetching raw data. Must not import domain -- a fetcher
                        returns bytes, it does not know what a player is.
  domain/               Types + pure transforms (players, fixtures, positions,
                        rules, identity). Must not import models.
  store/                Persisting domain objects (snapshots, parquet cache).
  features/             Derived metrics. Every derivation is a frame -> frame
                        function registered in features/registry.py.
  models/               Expected-points predictors. Each implements the Predictor protocol.
  optimise/             MILP squad/transfer optimisation (PuLP + CBC)
  backtest/             Historical replay harness + evaluation metrics
data/
  cache/                Local parquet/DuckDB cache. Gitignored.
  fixtures/             Small frozen JSON snapshots used by tests. Committed.
tests/                  pytest. Mirrors the fpl/ tree.
```

## Commands

```powershell
.\venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt   # includes requirements.txt
streamlit run streamlit_app.py
pytest                                # unit tests, must be fast + offline
pytest -m backtest                    # slow model-evaluation runs
ruff check . ; ruff format .
```

Note `;` not `&&` — this is Windows PowerShell 5.1, which has no `&&`.

## Non-negotiables

**Tests never hit the network.** Every source module takes an injectable fetcher.
Tests feed it frozen JSON from `data/fixtures/`. If a test needs new data, add a
snapshot file rather than mocking ad hoc in the test body.

**Models are evaluated, not just unit-tested.** A unit test proves the code runs.
A backtest proves the model is worth using. Every predictor added to `fpl/models/`
needs a `pytest -m backtest` case reporting its metrics against the baselines in
`fpl/backtest/baselines.py`. A model that does not beat the benchmark
(`SeasonMeanPredictor`) on held-out gameweeks does not get wired into the UI.

Five measured improvements have now failed to move selection significantly.
Prediction quality is not the binding constraint; squad construction, transfer
timing and captaincy are comparatively untouched.

**Evaluate on all four seasons, never one.** `scripts/backtest_seasons.py` is the
authority; `scripts/backtest.py` is single-season and kept only for quick
iteration. This is not pedantry — on 2025-26 alone the component model looked
*indistinguishable* from the benchmark at selection; across four seasons it is
*significantly worse*. One season had the sign wrong.

**But the seasons are not scored under the same rules.** Defensive contribution
points arrived in 2025-26 and continue in 2026-27; the three earlier seasons had
no such route to points. So 2025-26 is the only season whose rules match the one
being played, and on it the pooled ordering reverses — the season mean drops to
fourth. Read the per-season table, not just the pooled mean, and weight
2025-26 accordingly. A second current-rules season (2026-27) is what would
settle it.

**Never evaluate a live-only signal on historical data.** Injury status,
`chance_of_playing_next_round` and betting odds are published only for *now* —
nobody recorded who was injured in gameweek 12 of 2023-24. Running them over an
archive season does not fail, it returns "everyone fit" and the signal silently
contributes nothing, producing a backtest number that looks fine and means
nothing. `fpl/features/availability.py` raises `AvailabilityUnavailable` rather
than defaulting; keep that behaviour for any new live-only source. These signals
earn their place through live use and forward testing, not backtests — which
means the daily snapshots on the `data` branch are what will eventually make a
real evaluation possible.

**Model defensive contributions.** 2 points for clearing a threshold of defensive
actions: 10 CBIT for defenders, 12 CBIRT for midfielders and forwards,
goalkeepers ineligible. `ComponentPredictor` scores them and degrades to zero on
seasons lacking the column — correct for those seasons, but it means pre-2025-26
results understate any DC-aware model.

**Score over a horizon, and count turnover.** `fpl/backtest/horizon.py` scores
a prediction against the five-to-seven gameweeks you actually hold a squad for,
and reports how much the top fifteen churns between weeks. Both matter: the
season mean wins at every horizon and the gap *widens* tenfold from one week to
six, because its picks barely change (14% turnover against the component
model's 35%). See `docs/horizon-and-minutes.md`.

**Recency wins for minutes, stability wins for points.** The season average is
the best points predictor and the *worst* minutes forecaster. Scoring rate is a
stable property of a player; availability is a volatile property of their
situation. Do not apply one lesson to the other problem.

**Rank correlation is a diagnostic, not a target.** Ranking skill and selection
skill are inverted in this problem: the season mean is the worst ranker in the
field and the best selector, and every model that ranks better picks worse. Rank
correlation is dominated by the many players who score nothing; the top fifteen is
a question about the tail. Optimising ranking has so far made selection worse.

**Minutes are the binding constraint on squad selection, not points modelling.**
The season-opening backtest is unambiguous: the same model went from 19% of the
achievable ceiling to 55% purely by adding a minutes forecast, while every
refinement on top of that — component scoring, fixture difficulty, forcing the
full budget — moved it by a few points at most and none of them consistently.
A per-90 rate says how good a player is *while on the pitch*; multiplied by a
constant it buys substitutes. `fpl/models/minutes_forecast.py:PreseasonMinutes`
is the pre-season case. See `docs/season-opening-squad.md` §10.

**Forcing the full budget does nothing — it was a symptom, not a cause.**
`SquadConstraints.min_spend` exists and is correct, but every strategy with a
minutes term already spends £100.0m unprompted. Underspending was what a model
that over-rates bench players looks like, not a separate defect to constrain
away.

**Season-opening selection is four modules, not one.** `features/preseason_pool.py`
assembles the candidate pool (prices + career rates + defensive rates);
`models/preseason_strategies.py` is the registry of ways to value a player;
`optimise/preseason.py` constructs and explains the squad;
`backtest/preseason.py` only replays. Add a model to `strategies()` and it is
measured against every benchmark automatically — never construct one at a call
site, or the comparison silently stops being complete. `PreseasonContext` is
handed prior seasons and opening prices but never the target season, so the
point-in-time guarantee is structural rather than a rule each strategy has to
remember.

**Defensive contributions have three states, not two.** `not scored` (before
2025-26 — zero is *correct*), `forecast` (2026-27 onward, from 2025-26 data),
and `blind` (2025-26 itself: the rule applied but no prior season recorded the
actions, so 8.2% of points were invisible to any model). Gate on
`domain/rules.py:season_scores_defensive_contributions`, never on whether a
column exists — that would make "did not exist" and "existed but unrecorded"
indistinguishable. DC is worth 13.6% of defender points and a player's rate
persists within a season (r = 0.64). 2018-19 is the only other season carrying
the action counts and is unusable: no `position` column, and the threshold is
positional.

**The fixture curve is flat then decaying, and lives in `features/`.**
`[1.00, 1.00, 1.00, 0.70, 0.49, 0.34, 0.24]` over GW1–7: the opening three are
held for certain, the tail is not. It lives in `features/team_strength.py`
because two consumers must share it — inside `PreseasonPredictor` the summed
weights are a *uniform scalar* that cannot reorder players, so the shape only
reaches selection through `opening_run_difficulty`, which decides which
opponents count.

**Score the opening squad at three horizons, not one.** `SCORING_HORIZONS =
(3, 5, 7)` — certain hold, middle, and roughly where a free transfer a week has
rebuilt the squad. Compare shares of each horizon's *own* ceiling; raw points
across horizons are not comparable.

**Recommend twenty squads, not one.** `fpl/optimise/ranking.py` enumerates the
true ranked top N by **no-good cuts** — solve, forbid that exact combination of
fifteen, solve again — so entry twenty is the twentieth-best squad rather than
a perturbation. Excluding the *players* of each squad found instead would skip
thousands of better squads, because the second-best squad usually shares
thirteen or fourteen players with the best. The shortlist exists because the
spread between first and twentieth is typically inside the prediction's error:
showing one squad reads as an answer, showing twenty reads as the set of
near-equivalent options it actually is. The "Season opener" tab says which of
those two it is before showing the table.

**A squad-shaped tab honours no filter, and says so.** A squad must be legal —
fifteen players, two goalkeepers, at most three per club — so it cannot be
drawn from a pool narrowed to a few clubs or one position. That is a third tab
shape alongside player-shaped and team-shaped; `tests/test_views.py` encodes
which tabs may ignore filters, so a new one cannot quietly opt out.

**Expected points and optimisation stay separate.** The predictor answers "how many
points will this player score in GW N". The optimiser answers "given those numbers
and the FPL rules, what is the best squad". Never let a heuristic about budget or
team limits leak into a predictor.

**Squad selection is solved; transfer timing depends on the predictor.**
`fpl/optimise/squad.py` returns a provably optimal squad — trust it.
`fpl/optimise/transfers.py` beats holding when fed a *stable* predictor
(SeasonMean: +26 points over 15 gameweeks, 1 hit) and loses badly when fed a
*volatile* one (Component: −114, 14 hits), because it scales one gameweek's edge
by the horizon and so churns on noise. See `docs/optimiser-results.md`. Before
wiring transfer recommendations into the UI, run the season simulation with the
predictor you intend to use — the answer is not the same for all of them.

**The layering is tested, not just described.** `tests/test_architecture.py`
reads the imports and fails on any upward dependency. Seven layers: sources,
domain, store, features, models, optimise, backtest. Two violations existed
before it was written -- position vocabulary living in `models/`, and
`snapshot.py` fetching *and* writing from inside `sources/` -- so the test is
load-bearing rather than decorative.

**Add a derivation to the catalogue, not to a call site.** `features/registry.py`
declares what each derivation requires and provides. `enrich()` applies the
applicable ones and reports what it skipped, which is how live-only signals
(availability, set-piece duty) correctly produce fewer columns on historical
data instead of raising or inventing values.

**FPL rules live in one place** — `fpl/domain/rules.py`. Budget 100.0, 15-player
squad (2 GK / 5 DEF / 5 MID / 3 FWD), max 3 per club, valid starting XI formations,
transfer cost 4 points per extra transfer. Do not hardcode these anywhere else.

## Data model

`docs/data-model.md` holds the entity model and the grain of every dataset;
`docs/data-sources.md` maps each of its 91 elements to the source field or the
function behind it. The mapping lives in `fpl/domain/lineage.py` and is
enforced by `tests/test_lineage.py`, which fetches the real feeds and fails if
the model claims a field nobody publishes. Regenerate the document with
`python scripts/data_sources.py` rather than editing it.

Adding a field to the model means adding it to `lineage.py` too, or the test
fails. That is deliberate: a model listing a column that will always be empty
is worse than one that omits it.

## Refresh and orchestration

`docs/refresh-schedule.md` documents when each dataset refreshes and why. The
policy lives in `fpl/store/refresh.py`, and `tests/store/test_refresh.py`
checks it against the actual cron lines and cache TTLs — so the schedule cannot
drift from the document describing it.

Two scheduled runs, both `.github/workflows/snapshot.yml`. **06:00 UTC** writes
the append-only daily file; **11:30 UTC** refreshes only the gameweek snapshot,
because the earliest deadline is 12:30 UTC and a 06:00 capture is 6.5 hours
stale by then. Never make the second run overwrite the daily file: that would
replace the morning's injury news with the afternoon's and destroy the
point-in-time property.

## Data conventions

- Prices from the API are integer tenths (`now_cost: 55` → £5.5m). Convert once,
  at the source boundary, never in the UI.
- **Many API "numbers" are JSON strings** (`form`, `points_per_game`,
  `selected_by_percent`, `expected_goals`, `ict_index`, …). Untouched they sort
  lexicographically — `"9.5"` above `"12.0"`. `fpl/domain/players.py` coerces
  them at the boundary; add any new such column to `NUMERIC_STRING_COLUMNS`.
- Player identity is `element` id, which is **not stable across seasons**. Any
  cross-season join must go through the name+team mapping in `fpl/domain/identity.py`.
  `element_code` (in `history_past`) *is* stable and is preferred where available.
- The API serves only the current season, and only *now*. Past-season per-gameweek
  data comes from `fpl/sources/archive.py`; point-in-time captures of the live
  state come from `fpl/sources/snapshot.py`, written daily by
  `.github/workflows/snapshot.yml` onto the `data` branch. Never reconstruct
  "what was true before gameweek N" from today's API — that is lookahead.
- All cached data is timestamped. Anything older than the current gameweek deadline
  is stale and must be refetched.

## External sources

**Understat and FBref are off-limits.** Understat's `robots.txt` is
`User-agent: * / Disallow: /`; FBref sits behind a Cloudflare challenge that
403s even on `robots.txt`. Both were named in the roadmap; neither can be used,
and getting past a bot challenge is precisely the evasion these rules exist to
prevent. What they were wanted for — set-piece duties and shot quality — turns
out to be published officially: `penalties_order`,
`corners_and_indirect_freekicks_order`, `direct_freekicks_order` and the
`expected_*` family all come from `bootstrap-static`. Check the official API
before reaching for a scraper.

Respect `robots.txt` and site terms. Rate-limit every scraper, cache aggressively,
identify the client with a real User-Agent, and prefer official APIs where they
exist. Never commit API keys — they come from environment variables, read via
`fpl/config.py`, and are set as secrets in the hosting platform.

Social/pundit content is **sentiment input, never ground truth**. Anything derived
from it must be surfaced in the UI as an opinion signal, attributed to its source,
and kept out of the core expected-points model unless a backtest justifies it.

## UI conventions

The app is used on laptop, tablet, and phone. Assume a 375px-wide viewport works.
Data tables get a curated narrow column set on small screens. Prefer `st.tabs`
and vertical stacking over wide multi-column layouts. Test any new page at phone
width before considering it done.

### Adding a tab

One entry in `app/registry.py` and one render function taking a `ViewContext`.
Do not edit `streamlit_app.py`. The view declares which filters it honours and
is handed an already-filtered frame, so the contract below is structural rather
than something to remember.

### Filters

The sidebar filters are global: rendered once in `streamlit_app.py` before any
tab, and applied to every tab that can honour them. **Club** reaches all three
tabs including Fixtures; **position and price** reach the player-shaped tabs
only, because a fixture has neither. A control that silently applies to one tab
out of three is worse than no control, so any new tab must either consume the
filter or say why it cannot.

Selection widgets whose options depend on the filter must prune their remembered
state (`_prune_selection` in `app/scouting_view.py`) — Streamlit hands back a
stored value even when the current options no longer contain it.

### Charts

- **Never a dual-axis chart.** Two measures of different scale get two charts.
  Sliding two y-scales against each other can make any story appear, which is
  disqualifying for something used to spend money.
- Series colours come from `app/theme.py`, assigned **by fixed slot, never
  cycled**, so removing a series does not repaint the others. The palette is
  validated for colour-vision-deficiency separation and contrast in both light
  and dark; do not add a fifth colour — cap the series and say so.
- Two of the four light-mode series colours sit below 3:1 contrast, so any view
  with more than two series must also expose the numbers as a table.
- Charts are not unit-tested; the frames behind them are. Reshaping logic
  belongs in a tested helper, not inline in a `render` function.

## Style

- Python 3.12+, type hints on every public function in `fpl/`.
- Pandas for tabular work; return new frames rather than mutating in place.
- Docstrings explain *why* a calculation is done, not what the code says.
- Keep `requirements.txt` as declared direct dependencies with pinned versions —
  it is not a `pip freeze` dump.
