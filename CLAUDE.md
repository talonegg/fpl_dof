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
streamlit_app.py        Entry point. Must stay at the repo root -- Streamlit puts
                        the entry script's directory on sys.path, which is what
                        makes `import fpl` work on Streamlit Cloud with no install.
app/                    Streamlit UI only. Pages, widgets, layout, caching wrappers.
fpl/
  sources/              Fetching raw data (FPL API, historical archives, odds, etc.)
  domain/               Types + pure transforms (players, fixtures, gameweeks)
  features/             Feature engineering (form, fixture difficulty, minutes risk)
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

**Model defensive contributions.** 2 points for clearing a threshold of defensive
actions: 10 CBIT for defenders, 12 CBIRT for midfielders and forwards,
goalkeepers ineligible. `ComponentPredictor` scores them and degrades to zero on
seasons lacking the column — correct for those seasons, but it means pre-2025-26
results understate any DC-aware model.

**Rank correlation is a diagnostic, not a target.** Ranking skill and selection
skill are inverted in this problem: the season mean is the worst ranker in the
field and the best selector, and every model that ranks better picks worse. Rank
correlation is dominated by the many players who score nothing; the top fifteen is
a question about the tail. Optimising ranking has so far made selection worse.

**Expected points and optimisation stay separate.** The predictor answers "how many
points will this player score in GW N". The optimiser answers "given those numbers
and the FPL rules, what is the best squad". Never let a heuristic about budget or
team limits leak into a predictor.

**Squad selection is solved; transfer timing depends on the predictor.**
`fpl/optimise/squad.py` returns a provably optimal squad — trust it.
`fpl/optimise/transfers.py` beats holding when fed a *stable* predictor
(SeasonMean: +16 points over 15 gameweeks, 1 hit) and loses badly when fed a
*volatile* one (Component: −88, 13 hits), because it scales one gameweek's edge
by the horizon and so churns on noise. See `docs/optimiser-results.md`. Before
wiring transfer recommendations into the UI, run the season simulation with the
predictor you intend to use — the answer is not the same for all of them.

**FPL rules live in one place** — `fpl/domain/rules.py`. Budget 100.0, 15-player
squad (2 GK / 5 DEF / 5 MID / 3 FWD), max 3 per club, valid starting XI formations,
transfer cost 4 points per extra transfer. Do not hardcode these anywhere else.

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
