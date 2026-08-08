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
`fpl/backtest/baselines.py`. A model that does not beat `NaiveFormPredictor` on
held-out gameweeks does not get wired into the UI.

**Expected points and optimisation stay separate.** The predictor answers "how many
points will this player score in GW N". The optimiser answers "given those numbers
and the FPL rules, what is the best squad". Never let a heuristic about budget or
team limits leak into a predictor.

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

## Style

- Python 3.12+, type hints on every public function in `fpl/`.
- Pandas for tabular work; return new frames rather than mutating in place.
- Docstrings explain *why* a calculation is done, not what the code says.
- Keep `requirements.txt` as declared direct dependencies with pinned versions —
  it is not a `pip freeze` dump.
