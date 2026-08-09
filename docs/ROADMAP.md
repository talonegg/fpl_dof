# FPL DOF — Incremental Build Roadmap

Each phase is independently useful and shippable. Do not start a phase until the
previous one is deployed and green.

---

## Phase 0 — Foundations (1 sitting)

Nothing new for the user; everything after this depends on it.

- Restructure into `fpl/` (logic) + `app/` (UI). Move `data/reference.py` →
  `fpl/domain/reference.py`.
- Add `pytest`, `ruff`, `pyproject.toml`. First test: `add_readable_columns`
  against a frozen `bootstrap-static` snapshot in `data/fixtures/`.
- Replace the frozen `requirements.txt` with declared direct deps; add
  `requirements-dev.txt`.
- Wrap the API call in `@st.cache_data(ttl=3600)`. Currently every widget change
  re-hits the FPL API.
- GitHub Actions: run `ruff` + `pytest` on push.
- Deploy the skeleton to Streamlit Community Cloud so the deploy path is proven
  while the app is trivially small.

**Done when:** CI is green and the live URL loads on your phone.

---

## Phase 1 — Data layer with history

The current app only sees a snapshot of *today*. Every model needs per-gameweek history.

- `fpl/sources/fpl_api.py`: `bootstrap-static`, `fixtures/`, `element-summary/{id}/`
  (per-player GW history), `entry/{id}/` (your own team). All take an injectable
  fetcher.
- `fpl/sources/archive.py`: bulk historical seasons from the community
  `vaastav/Fantasy-Premier-League` dataset for backtesting.
- Local cache as parquet under `data/cache/` via a small `fpl/sources/cache.py`.
- A scheduled GitHub Action that fetches post-deadline snapshots and commits them
  to a `data` branch — free, versioned, no database needed.

**Done when:** you can load any past gameweek offline, and tests run with zero network.

---

## Phase 2 — Scouting UI

The human-assisted scouting layer. This is where you get daily value.

- Player detail view: GW-by-GW points, minutes, xG/xA vs actuals, price changes.
- Comparison mode: 2–4 players on shared axes.
- Fixture ticker: next 5–7 GWs per team, coloured by difficulty, sortable.
- Derived metrics the raw API lacks: points per 90, xGI per 90, minutes-share
  trend, value (points per £m), form-vs-fixture-adjusted.
- Watchlist persisted in `st.session_state` + a local JSON file.

**Testing:** every derived metric is a pure function in `fpl/features/` with a
unit test on a hand-computed example. Charts are not tested; the numbers behind
them are.

---

## Phase 3 — Expected points model

The core intellectual work. Build it in ascending order of ambition, keeping each
as a named baseline you must beat.

1. `NaiveFormPredictor` — last-N-GW average. The floor.
2. `MinutesAdjustedPredictor` — P(start) × points-per-90, with fixture difficulty.
3. `ComponentPredictor` — model the FPL scoring components separately (goals from
   xG rate, assists from xA, clean sheet probability from team defensive strength,
   bonus from BPS history) and sum them. This is the one likely to actually work.
4. Optional later: gradient boosting over the engineered features.

Every predictor implements the same protocol:
`predict(players, fixtures, horizon_gws) -> DataFrame[player_id, gw, xpts]`.

**Testing = backtesting.** `fpl/backtest/` replays past seasons gameweek by
gameweek with strict point-in-time data (no lookahead — this is the easiest bug
to write and the hardest to notice). Metrics: MAE, Spearman rank correlation,
and top-20 precision. Report all models side by side in a table committed to
`docs/model-results.md` on every change.

---

## Phase 4 — Squad optimiser

- MILP with PuLP/CBC in `fpl/optimise/squad.py`: maximise summed expected points
  over the horizon subject to budget, 2/5/5/3, max 3 per club, valid XI, captain
  doubling.
- `fpl/optimise/transfers.py`: given your current squad, find the transfer plan
  maximising net points over the horizon, charging 4 points per hit. Include
  "roll the transfer" as a candidate.

**Testing:** the optimiser is deterministic, so test it hard — constructed cases
with a known optimum, constraint-violation assertions on random inputs, and a
regression test on a fixed input/seed. Separately, a season-level backtest:
"model + optimiser starting from GW1 would have scored X vs the template team's Y".
That number is the honest measure of the whole system.

---

## Phase 5 — External data sources

Only now, once you have a backtest that can tell you whether a new signal helps.

- **Betting odds** (highest value): The Odds API free tier, or OddsPortal.
  Anytime-goalscorer and clean-sheet odds are a strong market-implied prior —
  fold them into the `ComponentPredictor` and measure the delta.
- **Advanced stats**: Understat / FBref for shot quality, touches in the box,
  set-piece duties.
- **Press/injury news**: FPL API's own `news` and `chance_of_playing_next_round`
  first — it is more reliable than scraping.
- ~~**Pundits/social**~~ — **deferred to the backlog.** See below.

Each source is a module implementing a common `Source` protocol with its own
rate limiter and cache TTL, so a failing source degrades the app rather than
breaking it.

**Testing:** contract tests against recorded HTTP responses, so a site changing
its markup fails CI loudly instead of silently poisoning the model.

---

## Phase 6 — Polish

- Responsive pass at 375px / 768px / 1280px.
- "My Team" import via your FPL entry id, with recommendations relative to your
  actual squad.
- Weekly digest: a scheduled Action that runs the model post-deadline and writes
  a markdown summary.

---

## Stack decisions

**Streamlit, kept swappable.** It is the fastest path from pandas to a deployed
app, free on Streamlit Community Cloud, and you already have it running. The
honest caveat: Streamlit's mobile UX is adequate, not great — wide dataframes
scroll awkwardly on a phone. The `fpl/` + `app/` split is the insurance policy:
if mobile becomes the primary use case, you replace `app/` with a React frontend
over a FastAPI wrapper around the same `fpl/` package, without touching any
modelling code.

**No database initially.** Parquet in a git branch, written by a scheduled Action,
is free, versioned, and diffable. Move to Supabase Postgres only when you need
concurrent writes or per-user state.

**Hosting:** Streamlit Community Cloud (free, sleeps when idle). Fly.io or Hugging
Face Spaces are the fallbacks if you outgrow it.

---

## Backlog

Things worth doing eventually, deliberately not scheduled.

### Pundit and social sentiment

YouTube Data API + transcripts, RSS from FPL blogs, LLM extraction into
structured `(player, sentiment, claim, source, date)` rows, surfaced as an
opinion panel.

Deferred because it is the weakest signal in Phase 5 and the most expensive to
build: it needs an extraction pipeline, an API key, and ongoing prompt
maintenance, and it is *live-only*, so it can never be justified by a backtest
the way `CLAUDE.md` requires. Everything cheaper has been done first.

If it is picked up: it stays an opinion panel attributed to its source, and it
does not enter the expected-points model.

### Set-piece and price-change history

`fpl/sources/snapshot.py` now records these daily. Once a season of captures
exists, "does a change in penalty duty move points" becomes answerable —
which is the only route to evaluating any live-only signal.

## The trap to avoid

The tempting order is scrape everything → build a big model → ship. The order
above is deliberately the reverse: get a working backtest harness early, so every
subsequent addition can be *measured*. Without it you will accumulate data sources
and features with no way to know whether any of them made your team better.
