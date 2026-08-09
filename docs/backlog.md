# Features backlog

Recommended next work, ordered by value per unit of effort. Written after the
Phase 6 architecture review, when adding a feature became cheap enough that
*choosing* which one matters more than building it.

Each item names what it unblocks and what it depends on. Anything whose
prerequisite has not landed is marked, because the ordering is not arbitrary.

**Last reviewed:** after the season-opening squad constructor shipped.

## What adding a feature costs now

- **A new tab**: one entry in `app/registry.py` and one render function. The
  entry point does not change, and the filter contract is declared rather than
  remembered.
- **A new derived column**: one entry in `fpl/features/registry.py`. Every
  caller picks it up; `provider_of()` answers where it came from.
- **A new predictor**: implement the `Predictor` protocol, add to
  `baselines.py`, and the comparison table includes it.
- **A new season-opening model**: add it to `strategies()` in
  `fpl/models/preseason_strategies.py` and it is measured against every
  benchmark, season and horizon automatically.
- **A new external source**: implement `Source`, and a failure degrades that
  panel rather than the app.

## Done since the last review

| Item | Where it landed | What it measured |
|---|---|---|
| Horizon-based selection metric | `fpl/backtest/horizon.py` | Season mean wins at every horizon; gap widens tenfold |
| Minutes model | `fpl/models/minutes_forecast.py` | Recency wins for minutes, stability for points |
| Season-opening squad constructor | four modules, see below | 19% → 52% of ceiling from the minutes term alone |
| Squad tab in the UI | `app/preseason_view.py` | Top twenty squads span 0.14% — the ranking is noise |
| Defensive contributions | `fpl/features/defensive.py` | 13.6% of defender points; forecastable from 2026-27 |
| Three-horizon scoring | `SCORING_HORIZONS = (3, 5, 7)` | Model edge is real at GW3, gone by GW7 |

The season-opening work is now four modules rather than a plan:
`features/preseason_pool.py` (candidate pool), `models/preseason_strategies.py`
(six registered models), `optimise/preseason.py` (constructor and recommender),
`backtest/preseason.py` (replay only).

### What that work changed about the priorities below

**Prediction quality is not the binding constraint, and this is now the sixth
measurement saying so.** Six refinements — component scoring, xG over actuals,
penalty correction, horizon scoring, fixture difficulty, forcing the full budget
— have failed to beat a per-90 rate scaled by expected minutes. Items that
propose *better expected points* should be read with that history in mind. Items
that address coverage, timing or presentation should be read more favourably
than they were a review ago.

## Tier 1 — do these next

### 1. Position-and-price prior for players with no history

**Effort: small. Value: the largest unmodelled gap in the constructor.**

**35% of the priced pool (239 of 690) carries no prior Premier League minutes**
and is dropped before the optimiser sees it. 160 of those are under £5.0m —
precisely the bench slots a real squad must fill. They are not rated poorly;
they are unrated, and the optimiser cannot buy them at all.

The design (§2, §5 of `season-opening-squad.md`) specifies a
`NewPlayerEstimator` seam and a price-band prior. It is the one gap where the
fix is cheap, the data already exists, and the current behaviour is a hard
restriction on the search space rather than a modelling nicety.

**Watch for:** this makes the pool larger, not better. The right test is whether
the backtest improves, not whether the squads look more complete.

### 2. Responsive pass at 375px

**Effort: small. Risk of not doing it: high.**

Still outstanding, and the Season opener tab has made it more pressing: it
renders a seven-column shortlist, a fifteen-row squad table and an expander.
`NARROW_COLUMNS` exists in `app/preseason_view.py` but nothing currently sets
`narrow=True` — the mechanism is built and unwired.

Needs Playwright (~150MB) or a manual pass at phone width.

### 3. "My Team" import

**Effort: medium. Blocked on:** your FPL entry id.

Every recommendation is currently in the abstract. The transfer optimiser
already works and the season-opening constructor already works; neither knows
what you actually own. This is the difference between a model and a tool.

## Tier 2 — worth doing once Tier 1 lands

### 4. Captaincy

**Effort: small. Never yet measured.**

Named in `CLAUDE.md` as comparatively untouched, and it remains so. The
optimiser picks a captain as a by-product of squad selection; nobody has asked
whether that choice is any good, or whether captaincy is where the recoverable
points are given that squad selection is solved and prediction is saturated.

A captain doubles a score, so it is the single highest-leverage weekly decision
and the cheapest thing left to evaluate.

### 5. Fix the penalty-goal distortion in the component model

**Effort: small.** Still open. xG includes penalties at ~0.76 each, so a
penalty taker's open-play rate is overstated. `fpl/features/penalties.py` has
the taker probabilities; the component model does not use them.

Lower priority than a review ago, because the component model is not currently
the recommended one — it loses to the simpler minutes-scaled rate.

### 6. Wire odds into team strength

**Effort: medium. Blocked on:** `ODDS_API_KEY`.

The market prices promotion and squad changes better than a blended historical
average can, and `TeamStrengthEstimator` is the seam. Live-only, so it earns its
place through forward testing rather than a backtest.

### 7. Transfer recommendations in the UI

**Effort: medium. Depends on:** item 3.

`fpl/optimise/transfers.py` beats holding when fed a *stable* predictor and
loses badly when fed a volatile one. Before wiring it in, run the season
simulation with the predictor you intend to ship — the answer is not the same
for all of them.

## Tier 3 — speculative, or waiting on data

### 8. A second current-rules season

**Waiting on:** 2026-27 completing. Nothing to build.

The single most valuable thing that could happen to this project, and it
requires only time. It would settle three open questions at once: whether the
season mean's pooled advantage survives current rules, whether defensive
contributions change model ordering once forecastable, and whether the models'
2025-26 defeat was caused by their blindness to that scoring route.

### 9. Measure the penalty-taker shares

**Waiting on:** a season of daily snapshots, now being captured.

### 10. Bonus points model

**Effort: medium.** `domain/bps.py` reconstructs 87% of BPS. The remaining 13%
is unpublished Opta data and is not closable.

### 11. DuckDB read layer

**Effort: medium.** Only worth it if parquet loading becomes the bottleneck. It
is not.

### 12. Weekly digest

**Effort: small.** A scheduled Action running post-deadline into a markdown
summary. Cheap, and more useful once item 3 lands.

### 13. Pundit and social sentiment

**Deferred, deliberately.** The weakest signal and the most expensive to build,
and *live-only* — it can never be justified by a backtest the way `CLAUDE.md`
requires. If picked up: an opinion panel attributed to its source, kept out of
the expected-points model.

## Explicitly not recommended

- **More expected-points refinements without a hypothesis about coverage or
  timing.** Six have now failed to move selection. The next one needs a reason
  to be different, stated before it is built.
- **Understat and FBref.** `robots.txt` forbids one and a bot challenge blocks
  the other. What they were wanted for is published officially.
- **A fifth chart colour.** The palette is validated for four; cap the series
  and say so.
- **Ranking-metric optimisation.** Rank correlation and selection skill are
  inverted in this problem. Every model that ranks better has picked worse.
