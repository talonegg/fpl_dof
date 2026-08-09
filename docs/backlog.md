# Features backlog

Recommended next work, ordered by value per unit of effort. Written after the
Phase 6 architecture review, when adding a feature became cheap enough that
*choosing* which one matters more than building it.

Each item names what it unblocks and what it depends on. Anything whose
prerequisite has not landed is marked, because the ordering is not arbitrary.

## What adding a feature costs now

- **A new tab**: one entry in `app/registry.py` and one render function. The
  entry point does not change, and the filter contract is declared rather than
  remembered.
- **A new derived column**: one entry in `fpl/features/registry.py`. Every
  caller picks it up; `provider_of()` answers where it came from.
- **A new predictor**: implement the `Predictor` protocol, add to
  `baselines.py`, and the comparison table includes it.
- **A new external source**: implement `Source`, and a failure degrades that
  panel rather than the app.

## Tier 1 — do these next

### 1. Responsive pass at 375px

**Effort: small. Risk of not doing it: high.**

`CLAUDE.md` requires a phone to work and the app has never been *looked at* in
a browser — every verification so far has been through `AppTest`, which
exercises the render path but sees no layout. The app has grown from one table
to three tabs, a global sidebar, charts and a footer table since that rule was
written.

Needs Playwright (~150 MB of browser binaries) to do honestly. Without it,
this can be restructured for known-good patterns but not *verified*, which is
the same position the last four phases have been in.

### 2. "My Team" import

**Effort: medium. Unblocks: everything about transfers.**

`entry/{id}/` and `entry/{id}/event/{gw}/picks/` give the squad you actually
own. Until then the optimiser answers "what is the best squad" rather than
"what should *you* do", which is a different and much more useful question.

Prerequisite for any transfer recommendation being meaningful.

### 3. Fix the penalty-goal distortion in the component model

**Effort: small. Directly improves the best-ranking model.**

Penalties score 12 BPS for every position, but the API reports only
`goals_scored`, so a midfielder's penalty is credited 18 and a forward's 24.
`penalties_order` identifies the takers, so their goals can be discounted
towards the penalty share. This is the largest *known* modelling error, and it
is correctable without new data.

## Tier 2 — worth doing once Tier 1 lands

### 4. Horizon-based selection metric

Every model comparison so far scores the predicted top 15 for *one* gameweek.
You pick a squad for five to seven. A horizon metric might reward the ranking
skill the component model demonstrably has and which has so far converted into
nothing — see `docs/model-results.md`.

This is the cheapest remaining shot at the central puzzle of the project.

### 5. Minutes model

Not playing is the largest single cause of a zero score, and it is the one
input where the data is complete: `starts`, `minutes`, `chance_of_playing`,
and now a daily capture of all three. Unlike bonus points, there is no ceiling
imposed by unpublished data.

### 6. Wire odds into the component model

Depends on an `ODDS_API_KEY`. The market is the strongest freely available
prior and `fpl/features/market.py` already converts it to per-team expected
goals and clean-sheet probability. Blocked only on the key, and on the honest
caveat that live-only signals cannot be backtested — so this earns its place by
forward testing, not by a backtest.

### 7. Squad tab in the UI

The optimiser is trustworthy and has no interface. A tab showing the optimal
squad for a chosen predictor, with the constraints visible, would make Phase 4
usable rather than merely correct. Cheap now that a tab is one registry entry.

**Not** the transfer planner: that stays out until a season simulation shows it
beating a hold for the predictor in use.

## Tier 3 — speculative, or waiting on data

### 8. Measure the penalty-taker shares

`TAKER_SHARE` is currently an assumption. It becomes measurable once a season
of daily captures exists, because `penalties_order` is finally being recorded.
Revisit around GW10.

### 9. Bonus points model

Capped at roughly seven-eighths of BPS by unpublished Opta data — see
`docs/data-model.md`. Worth building only after the minutes model, since that
ceiling does not apply there.

### 10. DuckDB read layer

Recommended in `docs/data-model.md`, but only when a query first spans seasons.
Adding it before then is infrastructure for its own sake.

### 11. Weekly digest

A scheduled Action running post-deadline and writing a markdown summary.
Pleasant, not load-bearing.

### 12. Pundit and social sentiment

Deferred from Phase 5 with reasoning recorded in `ROADMAP.md`: the weakest
signal in the phase, the most expensive to build, and live-only so it can never
satisfy the backtest rule.

## Explicitly not recommended

| Idea | Why not |
|---|---|
| Scraping Understat or FBref | `robots.txt` disallows one; the other 403s behind Cloudflare |
| A database | 22 MB, no concurrent writers — parquet on the `data` branch is correct until it is not |
| Persisting derived columns | They are pure functions of the sources; storing them creates something that can go stale |
| More predictors before a better metric | Five exist and the best one converts its advantage into nothing. The metric is the bottleneck, not the model |
| Transfer recommendations in the UI | Loses points with a volatile predictor; needs the season simulation to say otherwise first |
