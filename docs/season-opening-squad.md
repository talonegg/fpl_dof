# Season-opening squad recommendation: design

How to pick the fifteen players you start a season with, and how to know
whether the recommendation is any good.

Written against verified data rather than intentions: every feasibility claim
below was checked against the real archive before being relied on.

## 1. What already exists

| Layer | Reusable for this | Notes |
|---|---|---|
| `optimise/squad.py` | **Yes, unchanged** | MILP, provably optimal, ~1s on 573 players |
| `models/components.py` | **Structure yes, inputs no** | Assumes within-season history |
| `models/minutes_forecast.py` | **Yes** | 88% start accuracy, but needs current-season minutes |
| `domain/identity.py` | **Yes, essential** | Cross-season matching, 60% carryover |
| `domain/bps.py` | **Yes** | 87% of BPS reconstructable |
| `features/defensive.py` | **Yes** | DC formula verified exactly |
| `features/penalties.py` | **Yes** | Taker probability from `penalties_order` |
| `backtest/horizon.py` | **Yes, adapted** | Horizon scoring already built |

The optimiser is not the problem and does not need replacing. **Everything
hard about this task is in constructing the expected-points vector.**

## 2. Data feasibility — checked, not assumed

| Requirement | Feasible | Evidence |
|---|---|---|
| Historic start-of-season prices | **Yes** | GW1 prices present in all four seasons, £4.0–15.0 |
| Multi-season player history | **Yes** | 60% carry over season to season |
| Detect club transfers | **Yes** | 68 of 487 (14%) changed club 2024-25 → 2025-26 |
| Team defensive strength | **Yes, strong** | xGC/match 0.76–2.02 (2.6× spread), r = −0.825 with clean-sheet rate |
| xG, xA, xGC, BPS, starts | **Yes, 4 seasons** | Present 2022-23 onward |
| **Defensive contribution inputs** | **No — 1 season only** | CBI, tackles, recoveries exist **only in 2025-26** |
| Upcoming fixtures GW1–10 | **Yes** | Published months ahead |

### Two constraints that shape the design

**Defensive contributions cannot be weighted across seasons.** The underlying
stats arrived with the rule in 2025-26. There is exactly one season of CBIT
and CBIRT data, so the requested multi-season weighting is impossible for this
component — it gets a single-season estimate with wide uncertainty, and that
should be stated in the output rather than hidden. A second season (2026-27)
is what fixes it.

**15.2% of players have no Premier League history at all** (87 of 573). They
are overwhelmingly cheap — 68 of 87 are under £5m and none is above £7m — so
they are bench fodder rather than captaincy candidates. The design does not
try to model them: it assigns a position-and-price prior and flags them, which
is honest and costs almost nothing because they are not squad-defining picks.

## 3. The expected-points model

### 3.1 Overall shape

For each player, expected points for the opening run:

```
xPts(player) = Σ_gw  weight(gw) × xPts_per_match(player, opponent(gw), home(gw))
```

Two independent weightings, which the requirements ask for and which do
different jobs:

- **Fixture weighting** across the gameweeks ahead — near fixtures matter more.
- **Season weighting** across the seasons behind — recent form matters more.

### 3.2 Fixture weighting (gameweeks ahead)

Requirement: GW1–2 highest, GW4–5 lower, **after GW10 immaterial**.

```
weight(gw) = decay ^ (gw - 1)     for gw ≤ 10
weight(gw) = 0                    for gw > 10
```

With `decay = 0.78`, GW1 carries 1.00 and GW10 carries 0.09 — under a tenth of
the opening weight, which satisfies "immaterial" without a discontinuity that
would make the model flip on a single fixture. The cliff at 10 is a hard
truncation because the requirement is explicit; the decay does most of the work
before it arrives.

**Why not weight by squad-hold length instead?** Because a season-opening
squad is genuinely revisable — you get a free transfer a week. Weighting the
near fixtures reflects that the later ones will be re-decided with better
information.

### 3.3 Season weighting (seasons behind)

```
2025-26  0.50     most recent, and the only one under current scoring rules
2024-25  0.28
2023-24  0.15
2022-23  0.07
```

Geometric with ratio ≈ 0.55. Two adjustments on top:

- **Minutes-weighted, not season-weighted.** A player with 400 minutes in
  2025-26 and 3,000 in 2024-25 should not have the recent season dominate.
  Combine as `Σ (season weight × minutes) × rate / Σ (season weight × minutes)`.
- **Rule change respected.** 2025-26 is the only season scored under current
  rules. Defensive contributions get its weight alone; everything else blends.

### 3.4 Attacking returns: xG and xA with a finishing adjustment

Blend `xG90` and `xA90` across seasons using the weights above, then adjust for
persistent over- or under-performance:

```
finishing_multiplier = 1 + shrink × (career goals − career xG) / career xG
```

`shrink ≈ 0.25`, bounded to [0.85, 1.15].

**The shrinkage is the point and it should be aggressive.** Earlier work in
this project established that finishing over-performance is a *sell* signal:
shot quality persists, finishing streaks do not. A player 40% above their xG
gets credited 10%, not 40%. The requirement asks for historic
over/under-performance to be considered — considered, and largely regressed
away, which is what the evidence supports.

### 3.5 Clean sheets: the new team, not the old one

This is the requirement with the most modelling substance, and the data
supports it strongly — team xGC per match spans 0.76 to 2.02 and correlates
−0.825 with clean-sheet rate.

```
P(clean sheet) = f( team_xGC_next_season , opponent strength , home/away )
```

where `team_xGC_next_season` is the **new** club's expected concession rate,
estimated as:

1. The club's own blended xGC over prior seasons, if it was in the league.
2. For **promoted clubs**, the historical promoted-club average — they concede
   materially more than the league mean and using a mid-table prior would
   systematically over-rate their defenders.

A player moving from a 0.76 xGC club to a 2.02 xGC club should see their clean
sheet expectation fall by roughly the ratio of implied clean-sheet rates
(65% → 19%), which the correlation above supports directly. **Crucially the
player's own prior clean-sheet record is not used** — it measures their old
team, not them.

Poisson gives the conversion: `P(CS) = exp(−expected goals conceded)`, the same
machinery already in `features/market.py`.

### 3.6 Defensive contributions

Constrained by data, as noted. **External sources were investigated and none is
usable** — see §8. A partial workaround exists: a BPS-residual proxy that
recovers a weak signal for pre-2025-26 seasons. From the single season of real
data:

```
P(clears threshold) = f( CBIT90 or CBIRT90 , expected minutes )
```

Thresholds and positional eligibility are already exact and verified in
`features/defensive.py`. The estimate carries wide uncertainty and the output
should say so. For the 15% with no history, and for anyone whose 2025-26
minutes were thin, this term falls back to a position median.

### 3.7 BPS and bonus

`domain/bps.py` reconstructs 87% of BPS from published components. For a
season-opening forecast:

```
expected bonus = g( reconstructed BPS90 , position , expected minutes )
```

The 13% ceiling from unpublished Opta data applies and is not closable. Bonus
is a small share of total points, so this is the right place to accept an
approximation rather than the place to spend effort.

### 3.8 Minutes

The largest single term, and the one where the existing forecaster cannot be
used unchanged: `StartWeightedMinutes` needs current-season gameweeks and there
are none before GW1. Season-opening minutes must come from:

- prior-season minutes share, season-weighted;
- `chance_of_playing_next_round` and `status` for current injuries (live);
- a **new-signing prior** by price band — a £9m summer signing starts, a £4.5m
  one does not.

## 4. Optimiser: four approaches compared

| Approach | Optimality | Compute | Handles uncertainty | Verdict |
|---|---|---|---|---|
| **MILP, point estimate** | Exact | ~1 s | No | **Baseline — already built** |
| Greedy value-per-million | Poor | ms | No | Rejected: the budget couples every choice |
| **MILP, risk-adjusted objective** | Exact for its objective | ~1 s | Partly | **Recommended** |
| Stochastic MILP over scenarios | Exact | ~30 s for 50 scenarios | Yes | Worth testing, likely overkill |

**The solver is not the interesting choice — the objective is.** CBC already
returns a provable optimum in about a second, so swapping in a metaheuristic
would trade guaranteed optimality for nothing. Genetic algorithms and
simulated annealing are omitted for that reason: they solve a problem that is
not hard here.

What *is* worth varying:

**Risk-adjusted objective.** Maximise `xPts − λ × uncertainty` rather than
`xPts`. Uncertainty is available per player — minutes variance, thin history,
no history at all — and this is the natural way to stop the optimiser loading
up on 87 unmodelled cheap players whose point estimates are noise. Same solver,
one changed coefficient vector, negligible extra cost.

**Scenario-based stochastic MILP.** Sample N futures from the per-player
distributions, maximise expected points across all of them subject to one squad
choice. Handles uncertainty properly, costs ~30× the compute, and given every
finding in this project so far it will probably land inside the noise band.
Test it; do not assume it.

This project's own history argues for restraint here: **five successive
modelling improvements have failed to move selection significantly.** The
prior for "a more sophisticated optimiser objective will help" should be low,
and the test should be cheap before the build is expensive.

## 5. Estimator seams: where the known gaps plug in

Four gaps are known and none of them is closable today. Rather than hard-code
around them, each becomes a **named estimator with a protocol**, so closing a
gap later is swapping an implementation rather than editing the model.

This is the same pattern already proven twice in this codebase: the derivation
catalogue in `features/registry.py` and the view catalogue in `app/registry.py`.
Both turned "edit the call site and remember the rules" into "add an entry".

```python
class RateEstimator(Protocol):
    """Estimates a per-90 rate for a player from whatever it can see."""
    name: str
    def estimate(self, player_history: pd.DataFrame) -> pd.Series: ...
    def confidence(self, player_history: pd.DataFrame) -> pd.Series: ...
```

`confidence` is not decoration. It is what lets the optimiser's risk-adjusted
objective discount a player whose estimate rests on a proxy or a price band,
and it is what makes a v1 with weak estimators safe to ship.

### The four seams

| Seam | v1 implementation | Closes when | Swap cost |
|---|---|---|---|
| `DefensiveEstimator` | 2025-26 actuals; BPS-residual proxy before that | 2026-27 completes, giving two real seasons | One class |
| `NewPlayerEstimator` | Price-band prior by position | Transfermarkt ingest, if ever justified | One class |
| `TeamStrengthEstimator` | Blended xGC, promoted-club prior | Odds-implied strength once `ODDS_API_KEY` exists | One class |
| `MinutesEstimator` | Prior-season share + injury status | A second season of daily captures | One class |

Each ships with the weakest honest implementation and a test asserting the
protocol holds, so the successor has a contract to satisfy rather than a shape
to guess at.

### Registry, not conditionals

```python
ESTIMATORS = {
    "defensive": BpsResidualDefensive(),      # -> ActualCbitDefensive() in v2
    "new_player": PriceBandPrior(),           # -> ForeignLeaguePrior() if justified
    "team_strength": BlendedConcession(),     # -> MarketImpliedStrength() with a key
    "minutes": PriorSeasonMinutes(),          # -> CapturedMinutes() after a season
}
```

The model composes whatever is registered and reports which estimator produced
each term. When a squad recommendation looks wrong, the first question is which
estimator to blame — and that should be answerable without reading the code.

### Provenance travels with the number

Every player carries the estimator that produced each component, so the output
can distinguish "4.2 expected points, all from real data" from "4.2 expected
points, defensive term from a proxy and minutes from a price band". Those are
very different recommendations and the current design would otherwise present
them identically.

## 6. Implementation plan

Six stages. **Stages 1, 2 and 6 come first** — they test the pipeline against
a known quantity before any new modelling exists, so a negative result costs a
day rather than a week.

### Stage 1 — Cross-season aggregation (`fpl/features/career.py`)
Minutes-weighted blending of per-90 rates across seasons. Handles the 40% who
do not carry over by returning NaN and a confidence of zero, never a guess.
**Test:** an ever-present matches a hand computation; one thin season does not
dominate a heavy one.

### Stage 2 — Team strength (`fpl/features/team_strength.py`)
`TeamStrengthEstimator` seam. Blended xGC per club, promoted-club prior.
**Test:** the 2.6× spread reproduces; a transferred player's clean-sheet
expectation follows their new club, not their old.

### Stage 3 — Estimator seams (`fpl/models/estimators.py`)
The four protocols and their v1 implementations, including the BPS-residual
defensive proxy validated in §8. **Test:** each satisfies its protocol; each
reports lower confidence when working from a proxy than from real data.

### Stage 4 — Season-opening predictor (`fpl/models/preseason.py`)
Composes §3.4–3.8 through the registry, emitting expected points, confidence
and per-component provenance. **Test:** component-by-component against
hand-worked cases, as `ComponentPredictor` is.

### Stage 5 — Fixture-weighted opening run (`fpl/features/opening_run.py`)
The GW1–10 decay applied to the published fixture list. **Test:** an easy
opening run outranks an equal club with a hard one; GW11+ changes nothing.

### Stage 6 — Backtest (`fpl/backtest/preseason.py`)
The stage that decides whether any of the rest was worth building.

## 7. Backtest design

**Now possible** because GW1 prices exist for all four seasons.

For each season S in {2023-24, 2024-25, 2025-26}:

1. Build the squad using **only seasons before S** and S's GW1 prices.
2. Score the actual points those fifteen scored over GW1–10 of S.
3. Also score the full season, to see whether an opening-run bias costs
   anything later.

**Benchmarks, in ascending order of difficulty:**

| Benchmark | Why |
|---|---|
| Random legal squad | The floor; anything must beat it |
| Most-selected XV at GW1 | The template — what everyone actually owned |
| Highest prior-season points within budget | The obvious heuristic |
| **Optimal squad in hindsight** | The ceiling; the gap to it is the honest measure |

Reporting the hindsight ceiling matters. "Our squad scored 620" means nothing;
"620 against a template's 585 and a perfect 760" is a result.

**Only three seasons are testable**, and each gives one squad — three
observations. That is a very small sample and no significance test will rescue
it. The backtest can rule out a bad model; it cannot establish a good one.
Treat a positive result as permission to use it, not as proof it works.

### Testing the seams separately

Because the gaps are behind estimators, each can be ablated:

| Ablation | Question it answers |
|---|---|
| BPS-residual proxy vs no defensive term | Is the weak proxy better than nothing? |
| Price-band prior vs excluding new players | Do the 87 unmodelled players cost anything? |
| Promoted-club prior vs league average | How much does the promoted case matter? |

Three observations cannot support a *ranking* of these, but they can show
whether any is actively harmful — which is the question that matters before a
future release invests in closing that gap.

## 8. External sources investigated

### For pre-2025-26 defensive statistics

The FPL API is the constraint, not the mirror. CBIT arrived with the rule in
2025-26, so **every FPL-derived source has the same wall** — verified against
`FPL-Core-Insights`, whose defensive columns exist for 2025-2026 and whose
earlier-season files 404.

Opta-derived sources do have the data historically. None is usable:

| Source | Has pre-2025-26 CBIT | Usable | Why |
|---|---|---|---|
| FBref | yes | **no** | 403 Cloudflare challenge, even on `robots.txt` |
| Sofascore | yes | **no** | 403 on `robots.txt` |
| WhoScored | yes | **no** | `robots.txt` permits, but it is Opta's own property and its terms do not |
| StatsBomb open data | — | **no** | Premier League coverage is 2003/04 and 2015/16 only |
| SportMonks and similar | yes | paid | A commercial licence, not a free tier |

**No free, permitted source exists.**

### The BPS-residual proxy

Defensive actions *are* scored inside BPS — clearances at 1 per 2, tackles at 2
each, recoveries at 1 per 3 — and BPS is published for every season. The
residual after reconstructing the non-defensive components therefore carries a
defensive signal even where the raw counts are absent.

Validated on 2025-26, where both sides are known:

| Position | Correlation with actual DC | Top-30 DC rate | Everyone else | Lift |
|---|---|---|---|---|
| Defenders | 0.457 | 38% | 23% | **1.7×** |
| Midfielders | 0.669 | 33% | 13% | **2.6×** |
| Forwards | 0.504 | — | — | — |

**A weak prior, not a substitute.** It finds the defensively busy players
roughly twice as well as chance, which is worth having where the alternative is
nothing. It must never be presented as a measurement, and real 2025-26 data
always overrides it. This is the v1 `DefensiveEstimator`.

### For players with no Premier League history

| Source | Offers | Permitted | Verdict |
|---|---|---|---|
| **Transfermarkt** | appearances, goals, assists, minutes, fee, age | `robots.txt` empty | **Viable, best option** |
| worldfootball.net | appearances, goals across leagues | `Allow: /` | Viable fallback |
| FBref | xG, xA, defensive stats for most leagues | **no** — Cloudflare | Would have been ideal |

**Partially yes.** Basic productivity is obtainable within terms. Underlying
statistics are **not**, because the source that has them is the one that blocks
access. So an incoming player can be given a *rate* prior but not the xG-based
treatment everyone else gets — two estimators for two populations, flagged in
the output rather than blended silently.

A league-strength discount would also be needed, and its multipliers must come
from published research: this project has no data to fit them, and inventing
them would be worse than omitting them.

**v1 does not scrape.** All 87 unmodelled players are under £7m and 68 under
£5m, so a price-band prior is adequate. The `NewPlayerEstimator` seam exists so
that this can change without touching the model.

## 9. Honest risks

| Risk | Assessment |
|---|---|
| Three test observations | The real limit. Design decisions cannot be tuned on this without overfitting |
| DC from one season | Mitigated by the proxy, closed when 2026-27 completes |
| 15% of players unmodellable | Contained: 78% of them are under £5m |
| Promoted clubs | Prior-based; likely the largest single error source |
| Five prior improvements did nothing | The base rate for this helping is not high |

## 10. Recommendation

Build **stages 1, 2 and 6** first — cross-season rates, team strength, and the
backtest harness — then run the **existing** `ComponentPredictor` through the
pipeline with blended prior-season inputs.

That tests the whole pipeline against a known quantity before any new modelling
exists. If it cannot beat the template squad, the sophisticated version will
not either, and the finding arrives after a day's work rather than a week's.

The estimator seams mean the weak v1 implementations are not throwaway work:
each is a contract a stronger implementation can satisfy later, and each
reports its own confidence so the optimiser already knows which numbers to
trust.

### What a future release closes, in order of expected value

1. **Real defensive data** once 2026-27 completes — replaces the proxy, and
   needs no new source.
2. **Odds-implied team strength** once a key exists — the market prices
   promotion and squad changes better than a blended average can.
3. **Captured minutes** after a season of daily snapshots — the forecaster
   already exists and only lacks history.
4. **Foreign-league priors** — last, and only if an expensive signing ever
   needs rating.

Note that the first three need **no new scraping**: they arrive from data this
project already collects or is already entitled to. That ordering is
deliberate.
