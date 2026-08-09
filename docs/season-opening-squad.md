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
usable** — see §9. A partial workaround exists: a BPS-residual proxy that
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

## 5. Implementation plan

Five stages, each independently testable. Stop early if the evidence says so.

### Stage 1 — Cross-season aggregation (`fpl/features/career.py`)
Blend per-player rates across seasons with the minutes-weighted scheme.
Handle the 40% who do not carry over. **Test:** rates for a known ever-present
match a hand computation; a player with one thin season is not dominated by it.

### Stage 2 — Team strength (`fpl/features/team_strength.py`)
Per-club blended xGC and xG, with a promoted-club prior. **Test:** the
2.6× spread reproduces; a transferred player's clean-sheet expectation moves
with their new club, not their old.

### Stage 3 — Season-opening predictor (`fpl/models/preseason.py`)
Compose 3.4–3.8 into a `Predictor`, plus per-player uncertainty. **Test:**
component-by-component against hand-worked cases, as `ComponentPredictor` is.

### Stage 4 — Fixture-weighted horizon (`fpl/features/opening_run.py`)
Apply the decay across GW1–10 using the published fixture list. **Test:** a
club with an easy opening run outranks an equal club with a hard one; GW11+
changes nothing.

### Stage 5 — Backtest (`fpl/backtest/preseason.py`)
The stage that decides whether any of it was worth building.

## 6. Backtest design

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

## 7. Honest risks

| Risk | Assessment |
|---|---|
| Three test observations | The real limit. Design decisions cannot be tuned on this without overfitting |
| DC from one season | Unavoidable until 2026-27 completes |
| 15% of players unmodellable | Contained: 78% of them are under £5m |
| Promoted clubs | Prior-based; will be the largest single error source |
| Five prior improvements did nothing | The base rate for this helping is not high |

## 8. Recommendation

Build stages 1, 2 and 5 first — cross-season rates, team strength, and the
backtest harness — then run the **existing** `ComponentPredictor` through it
with blended prior-season inputs. That tests the whole pipeline against a
known quantity before any new modelling exists.

If that pipeline cannot beat the template squad, the sophisticated version
will not either, and the finding arrives after a day's work rather than a
week's.
