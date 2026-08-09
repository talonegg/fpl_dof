# Horizon scoring and minutes forecasting

Backlog items 4 and 5, both aimed at the central puzzle: five predictors
exist, the best one ranks decisively better than the rest, and that advantage
has converted into nothing on every selection measure tried.

Season 2025-26, gameweeks 6 onward, point-in-time throughout.

## Item 4 — does ranking convert over a horizon?

**The hypothesis.** Every comparison so far scored the predicted top fifteen
against *one* gameweek. You do not pick for one gameweek; you buy a squad and
keep it for five to seven, paying four points to change your mind. A single
week is dominated by hauls, which are close to unpredictable; over six weeks
hauls partly cancel and the underlying rate should show through. If the
component model has genuine ranking skill that the one-week metric cannot see,
this is where it should appear.

**It does not.** The answer is clear and it is the opposite of the hypothesis.

| Model | h=1 | h=3 | h=6 | Turnover (h=6) |
|---|---|---|---|---|
| **SeasonMean** | **4.259** | **3.995** | **3.904** | **0.138** |
| Component(4) | 4.222 | 3.794 | 3.544 | 0.353 |
| MinutesAdjusted(4) | 4.123 | 3.823 | 3.573 | 0.281 |
| NaiveForm(5) | 3.842 | 3.564 | 3.437 | 0.402 |

Points are per gameweek of the window, so the columns are directly comparable.

The season mean wins at every horizon, and **the gap widens tenfold** as the
horizon lengthens — from +0.036 at one week to +0.360 at six. Whatever the
component model's ranking advantage is, holding its picks for longer makes it
worse, not better.

**Rank correlation at six weeks: 0.818 for the season mean against 0.816 for
the component model.** The ranking advantage that held in 129 of 131 single
gameweeks disappears entirely at the horizon you actually play. It was an
artefact of one-week granularity.

### Turnover is the finding the one-week metric was hiding

The season mean replaces **14%** of its top fifteen between consecutive
gameweeks. The component model replaces **35%** — two and a half times as
much, for no better return.

At four points a transfer that is not a detail, and it explains the Phase 4
result rather than merely accompanying it: the transfer planner lost 88 points
with the component model and gained 16 with the season mean. The instability
was always there; the one-week metric simply had no way to charge for it.

**Conclusion.** Ranking skill does not convert, at any horizon. The component
model reorders players week to week; that reordering is expensive to act on
and does not identify better players to hold.

## Item 5 — forecasting minutes

**Why minutes deserve their own model.** Not playing is the largest single
cause of a zero score, and minutes are the one input where the data is
complete. Every other component runs into something unpublished — BPS into
Opta events, penalties into who took them. Minutes are recorded exactly, every
week, for every player. There is no ceiling here except modelling.

Minutes are also a *calibration* problem rather than a ranking one, so they are
scored differently: error and Brier score, not rank correlation.

| Forecaster | MAE | Brier | Start accuracy | Zero recall |
|---|---|---|---|---|
| **StartWeighted(8)** | 12.04 | **0.0896** | **88.2%** | 0.920 |
| StartWeighted(5) | **11.79** | 0.0910 | 88.0% | **0.922** |
| RecentMinutes(3) | 11.81 | 0.0970 | 87.6% | 0.917 |
| RecentMinutes(5) | 12.93 | 0.0999 | 86.3% | 0.907 |
| SeasonMinutes | 15.31 | 0.1062 | 84.3% | 0.871 |

**Recency wins for minutes — the exact opposite of points.** For points, the
season mean beat every form window tried. For minutes, the season average is
the *worst* forecaster on every measure.

That contrast is the useful result. Scoring rate is a stable property of a
player and is best estimated from as much data as possible; availability is a
volatile property of their current situation and is best estimated from the
most recent evidence. They are different problems and want opposite treatment
— which is exactly what `MinutesAdjustedPredictor` assumed, now with evidence.

`StartWeightedMinutes` separates whether a player starts from how long they
last once started. A starter usually substituted on the hour and a substitute
who usually gets half an hour both average about 45 minutes and are not the
same player.

### Feeding it into the component model

| Model | Top-15 return | Rank corr | MAE |
|---|---|---|---|
| Component(4, StartWeighted(8)) | **4.283** | 0.727 | **0.940** |
| SeasonMean | 4.259 | 0.691 | 1.059 |
| Component(4) baseline | 4.222 | 0.743 | 0.968 |

A better minutes forecast makes the component model nominally the best in the
field and cuts its prediction error decisively (t = −15.3 on MAE, better in 33
of 33 gameweeks).

**The selection improvement is not significant.** +0.061 against the baseline
at t = +0.72, and +0.024 against the season mean at t = +0.14 — both inside
the noise band, on 33 gameweeks.

So this is the fifth measured improvement that does not move the metric that
decides anything. The pattern is now consistent enough to be a conclusion in
itself: **prediction quality is not the binding constraint on selection.**

## What these two results imply

1. **Stop optimising rank correlation.** At the horizon that matters it does
   not even favour the model that wins on it weekly.
2. **Stability is worth more than accuracy.** The season mean's advantage is
   that it barely changes its mind, and the transfer simulation already priced
   that at over 100 points a season.
3. **Minutes forecasting is worth keeping** for its own sake — an 88% start
   accuracy and 92% zero recall is directly useful for avoiding blanks, whatever
   it does to the points metric.
4. **The remaining upside is probably not in the predictor.** Four phases of
   modelling have produced no significant selection gain. Squad construction,
   transfer timing and captaincy are untouched by comparison.

## Since this was written: the season-opening case

The horizons here (1 to 6 gameweeks, in-season) have a pre-season counterpart
in `fpl/backtest/preseason.py`, scored at **3, 5 and 7** gameweeks. The lesson
transferred, and sharpened:

- **In-season**, the season mean wins at every horizon and the gap *widens*
  with the horizon, because its picks barely change.
- **Pre-season**, the opposite: the model's advantage over the naive heuristic
  is largest at three gameweeks (42% against 38% of the achievable ceiling)
  and **gone by seven** (52% against 52%).

Both are the same underlying fact from two directions. A prior-season points
total is a good estimate of season-long value and a poor estimate of who starts
in August, so a model that forecasts minutes wins early and its edge decays as
the squad gets rebuilt by transfers.

The practical consequence is the one this document already argued for: **score
at more than one horizon**. Scoring the opening squad only at gameweek 7 — as
the first design did — reported a dead heat and would have concluded there was
nothing there.

## Minutes, pre-season

`PreseasonMinutes` is the pre-season member of the forecaster family. It cannot
read recent gameweeks because there are none, so it works from career
aggregates and regresses thin histories towards the population.

It is also where this project's largest measured effect lives: adding it took
the same model from **19% of the achievable ceiling to 52%**. The recency-versus-
stability finding above concerns *which* minutes forecaster to use; this
concerns whether to forecast minutes at all, and that question turns out to
dominate every refinement layered on top of it.
