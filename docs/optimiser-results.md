# Optimiser results

What the model *and* the optimiser together would actually have scored, playing
2025-26 properly: buy a squad once, then change it one transfer at a time,
paying 4 points for extras.

Reproduce with `fpl.backtest.season.simulate_season`. These numbers are
hand-recorded rather than generated, because a full season simulation takes
minutes; regenerate them when anything in `fpl/optimise/` or `fpl/models/`
changes.

- Season: **2025-26**
- Gameweeks 6–20 (15 scored gameweeks), point-in-time
- Recorded: 2026-08-08

## The headline

**The transfer planner loses points.** Holding the opening squad beats every
transfer strategy tried, for both models.

| Model | Horizon | Net points | Hits paid | Transfers | Points/GW |
|---|---|---|---|---|---|
| Component(4) | **0 (hold)** | **793** | 0 | 0 | **52.87** |
| Component(4) | 3 | 697 | 44 | 26 | 46.47 |
| Component(4) | 5 | 704 | 56 | 29 | 46.93 |
| Component(4) | 8 | 665 | 80 | 35 | 44.33 |
| SeasonMean | **0 (hold)** | **740** | 0 | 0 | **49.33** |
| SeasonMean | 3 | 721 | 4 | 16 | 48.07 |
| SeasonMean | 5 | 742 | 8 | 17 | 49.47 |
| SeasonMean | 8 | 738 | 16 | 19 | 49.20 |

Transferring costs the component model roughly **90–130 points over 15
gameweeks**. For the season mean it is roughly break-even, and notably it makes
far fewer transfers — 17 against 29.

## Why

`plan_transfers` computes a transfer's gain as
`(new squad expected points − current squad expected points) × horizon`. That
assumes the edge persists for the whole horizon. It does not: predictions move
every week, so much of the apparent gain is noise that reverts. The planner
then pays a real 4-point hit for an imaginary edge.

The component model suffers most precisely because it reacts more to new
information — the same responsiveness that makes it the best ranker in
`docs/model-results.md` makes it the worst transfer trigger.

This is a flaw in the *decision rule*, not in the solver. The squad optimiser
returns a provably optimal squad for the numbers it is given.

## What this does and does not show

- **Squad selection works.** The optimiser respects every constraint against
  700 real players and produces plausible squads at a binding £100.0m budget.
- **The transfer rule does not.** It should not be exposed in the UI until a
  simulation shows it beating a hold.
- **One season, one start point.** 15 gameweeks is a small sample and a single
  opening squad carries a lot of luck. Treat the gap between the two models'
  hold results as suggestive only.

## What would fix it

1. Require a transfer's gain to clear the hit **plus a margin**, rather than
   merely exceed it.
2. Predict the horizon directly instead of scaling one gameweek by it — the
   fixtures over the next five weeks are known and differ per player.
3. Damp the churn: prefer transfers whose edge has persisted across several
   gameweeks, not ones that appeared this week.
