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

**Whether transferring pays depends on the predictor**, and the earlier blanket
claim that it always loses was partly an artefact of a bug (see below).

| Model | Horizon | Net points | Hits paid | Transfers | Points/GW |
|---|---|---|---|---|---|
| Component(4) | **0 (hold)** | **793** | 0 | 0 | **52.87** |
| Component(4) | 5 | 705 | 52 | 28 | 47.00 |
| SeasonMean | 0 (hold) | 740 | 0 | 0 | 49.33 |
| SeasonMean | **5** | **756** | 4 | 16 | **50.40** |

- With the **stable** predictor, transferring now *beats* holding by 16 points
  over 15 gameweeks. It makes 16 transfers and pays only one hit.
- With the **volatile** predictor, transferring still costs about 88 points. It
  makes 28 transfers and pays thirteen hits.

The best single result remains Component holding its opening squad (793), but
that is one opening squad's luck — see the four-start-point check below.

### Corrected from an earlier version of this document

The first version reported that holding beat transferring for *both* models.
Three bugs in the simulation biased that, all in the same direction:

- Free transfers were **reset to 1** whenever any transfer was made rather than
  deducting what was spent, so the simulation paid hits a real manager would
  not.
- The **bank was never tracked**, so money unspent at the opening buy was
  forfeited for the season.
- The transfer search stopped at 4 transfers while 5 free ones can be banked.

Fixing them halved the season mean's hits (8 → 4) and turned its result from a
dead heat into a win. The component model's verdict was unaffected — its
problem is real, not accounting.

## Why the volatile model still loses

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

## Does the better model pick a better squad? No.

The table above makes Component's hold look far better than SeasonMean's (793
against 740). That is an artefact of one starting gameweek. Holding from four
different starts, through to gameweek 38:

| Start GW | Component(4) | SeasonMean | Gameweeks |
|---|---|---|---|
| 6 | 54.18 | 45.52 | 33 |
| 12 | 49.52 | 47.07 | 27 |
| 20 | **40.74** | **50.53** | 19 |
| 28 | 48.73 | 49.55 | 11 |
| **Mean** | **48.29** | **48.17** | |

Two wins each and a mean difference of 0.1 points per gameweek. There is **no
evidence** the component model picks better squads than the season mean.

What does differ is spread: Component ranges over 13.5 points per gameweek
(40.7–54.2), SeasonMean over 5.0 (45.5–50.5). The more responsive model does
not produce better squads, it produces more variable ones — which is the same
property that makes it a poor transfer trigger.

This is the third metric on which Component's clear superiority at *ranking*
has failed to convert into anything you can act on.

## What this does and does not show

- **Squad selection works.** The optimiser respects every constraint against
  700 real players and produces plausible squads at a binding £100.0m budget.
- **The transfer rule does not.** It should not be exposed in the UI until a
  simulation shows it beating a hold.
- **The choice of predictor is not settled.** Across four start points the two
  models are tied. Use whichever you like for squad selection; prefer the
  season mean if you dislike variance.
- **Still one season.** Four start points from the same season are not four
  independent samples — they share the same players, prices and injuries.
  Multiple seasons remain the biggest missing piece in the evaluation.

## What would fix it

1. Require a transfer's gain to clear the hit **plus a margin**, rather than
   merely exceed it.
2. Predict the horizon directly instead of scaling one gameweek by it — the
   fixtures over the next five weeks are known and differ per player.
3. Damp the churn: prefer transfers whose edge has persisted across several
   gameweeks, not ones that appeared this week.
