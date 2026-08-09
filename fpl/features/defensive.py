"""Defensive contributions, and what can be known about BPS.

Two scoring routes that both reward defensive work, and that differ in one
important way: **one is fully recoverable from published data and the other is
not.**

## Defensive contributions — fully recoverable

Introduced in 2025-26 and continuing in 2026-27. Two points for clearing a
threshold of defensive actions in a match:

- **Defenders**: 10 or more *CBIT* — clearances, blocks, interceptions, tackles
- **Midfielders and forwards**: 12 or more *CBIRT* — the same plus recoveries
- **Goalkeepers**: not eligible

The API publishes ``defensive_contribution`` as the raw action count, plus its
inputs separately. Verified against 2025-26: the identity holds *exactly* on
every appearance — 3,950 defender appearances all satisfy CBIT, and 6,775
midfield and forward appearances all satisfy CBIRT. So
:func:`defensive_actions` can be checked against the published figure rather
than trusted, which is what :func:`formula_agreement` is for.

## BPS — only partly recoverable

The bonus points system is scored from a much wider set of Opta events than
the API exposes: 16 of its 38 actions are observable here. The official table
and a reconstruction from it live in :mod:`fpl.domain.bps`, which measures the
gap directly rather than estimating it.
"""

from __future__ import annotations

import pandas as pd

from fpl.models.components import canonical_position

CLEARANCES_BLOCKS_INTERCEPTIONS = "clearances_blocks_interceptions"
TACKLES = "tackles"
RECOVERIES = "recoveries"
PUBLISHED_TOTAL = "defensive_contribution"

# Clearances, blocks, interceptions and tackles. Defenders are scored on these.
CBIT_COLUMNS = (CLEARANCES_BLOCKS_INTERCEPTIONS, TACKLES)
# The same plus recoveries. Midfielders and forwards are scored on these, with
# a higher bar, because recoveries are far more common in midfield.
CBIRT_COLUMNS = (CLEARANCES_BLOCKS_INTERCEPTIONS, TACKLES, RECOVERIES)

DEFENDER = "DEF"
GOALKEEPER = "GK"

# Actions needed to earn the two points.
DEFENSIVE_THRESHOLD = {"DEF": 10, "MID": 12, "FWD": 12}
DEFENSIVE_CONTRIBUTION_POINTS = 2

# The BPS table and the measured size of the gap now live in fpl/domain/bps.py,
# built from the official published values rather than fitted coefficients.


def defensive_actions(appearances: pd.DataFrame) -> pd.Series:
    """The action count each player is judged on, by position.

    CBIT for defenders, CBIRT for midfielders and forwards. Goalkeepers get
    NaN rather than a number: they are not eligible, and a zero would read as
    "did nothing" rather than "does not apply".
    """
    if appearances.empty:
        return pd.Series(dtype="float64")

    positions = appearances["position"].map(canonical_position)

    def total(columns) -> pd.Series:
        present = [column for column in columns if column in appearances.columns]
        if not present:
            return pd.Series(0.0, index=appearances.index)
        return appearances[present].fillna(0).sum(axis=1)

    cbit = total(CBIT_COLUMNS)
    cbirt = total(CBIRT_COLUMNS)

    actions = cbirt.where(positions != DEFENDER, cbit)
    return actions.where(positions != GOALKEEPER)


def threshold_for(appearances: pd.DataFrame) -> pd.Series:
    """The action count each player needs, NaN where they are ineligible."""
    positions = appearances["position"].map(canonical_position)
    return positions.map(DEFENSIVE_THRESHOLD)


def clears_threshold(appearances: pd.DataFrame) -> pd.Series:
    """Whether each appearance earned the defensive contribution points."""
    if appearances.empty:
        return pd.Series(dtype="bool")
    actions = defensive_actions(appearances)
    return (actions >= threshold_for(appearances)).fillna(False)


def defensive_points(appearances: pd.DataFrame) -> pd.Series:
    """Points earned from defensive contributions, per appearance."""
    return clears_threshold(appearances).astype(int) * DEFENSIVE_CONTRIBUTION_POINTS


def formula_agreement(appearances: pd.DataFrame) -> float:
    """How often the computed action count matches the published one.

    A data quality check rather than a metric. It should be 1.0; anything less
    means either the scoring rule changed or the inputs stopped meaning what
    they meant, and both are worth knowing before a model is built on them.
    Goalkeepers are excluded, being ineligible.
    """
    if appearances.empty or PUBLISHED_TOTAL not in appearances.columns:
        return float("nan")

    positions = appearances["position"].map(canonical_position)
    eligible = appearances[positions != GOALKEEPER]
    if eligible.empty:
        return float("nan")

    computed = defensive_actions(eligible)
    published = eligible[PUBLISHED_TOTAL].fillna(0)
    return float((computed == published).mean())


def add_defensive_metrics(appearances: pd.DataFrame) -> pd.DataFrame:
    """Attach the action count, threshold, and whether it was cleared."""
    if appearances.empty or "position" not in appearances.columns:
        return appearances

    df = appearances.copy()
    df["defensive_actions"] = defensive_actions(df)
    df["defensive_threshold"] = threshold_for(df)
    df["cleared_defensive_threshold"] = clears_threshold(df)
    df["defensive_points"] = defensive_points(df)
    return df
