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

from fpl.domain.identity import add_match_key
from fpl.domain.positions import canonical_position

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

# The threshold is only reachable by a player who was on the pitch for it.
FULL_APPEARANCE_MINUTES = 60
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


# -- Pre-season forecasting ------------------------------------------------

DEFENSIVE_CONTRIBUTION_POINTS = 2


def defensive_contribution_rate(season_data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Per-player rate of clearing the defensive-contribution threshold.

    What a pre-season forecast needs, and it is a different question from
    ``defensive_points``: not "did this player clear the threshold in this
    match" but "what share of their matches do they clear it in".

    Only seasons carrying the underlying action counts contribute. Seasons
    without them are skipped rather than counted as zero — a player is not a
    poor defensive contributor because nobody recorded his tackles.

    Returns ``match_key``, ``defensive_matches`` and ``defensive_rate`` (0 to
    1). Players absent from every season carrying the data are simply absent,
    which is how the caller can tell "will not clear it" from "unknown".
    """
    frames = []
    for data in season_data.values():
        if data.empty or "player_name" not in data.columns:
            continue
        # No action counts means this season cannot speak to the question. The
        # threshold is positional, so a season carrying the counts but not the
        # positions cannot either -- 2018-19 is exactly that case.
        if not {"clearances_blocks_interceptions", "recoveries", "tackles"} & set(data.columns):
            continue
        if not {"position", "minutes"} <= set(data.columns):
            continue

        played = data[data["minutes"] >= FULL_APPEARANCE_MINUTES].copy()
        if played.empty:
            continue

        played["cleared"] = clears_threshold(played).fillna(False).astype(float)
        frames.append(played[["player_name", "cleared"]])

    if not frames:
        return pd.DataFrame(columns=["match_key", "defensive_matches", "defensive_rate"])

    combined = pd.concat(frames, ignore_index=True)
    rate = combined.groupby("player_name", as_index=False).agg(
        defensive_matches=("cleared", "size"), defensive_rate=("cleared", "mean")
    )
    return add_match_key(rate, "player_name")


def expected_defensive_points(
    rate: pd.Series, start_probability: pd.Series, reliable_matches: pd.Series | None = None
) -> pd.Series:
    """Defensive-contribution points per match from a rate and a start chance.

    The threshold pays only to a player on the pitch long enough to reach it,
    so the rate is scaled by the chance of a full appearance rather than
    applied flat.

    ``reliable_matches`` regresses thin samples towards the population, the
    same correction the minutes forecaster needs and for the same reason: a
    player who cleared the threshold in his only match is not a certainty.
    """
    if rate.empty:
        return pd.Series(dtype="float64")

    adjusted = rate.fillna(0.0)
    if reliable_matches is not None:
        weight = (reliable_matches.fillna(0) / 20).clip(0, 1)
        population = float(adjusted[weight > 0].mean()) if (weight > 0).any() else 0.0
        adjusted = weight * adjusted + (1 - weight) * population

    return adjusted * start_probability.fillna(0.0) * DEFENSIVE_CONTRIBUTION_POINTS
