"""Whether a player is going to be on the pitch at all.

Not playing is the single largest cause of a zero score, and unlike most of
what this project models it is not a prediction — the FPL API publishes it.
Using it is closer to reading the rules than to forecasting.

The field to be careful with is ``chance_of_playing_next_round``. It is null
for the overwhelming majority of players, and null means **"no news"** — which
is excellent news for a fit player and tells you nothing about an injured one.
Reading null as "available" would mark a long-term absentee as fully fit; on
live data 505 of 573 players have a null chance, and 59 of those are flagged.
So ``status`` is the authority and the percentage only refines it.

Status codes, from the API:

``a`` available · ``d`` doubtful · ``i`` injured · ``s`` suspended
``u`` unavailable (left the club, ineligible) · ``n`` not in squad

**This is a live-only signal.** The historical archive carries no ``status``
and no ``chance_of_playing_next_round``, because the API only ever publishes
the current state — nobody recorded who was injured in gameweek 12 of 2023-24.
So availability must never be evaluated on archive seasons: it is not that the
answer would be uncertain, it is that every historical player would come back
"fully fit" and the signal would silently contribute nothing.

Rather than defaulting to fit, :func:`availability` raises when handed a frame
with neither field. A backtest that quietly assumed everyone was available
would produce a number that looks fine and means nothing.
"""

from __future__ import annotations

import pandas as pd

AVAILABLE = "a"
DOUBTFUL = "d"

# What each status implies when no percentage is published. A doubtful player
# with no number attached is a genuine coin-toss; the rest are definite.
STATUS_AVAILABILITY = {
    AVAILABLE: 1.0,
    DOUBTFUL: 0.5,
    "i": 0.0,  # injured
    "s": 0.0,  # suspended
    "u": 0.0,  # unavailable
    "n": 0.0,  # not in squad
}

# Below this a player is treated as not worth selecting at all. 0.75 is the
# API's own "expected to play" band, so anything under it carries real doubt.
SELECTABLE_THRESHOLD = 0.75

AVAILABILITY_COLUMNS = ("status", "chance_of_playing_next_round")


class AvailabilityUnavailable(ValueError):
    """The frame carries no availability data, so the question cannot be asked."""


def has_availability_data(players: pd.DataFrame) -> bool:
    """Whether this frame carries availability at all.

    False for every archive season. Use it to decide whether an availability
    signal applies, rather than applying it and getting silent ones back.
    """
    return any(column in players.columns for column in AVAILABILITY_COLUMNS)


def availability(players: pd.DataFrame) -> pd.Series:
    """Probability each player features in the next gameweek, 0 to 1.

    Uses ``chance_of_playing_next_round`` where the API publishes one, and
    falls back to ``status`` where it does not — never the other way round,
    because a null chance is an absence of news rather than a clean bill of
    health.

    Raises :class:`AvailabilityUnavailable` on a frame carrying neither field,
    which in practice means historical data. Returning "everyone is fit" there
    would be a silent, plausible, wrong answer.
    """
    if players.empty:
        return pd.Series(dtype="float64")

    if not has_availability_data(players):
        raise AvailabilityUnavailable(
            "this data carries no status or chance_of_playing_next_round — "
            "availability is a live-only signal and was never recorded "
            "historically, so it cannot be evaluated here"
        )

    if "status" in players.columns:
        from_status = players["status"].map(STATUS_AVAILABILITY)
        # An unrecognised status is not silently "fit" -- treat it as doubtful
        # so a new code cannot quietly promote an absentee.
        from_status = from_status.fillna(0.5)
    else:
        from_status = pd.Series(1.0, index=players.index)

    if "chance_of_playing_next_round" not in players.columns:
        return from_status

    published = pd.to_numeric(players["chance_of_playing_next_round"], errors="coerce") / 100.0
    return published.fillna(from_status).clip(0.0, 1.0)


def add_availability(players: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with ``availability`` and ``is_selectable`` columns."""
    df = players.copy()
    df["availability"] = availability(df)
    df["is_selectable"] = df["availability"] >= SELECTABLE_THRESHOLD
    return df


def selectable(players: pd.DataFrame, threshold: float = SELECTABLE_THRESHOLD) -> pd.DataFrame:
    """Only the players fit enough to be worth picking.

    Intended for the optimiser's pool. Recommending an injured player is not a
    modelling error to be measured, it is simply wrong, so this is a filter
    rather than a scoring adjustment.
    """
    if players.empty:
        return players
    return players[availability(players) >= threshold]


def discount_expected_points(pool: pd.DataFrame, column: str = "expected_points") -> pd.DataFrame:
    """Scale expected points by the chance of actually playing.

    Separate from :func:`selectable` on purpose. Filtering answers "may I pick
    this player"; discounting answers "what is he worth given the doubt", and
    only the second belongs anywhere near a number the optimiser maximises.
    """
    df = pool.copy()
    if column in df.columns:
        df[column] = df[column] * availability(df)
    return df


def flagged(players: pd.DataFrame) -> pd.DataFrame:
    """Players carrying any injury, suspension or availability news.

    Sorted by how bad it is, so the top of the table is who you cannot pick.
    """
    if players.empty or "status" not in players.columns:
        return players.iloc[0:0]

    concerns = players[players["status"] != AVAILABLE].copy()
    if concerns.empty:
        return concerns
    concerns["availability"] = availability(concerns)
    return concerns.sort_values(["availability", "web_name"])
