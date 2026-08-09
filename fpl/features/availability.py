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

import re
from datetime import date, datetime

import pandas as pd

AVAILABLE = "a"
DOUBTFUL = "d"

# The API publishes no structured return date -- it is prose inside `news`.
# Two phrasings carry one, and a third says explicitly that none is known:
#   "Groin injury - Expected back 21 Aug"
#   "Suspended until 29 Aug"
#   "Knee injury - Unknown return date"
RETURN_DATE_PATTERN = re.compile(
    r"(?:expected back|suspended until)\s+(\d{1,2})\s+([A-Za-z]{3,})", re.IGNORECASE
)
UNKNOWN_RETURN_PATTERN = re.compile(r"unknown return date", re.IGNORECASE)
# "Has joined X on loan", "has departed the club", "has returned to Y".
DEPARTED_PATTERN = re.compile(r"joined .+ on loan|departed the club|has returned to", re.IGNORECASE)

MONTHS = {
    "jan": 1,
    "feb": 2,
    "mar": 3,
    "apr": 4,
    "may": 5,
    "jun": 6,
    "jul": 7,
    "aug": 8,
    "sep": 9,
    "oct": 10,
    "nov": 11,
    "dec": 12,
}

# How to describe why a return date is or is not known. Kept as data so the UI
# can group on it rather than re-parsing prose.
RETURN_KNOWN = "Expected back"
RETURN_UNKNOWN = "Unknown return date"
RETURN_DEPARTED = "Left the club"
RETURN_NOT_APPLICABLE = "No return needed"
RETURN_UNSTATED = "No date given"

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


AVAILABLE_BAND = "Available"
DOUBTFUL_BAND = "Doubtful"
UNAVAILABLE_BAND = "Unavailable"

# Ordered best to worst, which is the order the filter should offer them in.
AVAILABILITY_BANDS = (AVAILABLE_BAND, DOUBTFUL_BAND, UNAVAILABLE_BAND)


def availability_band(players: pd.DataFrame) -> pd.Series:
    """Bucket each player into available, doubtful or unavailable.

    Three bands rather than a continuous number because that is how the
    decision is actually made: buy, watch, or ignore.
    """
    if players.empty:
        return pd.Series(dtype="object")

    chance = availability(players)
    return pd.Series(
        pd.cut(
            chance,
            bins=[-0.01, 0.0, SELECTABLE_THRESHOLD - 0.001, 1.0],
            labels=[UNAVAILABLE_BAND, DOUBTFUL_BAND, AVAILABLE_BAND],
        ),
        index=players.index,
    ).astype(str)


def parse_return_date(news: object, anchor: date | None = None) -> date | None:
    """Pull an expected return date out of a news string.

    The API writes dates without a year ("Expected back 21 Aug"), so the year
    has to be inferred: take the next occurrence on or after ``anchor``, which
    should be the date the news was published. Anchoring on *today* instead
    would push a stale January note into next year.
    """
    if not isinstance(news, str) or not news.strip():
        return None

    match = RETURN_DATE_PATTERN.search(news)
    if not match:
        return None

    day = int(match.group(1))
    month = MONTHS.get(match.group(2)[:3].lower())
    if month is None:
        return None

    anchor = anchor or datetime.now().date()
    for year in (anchor.year, anchor.year + 1):
        try:
            candidate = date(year, month, day)
        except ValueError:
            return None  # 30 Feb and similar
        if candidate >= anchor:
            return candidate
    return None


# Most news reads "<reason> - <when>", so the reason is simply what precedes
# the separator. The exceptions have no separator and need naming.
REASON_SEPARATOR = " - "
SUSPENSION_PATTERN = re.compile(r"suspended until", re.IGNORECASE)
LOAN_PATTERN = re.compile(r"joined\s+(.+?)\s+on loan", re.IGNORECASE)
PERMANENT_MOVE_PATTERN = re.compile(r"joined\s+(.+?)\s+permanently", re.IGNORECASE)
RETURNED_PATTERN = re.compile(r"returned to\s+(.+?)\.?$", re.IGNORECASE)
FREE_AGENT_PATTERN = re.compile(r"departed the club", re.IGNORECASE)

REASON_SUSPENSION = "Suspension"
REASON_LEFT = "Left the club"
REASON_UNKNOWN = "Unknown"


def unavailability_reason(news: object) -> str:
    """Why the player is unavailable: "Hamstring injury", "Suspension", …

    The reason and the timing are different questions and the API answers both
    in one sentence. This pulls out the first; :func:`parse_return_date` and
    :func:`return_status` handle the second.
    """
    if not isinstance(news, str) or not news.strip():
        return ""

    news = news.strip()

    if SUSPENSION_PATTERN.search(news):
        return REASON_SUSPENSION

    loan = LOAN_PATTERN.search(news)
    if loan:
        return f"On loan at {loan.group(1)}"

    permanent = PERMANENT_MOVE_PATTERN.search(news)
    if permanent:
        return f"Transferred to {permanent.group(1)}"

    returned = RETURNED_PATTERN.search(news)
    if returned:
        return f"Returned to {returned.group(1)}"

    if FREE_AGENT_PATTERN.search(news):
        return REASON_LEFT

    if REASON_SEPARATOR in news:
        reason = news.split(REASON_SEPARATOR, 1)[0].strip()
        return reason[:1].upper() + reason[1:] if reason else REASON_UNKNOWN

    # Something unrecognised: show it rather than swallowing it, so a new
    # phrasing surfaces as odd text instead of a silently blank column.
    return news[:1].upper() + news[1:]


def return_status(news: object) -> str:
    """Why a return date is or is not known, as a groupable label."""
    if not isinstance(news, str) or not news.strip():
        return RETURN_NOT_APPLICABLE
    if DEPARTED_PATTERN.search(news):
        return RETURN_DEPARTED
    if RETURN_DATE_PATTERN.search(news):
        return RETURN_KNOWN
    if UNKNOWN_RETURN_PATTERN.search(news):
        return RETURN_UNKNOWN
    return RETURN_UNSTATED


def add_return_dates(players: pd.DataFrame) -> pd.DataFrame:
    """Add ``return_date`` and ``return_status`` columns.

    ``return_date`` is NaT wherever no date is published, which is most of the
    time — ``return_status`` is what distinguishes "we know they are out but
    not until when" from "they have left the club", and those should never be
    read as the same thing.
    """
    df = players.copy()
    if "news" not in df.columns:
        df["return_date"] = pd.NaT
        df["return_status"] = RETURN_NOT_APPLICABLE
        df["reason"] = ""
        return df

    anchors = (
        pd.to_datetime(df["news_added"], errors="coerce", utc=True).dt.date
        if "news_added" in df.columns
        else pd.Series([None] * len(df), index=df.index)
    )
    df["return_date"] = pd.to_datetime(
        [
            parse_return_date(news, anchor if isinstance(anchor, date) else None)
            for news, anchor in zip(df["news"], anchors, strict=True)
        ]
    )
    df["return_status"] = df["news"].map(return_status)
    df["reason"] = df["news"].map(unavailability_reason)
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
