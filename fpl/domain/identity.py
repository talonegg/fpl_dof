"""Matching players across seasons.

``element`` ids are reassigned every season, and the historical archive does
not carry the stable ``code``. So joining last season's gameweek history onto
this season's player list has to go through names, which is inherently fuzzy:
accents, punctuation and middle names all vary between sources.

The approach here is deliberately conservative. Names are normalised hard
(accents stripped, punctuation removed, case folded) and matched exactly on
the result. Anything that does not match is *reported*, never guessed at --
a wrong join silently attributes one player's history to another, which is
far worse than a gap.
"""

from __future__ import annotations

import re
import unicodedata

import pandas as pd

# The archive abbreviates positions; the API spells them out.
ARCHIVE_POSITIONS = {
    "GK": "Goalkeeper",
    "GKP": "Goalkeeper",
    "DEF": "Defender",
    "MID": "Midfielder",
    "FWD": "Forward",
}

_PUNCTUATION = re.compile(r"[^\w\s]")
_WHITESPACE = re.compile(r"\s+")


def normalise_name(name: str) -> str:
    """Reduce a player name to a comparable key.

    ``"Aarón Anselmino"`` and ``"Aaron  Anselmino"`` both become
    ``"aaron anselmino"``. Non-strings (a NaN in a scraped column) become the
    empty string, which never matches anything -- the safe direction to fail.
    """
    if not isinstance(name, str):
        return ""
    decomposed = unicodedata.normalize("NFKD", str(name))
    without_accents = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    cleaned = _PUNCTUATION.sub(" ", without_accents)
    return _WHITESPACE.sub(" ", cleaned).strip().lower()


def add_match_key(df: pd.DataFrame, name_column: str) -> pd.DataFrame:
    """Return a copy with a ``match_key`` column derived from ``name_column``.

    An unusable name becomes NaN rather than the empty string. Empty strings
    are equal to each other, so a nameless archive row would otherwise join to
    a nameless current player and be counted as a successful match.
    """
    df = df.copy()
    keys = df[name_column].map(normalise_name)
    df["match_key"] = keys.where(keys.astype(bool))
    return df


def player_match_keys(players: pd.DataFrame) -> pd.DataFrame:
    """Build ``element`` -> ``match_key`` for the current season's players.

    The archive stores the full name, so that is what we key on rather than
    ``web_name``, which is a display abbreviation and far more collision-prone
    ("Gabriel" is three different players).
    """
    keyed = players.copy()
    keyed["full_name"] = keyed["first_name"].fillna("") + " " + keyed["second_name"].fillna("")
    keyed = add_match_key(keyed, "full_name")
    return keyed[["element", "match_key", "full_name"]]


def match_to_current_players(
    archive: pd.DataFrame, players: pd.DataFrame, name_column: str = "player_name"
) -> pd.DataFrame:
    """Attach this season's ``element`` id to archive rows, where it can be found.

    Rows whose player is not in the current player list keep a null
    ``element`` -- they left the league, and dropping them silently would hide
    that from anyone counting rows.
    """
    keyed_archive = add_match_key(archive, name_column)
    keys = player_match_keys(players).rename(columns={"element": "current_element"})
    keys = keys[keys["match_key"].notna()]

    # Two current players sharing a normalised name cannot be told apart, and
    # picking one silently attributes a whole season of history to the wrong
    # player. Drop both and let ambiguous_names() report them -- a gap is
    # recoverable, a wrong join is not.
    collisions = set(keys["match_key"][keys["match_key"].duplicated(keep=False)])
    keys = keys[~keys["match_key"].isin(collisions)]

    merged = keyed_archive.merge(keys[["match_key", "current_element"]], on="match_key", how="left")
    return merged


def ambiguous_names(players: pd.DataFrame) -> list[str]:
    """Current players whose normalised names collide with another player's.

    These can never be matched safely, and they do *not* show up in
    :func:`unmatched_names` -- that only inspects archive rows that found
    nothing. Check both before trusting a cross-season join.
    """
    keys = player_match_keys(players)
    keys = keys[keys["match_key"].notna()]
    duplicated = keys["match_key"].duplicated(keep=False)
    return sorted(keys.loc[duplicated, "full_name"].str.strip().unique().tolist())


def unmatched_names(matched: pd.DataFrame, name_column: str = "player_name") -> list[str]:
    """Names in a matched frame that found no current player, de-duplicated.

    Worth eyeballing after any cross-season join: a handful is normal
    (retirements, transfers abroad), a flood means the matching is broken.
    """
    missing = matched[matched["current_element"].isna()]
    return sorted(missing[name_column].dropna().unique().tolist())


def match_rate(matched: pd.DataFrame) -> float:
    """Fraction of rows that found a current player, between 0 and 1."""
    if matched.empty:
        return 0.0
    return float(matched["current_element"].notna().mean())


def normalise_positions(df: pd.DataFrame, column: str = "position") -> pd.DataFrame:
    """Expand archive position codes (``"MID"``) to our names (``"Midfielder"``)."""
    df = df.copy()
    df[column] = df[column].map(lambda value: ARCHIVE_POSITIONS.get(value, value))
    return df
