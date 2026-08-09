"""Matching team names between sources.

The FPL API says "Man Utd", "Spurs" and "Nott'm Forest". Bookmakers say
"Manchester United", "Tottenham Hotspur" and "Nottingham Forest". Neither is
wrong, and an exact join silently drops the teams whose names disagree —
which, since the disagreements cluster on the biggest clubs, means losing
precisely the players most worth pricing.

The approach mirrors :mod:`fpl.domain.identity`: normalise hard, match
exactly on the result, and *report* what did not match rather than guessing.
Aliases are only needed where normalisation alone cannot bridge the gap.
"""

from __future__ import annotations

import pandas as pd

from fpl.domain.identity import normalise_name

# Normalised external name -> normalised FPL name. Only the cases where the
# two sources genuinely disagree about the club's name, not mere formatting.
TEAM_ALIASES = {
    "manchester united": "man utd",
    "manchester city": "man city",
    "tottenham hotspur": "spurs",
    "tottenham": "spurs",
    # normalise_name turns "Nott'm Forest" into "nott m forest" -- the
    # apostrophe becomes a space -- so that, not "nottm forest", is the target.
    "nottingham forest": "nott m forest",
    "wolverhampton wanderers": "wolves",
    "newcastle united": "newcastle",
    "brighton and hove albion": "brighton",
    "brighton hove albion": "brighton",
    "afc bournemouth": "bournemouth",
    "west ham united": "west ham",
    "leeds united": "leeds",
    "leicester city": "leicester",
    "ipswich town": "ipswich",
    "luton town": "luton",
    "sheffield united": "sheffield utd",
    "west bromwich albion": "west brom",
    "norwich city": "norwich",
    "cardiff city": "cardiff",
    "stoke city": "stoke",
    "swansea city": "swansea",
    "hull city": "hull",
    "birmingham city": "birmingham",
}


def team_key(name: str) -> str:
    """Reduce a team name to something comparable across sources."""
    normalised = normalise_name(name)
    return TEAM_ALIASES.get(normalised, normalised)


def build_team_lookup(teams: list[dict]) -> dict[str, int]:
    """Map a comparable team key to the FPL team id.

    Both the full name and the short name are registered, because external
    sources variously resemble one or the other.
    """
    lookup: dict[str, int] = {}
    for team in teams:
        for field in ("name", "short_name"):
            value = team.get(field)
            if isinstance(value, str) and value:
                lookup[team_key(value)] = team["id"]
    return lookup


def match_teams(names: pd.Series, teams: list[dict]) -> pd.Series:
    """Resolve external team names to FPL team ids, NaN where unmatched."""
    lookup = build_team_lookup(teams)
    return names.map(lambda name: lookup.get(team_key(name)))


def unmatched_teams(names: pd.Series, teams: list[dict]) -> list[str]:
    """External names that resolved to no FPL team.

    Should be empty. Anything here means a fixture's odds will be silently
    dropped, so it is worth failing a run over rather than logging quietly.
    """
    lookup = build_team_lookup(teams)
    missing = {name for name in names.dropna().unique() if team_key(name) not in lookup}
    return sorted(missing)
