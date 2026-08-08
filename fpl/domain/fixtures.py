"""Fixture schedules and difficulty, from the team's point of view.

The raw ``fixtures/`` endpoint is match-shaped: one row per match, with a home
team and an away team. Almost every FPL question is team-shaped instead --
"who does Arsenal play next, and how hard is it" -- so the core transform here
explodes each match into two rows, one per team.

That reshaping is also what makes blanks and doubles fall out naturally: a team
with no match in a gameweek simply has no row, and a team with two matches has
two.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from fpl.domain.reference import build_team_map


def build_team_schedule(
    fixtures: list[dict[str, Any]], teams: list[dict[str, Any]]
) -> pd.DataFrame:
    """Return one row per (team, fixture), from that team's perspective.

    Columns: ``team``, ``team_name``, ``opponent``, ``opponent_name``,
    ``gameweek``, ``is_home``, ``difficulty``, ``kickoff_time``, ``finished``.

    Fixtures not yet assigned to a gameweek (``event`` is null, which happens
    for postponed matches awaiting rescheduling) are dropped -- they cannot be
    reasoned about on a gameweek horizon.
    """
    team_map = build_team_map(teams)

    rows = []
    for fixture in fixtures:
        if fixture.get("event") is None:
            continue
        for is_home in (True, False):
            team = fixture["team_h"] if is_home else fixture["team_a"]
            opponent = fixture["team_a"] if is_home else fixture["team_h"]
            difficulty = fixture["team_h_difficulty"] if is_home else fixture["team_a_difficulty"]
            rows.append(
                {
                    "fixture_id": fixture["id"],
                    "team": team,
                    "team_name": team_map.get(team),
                    "opponent": opponent,
                    "opponent_name": team_map.get(opponent),
                    "gameweek": fixture["event"],
                    "is_home": is_home,
                    "difficulty": difficulty,
                    "kickoff_time": fixture.get("kickoff_time"),
                    "finished": fixture.get("finished", False),
                }
            )

    schedule = pd.DataFrame(rows)
    if schedule.empty:
        return schedule
    schedule["gameweek"] = schedule["gameweek"].astype(int)
    return schedule.sort_values(["team", "gameweek", "fixture_id"]).reset_index(drop=True)


def upcoming(schedule: pd.DataFrame, from_gameweek: int, horizon: int) -> pd.DataFrame:
    """Slice ``schedule`` to the ``horizon`` gameweeks starting at ``from_gameweek``.

    The window is over *gameweeks*, not matches, so a team playing twice in one
    gameweek contributes both matches.
    """
    if schedule.empty:
        return schedule
    last_gameweek = from_gameweek + horizon - 1
    return schedule[schedule["gameweek"].between(from_gameweek, last_gameweek)]


def difficulty_summary(schedule: pd.DataFrame, from_gameweek: int, horizon: int) -> pd.DataFrame:
    """Rank teams by how kind their next ``horizon`` gameweeks look.

    Returns one row per team with ``fixture_count``, ``mean_difficulty`` and
    ``total_difficulty``.

    ``fixture_count`` is the column that matters as much as difficulty: over a
    5-gameweek window it is normally 5, but a team with a blank has 4 and a
    team with a double has 6. Mean difficulty alone hides that entirely, which
    is how people talk themselves into transferring in a player who then does
    not play.

    Teams are sorted by ``total_difficulty`` ascending, so a team with an extra
    easy fixture correctly outranks one with fewer.
    """
    window = upcoming(schedule, from_gameweek, horizon)
    if window.empty:
        return pd.DataFrame(
            columns=[
                "team",
                "team_name",
                "fixture_count",
                "mean_difficulty",
                "total_difficulty",
            ]
        )

    summary = (
        window.groupby(["team", "team_name"], as_index=False)
        .agg(
            fixture_count=("difficulty", "size"),
            mean_difficulty=("difficulty", "mean"),
            total_difficulty=("difficulty", "sum"),
        )
        .sort_values(["total_difficulty", "mean_difficulty"])
        .reset_index(drop=True)
    )
    return summary


def blanks_and_doubles(schedule: pd.DataFrame, from_gameweek: int, horizon: int) -> pd.DataFrame:
    """Find every (team, gameweek) in the window that is not exactly one match.

    Returns rows with ``team``, ``team_name``, ``gameweek``, ``fixture_count``
    and ``kind`` of ``"blank"`` or ``"double"``. Empty frame when the window is
    uneventful.
    """
    window = upcoming(schedule, from_gameweek, horizon)
    columns = ["team", "team_name", "gameweek", "fixture_count", "kind"]
    if window.empty:
        return pd.DataFrame(columns=columns)

    counts = window.groupby(["team", "team_name", "gameweek"], as_index=False).size()
    counts = counts.rename(columns={"size": "fixture_count"})

    # A blank leaves no row at all, so build the full grid and look for holes.
    teams = window[["team", "team_name"]].drop_duplicates()
    gameweeks = pd.DataFrame({"gameweek": range(from_gameweek, from_gameweek + horizon)})
    grid = teams.merge(gameweeks, how="cross")

    merged = grid.merge(counts, on=["team", "team_name", "gameweek"], how="left")
    merged["fixture_count"] = merged["fixture_count"].fillna(0).astype(int)

    notable = merged[merged["fixture_count"] != 1].copy()
    notable["kind"] = notable["fixture_count"].map(lambda n: "blank" if n == 0 else "double")
    return notable.sort_values(["gameweek", "team_name"]).reset_index(drop=True)[columns]


def next_gameweek(events: list[dict[str, Any]]) -> int | None:
    """The gameweek currently accepting transfers, from ``bootstrap["events"]``.

    Returns ``None`` once the season is over and no gameweek is next.
    """
    for event in events:
        if event.get("is_next"):
            return int(event["id"])
    # Before the season opens no event is flagged as next, so fall back to the
    # first one that has not finished.
    for event in events:
        if not event.get("finished"):
            return int(event["id"])
    return None
