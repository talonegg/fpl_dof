"""The best squad, the second best, the third — a ranked shortlist.

One optimal squad is a poor recommendation even when it is correct. The
expected-points vector behind it carries real uncertainty, so the gap between
first and twentieth is usually small enough that the ordering is not
meaningful on its own — and seeing that gap is what tells a reader how much to
trust the top of the list. A single answer hides exactly the thing they should
be judging it on.

Enumerated by **no-good cuts**: solve, forbid that exact combination of
fifteen, solve again. Each solve is still a provably optimal answer to a
slightly smaller problem, so the sequence is the true ranked top N rather than
a set of perturbations around one solution.

The naive alternative — excluding the *players* of each squad found — is
tempting and wrong. The second-best squad typically shares thirteen or fourteen
players with the best, so excluding its players would skip past thousands of
better squads and return fifteen unrelated ones.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from fpl.optimise.squad import InfeasibleSquad, Squad, SquadConstraints, optimise_squad

DEFAULT_SHORTLIST = 20


@dataclass
class RankedSquad:
    """One squad in the shortlist, with its distance from the best."""

    rank: int
    squad: Squad
    expected_points: float
    """Starting XI plus captain — what the manager banks."""

    score: float
    """The quantity the ranking is actually by: expected points plus the bench
    at its weight. Ordering by ``expected_points`` instead produces a list that
    looks wrongly sorted, because a squad with a stronger bench can win on the
    objective while showing fewer starting-XI points."""

    cost: float
    gap_to_best: float
    """Score behind the top squad. The number that says whether rank means anything."""

    changes_from_best: int
    """How many of the fifteen differ from the top squad."""

    @property
    def players(self) -> pd.DataFrame:
        return self.squad.players


@dataclass
class Shortlist:
    """A ranked set of squads, and what separates them."""

    squads: list[RankedSquad] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.squads)

    @property
    def spread(self) -> float:
        """Expected-points gap between the best and worst on the list.

        Small relative to the numbers themselves means the ranking is inside
        the noise of the prediction, and the shortlist should be read as a set
        of near-equivalent options rather than an order of merit.
        """
        if not self.squads:
            return 0.0
        return self.squads[0].score - self.squads[-1].score

    def table(self) -> pd.DataFrame:
        """One row per squad, for display."""
        if not self.squads:
            return pd.DataFrame()

        return pd.DataFrame(
            [
                {
                    "rank": entry.rank,
                    "score": entry.score,
                    "xi_points": entry.expected_points,
                    "gap": entry.gap_to_best,
                    "cost": entry.cost,
                    "formation": entry.squad.formation,
                    "changes_from_best": entry.changes_from_best,
                    "captain": _captain_name(entry.squad),
                }
                for entry in self.squads
            ]
        )


def _captain_name(squad: Squad) -> str:
    starting = squad.starting
    chosen = starting[starting["element"] == squad.captain]
    if chosen.empty or "player_name" not in chosen.columns:
        return ""
    return str(chosen["player_name"].iloc[0])


def rank_squads(
    players: pd.DataFrame,
    count: int = DEFAULT_SHORTLIST,
    constraints: SquadConstraints | None = None,
) -> Shortlist:
    """The ``count`` best legal squads, in order.

    ``players`` must already carry ``expected_points`` — this ranks squads, it
    does not decide what a player is worth. Stops early and returns what it
    found when no further legal squad exists, rather than padding the list.
    """
    if players.empty or count < 1:
        return Shortlist()

    base = constraints or SquadConstraints()
    forbidden: list[frozenset[int]] = list(base.forbidden_squads)

    ranked: list[RankedSquad] = []
    best_points: float | None = None
    best_players: set[int] = set()

    for rank in range(1, count + 1):
        attempt = replace_forbidden(base, tuple(forbidden))
        try:
            squad = optimise_squad(players, attempt)
        except InfeasibleSquad:
            # Every remaining combination is illegal: the shortlist is simply
            # shorter than asked for, which is honest.
            break

        elements = set(squad.players["element"])
        if best_points is None:
            best_points = squad.objective
            best_players = elements

        ranked.append(
            RankedSquad(
                rank=rank,
                squad=squad,
                expected_points=squad.expected_points,
                score=squad.objective,
                cost=squad.cost,
                gap_to_best=best_points - squad.objective,
                changes_from_best=len(elements - best_players),
            )
        )
        forbidden.append(frozenset(elements))

    return Shortlist(ranked)


def replace_forbidden(
    constraints: SquadConstraints, forbidden: tuple[frozenset[int], ...]
) -> SquadConstraints:
    """A copy of ``constraints`` with a different forbidden list.

    Copied rather than mutated so a caller's constraints object is not quietly
    changed by having been passed to a ranking run.
    """
    return SquadConstraints(
        budget=constraints.budget,
        squad_size=constraints.squad_size,
        starting_size=constraints.starting_size,
        max_per_club=constraints.max_per_club,
        composition=dict(constraints.composition),
        bench_weight=constraints.bench_weight,
        must_include=constraints.must_include,
        must_exclude=constraints.must_exclude,
        min_spend=constraints.min_spend,
        forbidden_squads=forbidden,
    )


def squad_differences(shortlist: Shortlist) -> pd.DataFrame:
    """Which players separate each squad from the best one.

    The useful view when the expected-points spread is narrow: if twenty squads
    sit within a couple of points, the question stops being "which is best" and
    becomes "which players am I actually choosing between".
    """
    if len(shortlist) < 2:
        return pd.DataFrame()

    best = set(shortlist.squads[0].players["element"])
    names = (
        shortlist.squads[0].players.set_index("element")["player_name"].to_dict()
        if "player_name" in shortlist.squads[0].players.columns
        else {}
    )

    rows = []
    for entry in shortlist.squads[1:]:
        current = entry.players
        elements = set(current["element"])
        if "player_name" in current.columns:
            names.update(current.set_index("element")["player_name"].to_dict())

        for element in sorted(elements - best):
            rows.append({"rank": entry.rank, "change": "in", "player": names.get(element, element)})
        for element in sorted(best - elements):
            rows.append(
                {"rank": entry.rank, "change": "out", "player": names.get(element, element)}
            )

    return pd.DataFrame(rows)
