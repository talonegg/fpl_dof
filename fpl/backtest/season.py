"""Playing a whole season with a model and an optimiser.

The model backtest answers "are the predictions any good". This answers the
question you actually care about: **what would this have scored?**

It plays the season properly rather than re-picking a perfect squad every week.
A squad is bought once, then changed one transfer at a time, paying 4 points
for extra ones. That constraint is the entire difficulty of FPL, and a
simulation that ignores it produces a fantasy number — you cannot act on a
strategy that requires fifteen free transfers a week.

Everything is point-in-time: each week's decisions see only earlier gameweeks,
because the pool is built from :func:`fpl.backtest.harness.replay`'s own
slicing discipline.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import pandas as pd

from fpl.backtest.harness import DEFAULT_FIRST_GAMEWEEK, known_fixtures, prepare_season
from fpl.domain.identity import normalise_positions
from fpl.domain.rules import FREE_TRANSFERS_PER_GAMEWEEK, MAX_ROLLED_FREE_TRANSFERS
from fpl.models.base import Predictor
from fpl.optimise.squad import InfeasibleSquad, SquadConstraints, optimise_squad
from fpl.optimise.transfers import plan_transfers


@dataclass
class GameweekOutcome:
    """What happened in one simulated gameweek."""

    gameweek: int
    points: float
    transfers: int
    hits_cost: int
    captain: int
    squad: list[int] = field(default_factory=list)

    @property
    def net_points(self) -> float:
        return self.points - self.hits_cost


@dataclass
class SeasonResult:
    """A whole simulated season."""

    model: str
    outcomes: list[GameweekOutcome]

    @property
    def total_points(self) -> float:
        return sum(outcome.net_points for outcome in self.outcomes)

    @property
    def gross_points(self) -> float:
        return sum(outcome.points for outcome in self.outcomes)

    @property
    def total_hits(self) -> int:
        return sum(outcome.hits_cost for outcome in self.outcomes)

    @property
    def transfers_made(self) -> int:
        return sum(outcome.transfers for outcome in self.outcomes)

    @property
    def points_per_gameweek(self) -> float:
        return self.total_points / len(self.outcomes) if self.outcomes else 0.0

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "gameweek": outcome.gameweek,
                    "points": outcome.points,
                    "hits_cost": outcome.hits_cost,
                    "net_points": outcome.net_points,
                    "transfers": outcome.transfers,
                }
                for outcome in self.outcomes
            ]
        )


def build_pool(history: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    """Attach the price, position and club a squad optimiser needs.

    Taken from each player's most recent appearance in ``history``, which is
    the latest anyone could have known them at the deadline.
    """
    if history.empty or predictions.empty:
        return pd.DataFrame()

    # An optimiser cannot work without these; older archive seasons lack some
    # of them, and filtering defensively only to dereference them two lines
    # later would raise a bare KeyError instead of saying what is wrong.
    required = ("price", "position", "team_name")
    missing = [column for column in required if column not in history.columns]
    if missing:
        raise ValueError(
            f"season data cannot be optimised over: missing {missing}. "
            "Older archive seasons do not carry these."
        )

    latest = history.sort_values("gameweek").groupby("element").last().reset_index()
    columns = [
        column
        for column in ("element", "player_name", "position", "team_name", "price")
        if column in latest.columns
    ]
    pool = predictions.merge(latest[columns], on="element", how="inner")
    pool = normalise_positions(pool)
    pool["team"] = pool["team_name"]
    return pool.dropna(subset=["price", "position", "team"])


def _actual_points(season: pd.DataFrame, gameweek: int, starting: list[int], captain: int) -> float:
    """What a starting eleven really scored, with the armband doubled."""
    actual = season[season["gameweek"] == gameweek].set_index("element")["total_points"]
    total = float(sum(actual.get(element, 0.0) for element in starting))
    return total + float(actual.get(captain, 0.0))


def simulate_season(
    season: pd.DataFrame,
    predictor: Predictor,
    first_gameweek: int = DEFAULT_FIRST_GAMEWEEK,
    last_gameweek: int | None = None,
    horizon: int = 5,
    constraints: SquadConstraints | None = None,
) -> SeasonResult:
    """Play the season with ``predictor``, one transfer a week.

    Returns what the strategy actually scored, net of transfer hits.
    """
    season = prepare_season(season)
    constraints = constraints or SquadConstraints()

    available = sorted(season["gameweek"].unique())
    last = last_gameweek if last_gameweek is not None else max(available)
    targets = [gw for gw in available if first_gameweek <= gw <= last]

    squad: list[int] | None = None
    free_transfers = FREE_TRANSFERS_PER_GAMEWEEK
    bank = 0.0
    outcomes: list[GameweekOutcome] = []

    for gameweek in targets:
        history = season[season["gameweek"] < gameweek]
        if history.empty:
            continue

        predictions = predictor.predict(history, gameweek, known_fixtures(season, gameweek))
        pool = build_pool(history, predictions)
        if pool.empty:
            continue

        transfers = 0
        hits_cost = 0

        if squad is None:
            try:
                chosen = optimise_squad(pool, constraints)
            except InfeasibleSquad:
                continue
            squad = [int(e) for e in chosen.players["element"]]
            # Whatever the opening squad did not spend stays available. Without
            # this the unspent balance is forfeited for the whole season, which
            # quietly penalises every later transfer.
            bank = max(0.0, constraints.budget - chosen.cost)
        else:
            owned_in_pool = pool[pool["element"].isin(squad)]
            if len(owned_in_pool) < constraints.squad_size:
                # A player has vanished from this gameweek's data. Hold, and
                # let them score nothing -- but keep playing the season, which
                # means selecting from whoever is left rather than abandoning
                # every remaining gameweek.
                pass
            else:
                plan = plan_transfers(
                    pool,
                    squad,
                    bank=bank,
                    free_transfers=free_transfers,
                    horizon=horizon,
                    constraints=constraints,
                )
                if plan.is_worth_it:
                    sold = pool[pool["element"].isin(plan.transfers_out)]["price"].sum()
                    bought = pool[pool["element"].isin(plan.transfers_in)]["price"].sum()
                    bank = max(0.0, bank + float(sold) - float(bought))
                    squad = [e for e in squad if e not in set(plan.transfers_out)]
                    squad += plan.transfers_in
                    transfers = plan.transfer_count
                    hits_cost = plan.points_cost

        # FPL banks unused transfers and *deducts* the ones spent; it does not
        # reset to one. Resetting charges hits a real manager would not pay.
        free_transfers = min(
            max(free_transfers - transfers, 0) + FREE_TRANSFERS_PER_GAMEWEEK,
            MAX_ROLLED_FREE_TRANSFERS,
        )

        # Pick the eleven and the captain from the squad we now hold.
        starting, captain = _pick_lineup(pool, squad, constraints)
        if starting is None:
            # Cannot field a legal eleven this week (players missing from the
            # data). Score nothing rather than ending the season silently.
            outcomes.append(
                GameweekOutcome(
                    gameweek=int(gameweek),
                    points=0.0,
                    transfers=transfers,
                    hits_cost=hits_cost,
                    captain=0,
                    squad=list(squad),
                )
            )
            continue

        outcomes.append(
            GameweekOutcome(
                gameweek=int(gameweek),
                points=_actual_points(season, gameweek, starting, captain),
                transfers=transfers,
                hits_cost=hits_cost,
                captain=captain,
                squad=list(squad),
            )
        )

    return SeasonResult(model=predictor.name, outcomes=outcomes)


def _pick_lineup(
    pool: pd.DataFrame, squad: list[int], constraints: SquadConstraints
) -> tuple[list[int] | None, int]:
    """Best legal eleven and captain from a held squad, or ``(None, 0)``.

    ``replace`` rather than a fresh ``SquadConstraints``: constructing a new one
    silently reverts every field the caller customised, so a squad of a
    non-default size became unpickable.
    """
    held = pool[pool["element"].isin(squad)]
    if held.empty:
        return None, 0

    try:
        lineup = optimise_squad(
            held,
            replace(
                constraints,
                budget=float(held["price"].sum()),
                max_per_club=constraints.squad_size,
                must_include=(),
                must_exclude=(),
            ),
        )
    except InfeasibleSquad:
        return None, 0

    return [int(e) for e in lineup.starting["element"]], lineup.captain
