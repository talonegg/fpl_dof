"""The catalogue of derivations, and how to apply them.

Every derived column in this project comes from a pure function of the form
``frame -> frame``. That was already true; what was missing was anywhere to
*look them up*. Callers hand-chained them::

    players = add_scouting_metrics(load_players(), load_schedule())
    return add_advanced_metrics(add_availability(players))

which is order-dependent, silently incomplete if you forget one, and gives no
answer to "where does ``expected_penalty_goals`` come from".

A derivation declares what it needs and what it produces, so the set can be
applied in one call, skipped cleanly when its inputs are absent, and reported
on. Adding a derivation means adding one entry here rather than finding every
call site.

Skipping matters more than it sounds. Several derivations are **live-only** —
availability and set-piece duty do not exist for historical seasons — so
applying the catalogue to an archive frame must quietly produce fewer columns
rather than raising or, worse, inventing values.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import pandas as pd

from fpl.features.advanced import add_advanced_metrics, has_set_piece_data
from fpl.features.availability import add_availability, has_availability_data
from fpl.features.penalties import add_penalty_metrics
from fpl.features.rates import add_scouting_metrics

Derivation_fn = Callable[[pd.DataFrame], pd.DataFrame]
Predicate = Callable[[pd.DataFrame], bool]


@dataclass(frozen=True)
class Derivation:
    """One derived set of columns, and the conditions for computing it."""

    name: str
    apply: Derivation_fn
    provides: tuple[str, ...]
    requires: tuple[str, ...] = ()
    applicable: Predicate | None = None
    note: str = ""

    def can_apply(self, frame: pd.DataFrame) -> bool:
        """Whether this derivation has what it needs."""
        if any(column not in frame.columns for column in self.requires):
            return False
        return self.applicable is None or self.applicable(frame)


@dataclass
class EnrichmentResult:
    """The enriched frame, and a record of what was and was not applied."""

    frame: pd.DataFrame
    applied: list[str] = field(default_factory=list)
    skipped: dict[str, str] = field(default_factory=dict)

    def summary(self) -> str:
        parts = [f"applied {len(self.applied)}"]
        if self.skipped:
            parts.append(f"skipped {len(self.skipped)} ({', '.join(sorted(self.skipped))})")
        return ", ".join(parts)


def _always(_: pd.DataFrame) -> bool:
    return True


DERIVATIONS: tuple[Derivation, ...] = (
    Derivation(
        name="rates",
        apply=add_scouting_metrics,
        requires=("minutes", "total_points", "price"),
        provides=(
            "total_points_per_90",
            "points_per_million",
            "minutes_share",
            "low_minutes",
        ),
        applicable=_always,
        note="per-90 rates and value; totals reward whoever played more",
    ),
    Derivation(
        name="availability",
        apply=add_availability,
        requires=(),
        provides=("availability", "is_selectable"),
        applicable=has_availability_data,
        note="live-only: the archive records no injury status",
    ),
    Derivation(
        name="advanced",
        apply=add_advanced_metrics,
        requires=(),
        provides=(
            "takes_penalties",
            "takes_corners",
            "takes_free_kicks",
            "set_piece_duties",
            "finishing_delta",
            "finishing_delta_per_90",
        ),
        applicable=has_set_piece_data,
        note="live-only: set-piece duty is never archived",
    ),
    Derivation(
        name="penalties",
        apply=add_penalty_metrics,
        requires=(),
        provides=("penalty_taker_probability", "expected_penalty_goals"),
        applicable=has_set_piece_data,
        note="live-only: depends on penalties_order",
    ),
)


def by_name() -> dict[str, Derivation]:
    """The catalogue, keyed by name."""
    return {derivation.name: derivation for derivation in DERIVATIONS}


def provider_of(column: str) -> Derivation | None:
    """Which derivation produces a given column.

    The answer to "where does this number come from", which is the question
    a hand-chained pipeline could not answer.
    """
    for derivation in DERIVATIONS:
        if column in derivation.provides:
            return derivation
    return None


def enrich(
    players: pd.DataFrame,
    derivations: tuple[Derivation, ...] = DERIVATIONS,
    **kwargs,
) -> EnrichmentResult:
    """Apply every applicable derivation, reporting what was skipped and why.

    Skips rather than raises: applying the catalogue to a historical frame
    should yield fewer columns, not an error, because the live-only signals
    genuinely do not exist there.
    """
    frame = players
    result = EnrichmentResult(frame=frame)

    for derivation in derivations:
        if not derivation.can_apply(frame):
            missing = [c for c in derivation.requires if c not in frame.columns]
            result.skipped[derivation.name] = (
                f"missing {missing}" if missing else "not applicable to this data"
            )
            continue

        extra = kwargs.get(derivation.name, {})
        frame = derivation.apply(frame, **extra) if extra else derivation.apply(frame)
        result.applied.append(derivation.name)

    result.frame = frame
    return result
