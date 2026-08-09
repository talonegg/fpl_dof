"""The contract every external source implements.

External sources fail in ways the FPL API does not: keys expire, free tiers run
out, sites change their markup, services go down mid-season. The whole design
here follows from one rule -- **a failing source degrades the app, it never
breaks it**. A missing odds feed should cost you the odds column, not the
scouting page.

So every source reports three things: whether it is `available` at all
(configured, keyed), what it returns when it works, and a `SourceResult` that
carries failure as data rather than as an exception escaping into a render.

Rate limiting is part of the contract, not an afterthought. Free tiers are
small and the fastest way to lose access to a source permanently is to hammer
it during development.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import pandas as pd


@dataclass
class SourceResult:
    """What a source returned, including the case where it returned nothing.

    Callers check ``ok`` and use ``data``; the UI shows ``error`` as a quiet
    note beside the affected panel rather than as a stack trace.
    """

    name: str
    data: pd.DataFrame = field(default_factory=pd.DataFrame)
    error: str | None = None
    fetched_at: float | None = None

    @property
    def ok(self) -> bool:
        return self.error is None

    @property
    def is_empty(self) -> bool:
        return self.data.empty


@runtime_checkable
class Source(Protocol):
    """An external provider of data the FPL API does not carry."""

    name: str

    @property
    def available(self) -> bool:
        """Whether this source is configured well enough to try."""
        ...

    def fetch(self) -> SourceResult:
        """Fetch, returning failure as data rather than raising."""
        ...


class RateLimiter:
    """The simplest thing that keeps a scraper polite: a minimum interval.

    Deliberately not a token bucket. Bursts are exactly what gets a client
    blocked, and no source here needs throughput -- it needs to still work in
    March.
    """

    def __init__(self, min_interval_seconds: float, clock=time.monotonic, sleep=time.sleep):
        self.min_interval_seconds = min_interval_seconds
        self._clock = clock
        self._sleep = sleep
        self._last_call: float | None = None

    def wait(self) -> float:
        """Block until the next call is allowed. Returns how long it waited."""
        now = self._clock()
        if self._last_call is None:
            self._last_call = now
            return 0.0

        elapsed = now - self._last_call
        remaining = self.min_interval_seconds - elapsed
        if remaining > 0:
            self._sleep(remaining)
            self._last_call = self._clock()
            return remaining

        self._last_call = now
        return 0.0


def guarded(name: str, fetch: Any) -> SourceResult:
    """Run ``fetch`` and turn any failure into a :class:`SourceResult`.

    Catching broadly is deliberate here and nowhere else: this is the boundary
    between the outside world and a render, and the outside world fails in
    ways that are not worth enumerating.
    """
    try:
        data = fetch()
    except Exception as error:  # noqa: BLE001 - the entire point of this function
        return SourceResult(name=name, error=f"{type(error).__name__}: {error}")
    return SourceResult(name=name, data=data, fetched_at=time.time())
