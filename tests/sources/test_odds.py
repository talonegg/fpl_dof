"""Tests for the betting-odds source.

The fetcher is injected, so these never call the real service — which matters
more here than elsewhere: the free tier is 500 requests a month, and a test
suite that spent them would take the source down for the season.
"""

from __future__ import annotations

import pandas as pd
import pytest

from fpl.config import Config
from fpl.sources.base import RateLimiter, Source, SourceResult, guarded
from fpl.sources.odds import OddsSource, flatten_odds

KEYED = Config(odds_api_key="test-key")


def test_flatten_gives_one_row_per_price(odds_payload):
    result = flatten_odds(odds_payload)

    # Arsenal: 2 books x (3 h2h + 2 totals) = 10. Burnley: 1 x 5 = 5.
    assert len(result) == 15


def test_flatten_keeps_the_bookmaker_so_devigging_can_be_per_book(odds_payload):
    result = flatten_odds(odds_payload)

    assert set(result["bookmaker"]) == {"williamhill", "betfair"}


def test_flatten_carries_the_totals_line(odds_payload):
    result = flatten_odds(odds_payload)

    totals = result[result["market"] == "totals"]
    assert set(totals["point"]) == {2.5}


def test_an_impossible_price_is_dropped():
    """A decimal price of 1 or less is not a real quote."""
    payload = [
        {
            "id": "m",
            "home_team": "A",
            "away_team": "B",
            "bookmakers": [
                {
                    "key": "book",
                    "markets": [
                        {
                            "key": "h2h",
                            "outcomes": [
                                {"name": "A", "price": 1.0},
                                {"name": "B", "price": 2.0},
                            ],
                        }
                    ],
                }
            ],
        }
    ]

    result = flatten_odds(payload)

    assert result["outcome"].tolist() == ["B"]


def test_flatten_of_nonsense_is_empty_rather_than_raising():
    assert flatten_odds({"error": "quota exceeded"}).empty
    assert flatten_odds([]).empty


def test_an_empty_payload_still_has_the_expected_columns():
    result = flatten_odds([])

    assert "price" in result.columns
    assert "bookmaker" in result.columns


def test_the_source_satisfies_the_protocol():
    assert isinstance(OddsSource(config=KEYED), Source)


def test_without_a_key_the_source_is_unavailable_and_says_so():
    result = OddsSource(config=Config()).fetch()

    assert not result.ok
    assert "ODDS_API_KEY" in result.error
    assert result.is_empty


def test_with_a_key_the_source_fetches(odds_payload):
    def fake_fetcher(url, params):
        return odds_payload

    result = OddsSource(config=KEYED, fetcher=fake_fetcher).fetch()

    assert result.ok
    assert len(result.data) == 15


def test_the_request_asks_for_decimal_odds_and_the_configured_markets(odds_payload):
    captured = {}

    def fake_fetcher(url, params):
        captured["url"] = url
        captured["params"] = params
        return odds_payload

    OddsSource(config=KEYED, fetcher=fake_fetcher).fetch()

    assert captured["url"].endswith("/sports/soccer_epl/odds")
    assert captured["params"]["oddsFormat"] == "decimal"
    assert captured["params"]["markets"] == "h2h,totals"


def test_a_failing_service_degrades_rather_than_raising():
    """The whole point of the Source contract."""

    def broken_fetcher(url, params):
        raise RuntimeError("odds API quota or rate limit exhausted (429)")

    result = OddsSource(config=KEYED, fetcher=broken_fetcher).fetch()

    assert not result.ok
    assert "429" in result.error
    assert result.is_empty


def test_guarded_turns_an_exception_into_a_result():
    def boom():
        raise ValueError("nope")

    result = guarded("thing", boom)

    assert not result.ok
    assert "ValueError" in result.error


def test_guarded_passes_success_through():
    result = guarded("thing", lambda: pd.DataFrame([{"a": 1}]))

    assert result.ok
    assert result.fetched_at is not None


def test_a_source_result_with_data_is_ok():
    assert SourceResult(name="x", data=pd.DataFrame([{"a": 1}])).ok


# --- Rate limiting: the fastest way to lose a free tier is to hammer it ---


class FakeClock:
    def __init__(self):
        self.now = 0.0
        self.slept = []

    def time(self):
        return self.now

    def sleep(self, seconds):
        self.slept.append(seconds)
        self.now += seconds


def test_the_first_call_is_not_delayed():
    clock = FakeClock()
    limiter = RateLimiter(2.0, clock=clock.time, sleep=clock.sleep)

    assert limiter.wait() == 0.0
    assert clock.slept == []


def test_a_rapid_second_call_waits():
    clock = FakeClock()
    limiter = RateLimiter(2.0, clock=clock.time, sleep=clock.sleep)

    limiter.wait()
    waited = limiter.wait()

    assert waited == pytest.approx(2.0)


def test_a_call_after_the_interval_does_not_wait():
    clock = FakeClock()
    limiter = RateLimiter(2.0, clock=clock.time, sleep=clock.sleep)

    limiter.wait()
    clock.now += 5.0

    assert limiter.wait() == 0.0
