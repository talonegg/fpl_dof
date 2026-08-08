"""Tests for the FPL API source layer.

These prove the wiring, not the network: the fetcher is injected, so a test
that accidentally reaches the internet will fail loudly in ``fake_fetcher``.
"""

from __future__ import annotations

import pytest

from fpl.sources import fpl_api


def test_fetch_bootstrap_uses_the_injected_fetcher(fake_fetcher, bootstrap):
    result = fpl_api.fetch_bootstrap(fetcher=fake_fetcher)

    assert result is bootstrap


def test_fetch_bootstrap_requests_the_expected_url():
    requested = []

    def recording_fetcher(url):
        requested.append(url)
        return {}

    fpl_api.fetch_bootstrap(fetcher=recording_fetcher)

    assert requested == ["https://fantasy.premierleague.com/api/bootstrap-static/"]


def test_fetch_fixtures_uses_the_injected_fetcher(fake_fetcher, fixtures_snapshot):
    result = fpl_api.fetch_fixtures(fetcher=fake_fetcher)

    assert result == fixtures_snapshot["fixtures"]


def test_fetch_player_summary_puts_the_element_id_in_the_url():
    requested = []

    def recording_fetcher(url):
        requested.append(url)
        return {}

    fpl_api.fetch_player_summary(427, fetcher=recording_fetcher)

    assert requested == ["https://fantasy.premierleague.com/api/element-summary/427/"]


def test_unexpected_urls_fail_the_test_rather_than_hitting_the_network(fake_fetcher):
    with pytest.raises(AssertionError):
        fake_fetcher("https://fantasy.premierleague.com/api/entry/12345/")
