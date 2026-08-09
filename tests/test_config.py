"""Tests for environment configuration.

A missing key must never raise on import: one unconfigured external source
should cost you that source, not the whole app.
"""

from __future__ import annotations

from fpl.config import DEFAULT_ODDS_CACHE_SECONDS, load_config, redact


def test_no_key_configured_means_odds_are_unavailable():
    config = load_config({})

    assert config.odds_api_key is None
    assert not config.has_odds


def test_a_key_makes_odds_available():
    config = load_config({"ODDS_API_KEY": "abcd1234"})

    assert config.has_odds


def test_a_blank_key_counts_as_unset():
    """An empty secret in a hosting platform is the common failure."""
    config = load_config({"ODDS_API_KEY": "   "})

    assert not config.has_odds


def test_the_cache_ttl_defaults_to_something_quota_safe():
    assert load_config({}).odds_cache_seconds == DEFAULT_ODDS_CACHE_SECONDS
    assert DEFAULT_ODDS_CACHE_SECONDS >= 3600


def test_the_cache_ttl_can_be_overridden():
    assert load_config({"ODDS_CACHE_SECONDS": "60"}).odds_cache_seconds == 60


def test_a_malformed_ttl_falls_back_rather_than_disabling_the_cache():
    """Treating it as zero would drain a 500-request month in minutes."""
    config = load_config({"ODDS_CACHE_SECONDS": "soon"})

    assert config.odds_cache_seconds == DEFAULT_ODDS_CACHE_SECONDS


def test_redact_keeps_only_the_last_few_characters():
    assert redact("supersecretkey") == "**********tkey"


def test_redact_hides_a_short_secret_entirely():
    assert redact("abc") == "***"


def test_redact_of_nothing_says_so():
    assert redact(None) == "<unset>"
