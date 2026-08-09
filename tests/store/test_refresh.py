"""Tests for the refresh policy.

These check that the declared policy matches what the code and the workflow
actually do. A policy document that disagrees with the cron line is worse than
none, because it is believed.
"""

from __future__ import annotations

import pathlib
import re

from fpl.store.refresh import (
    EARLIEST_DEADLINE_UTC,
    MORNING_RUN_UTC,
    POLICIES,
    PRE_DEADLINE_RUN_UTC,
    Cadence,
    by_cadence,
    by_dataset,
    scheduled_runs,
)

WORKFLOW = (
    pathlib.Path(__file__).resolve().parent.parent.parent / ".github" / "workflows" / "snapshot.yml"
)


def cron_lines() -> list[str]:
    return re.findall(r'- cron: "([^"]+)"', WORKFLOW.read_text(encoding="utf-8"))


def test_every_dataset_has_a_policy():
    assert len(by_dataset()) == len(POLICIES)


def test_every_policy_gives_a_reason():
    """A cadence without a reason cannot be reviewed later."""
    for policy in POLICIES:
        assert policy.reason, f"{policy.dataset} has no reason"
        assert policy.orchestrated_by, f"{policy.dataset} says nothing about what runs it"


def test_the_workflow_runs_at_the_declared_times():
    """The policy and the cron line must not drift apart."""
    crons = cron_lines()

    assert len(crons) == len(scheduled_runs())
    hours_minutes = {(int(c.split()[1]), int(c.split()[0])) for c in crons}
    declared = {(int(run.split(":")[0]), int(run.split(":")[1])) for run in scheduled_runs()}
    assert hours_minutes == declared


def test_the_pre_deadline_run_is_actually_before_the_earliest_deadline():
    """The whole point of the second run."""
    run_hour, run_minute = (int(part) for part in PRE_DEADLINE_RUN_UTC.split(":"))
    deadline_hour, deadline_minute = (int(part) for part in EARLIEST_DEADLINE_UTC.split(":"))

    assert (run_hour, run_minute) < (deadline_hour, deadline_minute)


def test_the_pre_deadline_run_is_not_so_early_as_to_be_pointless():
    """Within a couple of hours of the deadline, or it adds nothing."""
    run = int(PRE_DEADLINE_RUN_UTC.split(":")[0]) * 60 + int(PRE_DEADLINE_RUN_UTC.split(":")[1])
    deadline = int(EARLIEST_DEADLINE_UTC.split(":")[0]) * 60 + int(
        EARLIEST_DEADLINE_UTC.split(":")[1]
    )

    assert 0 < deadline - run <= 120


def test_the_morning_run_comes_first():
    assert MORNING_RUN_UTC < PRE_DEADLINE_RUN_UTC


def test_the_archive_never_expires():
    """A finished season cannot change."""
    archive = by_cadence(Cadence.IMMUTABLE)

    assert len(archive) == 1
    assert archive[0].max_age_seconds == float("inf")


def test_the_odds_cache_is_long_enough_to_protect_the_quota():
    """500 requests a month is about 16 a day; an hourly refresh would burn it."""
    odds = by_dataset()["odds"]

    assert odds.max_age_hours >= 6


def test_the_odds_policy_matches_the_configured_default():
    from fpl.config import DEFAULT_ODDS_CACHE_SECONDS

    assert by_dataset()["odds"].max_age_seconds == DEFAULT_ODDS_CACHE_SECONDS


def test_the_app_cache_matches_the_declared_bootstrap_policy():
    from app.data import CACHE_TTL_SECONDS

    assert by_dataset()["bootstrap (players, teams, events)"].max_age_seconds == (CACHE_TTL_SECONDS)


def test_the_daily_capture_is_daily_and_only_daily():
    """Two runs a day, but the second must not overwrite the morning's news."""
    daily = by_cadence(Cadence.DAILY)

    assert len(daily) == 1
    assert daily[0].max_age_seconds == 24 * 3600


def test_the_workflow_can_still_be_triggered_by_hand():
    """The first run had to be dispatched manually; keep that possible."""
    assert "workflow_dispatch" in WORKFLOW.read_text(encoding="utf-8")
