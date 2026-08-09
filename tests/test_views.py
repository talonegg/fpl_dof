"""Tests for the view registry.

The contract worth enforcing is `CLAUDE.md`'s: every tab either consumes the
global filter or declares why it cannot. That used to be a convention someone
had to remember at the call site; here it is a property of the registered view
and can be checked.
"""

from __future__ import annotations

import pandas as pd
import pytest

from app.registry import build_registry
from app.views import Registry, View, ViewContext
from fpl.features.filters import PlayerFilter

PLAYERS = pd.DataFrame(
    [
        {
            "element": 1,
            "web_name": "Fit",
            "position": "Midfielder",
            "team_name": "Arsenal",
            "price": 7.5,
            "status": "a",
            "chance_of_playing_next_round": None,
        },
        {
            "element": 2,
            "web_name": "Injured",
            "position": "Forward",
            "team_name": "Everton",
            "price": 5.0,
            "status": "i",
            "chance_of_playing_next_round": None,
        },
    ]
)

FILTER = PlayerFilter(
    positions=("Midfielder",),
    teams=("Arsenal",),
    price_range=(7.0, 8.0),
    availability_bands=("Available",),
)


def base_context() -> ViewContext:
    return ViewContext(
        players=PLAYERS,
        all_players=PLAYERS,
        player_filter=FILTER,
        schedule=lambda: pd.DataFrame(),
        history=lambda: pd.DataFrame(),
        next_gameweek=lambda: 1,
    )


def noop(_: ViewContext) -> None:
    return None


def test_a_view_honouring_everything_receives_a_filtered_frame():
    view = View(name="All", render=noop)

    context = view.context(base_context())

    assert context.filtered_count == 1
    assert context.players["web_name"].tolist() == ["Fit"]


def test_a_view_keeps_the_unfiltered_frame_for_counts():
    view = View(name="All", render=noop)

    context = view.context(base_context())

    assert context.total_count == 2


def test_a_view_that_ignores_a_filter_is_not_narrowed_by_it():
    """Fixtures is team-shaped: position cannot mean anything there."""
    view = View(name="Teams", render=noop, honours_position=False, honours_price=False)

    context = view.context(base_context())

    # Club and availability still apply, so Arsenal's fit player survives.
    assert context.players["web_name"].tolist() == ["Fit"]
    assert view.filter_for(FILTER).positions is None
    assert view.filter_for(FILTER).price_range is None


def test_ignored_filters_are_declared_rather_than_silent():
    """CLAUDE.md: a tab must consume the filter or say why it cannot."""
    view = View(name="Teams", render=noop, honours_position=False)

    assert view.ignored_filters == ("position",)


def test_a_view_honouring_everything_ignores_nothing():
    assert View(name="All", render=noop).ignored_filters == ()


def test_the_club_filter_reaches_every_player_and_team_shaped_view():
    """Club means something for player-shaped and team-shaped data alike.

    Squad-shaped views are the exception and must say so: a squad has to be
    legal — fifteen players, two goalkeepers, at most three per club — so it
    cannot be drawn from a pool the user has narrowed to a few teams.
    """
    for view in build_registry():
        if "squad-shaped" in view.note:
            continue
        assert view.honours_club, f"{view.name} does not honour the club filter"


def test_every_view_that_ignores_a_filter_explains_itself():
    for view in build_registry():
        if view.ignored_filters:
            assert view.note, f"{view.name} ignores filters without a note"


def test_only_the_declared_shapes_ignore_filters():
    """Every tab either consumes the global filters or is one of two known shapes."""
    ignoring = {view.name for view in build_registry() if view.ignored_filters}

    assert ignoring == {"Fixtures", "Season opener"}


def test_a_squad_shaped_view_honours_no_filter_at_all():
    """Half-applying them would answer a different question than the one asked."""
    opener = next(view for view in build_registry() if view.name == "Season opener")

    assert opener.ignored_filters == ("club", "position", "price", "availability")


def test_the_registry_has_the_expected_tabs():
    assert build_registry().names() == ["Players", "Scouting", "Fixtures", "Season opener"]


def test_a_duplicate_view_name_is_rejected():
    """Two tabs with one name is a copy-paste error, not a feature."""
    registry = Registry()
    registry.add(View(name="Players", render=noop))

    with pytest.raises(ValueError, match="already registered"):
        registry.add(View(name="Players", render=noop))


def test_registering_a_view_does_not_require_touching_the_entry_point():
    """The point of the registry: adding a tab is one entry."""
    registry = build_registry()
    before = len(registry)

    registry.add(View(name="Transfers", render=noop, note="future"))

    assert len(registry) == before + 1
    assert "Transfers" in registry.names()


def test_context_loaders_are_lazy():
    """A tab that is never opened should not pay to load a season of history."""
    calls = []
    context = ViewContext(
        players=PLAYERS,
        all_players=PLAYERS,
        player_filter=PlayerFilter(),
        schedule=lambda: calls.append("schedule") or pd.DataFrame(),
        history=lambda: calls.append("history") or pd.DataFrame(),
        next_gameweek=lambda: 1,
    )

    View(name="Idle", render=noop).context(context)

    assert calls == []
