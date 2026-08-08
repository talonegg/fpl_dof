"""Tests for the FPL rule constants and derived formations."""

from __future__ import annotations

from fpl.domain import rules


def test_squad_composition_sums_to_squad_size():
    assert sum(rules.SQUAD_COMPOSITION.values()) == rules.SQUAD_SIZE


def test_every_position_has_starting_xi_limits():
    assert set(rules.STARTING_XI_LIMITS) == set(rules.POSITIONS)


def test_valid_formations_match_the_known_set():
    formations = rules.valid_formations()

    assert set(formations) == {
        (3, 4, 3),
        (3, 5, 2),
        (4, 3, 3),
        (4, 4, 2),
        (4, 5, 1),
        (5, 2, 3),
        (5, 3, 2),
        (5, 4, 1),
    }


def test_every_formation_fills_the_starting_xi():
    for defenders, midfielders, forwards in rules.valid_formations():
        outfield = defenders + midfielders + forwards
        assert outfield + rules.STARTING_XI_LIMITS["Goalkeeper"].minimum == (rules.STARTING_XI_SIZE)


def test_no_formation_exceeds_what_the_squad_can_supply():
    for defenders, midfielders, forwards in rules.valid_formations():
        assert defenders <= rules.SQUAD_COMPOSITION["Defender"]
        assert midfielders <= rules.SQUAD_COMPOSITION["Midfielder"]
        assert forwards <= rules.SQUAD_COMPOSITION["Forward"]
