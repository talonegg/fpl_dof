"""The layering, enforced rather than described.

Architecture documented in a README decays; architecture with a test does not.
These are cheap — they read imports, they do not run anything — and they catch
the class of change that is individually reasonable and collectively turns a
layered design into a ball of mud.

The layers, lowest first:

``sources``   fetch from outside; know nothing about football
``domain``    types and pure transforms
``store``     persist domain objects
``features``  derived metrics
``models``    expected-points predictors
``optimise``  squad and transfer selection
``backtest``  evaluation harness
"""

from __future__ import annotations

import ast
import pathlib

import pytest

LAYERS = ("sources", "domain", "store", "features", "models", "optimise", "backtest")
LAYER_INDEX = {name: index for index, name in enumerate(LAYERS)}

# Cross-cutting, allowed from anywhere: no behaviour, just configuration.
CROSS_CUTTING = {"config"}

PACKAGE_ROOT = pathlib.Path(__file__).resolve().parent.parent / "fpl"
APP_ROOT = PACKAGE_ROOT.parent / "app"


def fpl_imports(path: pathlib.Path) -> set[str]:
    """The ``fpl`` sub-packages a module imports from."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            parts = node.module.split(".")
            if parts[0] == "fpl" and len(parts) > 1:
                imported.add(parts[1])
        elif isinstance(node, ast.Import):
            for alias in node.names:
                parts = alias.name.split(".")
                if parts[0] == "fpl" and len(parts) > 1:
                    imported.add(parts[1])
    return imported


def modules_in(layer: str) -> list[pathlib.Path]:
    return sorted((PACKAGE_ROOT / layer).glob("*.py"))


@pytest.mark.parametrize("layer", LAYERS)
def test_no_layer_imports_upwards(layer):
    """A lower layer must never depend on a higher one.

    This is the rule that keeps `domain` reusable: the moment it imports from
    `models`, you cannot use the domain types without dragging a predictor in.
    """
    for path in modules_in(layer):
        for imported in fpl_imports(path):
            if imported in CROSS_CUTTING or imported not in LAYER_INDEX:
                continue
            assert LAYER_INDEX[imported] <= LAYER_INDEX[layer], (
                f"{path.relative_to(PACKAGE_ROOT.parent)} imports upwards from "
                f"{layer} into {imported}"
            )


def test_sources_know_nothing_about_football():
    """A fetcher's job is to return bytes, not to build a player table.

    `snapshot.py` used to live here and fetched, transformed and wrote in one
    module, which is what forced `sources` to import from `domain`.
    """
    for path in modules_in("sources"):
        assert "domain" not in fpl_imports(path), (
            f"{path.name} builds domain objects; that belongs in domain or store"
        )


def test_the_brain_never_imports_the_screen():
    """CLAUDE.md's prime directive, as a test."""
    for path in PACKAGE_ROOT.rglob("*.py"):
        source = path.read_text(encoding="utf-8")
        assert "import streamlit" not in source, f"{path.name} imports streamlit"


def test_the_screen_does_no_arithmetic():
    """The UI may render and filter, but derivations belong in fpl/features."""
    banned = ("def per_90", "def add_scouting", "def availability(")
    for path in APP_ROOT.glob("*.py"):
        source = path.read_text(encoding="utf-8")
        for phrase in banned:
            assert phrase not in source, f"{path.name} defines a derivation"


def test_every_layer_exists_and_has_modules():
    for layer in LAYERS:
        assert modules_in(layer), f"{layer} has no modules"


def test_no_layer_imports_itself_circularly():
    """Two modules in one layer importing each other is allowed; a cycle is not."""
    import collections

    graph = collections.defaultdict(set)
    for layer in LAYERS:
        for path in modules_in(layer):
            graph[layer] |= {i for i in fpl_imports(path) if i in LAYER_INDEX} - {layer}

    for layer, imports in graph.items():
        for imported in imports:
            assert layer not in graph[imported], f"cycle between {layer} and {imported}"
