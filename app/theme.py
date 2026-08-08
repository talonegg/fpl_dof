"""Chart colours.

The categorical order is fixed and assigned by slot, never cycled: player 1 is
always slot 1, so removing a player from a comparison does not repaint the
others. Both sets are validated for colour-vision deficiency separation and
contrast against their own surface -- the dark values are the same hues stepped
for a dark background, not an automatic flip.

On the light surface, aqua and yellow fall below 3:1 contrast, so any view
using more than two series must also offer the numbers as a table.
"""

from __future__ import annotations

import streamlit as st

SERIES_LIGHT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
SERIES_DARK = ["#3987e5", "#d95926", "#199e70", "#c98500"]

MAX_COMPARISON_SERIES = len(SERIES_LIGHT)


def is_dark() -> bool:
    """Whether the viewer is using Streamlit's dark theme."""
    try:
        return st.context.theme.type == "dark"
    except (AttributeError, TypeError):
        # Older Streamlit, or no browser context (as under AppTest).
        return False


def series_colours(count: int) -> list[str]:
    """The first ``count`` categorical slots, in fixed order."""
    palette = SERIES_DARK if is_dark() else SERIES_LIGHT
    return palette[:count]
