"""Entry point for the FPL DOF app.

Deliberately lives at the repository root: Streamlit puts the entry script's
directory on ``sys.path``, so this is what makes ``import fpl`` work both
locally and on Streamlit Community Cloud without an editable install.
"""

import streamlit as st

from app.data import load_players
from app.players_view import render

st.set_page_config(
    page_title="FPL DOF",
    page_icon="⚽",
    layout="wide",
    # "auto" keeps the sidebar open on laptops but collapsed on phones.
    initial_sidebar_state="auto",
)

render(load_players())
