"""Entry point for the FPL DOF app.

Deliberately lives at the repository root: Streamlit puts the entry script's
directory on ``sys.path``, so this is what makes ``import fpl`` work both
locally and on Streamlit Community Cloud without an editable install.
"""

import streamlit as st

from app import fixtures_view, players_view
from app.data import load_next_gameweek, load_players, load_schedule

st.set_page_config(
    page_title="FPL DOF",
    page_icon="⚽",
    layout="wide",
    # "auto" keeps the sidebar open on laptops but collapsed on phones.
    initial_sidebar_state="auto",
)

# Tabs rather than columns: they stack cleanly on a phone.
players_tab, fixtures_tab = st.tabs(["Players", "Fixtures"])

with players_tab:
    players_view.render(load_players())

with fixtures_tab:
    gameweek = load_next_gameweek()
    if gameweek is None:
        st.info("The season is over — no upcoming fixtures.")
    else:
        fixtures_view.render(load_schedule(), from_gameweek=gameweek)
