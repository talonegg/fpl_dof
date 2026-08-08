# FPL DOF

Fantasy Premier League scouting, modelling and squad optimisation.

## Run it

```powershell
.\venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt
streamlit run streamlit_app.py
```

## Develop

```powershell
pytest                  # unit tests, offline, ~1s
pytest -m backtest      # slow model evaluation (added in Phase 3)
ruff check . ; ruff format .
```

## Layout

`fpl/` is the brain: pure, testable, never imports Streamlit.
`app/` is the screen: rendering only, no calculations.
`streamlit_app.py` at the root is the entry point — its location is what puts
the repo root on `sys.path`, so `import fpl` works locally and on Streamlit
Community Cloud without an editable install.

See [CLAUDE.md](CLAUDE.md) for conventions and [docs/ROADMAP.md](docs/ROADMAP.md)
for the build plan.

## Deploy

Streamlit Community Cloud, pointed at `streamlit_app.py` on the `main` branch.
It installs `requirements.txt` only.
