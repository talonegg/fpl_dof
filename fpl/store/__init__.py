"""Persisting domain objects.

Separate from ``sources`` on purpose. A source *fetches* — it talks to the
outside world and knows nothing about football. The store *keeps* — it writes
and reads domain objects that have already been built.

Conflating them is what made ``sources`` import upwards into ``domain``:
``snapshot.py`` was fetching, transforming and writing in one module, so the
fetch layer had to know what a player was.
"""
