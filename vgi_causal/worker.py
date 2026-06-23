"""VGI worker exposing causal treatment-effect estimation to DuckDB/SQL.

Assembles the causal table functions in ``vgi_causal`` into a single ``causal``
catalog and provides the process entry point. The repo-root ``causal_worker.py``
is a thin shim over this module for ``uv run``; installed users get the
``vgi-causal`` console script, which calls ``main`` here.

    ATTACH 'causal' (TYPE vgi, LOCATION 'uv run causal_worker.py');
    SELECT * FROM causal.ate((SELECT * FROM cohort), treatment := 't', outcome := 'y');
"""

from __future__ import annotations

import sys

from vgi import Worker
from vgi.catalog import Catalog, Schema

from vgi_causal.tables import TABLE_FUNCTIONS

_FUNCTIONS: list[type] = [*TABLE_FUNCTIONS]

_CAUSAL_CATALOG = Catalog(
    name="causal",
    default_schema="main",
    schemas=[
        Schema(
            name="main",
            comment="Causal treatment-effect estimation (ATE, ATT, propensity scores) for SQL",
            functions=list(_FUNCTIONS),
        ),
    ],
)


class CausalWorker(Worker):
    """Worker process hosting the ``causal`` catalog."""

    catalog = _CAUSAL_CATALOG


def main() -> None:
    """Run the worker (stdio by default; pass ``--http`` for the HTTP server)."""
    CausalWorker.main()


def main_http() -> None:
    """Run the worker over HTTP (injects ``--http`` into the worker CLI)."""
    argv = sys.argv[1:]
    if "--http" not in argv:
        argv = ["--http", *argv]
    sys.argv = [sys.argv[0], *argv]
    CausalWorker.main()
