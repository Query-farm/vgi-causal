"""End-to-end tests driving causal_worker.py as a real subprocess.

These spawn the worker via ``vgi.client.Client`` and invoke each function
through the real ``table_buffering_function`` RPC path -- exactly how DuckDB
drives a buffering function after ``ATTACH`` -- exercising bind, the sink
process RPC per batch, combine, and the finalize source stream over the wire.
"""

from __future__ import annotations

import sys
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest
from vgi import Arguments
from vgi.client import Client, ClientError

from .synthetic import TRUE_TAU, make_confounded

_WORKER = str(Path(__file__).resolve().parent.parent / "causal_worker.py")


@pytest.fixture(scope="module")
def client() -> Iterator[Client]:
    with Client(f"{sys.executable} {_WORKER}", worker_limit=1) as c:
        yield c


def _run(client: Client, name: str, table: pa.Table, **named: str) -> pa.Table:
    batches = list(
        client.table_buffering_function(
            function_name=name,
            input=iter(table.to_batches()),
            arguments=Arguments(named={k: pa.scalar(v) for k, v in named.items()}),
        )
    )
    return pa.Table.from_batches(batches)


def test_ate_e2e(client: Client) -> None:
    df = make_confounded()
    tbl = pa.Table.from_pandas(df[["t", "y", "x1", "x2"]], preserve_index=False)
    out = _run(client, "ate", tbl, treatment="t", outcome="y")
    d = out.to_pydict()
    by_method = dict(zip(d["method"], d["estimate"], strict=True))
    assert abs(by_method["aipw"] - TRUE_TAU) < 0.4


def test_propensity_e2e(client: Client) -> None:
    df = make_confounded(n=800)
    tbl = pa.Table.from_pandas(df[["id", "t", "x1", "x2"]], preserve_index=False)
    out = _run(client, "propensity_scores", tbl, treatment="t", id="id")
    p = np.asarray(out.to_pydict()["propensity"])
    assert ((p > 0.0) & (p < 1.0)).all()


def test_att_e2e(client: Client) -> None:
    df = make_confounded()
    tbl = pa.Table.from_pandas(df[["t", "y", "x1", "x2"]], preserve_index=False)
    out = _run(client, "att", tbl, treatment="t", outcome="y")
    assert abs(out.to_pydict()["estimate"][0] - TRUE_TAU) < 0.6


def test_missing_column_errors_e2e(client: Client) -> None:
    tbl = pa.table({"t": [0, 1, 0], "y": [1.0, 2.0, 1.5], "x1": [0.1, 0.2, 0.3]})
    with pytest.raises(ClientError):
        _run(client, "ate", tbl, treatment="nope", outcome="y")
