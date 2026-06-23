"""Table-function tests via the in-process buffering harness.

Drive each causal function through the real bind -> process(sink) -> combine ->
finalize lifecycle (no subprocess), checking the emitted Arrow result and that
the named column-role args resolve columns in the input relation.
"""

from __future__ import annotations

import numpy as np
import pyarrow as pa
import pytest

from vgi_causal.tables import Ate, Att, PropensityScores

from .harness import run_buffering
from .synthetic import TRUE_TAU, make_confounded


def _arrow(df) -> pa.Table:
    return pa.Table.from_pandas(df, preserve_index=False)


def test_ate_function_recovers_tau() -> None:
    df = make_confounded()
    out = run_buffering(
        Ate, _arrow(df[["t", "y", "x1", "x2"]]), named={"treatment": "t", "outcome": "y"}
    )
    d = out.to_pydict()
    assert out.schema.names == ["method", "estimate", "std_error", "ci_lower", "ci_upper"]
    by_method = dict(zip(d["method"], d["estimate"], strict=True))
    assert set(by_method) == {"ipw", "regression_adjustment", "aipw"}
    assert abs(by_method["aipw"] - TRUE_TAU) < 0.4


def test_propensity_function() -> None:
    df = make_confounded(n=800)
    out = run_buffering(
        PropensityScores, _arrow(df[["id", "t", "x1", "x2"]]), named={"treatment": "t", "id": "id"}
    )
    d = out.to_pydict()
    assert out.schema.names == ["id", "propensity", "treatment"]
    assert pa.types.is_int64(out.schema.field("id").type)
    p = np.asarray(d["propensity"])
    assert ((p > 0.0) & (p < 1.0)).all()
    assert out.num_rows == len(df)


def test_att_function() -> None:
    df = make_confounded()
    out = run_buffering(Att, _arrow(df[["t", "y", "x1", "x2"]]), named={"treatment": "t", "outcome": "y"})
    d = out.to_pydict()
    assert out.num_rows == 1
    assert abs(d["estimate"][0] - TRUE_TAU) < 0.6


def test_missing_column_raises() -> None:
    tbl = pa.table({"t": [0, 1], "y": [1.0, 2.0], "x1": [0.1, 0.2]})
    with pytest.raises(Exception, match="missing required column"):
        run_buffering(Ate, tbl, named={"treatment": "nope", "outcome": "y"})


def test_non_binary_treatment_raises() -> None:
    tbl = pa.table({"t": [0, 1, 2], "y": [1.0, 2.0, 3.0], "x1": [0.1, 0.2, 0.3]})
    with pytest.raises(Exception, match="binary 0/1"):
        run_buffering(Ate, tbl, named={"treatment": "t", "outcome": "y"})
