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
    out = run_buffering(Ate, _arrow(df[["t", "y", "x1", "x2"]]), named={"treatment": "t", "outcome": "y"})
    d = out.to_pydict()
    assert out.schema.names == ["method", "estimate", "std_error", "ci_lower", "ci_upper"]
    by_method = dict(zip(d["method"], d["estimate"], strict=True))
    assert set(by_method) == {"ipw", "regression_adjustment", "aipw"}
    assert abs(by_method["aipw"] - TRUE_TAU) < 0.4


def test_propensity_function() -> None:
    df = make_confounded(n=800)
    out = run_buffering(PropensityScores, _arrow(df[["id", "t", "x1", "x2"]]), named={"treatment": "t", "id": "id"})
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


class TestCursorSurvivesContinuation:
    """The finalize cursor must survive a wire round-trip between every tick.

    Over the stateless HTTP transport the framework serializes the finalize state
    after each tick, returns at most one data batch, then resumes by deserializing
    the token. ``run_buffering(..., serialize_state=True)`` emulates that. The old
    position-less ``DrainState{done}`` re-emitted row 0 forever (overrun guard);
    the offset cursor pages through the result and terminates.

    ``propensity_scores`` emits one row per input subject -- genuinely unbounded --
    so an 800-row cohort produces 800 output rows, far exceeding ``ROWS_PER_TICK``
    (64); the cursor must page across many continuation boundaries.
    """

    def test_propensity_pages_identically_under_serialization(self) -> None:
        from vgi_causal import buffering

        df = make_confounded(n=800)
        tbl = _arrow(df[["id", "t", "x1", "x2"]])
        named = {"treatment": "t", "id": "id"}

        # Sanity: the result genuinely spans several ROWS_PER_TICK ticks.
        assert len(df) > buffering.ROWS_PER_TICK

        baseline = run_buffering(PropensityScores, tbl, named=named).to_pydict()
        paged = run_buffering(PropensityScores, tbl, named=named, serialize_state=True).to_pydict()

        # (1) Same number of rows -- no truncation, no infinite re-emit.
        assert len(paged["id"]) == len(baseline["id"]) == len(df)
        # (2) Byte-identical rows in identical order (sort by id to be safe).
        b_order = np.argsort(np.asarray(baseline["id"]))
        p_order = np.argsort(np.asarray(paged["id"]))
        for col in ("id", "propensity", "treatment"):
            np.testing.assert_array_equal(np.asarray(baseline[col])[b_order], np.asarray(paged[col])[p_order])
        # (3) Each id emitted exactly once (no dupes).
        assert len(set(paged["id"])) == len(paged["id"]) == len(df)

    def test_small_rows_per_tick_pages_bounded_result(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Force the bounded estimators (ate=3 rows, att=1 row) to also page, so the
        # cursor is exercised across the continuation boundary for every function.
        from vgi_causal import buffering

        monkeypatch.setattr(buffering, "ROWS_PER_TICK", 2)
        df = make_confounded(n=400)

        ate_tbl = _arrow(df[["t", "y", "x1", "x2"]])
        baseline = run_buffering(Ate, ate_tbl, named={"treatment": "t", "outcome": "y"}).to_pydict()
        paged = run_buffering(Ate, ate_tbl, named={"treatment": "t", "outcome": "y"}, serialize_state=True).to_pydict()
        assert paged == baseline
        assert set(paged["method"]) == {"ipw", "regression_adjustment", "aipw"}

        att_tbl = _arrow(df[["t", "y", "x1", "x2"]])
        att_base = run_buffering(Att, att_tbl, named={"treatment": "t", "outcome": "y"}).to_pydict()
        att_paged = run_buffering(
            Att, att_tbl, named={"treatment": "t", "outcome": "y"}, serialize_state=True
        ).to_pydict()
        assert att_paged == att_base
        assert len(att_paged["estimate"]) == 1
