"""Shared plumbing for the table-buffering causal-inference functions.

Every causal estimator (ate, propensity_scores, att) must see the *whole* input
relation before it can produce any output: the propensity model is fit on all
rows, the ATE/ATT averages over the whole sample. They are therefore
``TableBufferingFunction`` (Sink+Source) functions. The sink phase serializes
each input batch to execution-scoped storage; finalize reassembles the full
table and runs the estimator once.

This module holds the single-bucket sink/combine implementation (``SinkBuffer``)
plus the Arrow (de)serialization and a ``pandas`` assembly helper, so each
function only writes its ``finalize`` logic.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, cast

import pandas as pd
import pyarrow as pa
from vgi.table_buffering_function import TableBufferingFunction, TableBufferingParams
from vgi_rpc import ArrowSerializableDataclass
from vgi_rpc.rpc import OutputCollector

_DATA_KEY = b"input_batches"

# Rows emitted per finalize tick. Bounded so the cursor (``offset``) is observable
# across the HTTP limit-1 continuation boundary: the stateless HTTP transport
# wire-serializes the finalize state after every tick, returns at most one data
# batch per response, then resumes by deserializing the token. Correctness no
# longer depends on the whole result fitting in one producer batch.
ROWS_PER_TICK = 64


@dataclass(kw_only=True)
class DrainState(ArrowSerializableDataclass):
    """Externalized finalize cursor: result batch (IPC bytes) plus next-row offset.

    Both fields wire-serialize through the HTTP continuation token, so a resumed
    finalize tick sees the advanced ``offset`` and emits the next bounded slice
    (or finishes) -- it never re-runs the estimator or restarts from row 0. This
    is what lets ``propensity_scores`` (one row per input subject, unbounded)
    page correctly over the stateless HTTP transport.

    ``result_ipc`` is empty until the first tick computes the estimate;
    ``started`` distinguishes "not yet computed" from "computed an empty result".
    """

    started: bool = False
    offset: int = 0
    result_ipc: bytes = b""


def result_to_ipc(batch: pa.RecordBatch) -> bytes:
    """Serialize the full computed result batch to Arrow IPC bytes for the cursor."""
    sink = pa.BufferOutputStream()
    # pyarrow.ipc ships no type stubs, so mypy sees these as untyped calls.
    with pa.ipc.new_stream(sink, batch.schema) as writer:  # type: ignore[no-untyped-call]
        writer.write_batch(batch)
    return cast(bytes, sink.getvalue().to_pybytes())


def ipc_to_table(value: bytes) -> pa.Table:
    """Inverse of :func:`result_to_ipc`: read the cursor's result back as a table."""
    # pyarrow.ipc ships no type stubs, so mypy sees this as an untyped call.
    reader = pa.ipc.open_stream(pa.BufferReader(value))  # type: ignore[no-untyped-call]
    return cast(pa.Table, reader.read_all())


def serialize_batch(batch: pa.RecordBatch) -> bytes:
    """Serialize one RecordBatch to a self-describing Arrow IPC stream."""
    sink = pa.BufferOutputStream()
    # pyarrow.ipc ships no type stubs, so mypy sees these as untyped calls.
    with pa.ipc.new_stream(sink, batch.schema) as writer:  # type: ignore[no-untyped-call]
        writer.write_batch(batch)
    return cast(bytes, sink.getvalue().to_pybytes())


def deserialize_batches(value: bytes) -> list[pa.RecordBatch]:
    """Inverse of :func:`serialize_batch` for one stored blob."""
    # pyarrow.ipc ships no type stubs, so mypy sees this as an untyped call.
    reader = pa.ipc.open_stream(pa.BufferReader(value))  # type: ignore[no-untyped-call]
    return cast("list[pa.RecordBatch]", reader.read_all().to_batches())


def input_schema_of(params: Any) -> pa.Schema:
    """Input schema from a process/finalize params object."""
    schema = params.init_call.bind_call.input_schema
    assert schema is not None
    return schema


class SinkBuffer[TArgs, TState](TableBufferingFunction[TArgs, TState]):
    """Single-bucket sink/combine: buffer every input batch under one key.

    Subclasses implement ``on_bind``, ``initial_finalize_state``, and
    ``finalize`` (calling ``buffered_frame(params)`` to get the full input as a
    ``pandas.DataFrame``).
    """

    @classmethod
    def process(cls, batch: pa.RecordBatch, params: TableBufferingParams[TArgs]) -> bytes:
        """Sink one input batch under the single buffering key.

        Args:
            batch: A batch of input rows to buffer.
            params: The buffering params for this execution.

        Returns:
            The execution id used as this sink's combine key.
        """
        if batch.num_rows:
            params.storage.state_append(_DATA_KEY, b"", serialize_batch(batch))
        return params.execution_id

    @classmethod
    def combine(cls, state_ids: list[bytes], params: TableBufferingParams[TArgs]) -> list[bytes]:
        """Collapse all sink keys to one finalize bucket.

        Args:
            state_ids: The sink keys produced by ``process``.
            params: The buffering params for this execution.

        Returns:
            A single-element list naming the one finalize bucket.
        """
        return [params.execution_id]

    @classmethod
    def buffered_frame(cls, params: TableBufferingParams[TArgs]) -> pd.DataFrame:
        """Reassemble all sunk batches into a single pandas DataFrame.

        Returns an empty (zero-row) frame -- with the right column names -- when
        no rows were sunk, so finalize can apply uniform empty-input handling.
        """
        input_schema = input_schema_of(params)
        batches: list[pa.RecordBatch] = []
        for _sid, value in params.storage.state_log_scan(_DATA_KEY, b""):
            batches.extend(deserialize_batches(value))
        if not batches:
            return cast(pd.DataFrame, pa.Table.from_batches([], schema=input_schema).to_pandas())
        return cast(pd.DataFrame, pa.Table.from_batches(batches, schema=input_schema).to_pandas())

    @classmethod
    def drain_result(
        cls,
        params: TableBufferingParams[TArgs],
        state: DrainState,
        out: OutputCollector,
        compute: Callable[[pd.DataFrame], dict[str, list[Any]]],
    ) -> None:
        """Compute the result once into the cursor, then stream a bounded slice.

        The first finalize tick runs ``compute`` over the buffered frame, packs the
        result batch into ``state.result_ipc`` (Arrow IPC bytes) and flips
        ``state.started``. Every tick then emits at most ``ROWS_PER_TICK`` rows from
        ``state.offset``, advances ``state.offset``, and calls ``out.finish()`` once
        the result is drained. Because ``state`` round-trips through the HTTP
        continuation token, a resumed tick sees the advanced offset and never
        re-runs ``compute`` or restarts from row 0.

        Args:
            params: The buffering params (args, output schema, storage).
            state: The finalize cursor (result bytes + offset).
            out: The collector to emit the result slice into.
            compute: Maps the buffered frame to the estimator's ``dict[str, list]``.
        """
        if not state.started:
            df = cls.buffered_frame(params)
            result = compute(df)
            batch = pa.RecordBatch.from_pydict(result, schema=params.output_schema)
            state.result_ipc = result_to_ipc(batch)
            state.started = True
            state.offset = 0

        table = ipc_to_table(state.result_ipc)
        total = table.num_rows
        if state.offset >= total:
            out.finish()
            return
        end = min(state.offset + ROWS_PER_TICK, total)
        chunk = table.slice(state.offset, end - state.offset).combine_chunks()
        out.emit(chunk.to_batches()[0])
        state.offset = end
