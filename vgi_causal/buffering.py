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

from dataclasses import dataclass
from typing import Any, cast

import pandas as pd
import pyarrow as pa
from vgi.table_buffering_function import TableBufferingFunction, TableBufferingParams
from vgi_rpc import ArrowSerializableDataclass

_DATA_KEY = b"input_batches"


@dataclass(kw_only=True)
class DrainState(ArrowSerializableDataclass):
    """Per-finalize-stream cursor: emit the single result batch once, then finish."""

    done: bool = False


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
