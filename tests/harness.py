"""In-process driver for the causal buffering (Sink+Source) functions.

Runs a ``TableBufferingFunction`` through its real bind -> init -> process(sink)
-> combine -> finalize lifecycle without spawning a worker process, so unit
tests stay fast and debuggable while still exercising the framework's argument
parsing, storage round-trip, and output schema.
"""

from __future__ import annotations

from typing import Any

import pyarrow as pa
from vgi.arguments import Arguments
from vgi.function_storage import BoundStorage, FunctionStorageSqlite
from vgi.invocation import FunctionType
from vgi.protocol import BindRequest, InitRequest
from vgi.table_buffering_function import TableBufferingParams


class _BatchTooLarge(Exception):
    """A finalize tick emitted more rows than one continuation response can carry."""


class _Collector:
    """Captures emitted batches from a finalize stream.

    When ``max_rows_per_tick`` is set (the ``serialize_state`` / HTTP-continuation
    model), an ``emit`` whose batch exceeds that cap raises :class:`_BatchTooLarge`
    *before* recording the batch -- mirroring the stateless transport, which rejects
    an over-cap response and resumes the tick from the pre-tick continuation token.
    A position-less finalize re-emits the whole result every tick and never fits;
    the offset cursor emits a bounded slice that does.
    """

    def __init__(self, max_rows_per_tick: int | None = None) -> None:
        self.batches: list[pa.RecordBatch] = []
        self.finished = False
        self._max_rows_per_tick = max_rows_per_tick
        self._tick_rows = 0

    def begin_tick(self) -> None:
        self._tick_rows = 0

    def emit(self, batch: pa.RecordBatch, *_a: Any, **_kw: Any) -> None:
        self._tick_rows += batch.num_rows
        if self._max_rows_per_tick is not None and self._tick_rows > self._max_rows_per_tick:
            raise _BatchTooLarge(
                f"finalize emitted {self._tick_rows} rows in one tick, exceeding the "
                f"{self._max_rows_per_tick}-row continuation cap"
            )
        self.batches.append(batch)

    def finish(self) -> None:
        self.finished = True

    def client_log(self, *_a: Any, **_kw: Any) -> None:
        pass


def run_buffering(
    func_cls: type,
    table: pa.Table,
    *,
    named: dict[str, str] | None = None,
    serialize_state: bool = False,
) -> pa.Table:
    """Drive a causal buffering function over a whole input ``table``.

    Args:
        func_cls: The ``TableBufferingFunction`` subclass to run.
        table: The input relation (the ``(SELECT ...)`` data) as an Arrow table.
        named: Named string column-role args (e.g. ``{"treatment": "t"}``).
        serialize_state: When ``True``, wire-serialize the finalize state between
            every ``finalize()`` tick (``deserialize_from_bytes(serialize_to_bytes())``),
            emulating the stateless HTTP continuation token. A position-less cursor
            (the old ``DrainState{done}``) re-emits row 0 forever under this mode and
            trips the overrun guard; the offset cursor pages and terminates.

    Returns:
        The emitted result as a single Arrow table (the function's output).

    Raises:
        AssertionError: If a finalize stream runs past the ~10000-tick guard
            (i.e. it never terminates -- the latent HTTP-continuation bug).
    """
    input_schema = table.schema
    args = Arguments(
        positional=(),
        named={k: pa.scalar(v) for k, v in (named or {}).items()},
    )

    bind_req = BindRequest(
        function_name=func_cls.Meta.name,
        arguments=args,
        function_type=FunctionType.TABLE_BUFFERING,
        input_schema=input_schema,
    )
    bind_resp = func_cls.bind(bind_req)

    init_req = InitRequest(bind_call=bind_req, output_schema=bind_resp.output_schema)
    init_resp = func_cls.global_init(init_req)
    execution_id = init_resp.execution_id

    storage = BoundStorage(FunctionStorageSqlite(":memory:"), execution_id)
    parsed_args = func_cls._parse_arguments(func_cls.FunctionArguments, args)

    def make_params() -> TableBufferingParams:
        return TableBufferingParams(
            args=parsed_args,
            init_call=init_req,
            init_response=init_resp,
            output_schema=bind_resp.output_schema,
            settings={},
            secrets={},
            storage=storage,
            execution_id=execution_id,
            attach_id=b"",
            transaction_id=None,
            function_name=func_cls.Meta.name,
        )

    # Sink phase: one process() call per input batch.
    state_ids: list[bytes] = []
    for batch in table.to_batches():
        state_ids.append(func_cls.process(batch, make_params()))

    # Combine phase.
    finalize_ids = func_cls.combine(state_ids, make_params())

    # Source phase: drain each finalize stream. Under ``serialize_state`` we model
    # the stateless HTTP continuation: re-serialize the finalize state between every
    # tick, and cap each tick at one response worth of rows (``ROWS_PER_TICK``). An
    # over-cap emit is rejected and the tick resumes from the PRE-tick token (the
    # mutation is discarded) -- exactly what makes a position-less finalize loop
    # forever and a position cursor page.
    from vgi_causal.buffering import ROWS_PER_TICK

    cap = ROWS_PER_TICK if serialize_state else None
    out = _Collector(max_rows_per_tick=cap)
    max_ticks = 10_000
    for fid in finalize_ids:
        params = make_params()
        state = func_cls.initial_finalize_state(fid, params)
        ticks = 0
        while not out.finished:
            ticks += 1
            assert ticks < max_ticks, (
                f"{func_cls.Meta.name}.finalize did not terminate within {max_ticks} ticks "
                f"(serialize_state={serialize_state}): the finalize cursor never advances "
                "across the continuation boundary -- the HTTP-continuation bug."
            )
            if not serialize_state:
                func_cls.finalize(params, fid, state, out)
                continue
            pre_tick_blob = state.serialize_to_bytes()
            saved_batches = len(out.batches)
            out.begin_tick()
            try:
                func_cls.finalize(params, fid, state, out)
            except _BatchTooLarge:
                # Continuation: drop the over-cap batches and the un-committed state
                # advance; the next attempt resumes from the pre-tick token.
                del out.batches[saved_batches:]
                state = type(state).deserialize_from_bytes(pre_tick_blob)
                continue
            state = type(state).deserialize_from_bytes(state.serialize_to_bytes())
        out.finished = False  # reset for the next finalize stream

    return pa.Table.from_batches(out.batches, schema=bind_resp.output_schema)
