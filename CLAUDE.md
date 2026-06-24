# CLAUDE.md — vgi-causal

Contributor/agent notes. User-facing docs live in `README.md`; this is the
"how it's built and where the sharp edges are" companion. Sibling style/tooling
to `vgi-conform` / `vgi-survival` (structure) and `vgi-scikit-learn` /
`vgi-survival` (the whole-relation buffering data-flow).

## What this is

A [VGI](https://query.farm) worker exposing **causal treatment-effect
estimation** to DuckDB/SQL: ATE (IPW, regression adjustment, doubly-robust
AIPW), ATT (IPW-ATT), and per-row propensity scores. Backed by scikit-learn +
statsmodels (BSD/permissive). `causal_worker.py` assembles every function into
one `causal` catalog (single `main` schema) over stdio.

## Layout

```
causal_worker.py       repo-root stdio entry shim; PEP 723 inline deps; main()
vgi_causal/
  causal.py            pure causal-inference logic over pandas frames; no Arrow/VGI; unit-testable
  buffering.py         SinkBuffer (single-bucket sink/combine) + Arrow<->pandas plumbing
  tables.py            the three TableBufferingFunction wrappers + output schemas + arg classes
  schema_utils.py      pa.Field comment / column-doc helper
  worker.py            assembles the catalog; main() / main_http()
tests/
  synthetic.py         confounded data generator with a KNOWN tau (the validation backbone)
  test_causal.py       pure-logic + edges (true-tau recovery, naive-vs-adjusted)
  test_tables.py       in-process buffering harness
  test_client.py       real Client RPC subprocess (how DuckDB drives it)
test/sql/causal.test   haybarn-unittest sqllogictest — authoritative E2E
Makefile               test / test-unit / test-sql / lint
```

To add a function: implement the math in `causal.py` (pure, takes a pandas
frame + role kwargs, returns a `dict[str, list]`, raises `CausalError` on bad
input), add a `pa.schema` + `@dataclass` args class + a `SinkBuffer` subclass in
`tables.py`, append it to `TABLE_FUNCTIONS`.

## THE core convention (read first): one relation in, named role args

These are **table functions**, not scalars. Each takes the whole input relation
as a single `(SELECT ...)` subquery — `Arg(0)`, typed `TableInput` — and the
column **roles** as NAMED string args (`treatment := 't'`, `outcome := 'y'`,
`id := 'id'`). **Every column not named by a role is a covariate/confounder**
and is adjusted for. The relation's columns *are* the data.

Causal estimates fit a model over and average across **every row** before any
output, so every function is a `TableBufferingFunction` (Sink+Source):

- `process(batch)` — sink each input batch to execution-scoped `BoundStorage`.
- `combine(state_ids)` — collapse to a single finalize key (one bucket).
- `finalize(...)` — on the first tick reassemble the full table
  (`buffered_frame()` → pandas) and run the estimator once into the cursor; each
  tick then emits a bounded slice and `out.finish()` once drained.

`SinkBuffer` in `buffering.py` implements `process`/`combine`/`buffered_frame`
plus `drain_result(...)` (the cursor loop); each function only writes `on_bind`
(its output schema) + a one-line `finalize` that hands `drain_result` the
estimator call.

### Why finalize streams an OFFSET cursor (HTTP continuation)

Over the **stateless http transport** the framework round-trips a producer's
per-finalize-stream state through a continuation token: after each `finalize()`
tick it wire-serializes the state (`ArrowSerializableDataclass.serialize_to_bytes()`),
the client returns it, and the worker resumes by deserializing it — emitting at
most one (the producer batch limit) data batch per response. subprocess/unix keep
the live state in-process so they hide the bug; only http (and the
`run_buffering(..., serialize_state=True)` unit harness) expose it.

A position-less `DrainState{done: bool}` that emits ALL rows in one `out.emit`
then sets `done` restarts from row 0 on every http resume and **loops forever**
once the output exceeds one producer batch — which `propensity_scores` (one row
per input subject, unbounded) routinely does. So `DrainState` carries an explicit
**offset cursor**: the already-computed result batch as IPC bytes (`result_ipc`,
fully serializable), a `started` flag, and an integer `offset`. The first tick
computes + packs the result; every tick emits at most `ROWS_PER_TICK` (64) rows
from `offset`, advances `offset`, and finishes when `offset >= total`. Because
`offset` survives the wire round-trip, a resumed tick emits the NEXT slice — never
re-runs the estimator, never restarts from row 0. `ate` (3 rows) and `att` (1 row)
are bounded but use the identical cursor for uniformity. Results are byte-identical
to the old emit-all path. The regression test is
`TestCursorSurvivesContinuation` in `test_tables.py` (re-serializes finalize state
between every tick, 10 000-tick overrun guard) plus the big-cohort paging asserts
in `causal.test` (which only terminate over http if the cursor works).

## Estimators (the math)

All in `causal.py`, pure functions over numpy arrays:

- **Propensity** `_fit_propensity`: L2-regularized `LogisticRegression` (lbfgs),
  scores clipped to `[1e-6, 1−1e-6]`. Regularization + clipping is what keeps
  **perfect separation** from crashing or producing infinite IPW weights.
- **IPW ATE** `_ipw_ate`: Hájek (self-normalized) contrast; SE from the
  influence function.
- **Regression-adjustment ATE** `_regression_adjustment_ate`: g-formula —
  `LinearRegression` on `[T, X]`, average `μ(1,X)−μ(0,X)`; SE = OLS SE of the
  treatment coefficient.
- **AIPW ATE** `_aipw_ate`: doubly-robust; per-row influence function gives the
  SE (and 95% Wald CI via `scipy.stats.norm`).
- **ATT** `att`: IPW-ATT weighting (`e/(1−e)` on controls), bootstrap SE
  (`np.random.default_rng(random_state)` — deterministic).

## Validation: a planted-effect synthetic (read `tests/synthetic.py`)

`make_confounded` plants a homogeneous effect `tau` and confounds it: treatment
propensity AND the outcome both depend on `x1, x2`. The tests assert the whole
point of causal adjustment:

- the **naive** difference of means is biased (`test_naive_is_biased`),
- every **adjusted** estimator recovers `tau` within tolerance
  (`test_ate_recovers_tau_all_methods`), and adjustment beats naive
  (`test_adjusted_beats_naive`),
- propensity scores are valid probabilities, higher for treated-likely units,
- ATT recovers `tau`.

The `.test` file plants the same story deterministically with
`generate_series` + a **noise-free linear** DGP, so `regression_adjustment`
recovers `tau` exactly (`round(estimate, 2) = 5.0`) — no RNG in the SQL assert.

## Sharp edges (learned the hard way)

1. **`haybarn-unittest` silently SKIPS `require vgi`.** Use an explicit
   `statement ok` / `LOAD vgi;` (the `.test` here does). `# group: [vgi_causal]`
   and `require-env VGI_CAUSAL_WORKER` gate the file.
2. **Buffering needs the input schema at bind.** The `(SELECT ...)` relation's
   schema arrives via `bind_call.input_schema`; `buffered_frame()` uses it to
   reassemble even when zero batches were sunk. `Client.table_buffering_function`
   peeks the first batch to learn that schema — so E2E tests always feed at least
   the typed columns.
3. **Treatment must be binary 0/1.** `_binary_treatment` rejects anything with
   values outside `{0,1}` (e.g. an id or a multi-valued column) with a clear
   error. The `.test` exercises this by aliasing `id AS t`.
4. **Every non-role column is a covariate.** Don't `SELECT *` a relation that
   still has the `id` into `ate`/`att` — `id` would be treated as a covariate.
   `propensity_scores` is the only one that takes (and excludes) `id`.
5. **Determinism.** Every estimator takes `random_state` (default 0); the ATT
   bootstrap uses a seeded `default_rng`. Keep it that way so CI and the `.test`
   are reproducible.
6. **dowhy is OPTIONAL and lazy.** `_dowhy_ate` imports dowhy *inside* the
   function and returns `None` if it's absent (caller falls back to in-house
   AIPW). Never import dowhy at module top level — it's slow and not a default
   dependency.

## Assumptions (state them; the worker can't enforce them)

Estimates are causal only under **unconfoundedness** (no unmeasured confounding
given the supplied covariates), **overlap** (`0 < e(X) < 1`), and SUTVA. See the
module docstring in `causal.py` and the README. Picking the covariate set is the
user's modeling call.
