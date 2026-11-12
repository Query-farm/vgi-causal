<p align="center">
  <img src="https://query.farm/media-kit/products/vgi/vgi-logo.png" alt="Vector Gateway Interface (VGI)" width="320">
</p>

<p align="center"><em>A <a href="https://query.farm">Query.Farm</a> VGI worker for DuckDB.</em></p>

# Causal Inference — ATE/ATT & Propensity (IPW/AIPW) in DuckDB

> **vgi-causal** · a [Query.Farm](https://query.farm) VGI worker

A [VGI](https://query.farm) worker that brings **causal treatment-effect
estimation** to DuckDB/SQL: average treatment effects (ATE) by inverse-
probability weighting, regression adjustment, and the doubly-robust AIPW
estimator; the average treatment effect on the treated (ATT); and per-row
propensity scores — backed by [scikit-learn](https://scikit-learn.org/) and
[statsmodels](https://www.statsmodels.org/) (both BSD/permissive licensed).

```sql
INSTALL vgi FROM community; LOAD vgi;
ATTACH 'causal' (TYPE vgi, LOCATION 'uvx --from git+https://github.com/Query-farm/vgi-causal@v0.4.0 vgi-causal-worker');

-- Average Treatment Effect: one row per estimator (ipw, regression_adjustment, aipw)
SELECT * FROM causal.ate((SELECT * FROM cohort),
                         treatment := 't', outcome := 'y');

-- Per-row propensity scores e(X) = P(T=1 | X); id is carried through
SELECT * FROM causal.propensity_scores((SELECT id, t, x1, x2 FROM cohort),
                                       treatment := 't', id := 'id')
ORDER BY id;

-- Average Treatment effect on the Treated
SELECT * FROM causal.att((SELECT * FROM cohort),
                         treatment := 't', outcome := 'y');
```

## Quick start

The worker isn't on PyPI; [uv](https://docs.astral.sh/uv/) runs it straight from a tagged release on GitHub. `uvx` builds an isolated environment on the first ATTACH and reuses it afterwards:

```sql
INSTALL vgi FROM community;
LOAD vgi;

ATTACH 'causal' AS causal (TYPE vgi, LOCATION 'uvx --from git+https://github.com/Query-farm/vgi-causal@v0.4.0 vgi-causal-worker');

-- Average treatment effect by IPW, regression adjustment, and doubly-robust AIPW.
WITH cohort AS (
  SELECT
    g                                                   AS id,
    (g - 30)::DOUBLE / 10.0                             AS x,
    CASE WHEN ((g * 7 + 3) % 10) < 5 THEN 1 ELSE 0 END  AS t,
    5.0 * (CASE WHEN ((g * 7 + 3) % 10) < 5 THEN 1 ELSE 0 END)
      + 2.0 * ((g - 30)::DOUBLE / 10.0)                 AS y
  FROM generate_series(0, 60) AS s(g)
)
SELECT method, round(estimate, 2) AS estimate FROM causal.main.ate((SELECT t, y, x FROM cohort), treatment := 't', outcome := 'y') ORDER BY method;

-- Average treatment effect on the treated (ATT).
WITH cohort AS (
  SELECT
    g                                                   AS id,
    (g - 30)::DOUBLE / 10.0                             AS x,
    CASE WHEN ((g * 7 + 3) % 10) < 5 THEN 1 ELSE 0 END  AS t,
    5.0 * (CASE WHEN ((g * 7 + 3) % 10) < 5 THEN 1 ELSE 0 END)
      + 2.0 * ((g - 30)::DOUBLE / 10.0)                 AS y
  FROM generate_series(0, 60) AS s(g)
)
SELECT round(estimate, 2) AS att FROM causal.main.att((SELECT t, y, x FROM cohort), treatment := 't', outcome := 'y');
```

Change `@v0.4.0` to move to a newer release.

### Other ways to run it

- **Docker** (the image bundles the worker and its dependencies):

  ```sql
  ATTACH 'causal' AS causal (TYPE vgi, LOCATION 'docker://ghcr.io/query-farm/vgi-causal:0.4.0');
  ```

- **From a local checkout** (`git clone https://github.com/Query-farm/vgi-causal`), point `LOCATION` at `uv run --project /path/to/vgi-causal vgi-causal-worker`.

### Build from source

To build the worker yourself, see [Development](#development).

## Data flow: one relation in, a result set out

Every function is a **table function** that consumes a *whole input relation* —
passed as a single `(SELECT ...)` subquery (the positional argument) — and emits
a result set. The roles of the columns inside that relation are passed as
**named string arguments**:

| named arg | meaning |
|-----------|---------|
| `treatment := 'col'` | the **binary 0/1** treatment column (1 = treated, 0 = control) |
| `outcome := 'col'`   | the numeric outcome column (`ate` / `att`) |
| `id := 'col'`        | (`propensity_scores` only) the row identifier to pass through, **excluded** from the covariates |

**Every other column in the relation is a covariate / confounder** and is
adjusted for. This mirrors how `vgi-scikit-learn` names `target` / `id`: the
relation *is* the data, and the named args just say which column plays which
role. Because a causal estimate fits a model over and averages across **every
row**, these are buffering (Sink+Source) functions — they buffer all input
batches, then run the estimator once.

## Functions

| function | returns | estimand |
|----------|---------|----------|
| `ate(rel, treatment, outcome)` | `(method, estimate, std_error, ci_lower, ci_upper)` — one row per method | ATE `E[Y(1) − Y(0)]` |
| `propensity_scores(rel, treatment, id)` | `(id, propensity, treatment)` — one row per input row | `e(X) = P(T=1 ∣ X)` |
| `att(rel, treatment, outcome)` | `(estimate, std_error)` — one row | ATT `E[Y(1) − Y(0) ∣ T=1]` |

### The three ATE estimators

`ate` emits one row per method:

- **`ipw`** — Inverse-Probability Weighting (stabilized/Hájek): weight treated
  rows by `1/e(X)` and control rows by `1/(1−e(X))`, where `e(X)` is the
  propensity from a logistic model.
- **`regression_adjustment`** — fit an outcome model `μ(T, X)` and average the
  predicted contrast `μ(1, X) − μ(0, X)` (the g-formula / standardization).
- **`aipw`** — Augmented IPW / **doubly-robust**: combines the outcome model and
  the propensity model; consistent if *either* is correctly specified. Its
  influence-function variance yields an honest standard error and Wald CI.

`att` uses IPW-ATT weighting (controls reweighted by the odds `e(X)/(1−e(X))`)
with a nonparametric bootstrap standard error.

## Causal interpretation — assumptions

These estimates are **causal only under the standard backdoor / selection-on-
observables assumptions**:

- **Unconfoundedness (conditional ignorability):** `(Y(0), Y(1)) ⟂ T | X` — the
  covariates you supply block every backdoor path; there is **no unmeasured
  confounding**.
- **Overlap (positivity):** `0 < e(X) < 1` for all `X`.
- **SUTVA / consistency:** a well-defined, non-interfering treatment.

If those don't hold, the numbers are still computed but are merely *adjusted
associations*, not causal effects. **The estimate is causal only under
unconfoundedness given the covariates you supply.** Choosing those covariates is
a modeling decision the worker cannot make for you.

## Treatment coding

`treatment` must be **binary 0/1** (or boolean). A non-binary / continuous
column is rejected with a clear error — these estimators are defined for a
binary treatment. Treated = `1`, control = `0`.

## Backend: scikit-learn + statsmodels (dowhy optional)

The default backend implements the estimators directly on scikit-learn
(`LogisticRegression` for propensity, `LinearRegression` for the outcome model)
and SciPy/statsmodels for the variance/quantiles. This keeps the worker fast,
deterministic (fixed `random_state`), and dependency-light — and the
end-to-end SQL tests reproducible.

[dowhy](https://github.com/py-why/dowhy) (Microsoft, MIT) is supported as an
**optional** doubly-robust backend: `pip install 'vgi-causal[dowhy]'`. It is
imported lazily (dowhy is slow to import) and only when explicitly requested, so
the default path never pays its cost.

## Development

```sh
uv sync --extra dev          # install deps (resolves vgi-python from ../vgi-python)
uv run --no-sync pytest -q   # unit + in-process + Client RPC tests
make test-sql                # end-to-end DuckDB sqllogictest via haybarn-unittest
```

## License

MIT. scikit-learn, statsmodels, scipy, numpy, pandas are BSD/permissive; the
optional dowhy backend is MIT.

---

## Authorship & License

Written by [Query.Farm](https://query.farm).

Copyright 2026 Query Farm LLC - https://query.farm

