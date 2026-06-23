<p align="center">
  <img src="https://raw.githubusercontent.com/Query-farm/vgi/main/docs/vgi-logo.png" alt="Vector Gateway Interface (VGI)" width="320">
</p>

<p align="center"><em>A <a href="https://query.farm">Query.Farm</a> VGI worker for DuckDB.</em></p>

# vgi-causal

A [VGI](https://query.farm) worker that brings **causal treatment-effect
estimation** to DuckDB/SQL: average treatment effects (ATE) by inverse-
probability weighting, regression adjustment, and the doubly-robust AIPW
estimator; the average treatment effect on the treated (ATT); and per-row
propensity scores — backed by [scikit-learn](https://scikit-learn.org/) and
[statsmodels](https://www.statsmodels.org/) (both BSD/permissive licensed).

```sql
INSTALL vgi FROM community; LOAD vgi;
ATTACH 'causal' (TYPE vgi, LOCATION 'uv run causal_worker.py');

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

## Install & run

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

Written by [Query.Farm](https://query.farm) — every VGI worker is designed and built by Query.Farm.

Copyright 2026 Query Farm LLC - https://query.farm

