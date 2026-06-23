"""Pure causal-inference logic over scikit-learn + statsmodels.

This module is the framework-free core: it takes a ``pandas.DataFrame`` (the
buffered input relation) plus the column roles, fits the propensity / outcome
models, and returns plain ``dict[str, list]`` column blocks ready to hand to
pyarrow. No VGI, no Arrow, no DuckDB here -- so every function is directly
unit-testable.

Estimands and estimators
------------------------
Given a binary treatment ``T in {0,1}``, an outcome ``Y``, and a vector of
covariates ``X`` (every other column in the relation), we estimate:

* **ATE** -- the Average Treatment Effect ``E[Y(1) - Y(0)]`` via three
  complementary estimators:

  - ``ipw`` -- Inverse-Probability Weighting (Horvitz-Thompson, stabilized):
    weight treated rows by ``1/e(X)`` and control rows by ``1/(1-e(X))`` where
    ``e(X) = P(T=1 | X)`` is the propensity score from a logistic model.
  - ``regression_adjustment`` -- fit an outcome model ``mu(T, X)`` (linear
    regression including the treatment indicator and covariates) and average
    the predicted contrast ``mu(1, X) - mu(0, X)`` over the sample (the
    g-formula / standardization estimator).
  - ``aipw`` -- the Augmented IPW / doubly-robust estimator, which combines the
    outcome model and the propensity model. It is consistent if *either* model
    is correctly specified, and its influence-function variance gives an
    honest standard error and Wald confidence interval.

* **ATT** -- the Average Treatment effect on the Treated
  ``E[Y(1) - Y(0) | T=1]`` via IPW-ATT weighting (control rows weighted by
  ``e(X)/(1-e(X))``), with a bootstrap standard error.

* **propensity scores** -- the fitted per-row ``e(X) = P(T=1 | X)``.

Causal interpretation / assumptions
-----------------------------------
These estimates are *causal* only under the standard backdoor / selection-on-
observables assumptions:

* **Unconfoundedness (conditional ignorability):** ``(Y(0), Y(1)) ⟂ T | X`` --
  the supplied covariates ``X`` block every backdoor path, i.e. there is no
  unmeasured confounding.
* **Overlap (positivity):** ``0 < e(X) < 1`` for all ``X`` -- every unit has a
  non-zero chance of either treatment.
* **SUTVA / consistency:** well-defined, non-interfering treatment.

If those do not hold, the numbers are still computed but are merely adjusted
associations, not causal effects. The estimate is causal *only under
unconfoundedness given the covariates you supply*.

scikit-learn and statsmodels are both BSD/permissive licensed; numpy/pandas are
BSD-licensed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Importing scikit-learn / statsmodels is comparatively expensive; do it once at
# module import so the per-call path is cheap. The worker imports this module at
# startup, so the cost is paid before the first SQL call. (dowhy, if installed,
# is even slower to import -- it is an *optional* backend imported lazily in
# ``_dowhy_ate`` only when explicitly requested.)
from scipy import stats
from sklearn.linear_model import LinearRegression, LogisticRegression

__all__ = [
    "CausalError",
    "ate",
    "att",
    "propensity_scores",
]

# A small epsilon to clip propensity scores away from {0, 1} so IPW weights stay
# finite even under near-perfect separation. This trades a touch of bias for
# numerical stability (a standard "trimming"/"clipping" device).
_PROPENSITY_CLIP = 1e-6


class CausalError(ValueError):
    """Raised for user-facing input problems (missing/empty/non-binary columns).

    A plain, explicit error so the worker surfaces a clear message to SQL
    instead of crashing with an opaque pandas/sklearn traceback.
    """


def _require_columns(df: pd.DataFrame, required: dict[str, str]) -> None:
    """Validate that each required role maps to a present column.

    Args:
        df: The input relation.
        required: Mapping of role name (e.g. ``"treatment"``) to the column name
            the caller passed for that role.

    Raises:
        CausalError: If any named column is absent from the relation.
    """
    have = set(df.columns)
    missing = {role: col for role, col in required.items() if col not in have}
    if missing:
        detail = ", ".join(f"{role} := '{col}'" for role, col in missing.items())
        raise CausalError(
            f"missing required column(s): {detail}; "
            f"input relation has columns: {', '.join(map(str, df.columns))}"
        )


def _numeric(df: pd.DataFrame, column: str, *, role: str) -> np.ndarray:
    """Coerce a column to a float64 numpy array or raise a clear error.

    Args:
        df: The input relation.
        column: Column name to coerce.
        role: Human-readable role label for the error message.

    Returns:
        The column as a contiguous float64 array.

    Raises:
        CausalError: If the column is not numeric / not coercible to float.
    """
    series = df[column]
    coerced = pd.to_numeric(series, errors="coerce")
    if coerced.isna().any() and not series.isna().any():
        raise CausalError(
            f"{role} column '{column}' must be numeric, but contains "
            f"non-numeric values (dtype {series.dtype})"
        )
    return np.asarray(coerced, dtype=float)


def _binary_treatment(df: pd.DataFrame, column: str) -> np.ndarray:
    """Coerce the treatment column to a strict 0/1 integer indicator.

    Accepts numeric (0/1) or boolean. Anything with values outside ``{0, 1}``
    (e.g. a multi-valued or continuous "treatment") is rejected -- these
    estimators are defined for a *binary* treatment only.

    Args:
        df: The input relation.
        column: Treatment column name.

    Returns:
        Integer array of 0/1 treatment indicators.

    Raises:
        CausalError: If the column is not binary 0/1 (or boolean).
    """
    series = df[column]
    if series.dtype == bool:
        return series.to_numpy().astype(int)
    coerced = pd.to_numeric(series, errors="coerce")
    if coerced.isna().any():
        raise CausalError(
            f"treatment column '{column}' must be binary 0/1, but contains "
            f"non-numeric values (dtype {series.dtype})"
        )
    values = np.asarray(coerced, dtype=float)
    unique = set(np.unique(values).tolist())
    if not unique <= {0.0, 1.0}:
        shown = ", ".join(str(v) for v in sorted(unique)[:6])
        raise CausalError(
            f"treatment column '{column}' must be binary 0/1, but has values "
            f"{{{shown}}}; recode it to 0 (control) / 1 (treated)"
        )
    return values.astype(int)


def _covariate_matrix(df: pd.DataFrame, covariates: list[str]) -> np.ndarray:
    """Build the float64 design matrix ``X`` from the covariate columns.

    Args:
        df: The input relation.
        covariates: Covariate column names (the relation columns that are
            neither treatment, outcome, nor id).

    Returns:
        An ``(n_rows, n_covariates)`` float64 array.

    Raises:
        CausalError: If any covariate column is non-numeric.
    """
    if not covariates:
        # No covariates means no adjustment is possible.
        raise CausalError(
            "no covariate columns found; the input relation must contain at "
            "least one covariate (confounder) column besides treatment/outcome"
        )
    cols = [_numeric(df, c, role="covariate") for c in covariates]
    return np.column_stack(cols)


def _fit_propensity(x: np.ndarray, treatment: np.ndarray, *, random_state: int) -> np.ndarray:
    """Fit a logistic propensity model and return clipped per-row e(X).

    Uses L2-regularized logistic regression (lbfgs), which keeps coefficients
    finite even under (near-)perfect separation -- so we never crash on a
    cleanly separated treatment. Scores are clipped into
    ``[_PROPENSITY_CLIP, 1 - _PROPENSITY_CLIP]`` to keep IPW weights finite.

    Args:
        x: Covariate design matrix.
        treatment: 0/1 treatment indicator.
        random_state: Seed for deterministic fitting.

    Returns:
        Per-row propensity scores ``e(X) = P(T=1 | X)`` in (0, 1).
    """
    # L2 regularization is the lbfgs default; leaving ``penalty`` unset (rather
    # than passing the now-deprecated penalty="l2") keeps coefficients finite
    # under (near-)perfect separation without a deprecation warning.
    model = LogisticRegression(
        C=1.0,
        solver="lbfgs",
        max_iter=1000,
        random_state=random_state,
    )
    model.fit(x, treatment)
    # predict_proba column order follows model.classes_; find the column for 1.
    classes = list(model.classes_)
    if 1 in classes:
        col = classes.index(1)
        scores = model.predict_proba(x)[:, col]
    else:
        # Degenerate: only one treatment class present -> constant score.
        scores = np.full(x.shape[0], float(classes[0]))
    return np.clip(scores, _PROPENSITY_CLIP, 1.0 - _PROPENSITY_CLIP)


def _check_both_arms(treatment: np.ndarray) -> None:
    """Require at least one treated and one control row.

    Raises:
        CausalError: If the treatment is constant (no contrast to estimate).
    """
    n_treated = int(treatment.sum())
    n_control = int(len(treatment) - n_treated)
    if n_treated == 0 or n_control == 0:
        raise CausalError(
            "treatment must contain both treated (1) and control (0) rows; "
            f"found {n_treated} treated and {n_control} control"
        )


def propensity_scores(
    df: pd.DataFrame,
    *,
    treatment: str,
    id: str,
    random_state: int = 0,
) -> dict[str, list]:
    """Fit the propensity model and emit per-row propensity scores.

    Every column other than ``treatment`` and ``id`` is treated as a covariate.
    The ``id`` column is carried through to the output and *excluded* from the
    covariates.

    Args:
        df: Input relation; must contain ``treatment`` and ``id`` columns plus
            one or more covariate columns.
        treatment: Name of the binary 0/1 treatment column.
        id: Name of the row-identifier column to pass through (excluded from
            the covariates).
        random_state: Seed for deterministic logistic-model fitting.

    Returns:
        Column block with keys ``id`` (the passthrough identifier as int64),
        ``propensity`` (float, the per-row ``e(X)`` in (0, 1)), and
        ``treatment`` (the 0/1 indicator as int), one entry per input row in
        input order.

    Raises:
        CausalError: On missing columns, empty input, non-binary treatment, or
            no covariate columns.
    """
    _require_columns(df, {"treatment": treatment, "id": id})
    if len(df) == 0:
        raise CausalError("propensity_scores requires a non-empty input relation")

    t = _binary_treatment(df, treatment)
    covariates = [c for c in df.columns if c not in (treatment, id)]
    x = _covariate_matrix(df, covariates)
    _check_both_arms(t)

    scores = _fit_propensity(x, t, random_state=random_state)
    ids = pd.to_numeric(df[id], errors="coerce")
    return {
        "id": [int(v) for v in ids],
        "propensity": [float(v) for v in scores],
        "treatment": [int(v) for v in t],
    }


def _ipw_ate(y: np.ndarray, t: np.ndarray, e: np.ndarray) -> tuple[float, float]:
    """Stabilized inverse-probability-weighted ATE and its standard error.

    Uses the Hajek (self-normalized) estimator: the weighted mean of treated
    outcomes minus the weighted mean of control outcomes, where weights are
    ``1/e`` for treated and ``1/(1-e)`` for control. The standard error comes
    from the influence-function / sandwich approximation.

    Returns:
        ``(estimate, std_error)``.
    """
    n = len(y)
    w1 = t / e
    w0 = (1.0 - t) / (1.0 - e)
    mu1 = np.sum(w1 * y) / np.sum(w1)
    mu0 = np.sum(w0 * y) / np.sum(w0)
    estimate = mu1 - mu0
    # Influence function for the Hajek IPW contrast.
    infl = (w1 * (y - mu1)) / (np.sum(w1) / n) - (w0 * (y - mu0)) / (np.sum(w0) / n)
    std_error = float(np.sqrt(np.sum(infl**2)) / n)
    return float(estimate), std_error


def _regression_adjustment_ate(
    y: np.ndarray, t: np.ndarray, x: np.ndarray
) -> tuple[float, float, np.ndarray, np.ndarray]:
    """G-formula ATE via an outcome regression including treatment + covariates.

    Fits ``mu(T, X)`` as a linear model on ``[T, X]`` and averages the
    individual contrast ``mu(1, X_i) - mu(0, X_i)``. Returns the estimate, a
    bootstrap-free analytic SE for the coefficient on ``T`` (which equals the
    contrast under a linear, no-interaction model), plus the per-row potential
    outcomes for reuse by AIPW.

    Returns:
        ``(estimate, std_error, mu1, mu0)``.
    """
    n = len(y)
    design = np.column_stack([t.astype(float), x])
    model = LinearRegression()
    model.fit(design, y)
    # Potential outcomes: set T=1 and T=0 for every row.
    x1 = np.column_stack([np.ones(n), x])
    x0 = np.column_stack([np.zeros(n), x])
    mu1 = model.predict(x1)
    mu0 = model.predict(x0)
    estimate = float(np.mean(mu1 - mu0))
    # SE of the treatment coefficient (= the contrast for a linear model w/o
    # treatment-covariate interactions), via the OLS covariance.
    full = np.column_stack([np.ones(n), design])
    resid = y - model.predict(design)
    dof = max(n - full.shape[1], 1)
    sigma2 = float(np.sum(resid**2) / dof)
    xtx_inv = np.linalg.pinv(full.T @ full)
    # Treatment coefficient sits at index 1 (after the intercept).
    var_t = sigma2 * xtx_inv[1, 1]
    std_error = float(np.sqrt(max(var_t, 0.0)))
    return estimate, std_error, mu1, mu0


def _aipw_ate(
    y: np.ndarray,
    t: np.ndarray,
    e: np.ndarray,
    mu1: np.ndarray,
    mu0: np.ndarray,
) -> tuple[float, float]:
    """Augmented IPW (doubly-robust) ATE with influence-function SE.

    Combines the outcome predictions ``mu1``/``mu0`` with the propensity ``e``.
    Consistent if *either* the outcome model or the propensity model is correct.
    The per-row influence function gives an honest standard error.

    Returns:
        ``(estimate, std_error)``.
    """
    n = len(y)
    psi = (mu1 - mu0) + t * (y - mu1) / e - (1.0 - t) * (y - mu0) / (1.0 - e)
    estimate = float(np.mean(psi))
    std_error = float(np.std(psi, ddof=1) / np.sqrt(n))
    return estimate, std_error


def ate(
    df: pd.DataFrame,
    *,
    treatment: str,
    outcome: str,
    random_state: int = 0,
    backend: str = "auto",
) -> dict[str, list]:
    """Average Treatment Effect (ATE) by three complementary estimators.

    Every column other than ``treatment`` and ``outcome`` is treated as a
    covariate (confounder) and adjusted for. Emits one row per method:
    ``ipw``, ``regression_adjustment``, and ``aipw`` (doubly-robust).

    Args:
        df: Input relation with ``treatment``, ``outcome`` and one or more
            numeric covariate columns.
        treatment: Name of the binary 0/1 treatment column.
        outcome: Name of the (numeric) outcome column.
        random_state: Seed for deterministic propensity-model fitting.
        backend: ``"auto"`` / ``"sklearn"`` use the in-house sklearn+statsmodels
            estimators (default). ``"dowhy"`` routes the AIPW estimate through
            the optional dowhy backend if it is installed.

    Returns:
        Column block with keys ``method`` (str), ``estimate``, ``std_error``,
        ``ci_lower``, ``ci_upper`` (all float) -- one row per method. The CI is
        the 95% Wald interval ``estimate ± 1.96 * std_error``.

    Raises:
        CausalError: On missing columns, empty input, non-binary treatment, a
            constant treatment, or no covariates.
    """
    _require_columns(df, {"treatment": treatment, "outcome": outcome})
    if len(df) == 0:
        raise CausalError("ate requires a non-empty input relation")

    t = _binary_treatment(df, treatment)
    y = _numeric(df, outcome, role="outcome")
    covariates = [c for c in df.columns if c not in (treatment, outcome)]
    x = _covariate_matrix(df, covariates)
    _check_both_arms(t)

    e = _fit_propensity(x, t, random_state=random_state)

    ipw_est, ipw_se = _ipw_ate(y, t, e)
    ra_est, ra_se, mu1, mu0 = _regression_adjustment_ate(y, t, x)
    aipw_est, aipw_se = _aipw_ate(y, t, e, mu1, mu0)

    if backend == "dowhy":
        dw = _dowhy_ate(df, treatment=treatment, outcome=outcome, covariates=covariates)
        if dw is not None:
            aipw_est, aipw_se = dw

    methods = ["ipw", "regression_adjustment", "aipw"]
    estimates = [ipw_est, ra_est, aipw_est]
    std_errors = [ipw_se, ra_se, aipw_se]
    z = float(stats.norm.ppf(0.975))
    ci_lower = [est - z * se for est, se in zip(estimates, std_errors, strict=True)]
    ci_upper = [est + z * se for est, se in zip(estimates, std_errors, strict=True)]
    return {
        "method": methods,
        "estimate": [float(v) for v in estimates],
        "std_error": [float(v) for v in std_errors],
        "ci_lower": [float(v) for v in ci_lower],
        "ci_upper": [float(v) for v in ci_upper],
    }


def att(
    df: pd.DataFrame,
    *,
    treatment: str,
    outcome: str,
    random_state: int = 0,
    n_boot: int = 200,
) -> dict[str, list]:
    """Average Treatment effect on the Treated (ATT) via IPW-ATT weighting.

    Estimates ``E[Y(1) - Y(0) | T=1]``. Treated rows contribute their observed
    outcome; control rows are reweighted by the odds ``e(X)/(1-e(X))`` so the
    reweighted control population matches the treated covariate distribution.
    The standard error is a nonparametric bootstrap over rows.

    Every column other than ``treatment`` and ``outcome`` is a covariate.

    Args:
        df: Input relation with ``treatment``, ``outcome`` and covariates.
        treatment: Name of the binary 0/1 treatment column.
        outcome: Name of the (numeric) outcome column.
        random_state: Seed for deterministic propensity fitting and bootstrap.
        n_boot: Number of bootstrap resamples for the standard error.

    Returns:
        Single-row column block with keys ``estimate`` (float) and
        ``std_error`` (float).

    Raises:
        CausalError: On missing columns, empty input, non-binary/constant
            treatment, or no covariates.
    """
    _require_columns(df, {"treatment": treatment, "outcome": outcome})
    if len(df) == 0:
        raise CausalError("att requires a non-empty input relation")

    t = _binary_treatment(df, treatment)
    y = _numeric(df, outcome, role="outcome")
    covariates = [c for c in df.columns if c not in (treatment, outcome)]
    x = _covariate_matrix(df, covariates)
    _check_both_arms(t)

    def _point(y_s: np.ndarray, t_s: np.ndarray, x_s: np.ndarray) -> float | None:
        if t_s.sum() == 0 or (1 - t_s).sum() == 0:
            return None
        e_s = _fit_propensity(x_s, t_s, random_state=random_state)
        treated_mean = float(np.mean(y_s[t_s == 1]))
        w = e_s / (1.0 - e_s)
        w_ctrl = w[t_s == 0]
        y_ctrl = y_s[t_s == 0]
        if np.sum(w_ctrl) == 0:
            return None
        control_mean = float(np.sum(w_ctrl * y_ctrl) / np.sum(w_ctrl))
        return treated_mean - control_mean

    estimate = _point(y, t, x)
    if estimate is None:
        raise CausalError("att could not be estimated (degenerate weighting)")

    rng = np.random.default_rng(random_state)
    n = len(y)
    boots: list[float] = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        val = _point(y[idx], t[idx], x[idx])
        if val is not None:
            boots.append(val)
    std_error = float(np.std(boots, ddof=1)) if len(boots) > 1 else 0.0
    return {"estimate": [float(estimate)], "std_error": [std_error]}


def _dowhy_ate(
    df: pd.DataFrame,
    *,
    treatment: str,
    outcome: str,
    covariates: list[str],
) -> tuple[float, float] | None:
    """Optional dowhy backend for the doubly-robust ATE.

    dowhy (Microsoft, MIT) is an *optional* dependency -- imported lazily here
    so the default sklearn/statsmodels path never pays its (slow) import cost.
    Returns ``None`` if dowhy is not installed, so the caller falls back to the
    in-house AIPW estimate.

    Returns:
        ``(estimate, std_error)`` or ``None`` if dowhy is unavailable.
    """
    try:
        from dowhy import CausalModel  # noqa: PLC0415  (lazy optional import)
    except Exception:
        return None

    common_causes = list(covariates)
    model = CausalModel(
        data=df,
        treatment=treatment,
        outcome=outcome,
        common_causes=common_causes,
    )
    identified = model.identify_effect(proceed_when_unidentifiable=True)
    estimate = model.estimate_effect(
        identified,
        method_name="backdoor.propensity_score_weighting",
    )
    return float(estimate.value), 0.0
