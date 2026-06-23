"""Pure-logic tests for the causal estimators against a KNOWN true effect.

The synthetic generator plants a homogeneous effect ``tau`` with confounding
(treatment assignment depends on the covariates). We assert that:

* the naive difference of means is biased (off from ``tau``),
* every covariate-adjusted estimator (ipw, regression_adjustment, aipw) recovers
  ``tau`` within tolerance,
* propensity scores are valid probabilities and higher for treated-likely units,
* ATT recovers the (here homogeneous) effect,
* edge cases raise clear errors instead of crashing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from vgi_causal import causal

from .synthetic import TRUE_TAU, make_confounded, naive_difference_of_means


def test_naive_is_biased() -> None:
    df = make_confounded()
    naive = naive_difference_of_means(df)
    # Confounding pushes the naive estimate well away from the truth.
    assert abs(naive - TRUE_TAU) > 0.5


def test_ate_recovers_tau_all_methods() -> None:
    df = make_confounded()
    res = causal.ate(df, treatment="t", outcome="y")
    by_method = dict(zip(res["method"], res["estimate"], strict=True))
    assert set(by_method) == {"ipw", "regression_adjustment", "aipw"}
    for method, est in by_method.items():
        assert abs(est - TRUE_TAU) < 0.4, f"{method} estimate {est} far from {TRUE_TAU}"


def test_adjusted_beats_naive() -> None:
    df = make_confounded()
    naive_err = abs(naive_difference_of_means(df) - TRUE_TAU)
    res = causal.ate(df, treatment="t", outcome="y")
    aipw = dict(zip(res["method"], res["estimate"], strict=True))["aipw"]
    assert abs(aipw - TRUE_TAU) < naive_err  # adjustment corrects the confounding


def test_ate_ci_covers_truth() -> None:
    df = make_confounded()
    res = causal.ate(df, treatment="t", outcome="y")
    for lo, hi, se in zip(res["ci_lower"], res["ci_upper"], res["std_error"], strict=True):
        assert se > 0
        assert lo <= TRUE_TAU <= hi


def test_propensity_scores_valid_and_ordered() -> None:
    df = make_confounded()
    res = causal.propensity_scores(df, treatment="t", id="id")
    p = np.asarray(res["propensity"])
    assert ((p > 0.0) & (p < 1.0)).all()
    assert len(p) == len(df)
    # Treated-likely units (high x1) should get higher mean propensity than
    # treated-unlikely (low x1). Compare the top vs bottom x1 deciles.
    order = np.argsort(df["x1"].to_numpy())
    k = len(df) // 10
    low = p[order[:k]].mean()
    high = p[order[-k:]].mean()
    assert high > low


def test_propensity_excludes_id_from_covariates() -> None:
    # If id were used as a covariate it would leak; result must still be valid.
    df = make_confounded(n=500)
    res = causal.propensity_scores(df, treatment="t", id="id")
    assert res["id"][:3] == [0, 1, 2]
    assert res["treatment"] == df["t"].tolist()


def test_att_recovers_tau() -> None:
    df = make_confounded()
    res = causal.att(df, treatment="t", outcome="y", n_boot=80)
    est = res["estimate"][0]
    assert abs(est - TRUE_TAU) < 0.5
    assert res["std_error"][0] > 0


# --- edges ---------------------------------------------------------------


def test_missing_column_raises() -> None:
    df = make_confounded(n=50)
    with pytest.raises(causal.CausalError, match="missing required column"):
        causal.ate(df, treatment="nope", outcome="y")


def test_non_binary_treatment_raises() -> None:
    df = make_confounded(n=50).copy()
    df["t"] = np.arange(len(df)) % 3  # values {0,1,2}
    with pytest.raises(causal.CausalError, match="binary 0/1"):
        causal.ate(df, treatment="t", outcome="y")


def test_empty_relation_raises() -> None:
    empty = pd.DataFrame({"t": [], "y": [], "x1": []})
    with pytest.raises(causal.CausalError, match="non-empty"):
        causal.ate(empty, treatment="t", outcome="y")


def test_no_covariates_raises() -> None:
    df = pd.DataFrame({"t": [0, 1, 0, 1], "y": [1.0, 2.0, 1.5, 2.5]})
    with pytest.raises(causal.CausalError, match="covariate"):
        causal.ate(df, treatment="t", outcome="y")


def test_constant_treatment_raises() -> None:
    df = make_confounded(n=50).copy()
    df["t"] = 1  # all treated
    with pytest.raises(causal.CausalError, match="both treated"):
        causal.ate(df, treatment="t", outcome="y")


def test_perfect_separation_does_not_crash() -> None:
    # Treatment perfectly determined by a covariate -> separation. Regularized
    # logistic + clipping must keep this finite instead of crashing.
    n = 200
    rng = np.random.default_rng(0)
    x1 = rng.normal(size=n)
    t = (x1 > 0).astype(int)  # perfectly separable
    y = 3.0 * t + x1 + rng.normal(scale=0.1, size=n)
    df = pd.DataFrame({"t": t, "y": y, "x1": x1})
    res = causal.ate(df, treatment="t", outcome="y")
    for est in res["estimate"]:
        assert np.isfinite(est)
    ps = causal.propensity_scores(pd.DataFrame({"id": np.arange(n), "t": t, "x1": x1}), treatment="t", id="id")
    assert all(0.0 < p < 1.0 for p in ps["propensity"])
