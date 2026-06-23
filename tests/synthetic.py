"""Synthetic confounded data generator with a KNOWN true treatment effect.

The data-generating process plants a homogeneous treatment effect ``tau`` and
introduces confounding by making treatment assignment depend on the covariates:

    e(X)  = sigmoid(a0 + a1*x1 + a2*x2)     # propensity depends on covariates
    T     ~ Bernoulli(e(X))                 # treated more where x1/x2 are high
    Y     = tau*T + f(x1, x2) + noise       # outcome ALSO depends on x1/x2

Because the covariates drive both T and Y, a naive difference of means
(``E[Y|T=1] - E[Y|T=0]``) is biased away from ``tau``. A covariate-adjusted
estimator (IPW / regression adjustment / AIPW) recovers ``tau`` -- which is the
whole point of the tests.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRUE_TAU = 3.0


def make_confounded(
    n: int = 4000,
    tau: float = TRUE_TAU,
    seed: int = 7,
) -> pd.DataFrame:
    """Generate a confounded cohort with a known treatment effect ``tau``.

    Args:
        n: Number of rows.
        tau: The planted (homogeneous) treatment effect.
        seed: RNG seed for reproducibility.

    Returns:
        A DataFrame with columns ``id``, ``x1``, ``x2`` (covariates/confounders),
        ``t`` (binary 0/1 treatment), and ``y`` (outcome).
    """
    rng = np.random.default_rng(seed)
    x1 = rng.normal(0.0, 1.0, size=n)
    x2 = rng.normal(0.0, 1.0, size=n)

    # Treatment propensity depends on the covariates -> confounding.
    logit = 0.0 + 1.2 * x1 - 0.8 * x2
    e = 1.0 / (1.0 + np.exp(-logit))
    t = (rng.uniform(size=n) < e).astype(int)

    # Outcome depends on the SAME covariates plus the treatment effect tau.
    y = tau * t + 2.0 * x1 + 1.5 * x2 + rng.normal(0.0, 1.0, size=n)

    return pd.DataFrame(
        {
            "id": np.arange(n, dtype=np.int64),
            "x1": x1,
            "x2": x2,
            "t": t,
            "y": y,
        }
    )


def naive_difference_of_means(df: pd.DataFrame) -> float:
    """Unadjusted contrast E[Y|T=1] - E[Y|T=0] (the BIASED estimator)."""
    return float(df.loc[df.t == 1, "y"].mean() - df.loc[df.t == 0, "y"].mean())
