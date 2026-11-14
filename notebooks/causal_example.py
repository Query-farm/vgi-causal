"""Generate exogenous simulation draws and present worker-computed causal results."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from vgi_notebooks import finish


def configure(adjust, subjects, seed):
    """Validate the three visible controls and create only exogenous draws."""
    if type(adjust) is not bool:
        raise ValueError("ADJUST_FOR_X2 must be True or False.")
    if type(subjects) is not int or not 500 <= subjects <= 10000:
        raise ValueError("N_SUBJECTS must be an integer from 500 to 10,000.")
    if type(seed) is not int or not 0 <= seed <= 1000000:
        raise ValueError("SEED must be an integer from 0 to 1,000,000.")
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "id": np.arange(subjects, dtype=np.int64),
            "x1": rng.normal(size=subjects),
            "x2": rng.normal(size=subjects),
            "assignment_draw": rng.uniform(size=subjects),
            "noise": rng.normal(size=subjects),
        }
    )


def plot_balance(balance_frame, model_label, subjects, seed):
    """Compare both covariate mean differences using a fixed pooled denominator."""
    fig, ax = plt.subplots(figsize=(11.5, 6.5))
    y = np.arange(len(balance_frame))
    for i, row in balance_frame.iterrows():
        ax.plot([row.unweighted_smd, row.weighted_smd], [i, i], color="#b8c3bd", linewidth=3)
    ax.scatter(balance_frame.unweighted_smd, y, s=90, color="#c27548", label="Unweighted", zorder=3)
    ax.scatter(balance_frame.weighted_smd, y, s=90, color="#2b8067", marker="D", label="IPW weighted", zorder=3)
    ax.axvline(0, color="#52675a", linewidth=1)
    ax.axvspan(-0.1, 0.1, color="#e5eee8", alpha=0.7, label="±0.1 reference band (diagnostic only)")
    limit = max(0.3, float(np.abs(balance_frame[["unweighted_smd", "weighted_smd"]]).max().max()) * 1.25)
    ax.set(
        xlim=(-limit, limit),
        ylim=(1.7, -0.7),
        yticks=y,
        yticklabels=balance_frame.covariate,
        xlabel="Treated − control standardized mean difference (fixed pooled SD)",
    )
    fig.legend(*ax.get_legend_handles_labels(), loc="upper left", bbox_to_anchor=(0.07, 0.82), ncol=2, fontsize=9)
    finish(
        fig,
        "Adjustment depends on what the model observes",
        f"Model: {model_label} · x2 stays visible to the simulation audit",
        f"Synthetic cohort · n={subjects:,} · seed={seed} · released VGI Causal v0.4.0\n"
        "Mean balance is a diagnostic; it cannot prove absence of unmeasured confounding.",
        name="01-balance",
    )


def plot_effects(results, model_label, subjects, seed):
    """Preserve worker estimates and intervals alongside the known synthetic effect."""
    labels = {
        "unadjusted": "Unadjusted",
        "ipw": "IPW",
        "regression_adjustment": "Regression adjustment",
        "aipw": "AIPW",
    }
    fig, ax = plt.subplots(figsize=(11.5, 6.5))
    for i, row in results.iterrows():
        color = "#c27548" if row.method == "unadjusted" else "#2b8067"
        ax.errorbar(
            row.estimate,
            i,
            xerr=[[row.estimate - row.ci_lower], [row.ci_upper - row.estimate]],
            fmt="o",
            color=color,
            markersize=8,
            capsize=5,
            linewidth=2,
        )
        ax.annotate(
            f"{row.estimate:.2f}",
            (row.estimate, i),
            xytext=(0, 12),
            textcoords="offset points",
            ha="center",
            fontsize=10,
        )
    ax.axvline(3.0, color="#4a6f9b", linestyle="--", linewidth=2, label="Planted effect = 3")
    left = min(3.0, results.ci_lower.min())
    right = max(3.0, results.ci_upper.max())
    pad = max(0.2, (right - left) * 0.14)
    ax.set(
        xlim=(left - pad, right + pad),
        ylim=(len(results) - 0.5, -0.65),
        yticks=np.arange(len(results)),
        yticklabels=[labels[x] for x in results.method],
        xlabel="Estimated treated − untreated outcome difference (arbitrary units)",
    )
    fig.legend(*ax.get_legend_handles_labels(), loc="upper left", bbox_to_anchor=(0.07, 0.82), fontsize=9)
    finish(
        fig,
        "Compare model estimates with the planted effect",
        f"Model: {model_label} · reported 95% intervals exclude omitted-confounder bias",
        f"Synthetic cohort · n={subjects:,} · seed={seed} · released VGI Causal v0.4.0\n"
        "Constant planted effect: +3 outcome units per subject. No measured treatment or population.",
        name="02-effects",
    )
