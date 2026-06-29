"""VGI worker exposing causal treatment-effect estimation to DuckDB/SQL.

Assembles the causal table functions in ``vgi_causal`` into a single ``causal``
catalog and provides the process entry point. The repo-root ``causal_worker.py``
is a thin shim over this module for ``uv run``; installed users get the
``vgi-causal`` console script, which calls ``main`` here.

    ATTACH 'causal' (TYPE vgi, LOCATION 'uv run causal_worker.py');
    SELECT * FROM causal.ate((SELECT * FROM cohort), treatment := 't', outcome := 'y');
"""

from __future__ import annotations

import json
import sys

from vgi import Worker
from vgi.catalog import Catalog, Schema

from vgi_causal.meta import COHORT_CTE
from vgi_causal.tables import TABLE_FUNCTIONS

_FUNCTIONS: list[type] = [*TABLE_FUNCTIONS]

_CATALOG_DESCRIPTION_LLM = (
    "Estimate the causal effect of a binary treatment on an outcome from observational "
    "cohort data, adjusting for confounders. Answers questions like 'what is the average "
    "treatment effect of this intervention?' (ate: IPW, regression adjustment, and "
    "doubly-robust AIPW), 'what is the effect on those who actually got treated?' (att, "
    "IPW-ATT), and 'how likely was each subject to be treated given their covariates?' "
    "(propensity_scores). Each function takes the whole cohort as a (SELECT ...) relation "
    "plus named role args (treatment, outcome, id); every other column is treated as a "
    "covariate and adjusted for. Treatment must be binary 0/1. Estimates are causal only "
    "under unconfoundedness, overlap, and SUTVA. Use for treatment-effect / impact / "
    "intervention analysis in SQL; not for forecasting or general regression."
)

_CATALOG_DESCRIPTION_MD = (
    "# Causal Inference in SQL: Treatment Effects, Propensity & Doubly-Robust Estimation\n\n"
    "![DoWhy logo](https://raw.githubusercontent.com/py-why/dowhy/main/docs/source/_static/dowhy-logo-small.png)\n\n"
    "**Estimate the causal effect of an intervention directly in DuckDB SQL** — average "
    "treatment effect (ATE), effect on the treated (ATT), and per-subject propensity "
    "scores — from observational cohort data, with confounders adjusted for automatically. "
    "This VGI extension turns causal questions like *\"did the treatment actually work, "
    "after controlling for who was likely to get it?\"* into a single SQL query.\n\n"
    "Causal inference goes beyond correlation and naive group means: a simple difference "
    "between treated and untreated subjects is biased whenever the same factors that drive "
    "treatment also drive the outcome (confounding). The `causal` catalog corrects that "
    "bias using the standard toolkit of observational causal inference — inverse "
    "probability weighting (IPW), regression adjustment (the g-formula), and "
    "doubly-robust augmented IPW (AIPW) — so analysts, data scientists, and "
    "epidemiologists can run treatment-effect, impact, and intervention analysis where "
    "their data already lives, without exporting to a notebook.\n\n"
    "Under the hood the estimators are powered by [scikit-learn](https://scikit-learn.org/stable/) "
    "([source](https://github.com/scikit-learn/scikit-learn)) for the propensity and "
    "outcome models and [statsmodels](https://www.statsmodels.org/stable/index.html) "
    "([source](https://github.com/statsmodels/statsmodels)) for the regression and "
    "inference machinery, with an optional doubly-robust backend via "
    "[DoWhy](https://www.pywhy.org/dowhy/) ([source](https://github.com/py-why/dowhy)). "
    "Propensities come from an L2-regularized logistic model with score clipping that keeps "
    "near-perfect separation from blowing up the weights, and standard errors come from "
    "influence functions (IPW/AIPW), OLS (regression adjustment), or a seeded bootstrap "
    "(ATT) — every estimate is deterministic and reproducible.\n\n"
    "## Functions\n\n"
    "Each function consumes a whole cohort as a `(SELECT ...)` relation plus named role "
    "arguments; **every column not named by a role is treated as a covariate/confounder "
    "and adjusted for**, and treatment must be binary `0/1`.\n\n"
    "- `ate(rel, treatment, outcome)` — Average Treatment Effect `E[Y(1)-Y(0)]`, returned "
    "one row per method: IPW, regression adjustment, and doubly-robust AIPW, each with a "
    "standard error and 95% confidence interval.\n"
    "- `att(rel, treatment, outcome)` — Average Treatment effect on the Treated "
    "`E[Y(1)-Y(0)|T=1]` via IPW-ATT weighting, with a bootstrap standard error.\n"
    "- `propensity_scores(rel, treatment, id)` — per-row fitted propensity "
    "`e(X)=P(T=1|X)` from the logistic model, keyed by your `id` column.\n\n"
    "```sql\n"
    "SELECT method, round(estimate, 2) AS estimate\n"
    "FROM causal.ate((SELECT t, y, x FROM cohort), treatment := 't', outcome := 'y')\n"
    "ORDER BY method;\n"
    "```\n\n"
    "Estimates are causal only under the usual assumptions — **unconfoundedness** (no "
    "unmeasured confounders given the supplied covariates), **overlap** "
    "(`0 < e(X) < 1`), and **SUTVA**. Choosing the covariate set is your modeling "
    "decision; the worker estimates, it does not assume the assumptions hold."
)

_SCHEMA_DESCRIPTION_LLM = (
    "Causal treatment-effect estimators over a buffered cohort relation: ate (average "
    "treatment effect via IPW / regression adjustment / doubly-robust AIPW), att (effect "
    "on the treated, IPW-ATT), and propensity_scores (per-row P(T=1|X)). Treatment is "
    "binary 0/1; every non-role column is an adjusted-for covariate."
)

_SCHEMA_DESCRIPTION_MD = (
    "Causal treatment-effect estimators (`ate`, `att`, `propensity_scores`) over a whole "
    "cohort relation, adjusting for confounders. Backed by scikit-learn and statsmodels."
)

_CATALOG_TAGS: dict[str, str] = {
    "vgi.title": "Causal Treatment-Effect Estimation",
    "vgi.keywords": json.dumps(
        [
            "causal inference",
            "treatment effect",
            "ate",
            "average treatment effect",
            "att",
            "propensity score",
            "ipw",
            "inverse probability weighting",
            "regression adjustment",
            "g-formula",
            "aipw",
            "doubly robust",
            "confounding",
            "intervention",
            "impact analysis",
            "observational data",
            "cohort",
        ]
    ),
    "vgi.doc_llm": _CATALOG_DESCRIPTION_LLM,
    "vgi.doc_md": _CATALOG_DESCRIPTION_MD,
    "vgi.author": "Query.Farm",
    "vgi.copyright": "Copyright 2026 Query Farm LLC - https://query.farm",
    "vgi.license": "MIT",
    "vgi.support_contact": "https://github.com/Query-farm/vgi-causal/issues",
    "vgi.support_policy_url": "https://github.com/Query-farm/vgi-causal/blob/main/README.md",
}

_SCHEMA_EXAMPLE_QUERIES = (
    COHORT_CTE + "SELECT method, round(estimate, 2) AS estimate "
    "FROM causal.main.ate((SELECT t, y, x FROM cohort), treatment := 't', outcome := 'y') "
    "ORDER BY method;\n" + COHORT_CTE + "SELECT round(estimate, 2) AS att "
    "FROM causal.main.att((SELECT t, y, x FROM cohort), treatment := 't', outcome := 'y');\n"
    + COHORT_CTE
    + "SELECT * FROM causal.main.propensity_scores((SELECT id, t, x FROM cohort), "
    "treatment := 't', id := 'id') ORDER BY id LIMIT 5;"
)

_SCHEMA_TAGS: dict[str, str] = {
    "vgi.title": "Causal Estimators (main)",
    "vgi.keywords": json.dumps(
        [
            "causal",
            "ate",
            "att",
            "propensity_scores",
            "treatment effect",
            "ipw",
            "regression adjustment",
            "aipw",
            "doubly robust",
            "propensity",
            "confounding",
        ]
    ),
    # VGI123 classifying tags use BARE keys (NOT vgi.-namespaced) for faceting.
    "domain": "statistics",
    "category": "causal-inference",
    "topic": "treatment-effect-estimation",
    # VGI139: source_url belongs only on the catalog object, not per-schema.
    "vgi.doc_llm": _SCHEMA_DESCRIPTION_LLM,
    "vgi.doc_md": _SCHEMA_DESCRIPTION_MD,
    # VGI506 representative, self-contained example queries for the schema.
    "vgi.example_queries": _SCHEMA_EXAMPLE_QUERIES,
}

_CAUSAL_CATALOG = Catalog(
    name="causal",
    default_schema="main",
    comment="Causal treatment-effect estimation (ATE, ATT, propensity scores) for SQL",
    source_url="https://github.com/Query-farm/vgi-causal",
    tags=_CATALOG_TAGS,
    schemas=[
        Schema(
            name="main",
            comment="Causal estimators: ate (IPW/RA/AIPW), att (IPW-ATT), and per-row propensity_scores",
            tags=_SCHEMA_TAGS,
            functions=list(_FUNCTIONS),
        ),
    ],
)


class CausalWorker(Worker):
    """Worker process hosting the ``causal`` catalog."""

    catalog = _CAUSAL_CATALOG


def main() -> None:
    """Run the worker (stdio by default; pass ``--http`` for the HTTP server)."""
    CausalWorker.main()


def main_http() -> None:
    """Run the worker over HTTP (injects ``--http`` into the worker CLI)."""
    argv = sys.argv[1:]
    if "--http" not in argv:
        argv = ["--http", *argv]
    sys.argv = [sys.argv[0], *argv]
    CausalWorker.main()
