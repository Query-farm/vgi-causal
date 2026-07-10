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
from vgi.catalog import Catalog, Schema, View

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
    'This VGI extension turns causal questions like *"did the treatment actually work, '
    'after controlling for who was likely to get it?"* into a single SQL query.\n\n'
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
    "Runnable, catalog-qualified demonstrations live in each object's example queries "
    "(and the schema's `vgi.example_queries`), so they are executed and coverage-checked "
    "rather than pasted inline here.\n\n"
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
    "## Causal treatment-effect estimation\n\n"
    "Estimate the causal effect of a **binary treatment** on an outcome from observational "
    "cohort data, adjusting for measured confounders — directly in DuckDB SQL, with no export "
    "to a notebook.\n\n"
    "### Key concepts\n\n"
    "- **Confounding** biases a naive difference of group means whenever the factors that "
    "drive treatment also drive the outcome; adjustment corrects it.\n"
    "- Estimators draw on the standard observational toolkit — inverse-probability weighting "
    "(IPW), regression adjustment (the g-formula), and doubly-robust augmented IPW (AIPW).\n"
    "- Each call consumes a whole `(SELECT ...)` cohort relation plus named role arguments; "
    "every column not named by a role is treated as an adjusted-for covariate, and treatment "
    "must be binary `0/1`.\n\n"
    "### When to use\n\n"
    "Reach for these estimators for treatment-effect, impact, and intervention analysis where "
    "the data already lives. Estimates are causal only under unconfoundedness, overlap, and "
    "SUTVA.\n\n"
    "Backed by scikit-learn and statsmodels, with deterministic, reproducible estimates."
)

# VGI413: an ordered category registry for the schema; every function declares a
# ``vgi.category`` naming one of these (see vgi_causal.tables).
_SCHEMA_CATEGORIES = json.dumps(
    [
        {
            "name": "Treatment Effects",
            "description": (
                "Aggregate causal effect estimators: the average treatment effect (ATE) over the "
                "whole cohort and the average treatment effect on the treated (ATT)."
            ),
        },
        {
            "name": "Propensity Diagnostics",
            "description": (
                "Per-subject propensity models e(X)=P(T=1|X) for overlap/positivity checks, "
                "matching, and inverse-probability weighting."
            ),
        },
        {
            "name": "Method Reference",
            "description": (
                "Browsable reference tables describing the estimators the worker exposes, so an "
                "agent can discover which methods each function emits before calling it."
            ),
        },
    ]
)

# VGI152: a fixed analyst-task suite so `vgi-lint simulate` can measure how well
# an agent actually uses this worker. Grader-only (never shown to the actor); each
# reference_sql runs against the seeded, deterministic COHORT_CTE, so results are
# reproducible run-to-run. ignore_column_names compares by VALUES.
_COHORT_PROMPT = (
    "You have an observational cohort defined by this DuckDB CTE (paste it verbatim at the "
    "top of your query):\n\n```sql\n" + COHORT_CTE.strip() + "\n```\n\n"
    "`t` is a binary 0/1 treatment indicator, `y` is the numeric outcome, and `x` is a "
    "covariate/confounder that influences both."
)

_AGENT_TEST_TASKS = json.dumps(
    [
        {
            "name": "ate_doubly_robust",
            "prompt": (
                _COHORT_PROMPT + " Estimate the average treatment effect of `t` on `y` using the doubly-robust "
                "(AIPW) estimator, adjusting for `x`. Return the point estimate rounded to 2 "
                "decimal places."
            ),
            "success_criteria": ("Calls causal.ate over the cohort and returns the AIPW row's estimate (~5.0)."),
            "reference_sql": (
                COHORT_CTE + "SELECT round(estimate, 2) AS estimate "
                "FROM causal.main.ate((SELECT t, y, x FROM cohort), "
                "treatment := 't', outcome := 'y') WHERE method = 'aipw'"
            ),
            "ignore_column_names": True,
        },
        {
            "name": "att_effect_on_treated",
            "prompt": (
                _COHORT_PROMPT + " Estimate the average treatment effect on the treated (ATT) of `t` on `y`, "
                "adjusting for `x`. Return the estimate rounded to 2 decimal places."
            ),
            "success_criteria": ("Calls causal.att over the cohort and returns the ATT estimate (~5.0)."),
            "reference_sql": (
                COHORT_CTE + "SELECT round(estimate, 2) AS att "
                "FROM causal.main.att((SELECT t, y, x FROM cohort), "
                "treatment := 't', outcome := 'y')"
            ),
            "ignore_column_names": True,
        },
        {
            "name": "propensity_gt_half_count",
            "prompt": (
                _COHORT_PROMPT + " Using the worker's propensity-score estimator, fit each subject's "
                "propensity to be treated, then return a single number: the count of subjects whose "
                "fitted propensity score is greater than 0.5."
            ),
            "success_criteria": ("Calls causal.propensity_scores and returns count(*) of rows with propensity > 0.5."),
            "reference_sql": (
                COHORT_CTE + "SELECT count(*) AS n "
                "FROM causal.main.propensity_scores((SELECT id, t, x FROM cohort), "
                "treatment := 't', id := 'id') WHERE propensity > 0.5"
            ),
            "ignore_column_names": True,
        },
        {
            "name": "methods_emitted_by_ate",
            "prompt": (
                "The worker exposes a browsable reference table that lists its estimator methods and "
                "which function emits each one. Using it, list the estimator method names produced by "
                "the `ate` function, one per row, ordered alphabetically."
            ),
            "success_criteria": (
                "Queries the methods reference table filtered to the ate function and returns its "
                "three method names (aipw, ipw, regression_adjustment)."
            ),
            "reference_sql": ("SELECT method FROM causal.main.methods WHERE function_name = 'ate' ORDER BY method"),
            "ignore_column_names": True,
        },
    ]
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
    # VGI152: fixed analyst-task suite for `vgi-lint simulate` (grader-only).
    "vgi.agent_test_tasks": _AGENT_TEST_TASKS,
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
    # VGI413: ordered category registry; each function names one via vgi.category.
    "vgi.categories": _SCHEMA_CATEGORIES,
    # VGI506 representative, self-contained example queries for the schema.
    "vgi.example_queries": _SCHEMA_EXAMPLE_QUERIES,
}

# VGI146: a genuine browsable table so an agent can inspect the worker's surface
# without first guessing a table function's arguments. It is a curated registry
# of the estimators each function emits, expressed as an inline VALUES relation so
# it scans with no external data or credentials. Keep this in sync with the
# `method` values produced by vgi_causal.causal.
_METHODS_VIEW_DEFINITION = """
SELECT * FROM (VALUES
    ('ipw',                   'ate',               'ATE',           FALSE,
     'Inverse-probability weighting (Hajek self-normalized contrast); reweights each subject by 1/e or 1/(1-e).'),
    ('regression_adjustment', 'ate',               'ATE',           FALSE,
     'g-formula: an OLS outcome model on the treatment and covariates, averaging predicted potential outcomes.'),
    ('aipw',                  'ate',               'ATE',           TRUE,
     'Doubly-robust augmented IPW; consistent if EITHER the propensity or the outcome model is correct.'),
    ('ipw_att',               'att',               'ATT',           FALSE,
     'IPW-ATT weighting: reweights controls by e/(1-e) to match the treated covariate distribution.'),
    ('propensity',            'propensity_scores', 'e(X)=P(T=1|X)', FALSE,
     'Regularized logistic propensity model; per-row scores clipped to (0,1) for overlap diagnostics.')
) AS t(method, function_name, estimand, doubly_robust, description)
"""

_METHODS_VIEW_EXAMPLES = json.dumps(
    [
        {
            "description": "Which estimators does the ate function emit, and which are doubly-robust?",
            "sql": (
                "SELECT method, doubly_robust FROM causal.main.methods WHERE function_name = 'ate' ORDER BY method"
            ),
        },
        {
            "description": "List every estimator with its estimand and the function that produces it.",
            "sql": "SELECT function_name, method, estimand FROM causal.main.methods ORDER BY function_name, method",
        },
    ]
)

_METHODS_VIEW = View(
    name="methods",
    definition=_METHODS_VIEW_DEFINITION,
    comment="Reference registry of the estimator methods each causal function emits",
    column_comments={
        "method": "Estimator identifier as it appears in output (e.g. the `method` column of `ate`).",
        "function_name": "The catalog function that emits this method: ate, att, or propensity_scores.",
        "estimand": "The causal quantity estimated: ATE, ATT, or the per-row propensity e(X).",
        "doubly_robust": "TRUE when the estimator is doubly-robust (consistent if either nuisance model is right).",
        "description": "One-line summary of the estimator and how it adjusts for confounding.",
    },
    tags={
        "vgi.title": "Estimator Method Reference",
        "vgi.doc_llm": (
            "A browsable registry of the estimator methods this worker exposes. One row per method with "
            "the function that emits it (ate / att / propensity_scores), the estimand it targets (ATE, "
            "ATT, or the per-row propensity e(X)), whether it is doubly-robust, and a one-line summary. "
            "Query it to discover which methods `ate` returns, or which estimators are doubly-robust, "
            "before calling a table function."
        ),
        "vgi.doc_md": (
            "# Estimator Method Reference\n\n"
            "A small, self-contained registry of the estimator methods the `causal` worker exposes — one "
            "row per method. Use it to discover the worker's surface before calling a table function: "
            "which methods `ate` emits, which estimator is doubly-robust, and what each one estimates.\n\n"
            "## Columns\n\n"
            "- `method` — the estimator id (matches the `method` column emitted by `ate`).\n"
            "- `function_name` — the function that produces the method (`ate`, `att`, or "
            "`propensity_scores`).\n"
            "- `estimand` — the causal quantity: `ATE`, `ATT`, or the per-row propensity `e(X)`.\n"
            "- `doubly_robust` — `TRUE` for AIPW (consistent if either nuisance model is correct).\n"
            "- `description` — a one-line summary of the estimator.\n\n"
            "The registry is backed by an inline `VALUES` relation, so it scans with no external data or "
            "credentials."
        ),
        "vgi.keywords": json.dumps(
            [
                "methods",
                "estimators",
                "reference",
                "registry",
                "ate",
                "att",
                "propensity",
                "ipw",
                "aipw",
                "regression adjustment",
                "doubly robust",
                "estimand",
            ]
        ),
        "vgi.category": "Method Reference",
        "vgi.example_queries": _METHODS_VIEW_EXAMPLES,
        # VGI123 classifying tags use BARE keys (NOT vgi.-namespaced) for faceting.
        "domain": "statistics",
        "topic": "treatment-effect-estimation",
    },
)

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
            views=[_METHODS_VIEW],
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
