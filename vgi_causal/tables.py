"""Causal-inference table functions for DuckDB via VGI.

Each function consumes a *whole* input relation -- passed as a ``(SELECT ...)``
subquery (positional ``Arg(0)``) -- and the column roles as NAMED string args
(``treatment := 't'``, ``outcome := 'y'``, ``id := 'id'``). Because a causal
estimate fits a model over and averages across every row, these are buffering
(Sink+Source) functions: they sink all input batches, then run the estimator
once in finalize.

    SELECT * FROM causal.ate((SELECT * FROM cohort), treatment := 't', outcome := 'y');
    SELECT * FROM causal.propensity_scores((SELECT id, t, x1, x2 FROM cohort),
                                           treatment := 't', id := 'id');
    SELECT * FROM causal.att((SELECT * FROM cohort), treatment := 't', outcome := 'y');

Treatment must be **binary 0/1**. Every column other than the named roles
(treatment/outcome, or treatment/id for propensity_scores) is a numeric
covariate/confounder and is adjusted for. The estimates are *causal* only under
unconfoundedness (no unmeasured confounding) given those covariates -- see
``vgi_causal.causal`` for the full math and assumptions.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Annotated, ClassVar

import pyarrow as pa
from vgi.arguments import Arg, TableInput
from vgi.invocation import BindResponse
from vgi.table_buffering_function import TableBufferingParams
from vgi.table_function import BindParams
from vgi_rpc.rpc import OutputCollector

from . import causal, meta
from .buffering import DrainState, SinkBuffer
from .meta import COHORT_CTE
from .schema_utils import field as cfield

# ---------------------------------------------------------------------------
# Output schemas
# ---------------------------------------------------------------------------

_ATE_SCHEMA = pa.schema(
    [
        cfield("method", pa.string(), "Estimator: ipw, regression_adjustment, or aipw.", nullable=False),
        cfield("estimate", pa.float64(), "Estimated average treatment effect E[Y(1)-Y(0)]."),
        cfield("std_error", pa.float64(), "Standard error of the estimate."),
        cfield("ci_lower", pa.float64(), "Lower bound of the 95% Wald confidence interval."),
        cfield("ci_upper", pa.float64(), "Upper bound of the 95% Wald confidence interval."),
    ]
)

_PROPENSITY_SCHEMA = pa.schema(
    [
        cfield("id", pa.int64(), "Passthrough row identifier (excluded from covariates).", nullable=False),
        cfield("propensity", pa.float64(), "Fitted propensity e(X)=P(T=1|X) in (0,1)."),
        cfield("treatment", pa.int32(), "Observed 0/1 treatment indicator for the row."),
    ]
)

_ATT_SCHEMA = pa.schema(
    [
        cfield("estimate", pa.float64(), "Average treatment effect on the treated E[Y(1)-Y(0)|T=1]."),
        cfield("std_error", pa.float64(), "Bootstrap standard error of the ATT estimate."),
    ]
)


# ---------------------------------------------------------------------------
# Executable examples (VGI509) -- self-contained, catalog-qualified, runnable.
# The COHORT_CTE inlines a confounded synthetic cohort so each query runs as
# written against the attached worker with no external table.
# ---------------------------------------------------------------------------

_ATE_EXECUTABLE_EXAMPLES = json.dumps(
    [
        {
            "description": "Average treatment effect by IPW, regression adjustment, and AIPW.",
            "sql": (
                COHORT_CTE + "SELECT method, round(estimate, 2) AS estimate "
                "FROM causal.main.ate((SELECT t, y, x FROM cohort), "
                "treatment := 't', outcome := 'y') ORDER BY method"
            ),
        },
        {
            "description": "Doubly-robust AIPW estimate with its 95% confidence interval.",
            "sql": (
                COHORT_CTE + "SELECT round(estimate, 2) AS estimate, "
                "round(ci_lower, 2) AS ci_lower, round(ci_upper, 2) AS ci_upper "
                "FROM causal.main.ate((SELECT t, y, x FROM cohort), "
                "treatment := 't', outcome := 'y') WHERE method = 'aipw'"
            ),
        },
    ]
)

_PROPENSITY_EXAMPLE_QUERIES = json.dumps(
    [
        {
            "description": "The five subjects most likely to have been treated, by fitted propensity.",
            "sql": (
                COHORT_CTE + "SELECT id, round(propensity, 3) AS propensity, treatment "
                "FROM causal.main.propensity_scores((SELECT id, t, x FROM cohort), "
                "treatment := 't', id := 'id') ORDER BY propensity DESC LIMIT 5"
            ),
        },
        {
            "description": "Count subjects whose fitted propensity exceeds one half (an overlap check).",
            "sql": (
                COHORT_CTE + "SELECT count(*) AS n_high_propensity "
                "FROM causal.main.propensity_scores((SELECT id, t, x FROM cohort), "
                "treatment := 't', id := 'id') WHERE propensity > 0.5"
            ),
        },
    ]
)

_ATT_EXAMPLE_QUERIES = json.dumps(
    [
        {
            "description": "Average treatment effect on the treated (IPW-ATT), with its bootstrap standard error.",
            "sql": (
                COHORT_CTE + "SELECT round(estimate, 2) AS estimate, round(std_error, 3) AS std_error "
                "FROM causal.main.att((SELECT t, y, x FROM cohort), treatment := 't', outcome := 'y')"
            ),
        },
    ]
)


# ---------------------------------------------------------------------------
# Argument dataclasses -- (SELECT ...) relation as Arg(0), roles as named args
# ---------------------------------------------------------------------------


@dataclass(slots=True, frozen=True)
class AteArgs:
    """Arguments for the ``ate`` table function."""

    data: Annotated[
        TableInput,
        Arg(
            0,
            doc=(
                "The cohort as a (SELECT ...) subquery. Every column other than the named "
                "treatment and outcome roles is treated as an adjusted-for covariate."
            ),
        ),
    ]
    treatment: Annotated[str, Arg("treatment", default="treatment", doc="Binary 0/1 treatment column.")]
    outcome: Annotated[
        str, Arg("outcome", default="outcome", doc="Outcome column whose treatment effect is estimated.")
    ]
    random_state: Annotated[
        int,
        Arg(
            "random_state",
            default=0,
            doc="Seed for the deterministic propensity-model fit; vary it to probe estimate stability.",
        ),
    ]


@dataclass(slots=True, frozen=True)
class PropensityArgs:
    """Arguments for the ``propensity_scores`` table function."""

    data: Annotated[
        TableInput,
        Arg(
            0,
            doc=(
                "The cohort as a (SELECT ...) subquery. The named id and treatment roles are "
                "excluded from the model; every other column is treated as a covariate."
            ),
        ),
    ]
    treatment: Annotated[str, Arg("treatment", default="treatment", doc="Binary 0/1 treatment column.")]
    id: Annotated[str, Arg("id", default="id", doc="Row id to pass through (excluded from covariates).")]
    random_state: Annotated[
        int,
        Arg(
            "random_state",
            default=0,
            doc="Seed for the deterministic propensity-model fit; vary it to probe score stability.",
        ),
    ]


@dataclass(slots=True, frozen=True)
class AttArgs:
    """Arguments for the ``att`` table function."""

    data: Annotated[
        TableInput,
        Arg(
            0,
            doc=(
                "The cohort as a (SELECT ...) subquery. Every column other than the named "
                "treatment and outcome roles is treated as an adjusted-for covariate."
            ),
        ),
    ]
    treatment: Annotated[str, Arg("treatment", default="treatment", doc="Binary 0/1 treatment column.")]
    outcome: Annotated[
        str, Arg("outcome", default="outcome", doc="Outcome column whose treatment effect is estimated.")
    ]
    random_state: Annotated[
        int,
        Arg(
            "random_state",
            default=0,
            doc="Seed for the deterministic propensity fit and bootstrap; vary it to probe SE stability.",
        ),
    ]


# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------


class Ate(SinkBuffer[AteArgs, DrainState]):
    """Average Treatment Effect over a buffered cohort relation (IPW/RA/AIPW)."""

    FunctionArguments: ClassVar[type] = AteArgs

    class Meta:
        """Catalog metadata for the ``ate`` function."""

        name = "ate"
        description = (
            "Average Treatment Effect E[Y(1)-Y(0)] adjusting for every covariate column. "
            "Emits one row per method: ipw, regression_adjustment, aipw (doubly-robust). "
            "treatment must be binary 0/1; causal only under unconfoundedness."
        )
        categories = ["causal", "estimator"]
        tags = {
            **meta.object_tags(
                "Average Treatment Effect Estimator",
                (
                    "# Average Treatment Effect (ATE)\n\n"
                    "Estimate the **average treatment effect** `E[Y(1) - Y(0)]` of a binary "
                    "treatment on a numeric outcome from an observational cohort, adjusting for "
                    "confounders.\n\n"
                    "**Inputs.** A `(SELECT ...)` cohort relation as the first positional argument, "
                    "plus named role args `treatment` (a binary 0/1 column) and `outcome` (a numeric "
                    "column). Every remaining column is treated as a covariate/confounder and is "
                    "adjusted for. An optional `random_state` (default 0) seeds the deterministic "
                    "propensity fit.\n\n"
                    "**Output.** Three rows, one per estimator: `ipw` (inverse-probability "
                    "weighting, Hajek/self-normalized), `regression_adjustment` (g-formula via OLS), "
                    "and `aipw` (doubly-robust). Each row has `estimate`, `std_error`, and a 95% "
                    "Wald confidence interval (`ci_lower`, `ci_upper`).\n\n"
                    "**When to use.** Answering 'what is the overall effect of this intervention?' "
                    "Comparing methods builds confidence: AIPW is doubly-robust, so it is consistent "
                    "if *either* the propensity or the outcome model is correct.\n\n"
                    "**Edge cases / assumptions.** Treatment must be exactly binary `{0, 1}` or the "
                    "call errors. Estimates are causal only under unconfoundedness (no unmeasured "
                    "confounding given the covariates), overlap (`0 < e(X) < 1`), and SUTVA. With "
                    "few rows or perfect separation the SEs widen but the regularized propensity "
                    "model still returns finite estimates."
                ),
                (
                    "# ate\n\n"
                    "Average Treatment Effect `E[Y(1) - Y(0)]` of a binary treatment on a numeric "
                    "outcome, adjusting for every covariate column in the input relation.\n\n"
                    "## Usage\n\n"
                    "Pass a cohort relation as the first positional argument and name the `treatment` "
                    "and `outcome` roles; every other column is an adjusted-for covariate. A complete, "
                    "runnable call is attached as this function's example query.\n\n"
                    "## Notes\n\n"
                    "- Emits one row per method: `ipw`, `regression_adjustment`, `aipw`.\n"
                    "- Each row carries the point estimate, its standard error, and a 95% Wald "
                    "confidence interval.\n"
                    "- `treatment` must be binary 0/1; every non-role column is an adjusted-for "
                    "covariate. Estimates are causal only under unconfoundedness, overlap, and SUTVA."
                ),
                [
                    "ate",
                    "average treatment effect",
                    "causal inference",
                    "treatment effect",
                    "ipw",
                    "inverse probability weighting",
                    "regression adjustment",
                    "g-formula",
                    "aipw",
                    "doubly robust",
                    "propensity",
                    "intervention",
                    "impact analysis",
                ],
            ),
            "vgi.category": "Treatment Effects",
            "vgi.executable_examples": _ATE_EXECUTABLE_EXAMPLES,
            # VGI515: the native duckdb_functions().examples carrier drops the
            # per-example description, so also ship the described JSON here.
            "vgi.example_queries": _ATE_EXECUTABLE_EXAMPLES,
            "vgi.result_columns_schema": json.dumps(
                [
                    {
                        "name": "method",
                        "type": "VARCHAR",
                        "description": "Estimator: ipw, regression_adjustment, or aipw. One row per method.",
                    },
                    {
                        "name": "estimate",
                        "type": "DOUBLE",
                        "description": "Estimated average treatment effect E[Y(1)-Y(0)].",
                    },
                    {"name": "std_error", "type": "DOUBLE", "description": "Standard error of the estimate."},
                    {
                        "name": "ci_lower",
                        "type": "DOUBLE",
                        "description": "Lower bound of the 95% Wald confidence interval.",
                    },
                    {
                        "name": "ci_upper",
                        "type": "DOUBLE",
                        "description": "Upper bound of the 95% Wald confidence interval.",
                    },
                ]
            ),
        }

    @classmethod
    def on_bind(cls, params: BindParams[AteArgs]) -> BindResponse:
        """Declare the output schema at bind time.

        Args:
            params: The bind-time params for this call.

        Returns:
            A bind response carrying the ATE output schema.
        """
        return BindResponse(output_schema=_ATE_SCHEMA)

    @classmethod
    def initial_finalize_state(cls, finalize_state_id: bytes, params: TableBufferingParams[AteArgs]) -> DrainState:
        """Create the per-finalize-stream cursor.

        Args:
            finalize_state_id: The finalize stream identifier.
            params: The buffering params for this execution.

        Returns:
            A fresh finalize cursor (result bytes + offset at 0).
        """
        return DrainState()

    @classmethod
    def finalize(
        cls,
        params: TableBufferingParams[AteArgs],
        finalize_state_id: bytes,
        state: DrainState,
        out: OutputCollector,
    ) -> None:
        """Run the ATE estimators over the buffered cohort and emit one batch.

        Args:
            params: The buffering params (args, output schema, storage).
            finalize_state_id: The finalize stream identifier.
            state: The finalize cursor (result bytes + offset).
            out: The collector to emit the result slice into.
        """
        a = params.args
        cls.drain_result(
            params,
            state,
            out,
            lambda df: causal.ate(df, treatment=a.treatment, outcome=a.outcome, random_state=a.random_state),
        )


class PropensityScores(SinkBuffer[PropensityArgs, DrainState]):
    """Per-row propensity scores e(X)=P(T=1|X) from a fitted logistic model."""

    FunctionArguments: ClassVar[type] = PropensityArgs

    class Meta:
        """Catalog metadata for the ``propensity_scores`` function."""

        name = "propensity_scores"
        description = (
            "Fit a logistic propensity model and emit per-row scores: (id, propensity, "
            "treatment). The id column is passed through and excluded from the covariates; "
            "every other column is a covariate. treatment must be binary 0/1."
        )
        categories = ["causal", "estimator"]
        tags = {
            **meta.object_tags(
                "Per-Row Propensity Score Estimator",
                (
                    "# Propensity Scores\n\n"
                    "Fit a logistic **propensity model** `e(X) = P(T = 1 | X)` over the cohort and "
                    "emit one row per input subject: its `id`, fitted `propensity`, and observed "
                    "`treatment`.\n\n"
                    "**Inputs.** A `(SELECT ...)` cohort relation as the first positional argument, "
                    "plus named role args `treatment` (a binary 0/1 column) and `id` (a row "
                    "identifier passed through and **excluded** from the covariates). Every other "
                    "column is a covariate fed to the logistic model. An optional `random_state` "
                    "(default 0) seeds the deterministic fit.\n\n"
                    "**Output.** One row per input row: `id` (`BIGINT`), `propensity` (`DOUBLE` in "
                    "`(0, 1)`), and `treatment` (`INTEGER` 0/1).\n\n"
                    "**When to use.** Diagnose overlap/positivity before estimating effects, build "
                    "matched or weighted cohorts, or trim units with extreme scores. The scores are "
                    "the same ones `ate`/`att` use internally for IPW.\n\n"
                    "**Edge cases / behaviors.** The logistic fit is L2-regularized and scores are "
                    "clipped to `[1e-6, 1 - 1e-6]`, so perfect separation does not produce 0/1 "
                    "scores or infinite weights. Treatment must be exactly binary `{0, 1}`. The "
                    "output is unbounded (one row per subject) and is paged through the worker's "
                    "offset cursor."
                ),
                (
                    "# propensity_scores\n\n"
                    "Per-row fitted propensity `e(X) = P(T = 1 | X)` from a regularized logistic "
                    "model.\n\n"
                    "## Usage\n\n"
                    "Pass a cohort relation as the first positional argument and name the `treatment` "
                    "and `id` roles; the `id` column is passed through and every other column is a "
                    "covariate. A complete, runnable call is attached as this function's example query.\n\n"
                    "## Notes\n\n"
                    "- Emits one row per input subject: `(id, propensity, treatment)`.\n"
                    "- The `id` column is passed through and excluded from the covariates; every "
                    "other column is a covariate.\n"
                    "- Scores are clipped to `(0, 1)`; use them for overlap diagnostics, matching, "
                    "or weighting. `treatment` must be binary 0/1."
                ),
                [
                    "propensity score",
                    "propensity",
                    "logistic regression",
                    "probability of treatment",
                    "e(X)",
                    "overlap",
                    "positivity",
                    "balance",
                    "matching",
                    "weighting",
                    "causal inference",
                ],
            ),
            "vgi.category": "Propensity Diagnostics",
            # VGI515: the native duckdb_functions().examples carrier drops the
            # per-example description, so also ship the described JSON here.
            "vgi.example_queries": _PROPENSITY_EXAMPLE_QUERIES,
            "vgi.result_columns_schema": json.dumps(
                [
                    {
                        "name": "id",
                        "type": "BIGINT",
                        "description": "Passthrough row identifier (the named id column, excluded from covariates).",
                    },
                    {
                        "name": "propensity",
                        "type": "DOUBLE",
                        "description": "Fitted propensity e(X)=P(T=1|X) in (0,1).",
                    },
                    {
                        "name": "treatment",
                        "type": "INTEGER",
                        "description": "Observed 0/1 treatment indicator for the row.",
                    },
                ]
            ),
        }

    @classmethod
    def on_bind(cls, params: BindParams[PropensityArgs]) -> BindResponse:
        """Declare the output schema at bind time.

        Args:
            params: The bind-time params for this call.

        Returns:
            A bind response carrying the propensity output schema.
        """
        return BindResponse(output_schema=_PROPENSITY_SCHEMA)

    @classmethod
    def initial_finalize_state(
        cls, finalize_state_id: bytes, params: TableBufferingParams[PropensityArgs]
    ) -> DrainState:
        """Create the per-finalize-stream cursor.

        Args:
            finalize_state_id: The finalize stream identifier.
            params: The buffering params for this execution.

        Returns:
            A fresh finalize cursor (result bytes + offset at 0).
        """
        return DrainState()

    @classmethod
    def finalize(
        cls,
        params: TableBufferingParams[PropensityArgs],
        finalize_state_id: bytes,
        state: DrainState,
        out: OutputCollector,
    ) -> None:
        """Fit the propensity model over the buffered cohort and emit per-row scores.

        Args:
            params: The buffering params (args, output schema, storage).
            finalize_state_id: The finalize stream identifier.
            state: The finalize cursor (result bytes + offset).
            out: The collector to emit the result slice into.
        """
        a = params.args
        cls.drain_result(
            params,
            state,
            out,
            lambda df: causal.propensity_scores(df, treatment=a.treatment, id=a.id, random_state=a.random_state),
        )


class Att(SinkBuffer[AttArgs, DrainState]):
    """Average Treatment effect on the Treated via IPW-ATT weighting."""

    FunctionArguments: ClassVar[type] = AttArgs

    class Meta:
        """Catalog metadata for the ``att`` function."""

        name = "att"
        description = (
            "Average Treatment effect on the Treated E[Y(1)-Y(0)|T=1] via IPW-ATT "
            "weighting, adjusting for every covariate column. Emits one row "
            "(estimate, std_error). treatment must be binary 0/1."
        )
        categories = ["causal", "estimator"]
        tags = {
            **meta.object_tags(
                "Average Treatment Effect on the Treated",
                (
                    "# Average Treatment effect on the Treated (ATT)\n\n"
                    "Estimate `E[Y(1) - Y(0) | T = 1]`: the average treatment effect **restricted to "
                    "the units that actually received treatment**, via inverse-probability-of-"
                    "treatment (IPW-ATT) weighting.\n\n"
                    "**Inputs.** A `(SELECT ...)` cohort relation as the first positional argument, "
                    "plus named role args `treatment` (a binary 0/1 column) and `outcome` (a numeric "
                    "column). Every remaining column is a covariate/confounder and is adjusted for. "
                    "An optional `random_state` (default 0) seeds the propensity fit and the bootstrap.\n\n"
                    "**Output.** A single row: `estimate` (the ATT) and `std_error` (a deterministic, "
                    "seeded bootstrap standard error).\n\n"
                    "**When to use.** When the policy question is about those who were treated -- "
                    "e.g. 'how much did the program help the people who enrolled?' -- rather than the "
                    "whole population (which is `ate`). The ATT and ATE coincide under a homogeneous "
                    "effect but diverge with effect heterogeneity.\n\n"
                    "**Edge cases / assumptions.** Controls are reweighted by `e/(1 - e)`; the "
                    "propensity model is regularized and clipped, so it stays finite under near-"
                    "separation. Treatment must be exactly binary `{0, 1}`. Causal only under "
                    "unconfoundedness, overlap on the treated, and SUTVA. The bootstrap is seeded "
                    "(`random_state`), so the SE is reproducible."
                ),
                (
                    "# att\n\n"
                    "Average Treatment effect on the Treated `E[Y(1) - Y(0) | T = 1]` via IPW-ATT "
                    "weighting, adjusting for every covariate column.\n\n"
                    "## Usage\n\n"
                    "Pass a cohort relation as the first positional argument and name the `treatment` "
                    "and `outcome` roles; every other column is an adjusted-for covariate. A complete, "
                    "runnable call is attached as this function's example query.\n\n"
                    "## Notes\n\n"
                    "- Emits a single row: `(estimate, std_error)`.\n"
                    "- `std_error` is a deterministic, seeded bootstrap SE.\n"
                    "- Use `att` for the effect on the treated subpopulation; use `ate` for the "
                    "whole-population effect. `treatment` must be binary 0/1."
                ),
                [
                    "att",
                    "average treatment effect on the treated",
                    "treated",
                    "ipw-att",
                    "inverse probability weighting",
                    "causal inference",
                    "treatment effect",
                    "subpopulation",
                    "bootstrap",
                    "intervention",
                ],
            ),
            "vgi.category": "Treatment Effects",
            # VGI515: the native duckdb_functions().examples carrier drops the
            # per-example description, so also ship the described JSON here.
            "vgi.example_queries": _ATT_EXAMPLE_QUERIES,
            "vgi.result_columns_schema": json.dumps(
                [
                    {
                        "name": "estimate",
                        "type": "DOUBLE",
                        "description": "Average treatment effect on the treated E[Y(1)-Y(0)|T=1].",
                    },
                    {
                        "name": "std_error",
                        "type": "DOUBLE",
                        "description": "Bootstrap standard error of the ATT estimate.",
                    },
                ]
            ),
        }

    @classmethod
    def on_bind(cls, params: BindParams[AttArgs]) -> BindResponse:
        """Declare the output schema at bind time.

        Args:
            params: The bind-time params for this call.

        Returns:
            A bind response carrying the ATT output schema.
        """
        return BindResponse(output_schema=_ATT_SCHEMA)

    @classmethod
    def initial_finalize_state(cls, finalize_state_id: bytes, params: TableBufferingParams[AttArgs]) -> DrainState:
        """Create the per-finalize-stream cursor.

        Args:
            finalize_state_id: The finalize stream identifier.
            params: The buffering params for this execution.

        Returns:
            A fresh finalize cursor (result bytes + offset at 0).
        """
        return DrainState()

    @classmethod
    def finalize(
        cls,
        params: TableBufferingParams[AttArgs],
        finalize_state_id: bytes,
        state: DrainState,
        out: OutputCollector,
    ) -> None:
        """Run the IPW-ATT estimator over the buffered cohort and emit one batch.

        Args:
            params: The buffering params (args, output schema, storage).
            finalize_state_id: The finalize stream identifier.
            state: The finalize cursor (result bytes + offset).
            out: The collector to emit the result slice into.
        """
        a = params.args
        cls.drain_result(
            params,
            state,
            out,
            lambda df: causal.att(df, treatment=a.treatment, outcome=a.outcome, random_state=a.random_state),
        )


TABLE_FUNCTIONS: list[type] = [Ate, PropensityScores, Att]
