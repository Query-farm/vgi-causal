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

from dataclasses import dataclass
from typing import Annotated, ClassVar

import pyarrow as pa
from vgi.arguments import Arg, TableInput
from vgi.invocation import BindResponse
from vgi.metadata import FunctionExample
from vgi.table_buffering_function import TableBufferingParams
from vgi.table_function import BindParams
from vgi_rpc.rpc import OutputCollector

from . import causal
from .buffering import DrainState, SinkBuffer
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
# Argument dataclasses -- (SELECT ...) relation as Arg(0), roles as named args
# ---------------------------------------------------------------------------


@dataclass(slots=True, frozen=True)
class AteArgs:
    """Arguments for the ``ate`` table function."""

    data: Annotated[TableInput, Arg(0, doc="Relation: treatment, outcome, and one+ covariate columns.")]
    treatment: Annotated[str, Arg("treatment", default="treatment", doc="Binary 0/1 treatment column.")]
    outcome: Annotated[str, Arg("outcome", default="outcome", doc="Numeric outcome column.")]


@dataclass(slots=True, frozen=True)
class PropensityArgs:
    """Arguments for the ``propensity_scores`` table function."""

    data: Annotated[TableInput, Arg(0, doc="Relation: id, treatment, and one+ covariate columns.")]
    treatment: Annotated[str, Arg("treatment", default="treatment", doc="Binary 0/1 treatment column.")]
    id: Annotated[str, Arg("id", default="id", doc="Row id to pass through (excluded from covariates).")]


@dataclass(slots=True, frozen=True)
class AttArgs:
    """Arguments for the ``att`` table function."""

    data: Annotated[TableInput, Arg(0, doc="Relation: treatment, outcome, and one+ covariate columns.")]
    treatment: Annotated[str, Arg("treatment", default="treatment", doc="Binary 0/1 treatment column.")]
    outcome: Annotated[str, Arg("outcome", default="outcome", doc="Numeric outcome column.")]


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
        examples = [
            FunctionExample(
                sql=("SELECT * FROM causal.ate((SELECT * FROM cohort), treatment := 't', outcome := 'y')"),
                description="ATE by IPW, regression adjustment, and doubly-robust AIPW",
            )
        ]
        tags = {
            "vgi.columns_md": (
                "| column | type | description |\n"
                "|---|---|---|\n"
                "| `method` | VARCHAR | Estimator: `ipw`, `regression_adjustment`, or "
                "`aipw`. One row per method. |\n"
                "| `estimate` | DOUBLE | Estimated average treatment effect E[Y(1)-Y(0)]. |\n"
                "| `std_error` | DOUBLE | Standard error of the estimate. |\n"
                "| `ci_lower` | DOUBLE | Lower bound of the 95% Wald confidence interval. |\n"
                "| `ci_upper` | DOUBLE | Upper bound of the 95% Wald confidence interval. |"
            )
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
        cls.drain_result(params, state, out, lambda df: causal.ate(df, treatment=a.treatment, outcome=a.outcome))


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
        examples = [
            FunctionExample(
                sql=(
                    "SELECT * FROM causal.propensity_scores((SELECT id, t, x1, x2 FROM cohort), "
                    "treatment := 't', id := 'id') ORDER BY id"
                ),
                description="Per-row propensity scores with id passthrough",
            )
        ]
        tags = {
            "vgi.columns_md": (
                "| column | type | description |\n"
                "|---|---|---|\n"
                "| `id` | BIGINT | Passthrough row identifier (the named `id` column, "
                "excluded from covariates). |\n"
                "| `propensity` | DOUBLE | Fitted propensity e(X)=P(T=1\\|X) in (0,1). |\n"
                "| `treatment` | INTEGER | Observed 0/1 treatment indicator for the row. |"
            )
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
        cls.drain_result(params, state, out, lambda df: causal.propensity_scores(df, treatment=a.treatment, id=a.id))


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
        examples = [
            FunctionExample(
                sql=("SELECT * FROM causal.att((SELECT * FROM cohort), treatment := 't', outcome := 'y')"),
                description="Average treatment effect on the treated (IPW-ATT)",
            )
        ]
        tags = {
            "vgi.columns_md": (
                "| column | type | description |\n"
                "|---|---|---|\n"
                "| `estimate` | DOUBLE | Average treatment effect on the treated "
                "E[Y(1)-Y(0)\\|T=1]. |\n"
                "| `std_error` | DOUBLE | Bootstrap standard error of the ATT estimate. |"
            )
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
        cls.drain_result(params, state, out, lambda df: causal.att(df, treatment=a.treatment, outcome=a.outcome))


TABLE_FUNCTIONS: list[type] = [Ate, PropensityScores, Att]
