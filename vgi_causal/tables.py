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
from vgi.table_buffering_function import OutputCollector, TableBufferingParams
from vgi.table_function import BindParams

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
    data: Annotated[TableInput, Arg(0, doc="Relation: treatment, outcome, and one+ covariate columns.")]
    treatment: Annotated[str, Arg("treatment", default="treatment", doc="Binary 0/1 treatment column.")]
    outcome: Annotated[str, Arg("outcome", default="outcome", doc="Numeric outcome column.")]


@dataclass(slots=True, frozen=True)
class PropensityArgs:
    data: Annotated[TableInput, Arg(0, doc="Relation: id, treatment, and one+ covariate columns.")]
    treatment: Annotated[str, Arg("treatment", default="treatment", doc="Binary 0/1 treatment column.")]
    id: Annotated[str, Arg("id", default="id", doc="Row id to pass through (excluded from covariates).")]


@dataclass(slots=True, frozen=True)
class AttArgs:
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
        name = "ate"
        description = (
            "Average Treatment Effect E[Y(1)-Y(0)] adjusting for every covariate column. "
            "Emits one row per method: ipw, regression_adjustment, aipw (doubly-robust). "
            "treatment must be binary 0/1; causal only under unconfoundedness."
        )
        categories = ["causal", "estimator"]
        examples = [
            FunctionExample(
                sql=(
                    "SELECT * FROM causal.ate((SELECT * FROM cohort), "
                    "treatment := 't', outcome := 'y')"
                ),
                description="ATE by IPW, regression adjustment, and doubly-robust AIPW",
            )
        ]

    @classmethod
    def on_bind(cls, params: BindParams[AteArgs]) -> BindResponse:
        return BindResponse(output_schema=_ATE_SCHEMA)

    @classmethod
    def initial_finalize_state(
        cls, finalize_state_id: bytes, params: TableBufferingParams[AteArgs]
    ) -> DrainState:
        return DrainState()

    @classmethod
    def finalize(
        cls,
        params: TableBufferingParams[AteArgs],
        finalize_state_id: bytes,
        state: DrainState,
        out: OutputCollector,
    ) -> None:
        if state.done:
            out.finish()
            return
        state.done = True
        a = params.args
        df = cls.buffered_frame(params)
        result = causal.ate(df, treatment=a.treatment, outcome=a.outcome)
        out.emit(pa.RecordBatch.from_pydict(result, schema=params.output_schema))


class PropensityScores(SinkBuffer[PropensityArgs, DrainState]):
    """Per-row propensity scores e(X)=P(T=1|X) from a fitted logistic model."""

    FunctionArguments: ClassVar[type] = PropensityArgs

    class Meta:
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

    @classmethod
    def on_bind(cls, params: BindParams[PropensityArgs]) -> BindResponse:
        return BindResponse(output_schema=_PROPENSITY_SCHEMA)

    @classmethod
    def initial_finalize_state(
        cls, finalize_state_id: bytes, params: TableBufferingParams[PropensityArgs]
    ) -> DrainState:
        return DrainState()

    @classmethod
    def finalize(
        cls,
        params: TableBufferingParams[PropensityArgs],
        finalize_state_id: bytes,
        state: DrainState,
        out: OutputCollector,
    ) -> None:
        if state.done:
            out.finish()
            return
        state.done = True
        a = params.args
        df = cls.buffered_frame(params)
        result = causal.propensity_scores(df, treatment=a.treatment, id=a.id)
        out.emit(pa.RecordBatch.from_pydict(result, schema=params.output_schema))


class Att(SinkBuffer[AttArgs, DrainState]):
    """Average Treatment effect on the Treated via IPW-ATT weighting."""

    FunctionArguments: ClassVar[type] = AttArgs

    class Meta:
        name = "att"
        description = (
            "Average Treatment effect on the Treated E[Y(1)-Y(0)|T=1] via IPW-ATT "
            "weighting, adjusting for every covariate column. Emits one row "
            "(estimate, std_error). treatment must be binary 0/1."
        )
        categories = ["causal", "estimator"]
        examples = [
            FunctionExample(
                sql=(
                    "SELECT * FROM causal.att((SELECT * FROM cohort), "
                    "treatment := 't', outcome := 'y')"
                ),
                description="Average treatment effect on the treated (IPW-ATT)",
            )
        ]

    @classmethod
    def on_bind(cls, params: BindParams[AttArgs]) -> BindResponse:
        return BindResponse(output_schema=_ATT_SCHEMA)

    @classmethod
    def initial_finalize_state(
        cls, finalize_state_id: bytes, params: TableBufferingParams[AttArgs]
    ) -> DrainState:
        return DrainState()

    @classmethod
    def finalize(
        cls,
        params: TableBufferingParams[AttArgs],
        finalize_state_id: bytes,
        state: DrainState,
        out: OutputCollector,
    ) -> None:
        if state.done:
            out.finish()
            return
        state.done = True
        a = params.args
        df = cls.buffered_frame(params)
        result = causal.att(df, treatment=a.treatment, outcome=a.outcome)
        out.emit(pa.RecordBatch.from_pydict(result, schema=params.output_schema))


TABLE_FUNCTIONS: list[type] = [Ate, PropensityScores, Att]
