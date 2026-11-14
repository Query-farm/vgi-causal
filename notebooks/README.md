# Causal inference examples

[Worker repository](https://github.com/Query-farm/vgi-causal) · [Shared tools and standard](https://github.com/Query-farm/vgi-example-notebooks)

[Adjustment depends on what the model can see](01_adjustment_needs_the_confounders.ipynb)
constructs a synthetic cohort with a known +3 effect, fits propensity scores and
three ATE estimators through SQL, audits balance and reconstructs the ATT point
estimate. The experiment omits one confounder while preserving every observation.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Query-farm/vgi-causal/blob/main/notebooks/01_adjustment_needs_the_confounders.ipynb)

The notebook includes the linked published worker title/subtitle, consistent
Query Farm Strata Sun/VGI branding, saved SVG figures and visible connection SQL.
GitHub displays code expanded; shared setup and presentation helpers keep the
analytical cells prominent.

## Run locally or in Colab

Open the Colab link and run all. Locally, from this directory:

```sh
uv sync --locked
uv run --locked vgi-notebooks --timeout 300
```

Choose `.venv/bin/python` as an interactive kernel. Standalone setup installs only
the shared base package and preserves compatible preloaded kernel/plot libraries.
The local environment additionally includes the execution runner. The small
presentation/input helper downloads from an immutable public revision with
SHA-256 verification. No local worker checkout or copied execution script is needed.

The visible connection uses the public release and its own dependency declarations:

```sql
ATTACH 'causal' (TYPE vgi,
 LOCATION 'uvx --python 3.13 --from git+https://github.com/Query-farm/vgi-causal@v0.4.0 vgi-causal-worker');
```

Initial setup requires internet access; uv provisions the separate Python/worker
environment. Estimation runs locally without credentials, external datasets or
API calls. The optional DoWhy backend is not installed or used.

## Simulation and interpretation

A seeded NumPy generator supplies independent standard-normal covariates/noise
and uniform assignment draws. Visible SQL defines logistic treatment assignment
from x1/x2, untreated outcome `2*x1 + 1.5*x2 + noise`, and a constant +3 effect.
The default contains4,000 synthetic subjects with seed7; none are real people.

`ADJUST_FOR_X2` controls the relation projections passed to every estimator. IDs
and simulation-only values never enter ATE/ATT covariates; propensity explicitly
excludes its ID role. The audit retains x2 even when models omit it, possible
because this is a simulation rather than an actually unmeasured variable.

The notebook states no-unmeasured-confounding, positivity, and consistency/no-
interference assumptions. Probability clipping makes weights finite but does not
prove adequate overlap. Balance uses a fixed unweighted pooled within-arm SD.
Worker95% Wald intervals and the unadjusted normal approximation do not account
for omitted-confounder bias. Agreement among estimators does not establish
identification.

The real `methods` view identifies each estimator's target and labels the ATE
result table. ATT targets the treated population, using propensity odds on
controls. SQL independently reconciles the worker's ATT point estimate; the
reported standard error uses the worker's200 seeded row-bootstrap resamples with
propensity refitting. The SQL check does not reproduce the bootstrap. True ATE
and ATT equal3 here only because individual effects are constant; heterogeneous
effects can give different target values.

Nine SQL invariants plus two ATT controls check population preservation, valid
probabilities, weight concentration, interval ordering, native method labels and
independent Hájek/ATT arithmetic. They do not require noisy estimates to equal
truth or intervals to contain it.

## Save the experiment separately

```sh
uv run --locked vgi-notebooks 01_adjustment_needs_the_confounders.ipynb \
 --parameters '{"ADJUST_FOR_X2":false}' --output-dir artifacts/omit-x2
```

Keep `N_SUBJECTS=4000` and `SEED=7`. The cohort, planted effect and unadjusted
contrast remain fixed while x2 balance, adjusted effects and ATT can change.
Restore `ADJUST_FOR_X2=True`. Parameter types/ranges are checked before model calls.

The shared runner requires separate override outputs and saves notebooks only
after success. Defaults, the detached experiment, changed figures and one fresh
preloaded-kernel bootstrap are checked; hosted Colab is not manually operated.
Review PNGs and experiment copies are ignored; native SVG outputs and GitHub SVG
assets are committed.

[Coverage evidence](coverage.json) retains all four actual SQL surfaces and
separates publicv0.4.0 execution from current-source compatibility. Current changes
are metadata/examples; estimation, buffering, schemas and dependencies match.
Existing green source CI is reused for these notebook-only changes.
