# Causal-inference notebooks

[Adjustment depends on what the model can see](01_adjustment_needs_the_confounders.ipynb)
constructs a synthetic cohort with a known +3 treatment effect. Visible SQL fits
propensity scores and three ATE estimators through the real worker, audits balance,
and compares estimates with the planted effect. The experiment omits one confounder
while preserving every observation.

## Run from this folder

Prerequisites: [uv](https://docs.astral.sh/uv/), Git and network access for first-time
package/extension installation. `uv` can provision Python. Copy this entire
`notebooks/` folder anywhere; no worker checkout or root environment is required.

```bash
cd notebooks
uv sync --frozen
uv run --frozen python execute.py
```

The locked notebook environment uses Haybarn 2.0.0rc11. The visible ATTACH runs
published Git revision `9ec727d6f00fe00d105b89a5c2ac05e9bd4f5dc5` through `uvx`,
Python 3.13 and SDK 0.51.1 in a separate worker runtime. To upgrade, change the
literal Git/SDK pins and rerun the default and experiment before saving new outputs.

For an interactive kernel:

```bash
uv run --frozen python -m ipykernel install --user \
  --name vgi-causal-notebooks --display-name 'Python (vgi-causal notebooks)'
```

Open the notebook in your preferred Jupyter interface, select that kernel and use
this folder as its working directory. Parameter and analytical SQL cells remain
visible; setup, plotting and cleanup are folded.

## Simulation and interpretation

All data are synthetic. A seeded NumPy generator supplies independent standard-normal
covariates/noise and uniform assignment draws. SQL defines logistic treatment
assignment from `x1` and `x2`, untreated outcome `2*x1 + 1.5*x2 + noise`, and a constant
+3 treatment effect. The default cohort contains 4,000 subjects with seed 7.
There is no external dataset, credential, live service or domain-specific outcome.

`ADJUST_FOR_X2` controls the visible SQL column projections passed to both worker
models. IDs and simulation-only quantities never enter the ATE covariate set. The
balance audit retains x2 even when the model omits it; that is possible because this
is a simulation, not a genuinely unmeasured real-world variable.

The notebook states no-unmeasured-confounding, positivity and consistency/no-interference
assumptions. Fitted propensity clipping ensures finite weights but does not prove
adequate overlap. Balance uses a fixed unweighted pooled within-arm standard deviation.
Worker-reported 95% Wald intervals and the unadjusted normal approximation do not
account for omitted-confounder bias. Agreement among estimators does not establish
causal identification. All worker calls are materialized before local result capture.

## Save the focused experiment separately

```bash
uv run --frozen python execute.py --parameters '{"ADJUST_FOR_X2":false}' \
  --output-dir validation/omit-x2
```

Keep `N_SUBJECTS = 4000` and `SEED = 7`. Compare weighted balance, errors from the
known effect and the effect plot; the unadjusted contrast should remain unchanged.
Reset `ADJUST_FOR_X2 = True` afterward. Parameter ranges are checked before model calls.

The runner uses a fresh temporary kernel, writes outputs only after successful
execution and requires a separate destination for parameter overrides. Experiment
folders include their branding and runtime companions. Logs, extracted charts and
experiment copies are ignored; the main notebook retains successful embedded outputs.
