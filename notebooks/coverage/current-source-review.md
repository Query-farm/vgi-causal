# Causal catalog review

Actual public v0.4.0 registers ate, att, propensity_scores and methods: four
logical surfaces. Full functions/tables/views exports are retained. There are
no excluded authentication/write surfaces.

Current396fb15 keeps the same registrations, named role arguments and output
schemas. The full runtime/dependency diff changes composed examples and catalog
metadata, including the example COHORT_CTE construction. The actual estimation,
200-resample ATT bootstrap, propensity clipping, relation buffering, transport
and dependency declarations are unchanged. The notebooks construct their own
synthetic cohort and do not execute that changed metadata CTE.

Current-source interface/behavior compatibility is reviewed; no current binary
was built or executed. Saved notebook receipts establish only publicv0.4.0 use.
Green source CI43561948160 at396fb15 is reused for notebook-only updates.
