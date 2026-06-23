"""Causal treatment-effect estimation as a VGI worker for DuckDB/SQL.

The implementation is split so each concern stays focused:

- ``causal``    -- pure causal-inference logic (propensity scores, IPW /
  regression-adjustment / doubly-robust AIPW for the ATE, IPW-ATT for the ATT)
  over ``pandas`` frames with scikit-learn + statsmodels; no Arrow or VGI
  dependency, directly unit-testable.
- ``buffering`` -- the single-bucket Sink+Source plumbing every function shares
  (buffer all input batches, then fit/estimate once).
- ``tables``    -- the VGI ``TableBufferingFunction`` wrappers: relation in via
  ``(SELECT ...)`` (``Arg(0)``), column roles as named string args.

``causal_worker.py`` at the repo root assembles these into the ``causal``
catalog and runs the worker over stdio (or HTTP).
"""

from __future__ import annotations

__version__ = "0.1.0"
