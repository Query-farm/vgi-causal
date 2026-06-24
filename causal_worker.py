# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "vgi-python[http]>=0.8.4",
#     "scikit-learn>=1.3",
#     "statsmodels>=0.14",
#     "scipy",
#     "numpy",
#     "pandas",
#     "pyarrow",
# ]
# ///
"""Stdio entry shim for the causal VGI worker.

Lets the worker run straight from a source checkout (``uv run
causal_worker.py``) and keeps ``import causal_worker`` working for tests. The
implementation lives in ``vgi_causal.worker``; installed users invoke the
``vgi-causal`` console script (which points at ``vgi_causal.worker:main``).

    ATTACH 'causal' (TYPE vgi, LOCATION 'uv run causal_worker.py');
    SELECT * FROM causal.ate((SELECT * FROM cohort), treatment := 't', outcome := 'y');
"""

from vgi_causal.worker import CausalWorker, main

__all__ = ["CausalWorker", "main"]

if __name__ == "__main__":
    main()
