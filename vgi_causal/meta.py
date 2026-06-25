"""Shared per-object discovery/description metadata for the causal worker.

The ``vgi-lint`` strict profile (0.23.0+) expects, on **every** function and
table, a set of discovery/description tags. This module centralizes:

- ``object_tags(...)`` — the four standard per-object tags
  (``vgi.title`` VGI124, ``vgi.doc_llm`` VGI112,
  ``vgi.doc_md`` VGI113, ``vgi.keywords`` VGI126/VGI138).
  ``vgi.source_url`` is intentionally NOT emitted per object (VGI139): it
  belongs only on the catalog object, which the worker sets directly.
- ``COHORT_CTE`` — a self-contained, confounded synthetic cohort expressed as a
  SQL CTE, so every documented example query is runnable as written (no external
  table required) and the linter can execute it.

The synthetic cohort plants a homogeneous treatment effect ``tau = 5`` and
confounds it: the covariate ``x`` drives both treatment assignment and the
outcome ``y = 5*t + 2*x``. With a noise-free linear DGP, regression adjustment
and AIPW recover ``tau`` exactly, while the naive difference of means is biased.
This mirrors the deterministic story in ``test/sql/causal.test``.
"""

from __future__ import annotations

import json

#: A self-contained confounded cohort (id, x, t, y) as a CTE. ``x`` confounds
#: treatment ``t`` and outcome ``y = 5*t + 2*x``; the planted ATE is ``tau = 5``.
COHORT_CTE = (
    "WITH cohort AS (\n"
    "  SELECT\n"
    "    g                                                   AS id,\n"
    "    (g - 30)::DOUBLE / 10.0                             AS x,\n"
    "    CASE WHEN ((g * 7 + 3) % 10) < 5 THEN 1 ELSE 0 END  AS t,\n"
    "    5.0 * (CASE WHEN ((g * 7 + 3) % 10) < 5 THEN 1 ELSE 0 END)\n"
    "      + 2.0 * ((g - 30)::DOUBLE / 10.0)                 AS y\n"
    "  FROM generate_series(0, 60) AS s(g)\n"
    ")\n"
)


def keywords_json(keywords: list[str]) -> str:
    """Serialize search keywords as a JSON array string (VGI138).

    ``vgi.keywords`` is transported as a single tag string but must hold a JSON
    array of strings (e.g. ``["ate", "propensity score"]``), not a
    comma-separated list.

    Args:
        keywords: Search terms / synonyms for the object.

    Returns:
        The keywords encoded as a compact JSON array string.
    """
    return json.dumps(keywords)


def object_tags(
    title: str,
    doc_llm: str,
    doc_md: str,
    keywords: list[str],
) -> dict[str, str]:
    """Build the four standard per-object discovery/description tags.

    ``vgi.source_url`` is intentionally omitted here (VGI139): a per-object
    source URL is redundant and only the catalog object should carry one.

    Args:
        title: Human-friendly display name (``vgi.title``); MUST add a word
            beyond the machine name or VGI125 fires.
        doc_llm: Markdown narrative aimed at LLM/agent audiences.
        doc_md: Markdown narrative for human docs.
        keywords: Search terms / synonyms, emitted as a JSON array (VGI138).

    Returns:
        A dict of the four standard per-object tags.
    """
    return {
        "vgi.title": title,
        "vgi.doc_llm": doc_llm,
        "vgi.doc_md": doc_md,
        "vgi.keywords": keywords_json(keywords),
    }
