"""Shared per-object discovery/description metadata for the causal worker.

The ``vgi-lint`` strict profile (0.23.0+) expects, on **every** function and
table, a set of discovery/description tags. This module centralizes:

- ``object_tags(...)`` — the five standard per-object tags
  (``vgi.title`` VGI124, ``vgi.description_llm`` VGI112,
  ``vgi.description_md`` VGI113, ``vgi.keywords`` VGI126,
  ``vgi.source_url`` VGI128).
- ``source_url(...)`` — the canonical GitHub blob URL for a source file.
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

#: Base GitHub blob URL for source files in this repo (pinned to ``main``).
SOURCE_BASE = "https://github.com/Query-farm/vgi-causal/blob/main/vgi_causal"

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


def source_url(relative_path: str) -> str:
    """Build the ``vgi.source_url`` for a file under ``vgi_causal/``.

    Args:
        relative_path: Implementing file relative to ``vgi_causal`` (e.g.
            ``"tables.py"``).

    Returns:
        The canonical GitHub blob URL for that source file.
    """
    return f"{SOURCE_BASE}/{relative_path}"


def object_tags(
    title: str,
    description_llm: str,
    description_md: str,
    keywords: str,
    relative_path: str,
) -> dict[str, str]:
    """Build the five standard per-object discovery/description tags.

    Args:
        title: Human-friendly display name (``vgi.title``); MUST add a word
            beyond the machine name or VGI125 fires.
        description_llm: Markdown narrative aimed at LLM/agent audiences.
        description_md: Markdown narrative for human docs.
        keywords: Comma-separated search terms/synonyms.
        relative_path: Implementing file relative to ``vgi_causal``.

    Returns:
        A dict of the five standard per-object tags.
    """
    return {
        "vgi.title": title,
        "vgi.description_llm": description_llm,
        "vgi.description_md": description_md,
        "vgi.keywords": keywords,
        "vgi.source_url": source_url(relative_path),
    }
