"""Drift guards for the published example cohort.

``COHORT_CTE`` is pasted verbatim into the README quick start and into the
private agent-test graders (``vgi-agent-tests.yaml``). Those copies once drifted
into a cohort where ``t`` did not depend on ``x`` at all (no confounding), while
every description still claimed ``x`` confounds the effect. Keep the copies
byte-identical to the single source in ``vgi_causal.meta``.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from vgi_causal import worker
from vgi_causal.meta import COHORT_CTE

ROOT = Path(__file__).resolve().parent.parent


def test_cohort_cte_is_confounded() -> None:
    # Treatment assignment must depend on x (through sign(g - 30)); without it the
    # "naive difference of means is biased" story in every example is false.
    assert "sign(g - 30)" in COHORT_CTE
    assert COHORT_CTE.count("sign(g - 30)") == 2


def test_readme_embeds_current_cohort() -> None:
    readme = (ROOT / "README.md").read_text()
    # The composed example continues the WITH list (``),\nps AS``), so match the
    # CTE body without its closing parenthesis.
    body = COHORT_CTE.rstrip().removesuffix(")")
    assert readme.count(body) >= 3


def test_agent_graders_embed_current_cohort() -> None:
    text = (ROOT / "vgi-agent-tests.yaml").read_text()
    escaped = json.dumps(COHORT_CTE)[1:-1]
    graded = re.findall(r'reference_sql: "(.*)"', text)
    cohort_graders = [g for g in graded if g.startswith("WITH cohort AS")]
    assert cohort_graders, "expected cohort-based graders"
    for g in cohort_graders:
        assert g.startswith(escaped)


def test_agent_task_names_match_graders() -> None:
    published = {t["name"] for t in json.loads(worker._AGENT_TEST_TASKS)}
    graded = set(re.findall(r"- name: (\S+)", (ROOT / "vgi-agent-tests.yaml").read_text()))
    assert published == graded
