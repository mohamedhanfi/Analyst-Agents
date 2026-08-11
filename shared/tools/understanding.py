"""CrewAI @tool wrappers — stage 2 Understanding tools.

All tools consume DataProfile JSON metadata only (golden rule: never raw
cells). The 20-row PII-redacted sample is the only raw-adjacent content that
may reach the LLM, and it is not used by column_profiler_tool.
"""
from __future__ import annotations

import json

from crewai.tools import tool

from shared.core.understanding import (
    ColumnProfiler,
    build_analysis_plan,
    build_domain_facts,
)
from shared.schemas import DataProfile


@tool("column_profiler_tool")
def column_profiler_tool(profile_json: str) -> str:
    """Per-column facts + §2.2 role guesses from a DataProfile.

    profile_json: the JSON content of metadata/data_profile.json (metadata
    only — no raw cell content).
    Returns JSON {"columns": [{name, dtype, nunique, nullable,
    suggested_role, alternate_roles}]}.
    """
    profile = DataProfile.model_validate(json.loads(profile_json))
    facts = ColumnProfiler().profile_columns(profile)
    return json.dumps({"columns": [f.to_dict() for f in facts]},
                      ensure_ascii=False, indent=2)


@tool("domain_classifier_tool")
def domain_classifier_tool(profile_json: str) -> str:
    """Facts the LLM uses to name the domain + business entities (§2.2).

    profile_json: the JSON content of metadata/data_profile.json.
    Returns profiled column facts + the PII-redacted 20-row sample, plus the
    "domain_decision" skeleton the agent must fill (detected_domain,
    domain_confidence in [0,1], entities as a list of names).
    """
    profile = DataProfile.model_validate(json.loads(profile_json))
    return json.dumps(build_domain_facts(profile), ensure_ascii=False,
                      indent=2)


@tool("dsl_plan_builder_tool")
def dsl_plan_builder_tool(raw_plan_json: str) -> str:
    """Validate + normalize the LLM's proposed analysis plan (§2.2).

    raw_plan_json: the LLM's raw JSON plan with candidate_kpis, each
    {"kpi_id", "name", "operation"}.
    Every operation is gated against the DSL whitelist
    (shared/dsl_validator.py). Invalid candidates are dropped and returned
    in "errors". Returns {"plan": <validated AnalysisPlan>, "errors": [...]}.
    """
    plan, errors = build_analysis_plan(raw_plan_json)
    return json.dumps({
        "plan": plan.model_dump(exclude_none=True),
        "errors": errors,
    }, ensure_ascii=False, indent=2)
