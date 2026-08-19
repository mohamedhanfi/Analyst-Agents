"""ColumnProfiler — per-column facts + §2.2 role rules (stage 2).

Runs on DataProfile metadata ONLY (dtype/nunique/missing/row_count) — never
touches raw cells. The LLM later reclassifies by name/semantics where the
rules are ambiguous; `alternate_roles` carries those candidates.

Also builds the AnalysisPlan from the LLM's raw KPI proposals, gating every
operation through shared/dsl_validator (Day 1 whitelist) and dropping what
does not comply — the plan is whitelist-only by construction.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Union

from shared.core.semantic_guards import (
    IDENTIFIER_NAME_RE,          # canonical pattern (semantic sweep §0)
    IdentifierSignal,
    is_code_like,
    is_identifier_like,
    is_ordinal_like,
)
from shared.dsl_validator import validate_operation
from shared.schemas import (
    AnalysisPlan,
    BusinessContext,
    ColumnRole,
    ColumnUnderstanding,
    DataProfile,
    DslOperation,
    DatasetUnderstanding,
    KpiCandidate,
)

# Re-exports for callers that predate the shared semantic_guards module —
# the canonical implementations live there now so no stage duplicates them.
detect_identifier_like = is_identifier_like

ALLOWED_STATISTICAL_TESTS = frozenset(
    {"descriptive", "correlation", "trend", "anova"})


@dataclass
class ColumnFacts:
    """One column's facts + the rule-based role guess (§2.2)."""
    name: str
    dtype: str
    nunique: int
    nullable: bool
    suggested_role: ColumnRole
    alternate_roles: List[ColumnRole] = field(default_factory=list)
    # Identifier-detection heuristic (task A): confidence + signals.
    identifier_score: float = 0.0
    identifier_signals: List[str] = field(default_factory=list)
    # Semantic sweep 2.2/2.3: encoded-categorical and ordinal flags so
    # downstream stages phrase the column correctly.
    code_like: bool = False
    ordinal: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "dtype": self.dtype,
            "nunique": self.nunique,
            "nullable": self.nullable,
            "suggested_role": self.suggested_role,
            "alternate_roles": list(self.alternate_roles),
            "identifier_score": self.identifier_score,
            "code_like": self.code_like,
            "ordinal": self.ordinal,
        }





def _is_numeric(dtype: str) -> bool:
    return dtype.startswith(("int", "float"))


def _is_temporal(dtype: str) -> bool:
    return "datetime" in dtype or "date" in dtype


ID_NAME_PATTERN = re.compile(r"(?:_|^)(id|code|key|sku)$", re.IGNORECASE)
TEMPORAL_NAME_KEYWORDS = ("date", "datetime", "time", "year", "month",
                          "week", "day", "timestamp", "period", "quarter")

# Semantic identifier patterns live in shared.core.semantic_guards (§0) —
# imported at the top of this module so every layer shares one source of
# truth (KPI sanity gate A.3, correlation gate 5.2, insight gate 6.2, QA 8.1).


def _looks_like_id(name: str) -> bool:
    name = name.strip().lower()
    if name in {"id", "rowid", "uuid"}:
        return True
    return bool(ID_NAME_PATTERN.search(name))


def _looks_temporal_name(name: str) -> bool:
    base = name.strip().lower().rstrip("s")
    return any(kw in base for kw in TEMPORAL_NAME_KEYWORDS)


def infer_role(dtype: str, nunique: int, row_count: int,
               name: str = "") -> tuple[ColumnRole, List[ColumnRole]]:
    """Apply the §2.2 table; return (suggested_role, alternate_roles).

    Dtype checks win over the all-unique heuristic: a datetime column is
    always temporal and a numeric column is always a measure (an all-unique
    numeric is only an identifier when its name looks like an id — e.g.
    order_id, zip_code — otherwise it is a measure with "identifier" as an
    alternate for the LLM to consider).
    """
    if row_count == 0:
        return "dimension", []
    if _is_temporal(dtype):
        return "temporal", []
    if _is_numeric(dtype):
        if nunique == row_count:
            if _looks_like_id(name):
                return "identifier", ["measure"]
            return "measure", ["identifier"]
        if nunique > 20:
            return "measure", []
        return "measure", ["categorical"]
    if dtype == "bool":
        return "dimension", ["categorical"]
    if _looks_temporal_name(name):
        return "temporal", ["dimension"]
    if nunique == row_count:
        return "identifier", []
    if nunique <= 20:
        return "dimension", []
    if nunique > 50:
        return "free_text", []
    return "dimension", ["free_text"]


class ColumnProfiler:
    """Facts + role guesses derived from a DataProfile (metadata only).

    ``df`` (optional) adds the value-shape signals of the identifier
    heuristic (task A.1) — pass the extracted DataFrame when available;
    without it only the name-pattern signal applies.
    """

    def profile_columns(self, profile: DataProfile,
                        df: Union["pd.DataFrame", None] = None,
                        identifier_threshold: float = 0.7
                        ) -> List[ColumnFacts]:
        row_count = int(profile.row_count)
        facts: List[ColumnFacts] = []
        for name in profile.columns:
            dtype = str(profile.column_types.get(name, "object"))
            nunique = int(profile.nunique.get(name, 0))
            nullable = int(profile.missing_values.get(name, 0)) > 0
            role, alternates = infer_role(dtype, nunique, row_count, name)
            series = (df[name] if df is not None
                      and name in df.columns else None)
            signal = detect_identifier_like(name, series)
            code_like = is_code_like(name, series)
            # 2.2 beats A.1 for *encoded codes*: a low-cardinality repeating
            # integer (gender_code 0/1, status 1/2/3) is a categorical even
            # when its name matches an identifier pattern (code$); IDs are
            # all-unique, which is_code_like excludes.
            if (_is_numeric(dtype) and role == "measure" and code_like):
                role = "categorical"
                alternates = ["measure"]
            # A.1: a column with a confident identifier signal is
            # never defaulted to free_text/measure — override to identifier.
            if (_is_numeric(dtype) and role == "measure"
                    and signal.score >= identifier_threshold):
                role = "identifier"
                alternates = ["measure"]
            elif (not _is_numeric(dtype) and not _is_temporal(dtype)
                    and role in ("free_text", "dimension")
                    and signal.score >= identifier_threshold):
                role = "identifier"
                alternates = ["dimension", "free_text"]
            facts.append(ColumnFacts(
                name=name, dtype=dtype, nunique=nunique,
                nullable=nullable, suggested_role=role,
                alternate_roles=alternates,
                identifier_score=signal.score,
                identifier_signals=list(signal.signals),
                code_like=code_like,
                ordinal=is_ordinal_like(name, series),
            ))
        return facts


def build_domain_facts(profile: DataProfile) -> Dict[str, Any]:
    """Package profiled facts + the PII-redacted sample for the LLM.

    The sample comes straight from DataProfile.sample (already [REDACTED] by
    the profiler — the only raw-adjacent content allowed to reach the LLM).
    `domain_decision` is the skeleton the LLM fills and returns.
    """
    facts = ColumnProfiler().profile_columns(profile)
    return {
        "domain_facts": {
            "row_count": int(profile.row_count),
            "columns": [f.to_dict() for f in facts],
            "sample": profile.sample,
        },
        "domain_decision": {
            "detected_domain": None,
            "domain_confidence": None,
            "entities": [],
        },
    }


def _default_kpi_name(op: Dict[str, Any]) -> str:
    function = str(op.get("function", "kpi")).capitalize()
    if function == "Ratio":
        return "Ratio"
    if function == "Correlation":
        return f"Correlation {op.get('column_a')} x {op.get('column_b')}"
    if function == "Growth":
        return f"Growth of {op.get('column')}"
    return f"{function} of {op.get('column')}"


def build_analysis_plan(raw_plan: Union[AnalysisPlan, dict, str, None]
                        ) -> tuple[AnalysisPlan, List[str]]:
    """Validate + normalize the LLM's proposed plan (never raises).

    Every KPI operation is gated through shared.dsl_validator.validate_
    operation; invalid candidates are DROPPED and their reasons returned in
    `errors`. Statistical tests outside ALLOWED_STATISTICAL_TESTS are
    dropped too. Missing kpi_id/name get deterministic defaults.
    """
    errors: List[str] = []
    if raw_plan is None:
        return AnalysisPlan(), ["plan is missing 'candidate_kpis'"]
    if isinstance(raw_plan, str):
        try:
            raw_plan = json.loads(raw_plan)
        except (json.JSONDecodeError, TypeError):
            return AnalysisPlan(), ["plan must be valid JSON"]
    if isinstance(raw_plan, AnalysisPlan):
        data = raw_plan.model_dump(exclude_none=True)
    elif isinstance(raw_plan, dict):
        data = raw_plan
    else:
        return AnalysisPlan(), ["plan must be an object with 'candidate_kpis'"]

    candidates = data.get("candidate_kpis")
    if candidates is None:
        errors.append("plan is missing 'candidate_kpis'")
        candidates = []

    kpis: List[KpiCandidate] = []
    for index, candidate in enumerate(candidates):
        if not isinstance(candidate, dict):
            errors.append(f"candidate #{index}: must be an object")
            continue
        operation = candidate.get("operation")
        if operation is None:
            errors.append(f"candidate #{index}: missing 'operation'")
            continue
        label = candidate.get("kpi_id") or f"#{index}"
        op_errors = validate_operation(operation)
        if op_errors:
            errors.extend(f"{label}: {e}" for e in op_errors)
            continue
        kpis.append(KpiCandidate(
            kpi_id=candidate.get("kpi_id") or f"KPI-{index + 1:03d}",
            name=candidate.get("name") or _default_kpi_name(operation),
            operation=DslOperation.model_validate(operation),
        ))

    tests = data.get("statistical_tests") or []
    allowed_tests: List[str] = []
    if not isinstance(tests, list):
        errors.append("'statistical_tests' must be a list")
        tests = []
    for test in tests:
        if not isinstance(test, str):
            errors.append(f"statistical test must be a string, got {type(test).__name__}")
            continue
        if test in ALLOWED_STATISTICAL_TESTS:
            allowed_tests.append(test)
        else:
            errors.append(f"unknown statistical test '{test}' (allowed: "
                          f"{', '.join(sorted(ALLOWED_STATISTICAL_TESTS))})")

    plan = AnalysisPlan(
        candidate_kpis=kpis,
        statistical_tests=allowed_tests,
        has_temporal_data=bool(data.get("has_temporal_data", False)),
        limitations=list(data.get("limitations") or []),
    )
    return plan, errors


def apply_role_overrides(facts: List[ColumnFacts],
                         overrides: Dict[str, str]
                         ) -> List[ColumnUnderstanding]:
    """Merge LLM reclassification with Python-authoritative role rules.

    Python rules are authoritative. The LLM may only reclassify a column
    into one of its `alternate_roles` (the ambiguous cases from the §2.2
    table) or into "identifier" by name/semantics (e.g. numeric zip_code);
    any other override is rejected and the rule-based role kept.

    Every accepted/rejected override is logged with its reason (§8 audit):
    the returned objects carry `override_source` + `override_reason`.
    """
    columns: List[ColumnUnderstanding] = []
    for f in facts:
        role = f.suggested_role
        override = overrides.get(f.name)
        if override is not None:
            allowed = {f.suggested_role, *f.alternate_roles, "identifier"}
            if _is_numeric(f.dtype):
                allowed |= {"measure", "categorical"}
            elif _is_temporal(f.dtype):
                allowed |= {"temporal", "dimension"}
            else:
                allowed |= {"dimension", "categorical", "free_text",
                            "temporal"}
            if override in allowed and override in ColumnRole.__args__:
                role = override
                reason = (f"accepted: '{f.suggested_role}' -> '{override}' "
                          f"(in alternate_roles or identifier)")
            else:
                reason = (f"rejected: '{override}' not allowed for "
                          f"'{f.suggested_role}' — rule-based role kept")
        else:
            reason = "no override — rule-based role kept"
        # Task A: a heuristic-identified identifier logs its signals, so
        # every identifier decision is auditable in the understanding JSON.
        if (override is None and f.suggested_role == "identifier"
                and f.identifier_score > 0 and not f.identifier_signals):
            reason = (f"identifier heuristic: name pattern match "
                      f"(score {f.identifier_score:.2f})")
        elif (override is None and f.suggested_role == "identifier"
                and f.identifier_score > 0):
            reason = ("identifier heuristic: "
                      + "; ".join(f.identifier_signals)
                      + f" (score {f.identifier_score:.2f})")
        columns.append(ColumnUnderstanding(
            name=f.name, role=role, dtype=f.dtype,
            nunique=f.nunique, nullable=f.nullable,
            override_source="llm" if override is not None else "rules",
            override_reason=reason,
            identifier_score=f.identifier_score,
            ordinal=f.ordinal,
        ))
    return columns


DOMAIN_KEYWORDS: Dict[str, List[str]] = {
    "sales": ["sale", "revenue", "order", "product", "invoice", "customer"],
    "finance": ["finance", "account", "expense", "budget", "payment", "invoice"],
    "hr": ["employee", "salary", "hiring", "hr ", "attrition", "staff"],
    "marketing": ["campaign", "marketing", "advertis", "lead", "click"],
    "retail": ["retail", "store", "inventory", "sku", "shop"],
    "healthcare": ["patient", "clinic", "hospital", "diagnos", "treatment"],
    "logistics": ["shipment", "logistics", "delivery", "warehouse", "route"],
}


def detect_domain_heuristic(context: BusinessContext) -> tuple[str, float]:
    """Deterministic domain guess from the business context answers.

    Scans the goal summary + answers for domain keywords. Returns
    ("generic", 0.0) in Generic Mode or when nothing matches — the LLM crew
    path may refine this with the same keyword hints.
    """
    if context.generic_mode:
        return "generic", 0.0
    text = " ".join([context.goal_summary, *context.answers.values(),
                     *context.business_questions]).lower()
    best, best_score = None, 0
    for domain, keywords in DOMAIN_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in text)
        if score > best_score:
            best, best_score = domain, score
    if best and best_score > 0:
        return best, min(0.5 + 0.1 * best_score, 0.9)
    return "generic", 0.0


def default_plan(profile: DataProfile,
                 df: Union["pd.DataFrame", None] = None) -> AnalysisPlan:
    """Deterministic fallback plan — used by the non-LLM path and when the
    crew fails to produce a valid plan. Whitelist ops only, by construction.

    ``df`` (optional) enables the value-shape identifier/code heuristics
    (A.1/2.2) so the plan never proposes aggregations on columns that the
    df-aware profiler classified as identifiers or codes.
    """
    facts = ColumnProfiler().profile_columns(profile, df=df)
    measures = [f.name for f in facts if f.suggested_role == "measure"]
    identifiers = [f.name for f in facts if f.suggested_role == "identifier"]
    temporals = [f.name for f in facts if f.suggested_role == "temporal"]
    dimensions = [f.name for f in facts
                  if f.suggested_role in ("dimension", "categorical")]

    kpis: List[KpiCandidate] = []
    for col in measures:
        kpis.append(KpiCandidate(
            kpi_id=f"KPI-{len(kpis) + 1:03d}", name=f"Total {col}",
            operation=DslOperation(function="sum", column=col)))
        kpis.append(KpiCandidate(
            kpi_id=f"KPI-{len(kpis) + 1:03d}", name=f"Average {col}",
            operation=DslOperation(function="mean", column=col)))
    if identifiers:
        kpis.append(KpiCandidate(
            kpi_id=f"KPI-{len(kpis) + 1:03d}",
            name=f"{identifiers[0]} count",
            operation=DslOperation(function="count", column=identifiers[0])))
    if measures and temporals:
        # Only compute growth when the temporal column is a time-bucket
        # (fewer unique values than rows, e.g. month, quarter) — skip for
        # point-in-time columns like hire_date where growth is meaningless.
        for tcol in temporals:
            tunique = profile.nunique.get(tcol, 0)
            if tunique < profile.row_count * 0.8:
                kpis.append(KpiCandidate(
                    kpi_id=f"KPI-{len(kpis) + 1:03d}",
                    name=f"{measures[0]} YoY growth",
                    operation=DslOperation(function="growth",
                                           column=measures[0],
                                           over_column=tcol,
                                           period="YoY")))
                break
    if len(measures) >= 2:
        # Limit to top-3 correlation pairs to avoid redundant evidence.
        pairs_added = 0
        for i in range(len(measures)):
            if pairs_added >= 3:
                break
            for j in range(i + 1, len(measures)):
                if pairs_added >= 3:
                    break
                kpis.append(KpiCandidate(
                    kpi_id=f"KPI-{len(kpis) + 1:03d}",
                    name=f"Correlation {measures[i]} x {measures[j]}",
                    operation=DslOperation(function="correlation",
                                           column_a=measures[i],
                                           column_b=measures[j],
                                           method="pearson")))
                pairs_added += 1

    tests = ["descriptive"]
    if len(measures) >= 2:
        tests.append("correlation")
    if temporals:
        tests.append("trend")
    if dimensions and measures:
        tests.append("anova")

    return AnalysisPlan(
        candidate_kpis=kpis,
        statistical_tests=tests,
        has_temporal_data=bool(temporals),
    )


def assemble_understanding(profile: DataProfile,
                           facts: List[ColumnFacts],
                           role_overrides: Dict[str, str],
                           domain: tuple[str, float, List[str]],
                           context: BusinessContext,
                           limitations: List[str]) -> DatasetUnderstanding:
    """Assemble metadata/dataset_understanding.json from all inputs.

    `domain` is (detected_domain, confidence, entities) — Python-validated
    before it gets here.
    """
    columns = apply_role_overrides(facts, role_overrides)
    detected_domain, confidence, entities = domain
    return DatasetUnderstanding(
        detected_domain=detected_domain,
        domain_confidence=float(confidence),
        entities=list(entities),
        temporal_columns=[c.name for c in columns if c.role == "temporal"],
        dimensions=[c.name for c in columns
                    if c.role in ("dimension", "categorical")],
        measures=[c.name for c in columns if c.role == "measure"],
        identifiers=[c.name for c in columns if c.role == "identifier"],
        columns=columns,
        has_temporal_data=any(c.role == "temporal" for c in columns),
        limitations=list(limitations),
    )
