"""Stage 8 QA — deterministic KPI recomputation and reference validation.

QA recomputes 100% of KPIs from the cleaned CSV using the same DSL
executor the analysis stage used.  Python is authoritative; any
discrepancy beyond tolerance is a critical finding.

CLI: python -m analysis.qa_recompute <run_dir>
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from analysis.dsl_executor import execute_plan
from analysis.evidence import EvidenceRegistry
from shared.core.semantic_guards import IDENTIFIER_NAME_RE

_TOLERANCE = 0.0001  # 0.01%

# 8.1: functions that value-aggregate — never allowed on identifier columns.
_VALUE_FUNCTIONS = frozenset({"sum", "mean", "median", "std", "min", "max",
                              "correlation"})


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class QaCheck:
    """One QA check result."""
    check: str
    severity: str  # "critical" | "warning" | "info"
    message: str


# ---------------------------------------------------------------------------
# KPI recomputation
# ---------------------------------------------------------------------------


def _load_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default if default is not None else {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return default if default is not None else {}


def _op_key(op: Dict[str, Any]) -> str:
    """Deterministic key for an operation (for matching reported ↔ recomputed)."""
    parts = [
        str(op.get("function", "")),
        str(op.get("column", "")),
        str(op.get("column_a", "")),
        str(op.get("column_b", "")),
        str(op.get("method", "")),
        str(op.get("over_column", "")),
        str(op.get("period", "")),
        str(op.get("group_by", "")),
        str(op.get("filter", "")),
    ]
    return "|".join(parts)


def recompute_kpis(run_dir: Path) -> Dict[str, Any]:
    """Recompute all KPIs from cleaned data + analysis plan.

    Returns ``{"recomputed": {kpi_id: value}, "errors": [...]}``.
    """
    cleaned = run_dir / "data" / "processed" / "cleaned_data.csv"
    if not cleaned.exists():
        return {"recomputed": {}, "errors": [f"cleaned data missing: {cleaned}"]}

    plan_path = run_dir / "metadata" / "analysis_plan.json"
    plan = _load_json(plan_path)
    if not plan or not plan.get("candidate_kpis"):
        return {"recomputed": {}, "errors": ["analysis_plan.json missing or empty"]}

    try:
        df = pd.read_csv(cleaned, encoding="utf-8-sig")
    except Exception as exc:  # noqa: BLE001
        return {"recomputed": {}, "errors": [f"failed to read cleaned CSV: {exc}"]}

    registry = EvidenceRegistry(file_hash="qa_recompute", sheet=None,
                                transformations=["cleaned_data"])
    try:
        results = execute_plan(df, plan, registry)
    except Exception as exc:  # noqa: BLE001
        return {"recomputed": {}, "errors": [f"execute_plan failed: {exc}"]}

    recomputed: Dict[str, Any] = {}
    for r in results:
        if r.value is not None:
            recomputed[r.kpi_id] = {
                "value": r.value,
                "name": r.name,
                "operation": r.operation.model_dump()
                if hasattr(r.operation, "model_dump")
                else (r.operation if isinstance(r.operation, dict) else {}),
            }
    return {"recomputed": recomputed, "errors": []}


def compare_kpis(reported: List[Dict[str, Any]],
                 recomputed: Dict[str, Any],
                 tolerance: float = _TOLERANCE) -> List[QaCheck]:
    """Compare reported KPIs to recomputed values.

    Matching is by operation signature (function + column + params), not
    by kpi_id, because IDs may differ between runs.
    """
    checks: List[QaCheck] = []
    if not recomputed:
        checks.append(QaCheck(
            check="kpi_recomputation",
            severity="critical",
            message="no KPIs could be recomputed"))
        return checks

    # Build operation-key → recomputed lookup
    recomp_by_op: Dict[str, Any] = {}
    for kid, info in recomputed.items():
        key = _op_key(info.get("operation", {}))
        recomp_by_op[key] = info

    matched = 0
    for kpi in reported:
        op = kpi.get("operation", {})
        if isinstance(op, dict):
            op_key = _op_key(op)
        else:
            op_key = _op_key(op.model_dump() if hasattr(op, "model_dump") else {})

        reported_val = kpi.get("value")
        if reported_val is None:
            checks.append(QaCheck(
                check="kpi_null_value",
                severity="warning",
                message=f"KPI {kpi.get('kpi_id', '?')} has null value"))
            continue

        if op_key not in recomp_by_op:
            checks.append(QaCheck(
                check="kpi_not_recomputed",
                severity="warning",
                message=f"KPI {kpi.get('kpi_id', '?')} ({kpi.get('name', '')}) "
                        f"has no matching recomputation"))
            continue

        matched += 1
        recomp_val = recomp_by_op[op_key]["value"]
        try:
            reported_f = float(reported_val)
            recomp_f = float(recomp_val)
        except (TypeError, ValueError):
            checks.append(QaCheck(
                check="kpi_type_error",
                severity="critical",
                message=f"KPI {kpi.get('kpi_id', '?')}: cannot compare "
                        f"reported={reported_val!r} vs recomputed={recomp_val!r}"))
            continue

        base = max(abs(reported_f), abs(recomp_f), 1.0)
        diff_pct = abs(reported_f - recomp_f) / base
        if diff_pct > tolerance:
            checks.append(QaCheck(
                check="kpi_mismatch",
                severity="critical",
                message=f"KPI {kpi.get('kpi_id', '?')} ({kpi.get('name', '')}): "
                        f"reported={reported_f} vs recomputed={recomp_f} "
                        f"(diff {diff_pct:.4%} > tolerance {tolerance:.4%})"))

    if not reported:
        checks.append(QaCheck(
            check="kpi_no_reported",
            severity="warning",
            message="no KPIs in kpis.json to compare"))
    elif matched == 0 and not any(c.severity == "critical" for c in checks):
        checks.append(QaCheck(
            check="kpi_no_match",
            severity="warning",
            message=f"0 of {len(reported)} reported KPIs matched a recomputed op"))

    return checks


# ---------------------------------------------------------------------------
# Reference validation
# ---------------------------------------------------------------------------


def validate_references(run_dir: Path) -> List[QaCheck]:
    """Check cross-artifact reference integrity."""
    checks: List[QaCheck] = []
    out = run_dir / "outputs"

    # Load artifacts
    evidence = _load_json(out / "evidence_registry.json", [])
    if isinstance(evidence, dict):
        evidence = evidence.get("entries", [])
    evidence_ids = {e.get("evidence_id") for e in evidence if isinstance(e, dict)}

    kpis = _load_json(out / "kpis.json", {"kpis": []})
    if isinstance(kpis, dict):
        kpis = kpis.get("kpis", [])

    insights_raw = _load_json(out / "insights.json",
                              {"insights": [], "recommendations": []})
    insights = insights_raw.get("insights", []) if isinstance(insights_raw, dict) else []
    recs = insights_raw.get("recommendations", []) if isinstance(insights_raw, dict) else []

    # Check KPI evidence refs
    for kpi in kpis:
        eid = kpi.get("evidence_id")
        if eid and eid not in evidence_ids:
            checks.append(QaCheck(
                check="kpi_evidence_ref",
                severity="critical",
                message=f"KPI {kpi.get('kpi_id', '?')} references "
                        f"evidence {eid!r} not in registry"))

    # Check insight evidence refs
    for ins in insights:
        for eid in ins.get("evidence_ids", []):
            if eid and eid not in evidence_ids:
                checks.append(QaCheck(
                    check="insight_evidence_ref",
                    severity="critical",
                    message=f"Insight {ins.get('insight_id', '?')} references "
                            f"evidence {eid!r} not in registry"))

    # Check recommendation → insight refs
    insight_ids = {i.get("insight_id") for i in insights}
    for rec in recs:
        ref = rec.get("insight_id")
        if ref and ref not in insight_ids:
            checks.append(QaCheck(
                check="rec_insight_ref",
                severity="critical",
                message=f"Recommendation {rec.get('recommendation_id', '?')} "
                        f"references insight {ref!r} not in insights"))

    # Check chart files exist
    charts_meta = _load_json(run_dir / "metadata" / "chart_metadata.json",
                             {"charts": []})
    charts = charts_meta.get("charts", []) if isinstance(charts_meta, dict) else []
    charts_dir = run_dir / "outputs" / "charts"
    for ch in charts:
        path = ch.get("chart_path")
        if path:
            full = run_dir / path
            if not full.exists():
                full = charts_dir / f"{ch.get('chart_id', '')}.svg"
                if not full.exists():
                    checks.append(QaCheck(
                        check="chart_file_missing",
                        severity="warning",
                        message=f"Chart {ch.get('chart_id', '?')} file "
                                f"not found at {path}"))

    # Check report.html exists and has sections
    report_path = run_dir / "report.html"
    if not report_path.exists():
        checks.append(QaCheck(
            check="report_missing",
            severity="warning",
            message="report.html not found"))
    else:
        html = report_path.read_text(encoding="utf-8")
        for sid in ["s1", "s2", "s3", "s4", "s5", "s6"]:
            if f'id="{sid}"' not in html:
                checks.append(QaCheck(
                    check="report_section_missing",
                    severity="warning",
                    message=f"report.html missing section #{sid}"))

    return checks


# ---------------------------------------------------------------------------
# Semantic relevance (8.1) — hard-fail, independent of numeric accuracy
# ---------------------------------------------------------------------------


def check_semantic_relevance(run_dir: Path) -> List[QaCheck]:
    """8.1: last line of defense — no surfaced KPI/insight may value-
    aggregate an identifier-like column, regardless of how accurate the
    numbers are. Critical severity -> NEEDS_REVISION (§2.8)."""
    checks: List[QaCheck] = []
    understanding = _load_json(run_dir / "metadata"
                               / "dataset_understanding.json")
    columns = understanding.get("columns") or []
    idents = {str(c.get("name")) for c in columns
              if c.get("role") == "identifier"}

    def _flagged(col: Any) -> bool:
        if not col:
            return False
        return col in idents or bool(IDENTIFIER_NAME_RE.search(str(col)))

    kpis = _load_json(run_dir / "outputs" / "kpis.json", {"kpis": []})
    kpis = kpis.get("kpis", []) if isinstance(kpis, dict) else []
    kpi_ops: Dict[str, Dict[str, Any]] = {}
    for kpi in kpis:
        op = kpi.get("operation") or {}
        kpi_ops[kpi.get("kpi_id")] = op
        if op.get("function") not in _VALUE_FUNCTIONS:
            continue
        cols = [op.get("column"), op.get("column_a"), op.get("column_b")]
        flagged = [c for c in cols if _flagged(c)]
        if flagged:
            checks.append(QaCheck(
                check="semantic_identifier_reference",
                severity="critical",
                message=(f"KPI {kpi.get('kpi_id', '?')} "
                         f"({kpi.get('name', '')}) value-aggregates "
                         f"identifier-like column '{flagged[0]}'")))

    insights_raw = _load_json(run_dir / "outputs" / "insights.json",
                              {"insights": []})
    insights = insights_raw.get("insights", []) \
        if isinstance(insights_raw, dict) else []
    for ins in insights:
        for kpi_id in ins.get("related_kpis") or []:
            op = kpi_ops.get(kpi_id) or {}
            if op.get("function") not in _VALUE_FUNCTIONS:
                continue
            cols = [op.get("column"), op.get("column_a"),
                    op.get("column_b")]
            flagged = [c for c in cols if _flagged(c)]
            if flagged:
                checks.append(QaCheck(
                    check="semantic_identifier_reference",
                    severity="critical",
                    message=(f"Insight {ins.get('insight_id', '?')} rests "
                             f"on KPI {kpi_id} which value-aggregates "
                             f"identifier-like column '{flagged[0]}'")))

    return checks


# ---------------------------------------------------------------------------
# Narrative consistency — text claims vs dataset facts (§2.8 gap fix)
# ---------------------------------------------------------------------------

# Catches invented claims like "lack of time-stamped data" when temporal
# columns exist. The gap between the negation and the temporal word may not
# contain "missing"/"gap(s)", so honest sentences ("no missing dates")
# never match. Hyphens include Unicode variants (U+2010–U+2015) because
# LLMs sometimes emit non-breaking hyphens ("time‐stamped").
_TEMPORAL_DENIAL_RE = re.compile(
    r"\b(?:lack\w*|no|none|without|absent)\b"
    r"(?:(?!\b(?:missing|gaps?)\b)[^.]){0,60}"
    r"\b(?:time[\u2010-\u2015 -]?stamp(?:ed)?|timestamps?|dates?"
    r"|date\s+columns?|temporal\s+(?:data|columns?))\b",
    re.IGNORECASE)
_NEGATION_RE = re.compile(r"\b(?:no|none|zero|not)\b", re.IGNORECASE)
_REMOVAL_VERB_RE = re.compile(
    r"\b(?:remov\w+|dropp\w+|delet\w+|exclud\w+)\b", re.IGNORECASE)


def _cleaning_removed_rows(cleaning: Dict[str, Any],
                           lineage: Dict[str, Any]) -> int:
    """Rows removed by preparation steps (dedup/drop_*); lineage first."""
    total = 0
    steps = (lineage or {}).get("steps")
    for step in steps if isinstance(steps, list) else []:
        if not isinstance(step, dict):
            continue
        for op in step.get("ops") or []:
            if not isinstance(op, dict):
                continue
            if str(op.get("op", "")) in ("dedup", "drop_negative",
                                         "drop_missing", "drop_outlier"):
                try:
                    total += int(op.get("rows_affected") or 0)
                except (TypeError, ValueError):
                    continue
    if total:
        return total
    if not isinstance(cleaning, dict):
        return 0
    try:
        before = int(cleaning.get("rows_before"))
        after = int(cleaning.get("rows_after"))
    except (TypeError, ValueError):
        return 0
    return max(0, before - after)


def check_report_consistency(run_dir: Path) -> List[QaCheck]:
    """Cross-check narrative text against dataset facts.

    Numeric QA cannot catch LLM-invented claims such as 'lack of
    time-stamped data' while a date column exists, or 'no rows were
    removed' while preparation dropped rows. These deterministic checks
    flag such contradictions at warning level."""
    checks: List[QaCheck] = []
    text = ""
    summary_path = run_dir / "outputs" / "exec_summary.txt"
    if summary_path.is_file():
        try:
            text = summary_path.read_text(encoding="utf-8")
        except OSError:
            text = ""
    if not text.strip():
        return checks

    # 1. Temporal denial vs actual temporal columns
    understanding = _load_json(run_dir / "metadata"
                               / "dataset_understanding.json")
    temporal_cols = [c for c in (understanding.get("temporal_columns")
                                 or []) if c]
    has_temporal = bool(understanding.get("has_temporal_data")) \
        or bool(temporal_cols)
    if has_temporal and _TEMPORAL_DENIAL_RE.search(text):
        checks.append(QaCheck(
            check="exec_summary_temporal_contradiction",
            severity="warning",
            message=("executive summary denies time-stamped data but the "
                     "dataset has temporal column(s): "
                     + (", ".join(str(c) for c in temporal_cols)
                        or "detected"))))

    # 2. "Nothing was removed" claims vs rows actually dropped
    suspect = next((s for s in re.split(r"(?<=[.!?])\s+", text)
                    if _NEGATION_RE.search(s) and _REMOVAL_VERB_RE.search(s)
                    and not re.search(r"\d", s)), None)
    if suspect:
        removed = _cleaning_removed_rows(
            _load_json(run_dir / "metadata" / "cleaning_result.json"),
            _load_json(run_dir / "metadata" / "lineage.json"))
        if removed > 0:
            checks.append(QaCheck(
                check="exec_summary_cleaning_contradiction",
                severity="warning",
                message=("executive summary claims no data was removed but "
                         f"preparation dropped {removed} row(s)")))
    return checks


# ---------------------------------------------------------------------------
# Full QA recomputation + validation
# ---------------------------------------------------------------------------


def run_all_checks(run_dir: Path) -> List[QaCheck]:
    """Run all deterministic QA checks and return combined results."""
    run_dir = Path(run_dir)
    checks: List[QaCheck] = []

    # 1. Recompute KPIs
    kpis_raw = _load_json(run_dir / "outputs" / "kpis.json", {"kpis": []})
    reported_kpis = kpis_raw.get("kpis", []) if isinstance(kpis_raw, dict) else []
    recomp = recompute_kpis(run_dir)
    checks.extend(compare_kpis(reported_kpis, recomp["recomputed"]))
    for err in recomp["errors"]:
        checks.append(QaCheck(
            check="kpi_recomputation_error",
            severity="critical",
            message=err))

    # 2. Validate references
    checks.extend(validate_references(run_dir))

    # 3. Semantic relevance hard-fail (8.1)
    checks.extend(check_semantic_relevance(run_dir))

    # 4. Narrative consistency vs dataset facts
    checks.extend(check_report_consistency(run_dir))

    return checks


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="qa_recompute",
        description="Deterministic QA: recompute KPIs + validate references")
    parser.add_argument("run_dir", help="run directory")
    args = parser.parse_args(argv)

    checks = run_all_checks(Path(args.run_dir))
    result = [
        {"check": c.check, "severity": c.severity, "message": c.message}
        for c in checks
    ]
    print(json.dumps(result, ensure_ascii=False, indent=2))
    crits = sum(1 for c in checks if c.severity == "critical")
    warns = sum(1 for c in checks if c.severity == "warning")
    print(f"\n{len(checks)} checks: {crits} critical, {warns} warning")
    return 1 if crits else 0


if __name__ == "__main__":
    sys.exit(main())
