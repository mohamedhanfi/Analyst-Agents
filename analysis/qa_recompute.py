"""Stage 8 QA — deterministic KPI recomputation and reference validation.

QA recomputes 100% of KPIs from the cleaned CSV using the same DSL
executor the analysis stage used.  Python is authoritative; any
discrepancy beyond tolerance is a critical finding.

CLI: python -m analysis.qa_recompute <run_dir>
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from analysis.dsl_executor import execute_plan
from analysis.evidence import EvidenceRegistry

_TOLERANCE = 0.0001  # 0.01%


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
