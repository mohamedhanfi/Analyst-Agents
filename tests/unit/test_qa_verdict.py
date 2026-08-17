"""Stage 8 QA — score + verdict tests."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from analysis.qa_recompute import QaCheck
from analysis.qa_verdict import build_qa_verdict, compute_score, decide_verdict


def _checks(*severity_counts: tuple) -> list[QaCheck]:
    """Build a list of QaCheck from (severity, count) tuples."""
    checks = []
    for sev, n in severity_counts:
        for i in range(n):
            checks.append(QaCheck(check=f"c_{sev}_{i}", severity=sev,
                                  message=f"msg {sev} {i}"))
    return checks


# ---------------------------------------------------------------------------
# compute_score
# ---------------------------------------------------------------------------


def test_score_clean() -> None:
    assert compute_score([]) == 100.0


def test_score_one_critical() -> None:
    checks = _checks(("critical", 1))
    assert compute_score(checks) == 85.0


def test_score_one_warning() -> None:
    checks = _checks(("warning", 1))
    assert compute_score(checks) == 97.5


def test_score_one_info() -> None:
    checks = _checks(("info", 1))
    assert compute_score(checks) == 99.5


def test_score_mixed() -> None:
    checks = _checks(("critical", 2), ("warning", 3), ("info", 1))
    # 100 - 30 - 7.5 - 0.5 = 62.0
    assert compute_score(checks) == 62.0


def test_score_floor_zero() -> None:
    checks = _checks(("critical", 10))
    assert compute_score(checks) == 0.0


# ---------------------------------------------------------------------------
# decide_verdict
# ---------------------------------------------------------------------------


def test_verdict_approved_clean() -> None:
    assert decide_verdict([]) == "APPROVED"


def test_verdict_approved_with_warnings() -> None:
    checks = _checks(("warning", 2))
    assert decide_verdict(checks) == "APPROVED_WITH_WARNINGS"


def test_verdict_needs_revision_critical() -> None:
    checks = _checks(("critical", 1))
    assert decide_verdict(checks) == "NEEDS_REVISION"


def test_verdict_needs_revision_fallback_reason() -> None:
    assert decide_verdict([], ["fallback_used"]) == "NEEDS_REVISION"


def test_verdict_needs_revision_limit_exceeded() -> None:
    assert decide_verdict([], ["cleaning_retry_limit_exceeded"]) == "NEEDS_REVISION"


def test_verdict_needs_revision_invalid_evidence() -> None:
    assert decide_verdict([], ["invalid_evidence_refs"]) == "NEEDS_REVISION"


def test_verdict_critical_overrides_warnings() -> None:
    checks = _checks(("warning", 5), ("critical", 1))
    assert decide_verdict(checks) == "NEEDS_REVISION"


def test_verdict_no_false_positive() -> None:
    checks = _checks(("info", 3))
    assert decide_verdict(checks) == "APPROVED"


# ---------------------------------------------------------------------------
# build_qa_verdict
# ---------------------------------------------------------------------------


def test_build_verdict_clean() -> None:
    v = build_qa_verdict([])
    assert v.verdict == "APPROVED"
    assert v.score == 100.0
    assert v.critical == []
    assert v.warnings == []


def test_build_verdict_with_issues() -> None:
    checks = [
        QaCheck(check="kpi_mismatch", severity="critical",
                message="KPI K1 mismatch"),
        QaCheck(check="chart_missing", severity="warning",
                message="Chart file missing"),
    ]
    v = build_qa_verdict(checks, ["fallback_used"])
    assert v.verdict == "NEEDS_REVISION"
    assert v.score == 82.5  # 100 - 15 - 2.5
    assert len(v.critical) == 1
    assert len(v.warnings) == 1
    assert "fallback_used" in v.reason_codes


def test_build_verdict_model_dump_roundtrip() -> None:
    v = build_qa_verdict([QaCheck(check="x", severity="warning", message="y")])
    d = v.model_dump()
    assert d["verdict"] == "APPROVED_WITH_WARNINGS"
    assert d["score"] == 97.5
    v2 = type(v).model_validate(d)
    assert v2.verdict == v.verdict
    assert v2.score == v.score


# ---------------------------------------------------------------------------
# run_verdict (integration with disk)
# ---------------------------------------------------------------------------


def test_run_verdict_empty_dir(tmp_path: Path) -> None:
    from analysis.qa_verdict import run_verdict
    run_dir = tmp_path / "run"
    (run_dir / "outputs").mkdir(parents=True)
    (run_dir / "metadata").mkdir(parents=True)
    (run_dir / "knowledge").mkdir(parents=True)

    v = run_verdict(run_dir)
    assert v.verdict == "NEEDS_REVISION"  # critical: no KPIs recomputed
    out_file = run_dir / "metadata" / "qa_verdict.json"
    assert out_file.exists()
    data = json.loads(out_file.read_text(encoding="utf-8"))
    assert data["verdict"] == "NEEDS_REVISION"
