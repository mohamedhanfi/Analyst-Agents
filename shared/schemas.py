"""Pydantic models for every JSON artifact (§2).

Defining the contracts once so agents, tools and QA stay aligned.
"""
from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Stage 1 — Ingestion (§2.1)
# ---------------------------------------------------------------------------


class SheetInfo(BaseModel):
    """One XLSX worksheet discovered by FileReader (metadata only)."""
    name: str
    row_count: int          # includes the header row (matches openpyxl max_row)


class ReadResult(BaseModel):
    """Result of FileReader.read — metadata, never raw data.

    CSV: extracted immediately (single sheet) -> extracted_path set.
    XLSX: discovery only -> sheets populated, extracted_path None until
    FileReader.extract_sheet() is called for the selected sheet.
    """
    file_type: Literal["csv", "xlsx"]
    file_name: str
    file_hash: str                          # "sha256:<hex>"
    row_count: int | None = None
    column_count: int | None = None
    separator: str | None = None            # csv only
    encoding: str | None = None             # csv only
    sheets: List[SheetInfo] = Field(default_factory=list)
    selected_sheet: str | None = None       # xlsx only, after extraction
    extracted_path: str | None = None


class ValidationError(BaseModel):
    """One failed check inside ValidationResult."""
    code: str
    message: str
    details: str | None = None


class ValidationResult(BaseModel):
    """Deep file validation: extension + MIME + signature + parser."""
    file_name: str
    file_path: str
    validation_status: Literal["passed", "failed"]
    extension_ok: bool = False
    rows_ok: bool = False
    size_ok: bool = False
    signature_ok: bool = False
    parser_ok: bool = False
    row_count: int | None = None
    file_size_mb: float | None = None
    errors: List[ValidationError] = Field(default_factory=list)


class DataProfile(BaseModel):
    """metadata/data_profile.json — shape, types, nulls, dups, PII."""
    file_name: str
    file_hash: str                                # "sha256:<hex>"
    row_count: int
    column_count: int
    columns: List[str]
    column_types: Dict[str, str]                  # col -> pandas dtype name
    missing_values: Dict[str, int] = Field(default_factory=dict)
    duplicate_rows: int = 0
    pii_columns: List[str] = Field(default_factory=list)
    nunique: Dict[str, int] = Field(default_factory=dict)
    sample: List[Dict[str, Any]] = Field(default_factory=list)   # 20 rows, PII redacted
    validation_status: Literal["passed", "failed"] = "passed"


class BusinessContext(BaseModel):
    """knowledge/business_context.json — user answers, or Generic Mode."""
    file_name: str
    sheet_used: Optional[str] = None              # which sheet was analysed
    business_questions: List[str] = Field(default_factory=list)
    answers: Dict[str, str] = Field(default_factory=dict)
    goal_summary: str = ""
    context_confidence: float = 0.0               # 0.0 => Generic Mode (§3.5)
    generic_mode: bool = False                    # true when user timed out


# ---------------------------------------------------------------------------
# Stage 2 — Understanding (§2.2)
# ---------------------------------------------------------------------------

ColumnRole = Literal["identifier", "temporal", "measure",
                     "categorical", "dimension", "free_text"]


class ColumnUnderstanding(BaseModel):
    name: str
    role: ColumnRole
    dtype: str
    nunique: int
    nullable: bool


class DatasetUnderstanding(BaseModel):
    """metadata/dataset_understanding.json"""
    detected_domain: str
    domain_confidence: float
    entities: List[str] = Field(default_factory=list)
    temporal_columns: List[str] = Field(default_factory=list)
    dimensions: List[str] = Field(default_factory=list)
    measures: List[str] = Field(default_factory=list)
    identifiers: List[str] = Field(default_factory=list)
    columns: List[ColumnUnderstanding] = Field(default_factory=list)
    has_temporal_data: bool = False
    limitations: List[str] = Field(default_factory=list)


class DslOperation(BaseModel):
    function: Literal["sum", "mean", "median", "count", "nunique", "min",
                      "max", "std", "growth", "correlation", "ratio"]
    column: Optional[str] = None
    column_a: Optional[str] = None                # correlation
    column_b: Optional[str] = None                # correlation
    method: Optional[Literal["pearson", "spearman"]] = None
    over_column: Optional[str] = None             # growth
    period: Optional[Literal["YoY", "MoM", "WoW"]] = None
    basis: Optional[Literal["previous_period", "start_of_period"]] = None  # None => executor applies "previous_period"
    as_percent: Optional[bool] = None
    group_by: Optional[List[str]] = None
    filter: Optional[Dict[str, Any]] = None
    numerator: Optional["DslOperation"] = None    # ratio
    denominator: Optional["DslOperation"] = None  # ratio


class KpiCandidate(BaseModel):
    kpi_id: str
    name: str
    operation: DslOperation


class AnalysisPlan(BaseModel):
    """metadata/analysis_plan.json — DSL ops only, no freeform formulas."""
    candidate_kpis: List[KpiCandidate] = Field(default_factory=list)
    statistical_tests: List[str] = Field(default_factory=list)
    has_temporal_data: bool = False
    limitations: List[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Stage 3 — Data Quality (§2.3)
# ---------------------------------------------------------------------------


class DataQualityReport(BaseModel):
    """metadata/data_quality_report.json"""
    status: Literal["passed", "needs_repair"]
    invalid: Dict[str, List[str]] = Field(default_factory=dict)
    missingness: Dict[str, Any] = Field(default_factory=dict)
    duplicates: int = 0
    issues: List[Dict[str, str]] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Stage 4 — Cleaning (§2.4)
# ---------------------------------------------------------------------------


class CleaningResult(BaseModel):
    """metadata/cleaning_result.json"""
    attempt: int = 1
    rows_before: int
    rows_after: int
    duplicates_removed: int = 0
    type_casts: Dict[str, str] = Field(default_factory=dict)
    flags_created: List[str] = Field(default_factory=list)
    outliers: Dict[str, int] = Field(default_factory=dict)
    status: Literal["passed", "failed"] = "passed"


# ---------------------------------------------------------------------------
# Stage 5 — Analysis (§2.5)
# ---------------------------------------------------------------------------

ChartKind = Literal["bar", "barh", "line", "doughnut",
                    "histogram", "scatter", "heatmap"]


class KpiResult(BaseModel):
    """One computed KPI in outputs/kpis.json.

    value is a float for measures, int for counts, and may be a str when a
    min/max lands on a temporal column; None when the computation failed.
    """
    kpi_id: str
    name: str
    operation: DslOperation
    value: float | int | str | None = None
    evidence_id: str | None = None
    computed_by: str = "pandas"


class StatisticalResult(BaseModel):
    """One result in outputs/statistical_results.json.

    extra carries test-specific fields (F, df, eta^2, groups compared, ...)
    that do not fit the common surface — tightened when the suite lands.
    """
    test_id: str
    category: str
    test_name: str
    variables: List[str] = Field(default_factory=list)
    statistic: float | None = None
    p_value: float | None = None
    ci_low: float | None = None
    ci_high: float | None = None
    effect_size: float | None = None
    n: int | None = None
    evidence_id: str | None = None
    extra: Dict[str, Any] = Field(default_factory=dict)


class ChartMetadata(BaseModel):
    """One planner decision in metadata/chart_metadata.json (shape rule hit)."""
    chart_id: str
    kind: ChartKind
    reason: str
    columns: List[str] = Field(default_factory=list)
    title: str = ""
    reliability: str | None = None     # "low_n" when the planner downgrades
    chart_path: str | None = None
    evidence_id: str | None = None
    computed_by: str = "pandas"


class EvidenceSource(BaseModel):
    """Full lineage of one computed value (§2.6)."""
    file_hash: str | None = None
    sheet: str | None = None
    transformations: List[str] = Field(default_factory=list)
    filter: str | None = None
    aggregation: str | None = None
    comparison: str | None = None
    result: Any | None = None


class EvidenceEntry(BaseModel):
    """One row of outputs/evidence_registry.json."""
    evidence_id: str
    source: EvidenceSource


# ---------------------------------------------------------------------------
# Stage 6 — Insights & Recommendations (§2.6)
# ---------------------------------------------------------------------------

ClaimType = Literal["DESCRIPTIVE", "COMPARATIVE", "CORRELATIONAL",
                    "PREDICTIVE", "CAUSAL"]


class Insight(BaseModel):
    """One evidence-grounded insight in outputs/insights.json."""
    insight_id: str
    claim_type: ClaimType
    title: str
    description: str
    confidence: Literal["high", "medium", "low"]
    evidence_ids: List[str] = Field(default_factory=list)
    required_evidence: List[str] = Field(default_factory=list)
    related_kpis: List[str] = Field(default_factory=list)


class Recommendation(BaseModel):
    """Hedged recommendation in outputs/insights.json (§2.6 chain)."""
    recommendation_id: str
    insight_id: str          # must reference an existing insight_id
    title: str
    description: str


# ---------------------------------------------------------------------------
# Stage 7 — Report (§2.7)
# ---------------------------------------------------------------------------


class ReportResult(BaseModel):
    """metadata/report_result.json"""
    status: Literal["rendered", "failed"]
    report_path: str | None = None
    locale: str = "en"
    sections: List[str] = Field(default_factory=list)
    error: str | None = None


# ---------------------------------------------------------------------------
# Stage 8 — QA (§2.8)
# ---------------------------------------------------------------------------


class QaVerdict(BaseModel):
    """metadata/qa_verdict.json — deterministic verdict + informational score.

    score is informational only; the verdict is decided purely by the
    logical conditions (§2.8).
    """
    verdict: Literal["APPROVED", "APPROVED_WITH_WARNINGS", "NEEDS_REVISION"]
    score: float
    critical: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    info: List[str] = Field(default_factory=list)
    reason_codes: List[str] = Field(default_factory=list)