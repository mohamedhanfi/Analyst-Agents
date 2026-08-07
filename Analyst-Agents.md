# Insight Forge — Implementation Guide (Production-Ready)

**Version:** 3.1.0
**For:** Development Team
**Changelog from 3.0.0:** Added run isolation, orchestration/interface layer, privacy redaction, resource limits, fixed retry semantics, added cleaning↔analysis validation loop, full-KPI QA recomputation, CSV injection protection, and file retention policy. See "Fixes Applied" callouts throughout.

---

## Golden Rule

> **The LLM decides WHAT to do. Python libraries DO the actual work.**

The LLM never calculates numbers, never draws charts, never cleans data, never reads files directly. It only:

- Decides which analysis to run
- Interprets results
- Writes insights and recommendations
- Generates report text

Everything computational is done by **pandas, numpy, matplotlib, scipy, openpyxl, jinja2**.

---

## 0. Technical Foundations (New)

This section did not exist in v3.0 and was the biggest gap — the agent tables described *what* each agent does, but not how the system actually runs, isolates data, or protects users. Read this before implementing any agent.

### 0.1 Orchestration

Don't reach for a heavy agent framework (LangGraph/CrewAI/AutoGen) for this pipeline — the flow is a **strict linear sequence with one conditional branch (QA verdict)**, not a dynamic graph. A heavy framework adds complexity without benefit here.

Use a simple custom orchestrator:

```
class Orchestrator:
    def run(self, run_id, file_path):
        state = PipelineState(run_id=run_id)
        for agent in [Ingestion, Understanding, Cleaning, Analyst, Insight, Report, QA]:
            result = agent.execute(state)
            state.update(agent.name, result)
            manifest.update(agent.name, result.status)
            if result.status == "stopped":
                break
        return state
```

- `PipelineState` is a single in-memory Pydantic object holding all context (Data Profile, Business Context, Analysis Plan, etc.) — this is what "passes as Context" in Section 8 actually *is*.
- Each agent reads what it needs from `state` and returns a typed result; the orchestrator is the only thing that writes to `state` and to the manifest.

### 0.2 Interface Layer

The original guide never says how a user actually triggers this. Define it explicitly:

- `POST /analyze` — accepts a file upload, creates a `run_id`, starts the pipeline as a background job, returns `{ "run_id": "..." }` immediately (this pipeline can take minutes; do not block the HTTP request).
- `GET /runs/{run_id}` — returns manifest status (`running`, `awaiting_input`, `completed`, `needs_revision`, `failed`).
- `POST /runs/{run_id}/answer` — submits the user's answer to a pending HumanInputTool question (see 0.3).
- `GET /runs/{run_id}/report` — returns the final HTML report once completed.

### 0.3 Run Isolation (Fixes Problem #1)

**v3.0 problem:** all agents read/write to fixed paths like `data/extracted/`, so two runs at the same time overwrite each other's data.

**Fix:** every path is namespaced under the run:

```
runs/
└── {run_id}/
    ├── knowledge/
    ├── data/
    │   ├── raw/
    │   ├── extracted/
    │   └── processed/
    └── outputs/
        ├── report.html
        ├── kpis.json
        ├── charts/
        └── metadata/
```

No agent ever writes outside its own `runs/{run_id}/` folder. This also makes cleanup (0.6) trivial — delete the folder.

### 0.4 Privacy & Redaction (Fixes Problem #3)

Sample rows (Agent 1 & 2) and any row-level values shown to the LLM go through a **Python redaction step before leaving the process**:

```
Step 1: Python scans sample rows with regex for: emails, phone numbers,
        national ID patterns, credit card numbers
Step 2: Detected values replaced with placeholders: "[EMAIL]", "[PHONE]"
Step 3: Redacted sample is what gets sent to the LLM — never the raw sample
Step 4: Original raw file stays local; only the redacted summary crosses
        the LLM API boundary
```

This matters more here than in a typical app because business datasets frequently contain customer PII in columns the pipeline doesn't even know it's touching (e.g., a "notes" free-text column).

### 0.5 Resource Limits (Fixes Problems #4 and #5)

| Limit | Value | Enforced By |
|-------|-------|-------------|
| Max file size | 50 MB | Ingestion, before reading |
| Max rows | 200,000 | Ingestion; if exceeded, analyze a random 200k-row sample and flag `"sampled": true` in Data Profile |
| Max sample rows sent to LLM | 15 rows | Understanding, Insight |
| Max charts per run | 8 | Analyst — LLM chart selection is capped; if it proposes more, Python truncates and logs a warning |
| Max KPIs per run | 12 | Analyst |
| LLM call timeout | 60s per call | All agents |
| Human input timeout | 5 minutes, then proceed with defaults (`industry: "unknown"`, no goals specified) and flag `"context_incomplete": true` | Ingestion |

### 0.6 File Retention

Uploaded files and run artifacts are deleted **24 hours after run completion** (or immediately after report delivery if the user has no account / isn't logged in). This is a default, not a suggestion — business data sitting indefinitely on disk is a real liability. Configure via `RUN_RETENTION_HOURS`.

### 0.7 CSV / Formula Injection Protection (Fixes Problem #9)

Before any user-supplied string value is written into `report.html` or any Excel-compatible export, Python sanitizes cells starting with `=`, `+`, `-`, `@` by prefixing a single quote — otherwise a malicious CSV upload (e.g., a `product` column containing `=cmd|'/c calc'!A1`) can execute when a downstream user opens an exported file in Excel.

```
def sanitize_cell(value: str) -> str:
    if value and value[0] in ('=', '+', '-', '@'):
        return "'" + value
    return value
```

Applied in: Ingestion (on read), Report (on render).

---

## Pipeline at a Glance

```
USER UPLOADS CSV / XLSX
        │
        ▼
   [1] INGESTION          ← Python reads file, LLM asks user questions
        │
        ▼
   [2] UNDERSTANDING      ← Python profiles data, LLM interprets domain
        │
        ▼
   [3] CLEANING           ← Python cleans data, LLM decides strategy
        │                    → re-validates Analysis Plan against
        │                      actually-remaining columns (Fix #7)
        ▼
   [4] ANALYSIS           ← Python computes everything, LLM selects what to compute
        │
        ▼
   [5] INSIGHTS           ← LLM interprets results, writes insights
        │
        ▼
   [6] REPORT             ← Python renders HTML, LLM writes summary
        │
        ▼
   [7] QA                 ← Python validates ALL numbers, LLM checks logic
        │
        ▼
   APPROVED / NEEDS REVISION
```

---

## Agent 1: Ingestion

### What it does
Receives the file, checks if it is CSV or XLSX, reads it, asks the user about their business, and produces a data profile.

### Who does what

| Task | Done By | How |
|------|---------|-----|
| Check file extension | **Python** | Simple string check |
| Check file size / row count limits | **Python** | Reject or sample per §0.5 |
| Read CSV | **Python** | `pandas.read_csv()` |
| Read XLSX + list sheets | **Python** | `openpyxl` or `pandas.read_excel(sheet_name=None)` |
| Ask user which sheet | **LLM + HumanInputTool** | LLM formulates question, tool displays it, 5-min timeout (§0.5) |
| Ask user business context | **LLM + HumanInputTool** | LLM asks industry, goals, KPIs |
| Merge sheets if needed | **Python** | `pandas.concat()` + add `source_sheet` column |
| Count rows, columns, nulls | **Python** | `df.shape`, `df.isnull().sum()` |
| Detect column types | **Python** | `df.dtypes` |
| Sanitize formula-injection cells | **Python** | Prefix `'` on cells starting with `=+-@` (§0.7) |
| Build Data Profile | **Python** | Structured dict from pandas results |

### Input
- Raw file from user (CSV or XLSX)
- User answers via HumanInputTool

### Output

| Artifact | Format | Saved To |
|----------|--------|----------|
| Extracted data | CSV | `runs/{run_id}/data/extracted/` |
| Data Profile | JSON | `runs/{run_id}/outputs/metadata/data_profile.json` |
| Business Context | JSON | `runs/{run_id}/knowledge/business_context.json` |

### Data Profile structure

```
{
  "file_name": "sales.xlsx",
  "file_type": "xlsx",
  "selected_sheets": ["Sales 2024"],
  "row_count": 2300,
  "column_count": 8,
  "columns": ["date", "product", "category", "revenue", "quantity", "city"],
  "column_types": {"date": "datetime64", "revenue": "float64"},
  "missing_values": {"city": 45, "category": 0},
  "duplicate_rows": 12,
  "sampled": false,
  "validation_status": "passed"
}
```

### Business Context structure

```
{
  "company_industry": "E-commerce",
  "business_type": "B2C",
  "target_audience": "Young adults",
  "analysis_goal": "Increase sales",
  "important_kpis": ["revenue", "order count"],
  "market_notes": "Weekend promotions drive sales",
  "context_incomplete": false
}
```

### If file is unsupported
Pipeline stops. Return error message to user. No further agents run.

### If file is empty or has < 5 rows
Pipeline stops. Return error message to user.

### If file exceeds size/row limits
File is randomly sampled to 200,000 rows (§0.5); `Data Profile.sampled = true`. Pipeline continues — this is a warning, not a stop condition.

---

## Agent 2: Data Understanding

### What it does
Looks at the data profile and a **redacted** sample of the actual data. Figures out what kind of dataset this is, what columns mean, and creates an Analysis Plan for downstream agents.

### Who does what

| Task | Done By | How |
|------|---------|-----|
| Get unique values per column | **Python** | `df.nunique()` |
| Get sample values | **Python** | `df.head()`, `df.sample()` — max 15 rows (§0.5) |
| Redact PII in sample | **Python** | Regex redaction before LLM sees it (§0.4) |
| Detect date columns | **Python** | `pd.api.types.is_datetime64_any_dtype()` |
| Detect numeric columns | **Python** | `df.select_dtypes(include='number')` |
| Detect high-cardinality columns | **Python** | `df.nunique() > threshold` |
| Classify column roles | **Python rules + LLM** | Python computes stats, LLM classifies based on stats |
| Detect business domain | **LLM** | LLM reads column names + business context, decides domain |
| Identify entities | **LLM** | LLM reasons about relationships |
| Detect time granularity | **Python** | Check min diff between dates |
| Propose candidate KPIs | **LLM** | LLM suggests KPIs based on domain + columns (max 12, §0.5) |
| Build Analysis Plan | **LLM** | LLM compiles everything into a plan |

### Column Role Classification Logic (Python)

```
For each column:
  if nunique == row_count → identifier
  if dtype is datetime → temporal
  if dtype is numeric and nunique > 20 → measure
  if dtype is numeric and nunique <= 20 → could be measure or categorical
  if dtype is object and nunique <= 20 → dimension
  if dtype is object and nunique > 50 → free_text
```

LLM reviews the Python classification and can adjust based on column names. Example: a numeric column named "zip_code" should be reclassified as identifier, not measure.

### Input
- Data Profile from Agent 1
- Business Context from Agent 1
- Redacted sample rows (Python reads first 15 rows, redacts, passes to LLM)

### Output

| Artifact | Format | Saved To |
|----------|--------|----------|
| Dataset Understanding | JSON | `runs/{run_id}/outputs/metadata/dataset_understanding.json` |
| Analysis Plan | JSON | `runs/{run_id}/outputs/metadata/analysis_plan.json` |

### Dataset Understanding structure

```
{
  "detected_domain": "Sales",
  "domain_confidence": 0.87,
  "entities": ["Product", "Customer", "Order"],
  "temporal_columns": ["order_date"],
  "time_granularity": "daily",
  "date_range": {"start": "2024-01-01", "end": "2025-06-30"},
  "dimensions": ["product", "category", "city"],
  "measures": ["revenue", "quantity"],
  "identifiers": ["order_id"],
  "column_roles": {
    "order_date": "temporal",
    "product": "dimension",
    "revenue": "measure",
    "order_id": "identifier"
  }
}
```

### Analysis Plan structure

```
{
  "domain": "Sales",
  "candidate_kpis": [
    {"id": "KPI-001", "name": "Total Revenue", "formula": "SUM(revenue)", "columns": ["revenue"]},
    {"id": "KPI-002", "name": "Order Count", "formula": "COUNT(order_id)", "columns": ["order_id"]},
    {"id": "KPI-003", "name": "Avg Order Value", "formula": "MEAN(revenue)", "columns": ["revenue"]}
  ],
  "statistical_tests": ["descriptive", "correlation", "trend"],
  "chart_types": ["line", "bar", "pie"],
  "has_temporal_data": true,
  "limitations": ["No customer demographic data"]
}
```

### Key point
The LLM does NOT look at all 2300 rows, and it never sees raw PII. Python gives it:
- Column names, types
- 15 redacted sample rows
- Unique value counts, missing value counts

---

## Agent 3: Cleaning

### What it does
Cleans the raw data based on the column roles from Agent 2. All cleaning operations are done by pandas. The LLM only decides the strategy.

### Who does what

| Task | Done By | How |
|------|---------|-----|
| Decide cleaning strategy | **LLM** | Reads Data Profile + column roles, decides what to do |
| Fill missing values | **Python** | `df.fillna()` with median/mode/"Unknown" |
| Cast column types | **Python** | `df.astype()`, `pd.to_datetime()` |
| Remove duplicates | **Python** | `df.drop_duplicates()` |
| Flag outliers | **Python** | IQR method: `Q1 - 1.5*IQR` to `Q3 + 1.5*IQR` |
| Drop columns if needed | **Python** | `df.drop(columns=[...])` |
| **Re-validate Analysis Plan** | **Python** | If any column the Plan depends on was dropped, mark affected KPIs/charts `"excluded_reason": "column dropped in cleaning"` in a `plan_diff` — Agent 4 reads this before computing (Fix #7) |
| Save cleaned data | **Python** | `df.to_csv()` |
| Log all operations | **Python** | Build CleaningResult dict |

### Cleaning Strategy Rules (decided by LLM, executed by Python)

| Column Role | Missing < 5% | Missing 5-30% | Missing > 30% | Missing > 70% |
|-------------|-------------|---------------|---------------|---------------|
| measure | Fill with median | Fill with median + flag | Exclude from KPIs + flag | Drop column |
| dimension | Fill with mode | Fill with "Unknown" | Keep + flag | Keep + flag |
| temporal | Drop row | Drop row | Drop row | Drop column |
| identifier | Drop row | Drop row | Drop row | Drop column |

### Analysis Plan Validation Loop (New — Fix #7)

```
Step 1: Python compares Analysis Plan's referenced columns against
        cleaned_data.csv's actual remaining columns
Step 2: For any KPI/chart referencing a dropped column:
        mark it "excluded_reason": "column dropped in cleaning"
Step 3: Write plan_diff.json alongside cleaning_result.json
Step 4: Agent 4 (Analyst) reads plan_diff.json FIRST and skips
        excluded items automatically — no wasted LLM call attempting
        a KPI on a column that no longer exists
```

### Input
- Raw data file: `runs/{run_id}/data/extracted/`
- Dataset Understanding from Agent 2
- Analysis Plan from Agent 2
- Data Profile from Agent 1

### Output

| Artifact | Format | Saved To |
|----------|--------|----------|
| Cleaned dataset | CSV | `runs/{run_id}/data/processed/cleaned_data.csv` |
| Cleaning Result | JSON | `runs/{run_id}/outputs/metadata/cleaning_result.json` |
| Plan Diff | JSON | `runs/{run_id}/outputs/metadata/plan_diff.json` |

### Cleaning Result structure

```
{
  "rows_before": 2300,
  "rows_after": 2276,
  "duplicates_removed": 12,
  "missing_values_handled": {
    "city": "filled_with_mode",
    "revenue": "filled_with_median"
  },
  "type_conversions": {
    "order_date": "string -> datetime",
    "revenue": "string -> float"
  },
  "outliers_flagged": {
    "revenue": 15,
    "quantity": 3
  },
  "columns_dropped": [],
  "output_file": "data/processed/cleaned_data.csv"
}
```

### Important
- LLM does NOT touch the data directly
- LLM outputs a cleaning strategy (JSON)
- Python executes the strategy and re-validates the Analysis Plan
- Python returns a log of what was done

---

## Agent 4: Analysis & Visualization

### What it does
This is the heaviest agent. It calculates KPIs, runs statistical tests, and generates charts. **All computation is done by Python. The LLM only selects what to compute (within plan_diff-approved items) and interprets results.**

### Who does what

| Task | Done By | How |
|------|---------|-----|
| Read plan_diff.json, skip excluded items | **Python** | Filters Analysis Plan before LLM sees it |
| Select which KPIs to compute | **LLM** | From non-excluded candidates only, capped at 12 (§0.5) |
| Compute KPIs | **Python** | `df['revenue'].sum()`, `df['order_id'].count()`, etc. |
| Compute descriptive stats | **Python** | `df.describe()`, `.mean()`, `.median()`, `.std()` |
| Compute quartiles / IQR | **Python** | `df.quantile([0.25, 0.75])` |
| Compute distribution | **Python** | `scipy.stats.skew()`, `scipy.stats.kurtosis()` |
| Compute correlation | **Python** | `df.corr()` |
| Compute trend over time | **Python** | `df.groupby(pd.Grouper(freq='M')).sum()` |
| Detect outliers | **Python** | IQR method or `scipy.stats.zscore()` |
| Select chart types | **LLM** | Based on data patterns and Analysis Plan, capped at 8 (§0.5) |
| Draw charts | **Python** | `matplotlib.pyplot` |
| Save charts as PNG | **Python** | `plt.savefig()` |
| Register evidence | **Python** | Build evidence registry dict |

### KPI Computation Flow

```
Step 1: LLM reads Analysis Plan (already filtered by plan_diff.json)
Step 2: LLM outputs a list of KPI definitions (JSON), max 12
Step 3: Python tool validates every source column exists in
        cleaned_data.csv BEFORE computing — if not, reject that
        single KPI with a logged reason, do not retry with the LLM
        (this is a data-existence check, not an LLM reasoning error)
Step 4: Python computes each remaining KPI using pandas
Step 5: Python returns results
Step 6: LLM interprets results (does NOT recompute)
```

### Statistical Analysis Flow

```
Step 1: Python runs df.describe() on all measure columns
Step 2: Python runs df.corr() on all measure columns
Step 3: Python runs trend aggregation if temporal column exists
Step 4: Python returns all results as structured data
Step 5: LLM reads the results and writes interpretations
```

### Chart Selection Rules (LLM decides, Python draws, capped at 8 total)

| Data Pattern | Chart Type | Python Function |
|-------------|-----------|----------------|
| Measure over time | Line | `plt.plot()` |
| Measure by category (≤ 15) | Bar | `plt.bar()` |
| Measure by category (> 15) | Horizontal Bar (Top 15) | `plt.barh()` |
| Part to whole (≤ 7 cats) | Pie | `plt.pie()` |
| Distribution | Histogram | `plt.hist()` |
| Two measures relationship | Scatter | `plt.scatter()` |
| Correlation matrix (3+ measures) | Heatmap | `sns.heatmap()` |

### Chart Generation Flow

```
Step 1: LLM outputs chart specifications (JSON), max 8:
        {
          "chart_id": "CHART-001",
          "type": "line",
          "x_column": "order_date",
          "y_column": "revenue",
          "title": "Revenue Over Time",
          "aggregation": "monthly_sum"
        }

Step 2: Python tool validates columns exist, rejects if not (no retry)
Step 3: If LLM proposed more than 8 charts, Python keeps the first 8
        and logs a truncation warning
Step 4: Python reads cleaned_data.csv, aggregates as specified
Step 5: Python draws chart with matplotlib
Step 6: Python saves to outputs/charts/chart_001.png
Step 7: Python returns file path + metadata
```

### Input
- Cleaned data: `runs/{run_id}/data/processed/cleaned_data.csv`
- Analysis Plan + Plan Diff from Agents 2/3
- Dataset Understanding from Agent 2
- Business Context from Agent 1

### Output

| Artifact | Format | Saved To |
|----------|--------|----------|
| KPI Results | JSON | `runs/{run_id}/outputs/kpis.json` |
| Statistical Results | JSON | `runs/{run_id}/outputs/statistical_results.json` |
| Chart Images | PNG | `runs/{run_id}/outputs/charts/` |
| Chart Metadata | JSON | `runs/{run_id}/outputs/metadata/chart_metadata.json` |
| Evidence Registry | JSON | `runs/{run_id}/outputs/evidence_registry.json` |

### KPI Result structure

```
{
  "kpi_id": "KPI-001",
  "name": "Total Revenue",
  "value": 2450000.50,
  "unit": "currency",
  "formula": "SUM(revenue)",
  "source_columns": ["revenue"],
  "evidence_id": "EV-001",
  "computed_by": "pandas",
  "status": "computed"
}
```

### Statistical Result structure

```
{
  "result_id": "STAT-001",
  "test_type": "descriptive",
  "column": "revenue",
  "values": {
    "mean": 1065.2, "median": 980.0, "std": 450.3,
    "min": 50.0, "max": 5200.0,
    "q1": 720.0, "q3": 1350.0, "iqr": 630.0,
    "skewness": 1.2
  },
  "interpretation": "Revenue is right-skewed...",
  "evidence_id": "EV-002",
  "computed_by": "pandas + scipy"
}
```

### Evidence Entry structure

```
{
  "evidence_id": "EV-001",
  "type": "kpi",
  "source_columns": ["revenue"],
  "value": "2450000.50",
  "method": "SUM(revenue) via pandas",
  "computed_by": "python",
  "created_at": "2025-07-01T12:00:00Z"
}
```

### Important
- LLM never sees all 2300 rows
- Python computes everything and returns summary results
- LLM only interprets the summary results
- Every computed value gets an evidence_id
- Evidence registry is built by Python, not LLM

---

## Agent 5: Insights & Recommendations

### What it does
Reads the KPIs, statistical results, and evidence produced by Agent 4. Writes human-readable insights and recommendations. This is the most LLM-heavy agent because its job is interpretation and writing.

### Who does what

| Task | Done By | How |
|------|---------|-----|
| Read KPI results | **Python** | Load `outputs/kpis.json` |
| Read statistical results | **Python** | Load `outputs/statistical_results.json` |
| Read evidence registry | **Python** | Load `outputs/evidence_registry.json` |
| Cross-reference with business context | **LLM** | LLM reads all inputs |
| Identify patterns and trends | **LLM** | LLM reasons over the numbers |
| Write insights | **LLM** | LLM generates text |
| Assign confidence levels | **LLM** | Based on evidence strength |
| Write recommendations | **LLM** | LLM generates actionable advice |
| Validate evidence references | **Python** | Check that every evidence_id exists |

### Anti-Hallucination Rules

| Rule | Enforcement |
|------|------------|
| Every insight must reference ≥ 1 evidence_id | Pydantic validator |
| Every evidence_id must exist in registry | Python validation before saving |
| No causation claims without data support | Prompt constraint + QA check |
| No external knowledge | Prompt constraint |
| If no evidence, write as Limitation | Prompt constraint |

### What the LLM receives as input (NOT raw data)

The LLM does NOT see the dataset. It sees:

```
KPIs:
- Total Revenue: 2,450,000 (EV-001)
- Order Count: 2,276 (EV-002)
- Avg Order Value: 1,076.45 (EV-003)

Statistical Results:
- Revenue mean: 1,065.2, median: 980.0, skewness: 1.2 (EV-004)
- Revenue-Quantity correlation: 0.85 (EV-005)
- Q4 revenue up 27.4% vs Q3 (EV-006)

Business Context:
- Industry: E-commerce
- Goal: Increase sales
- Important KPIs: revenue, order count
```

### Input
- KPI Results, Statistical Results, Evidence Registry from Agent 4 (JSON)
- Business Context from Agent 1 (JSON)
- Dataset Understanding from Agent 2 (JSON)

### Output

| Artifact | Format | Saved To |
|----------|--------|----------|
| Insights + Recommendations + Limitations | JSON | `runs/{run_id}/outputs/insights.json` |

### Insight structure

```
{
  "insight_id": "INS-001",
  "title": "Revenue shows strong Q4 growth",
  "description": "Revenue increased by 27.4% between Q3 and Q4...",
  "confidence": "high",
  "evidence_ids": ["EV-001", "EV-006"],
  "related_kpis": ["KPI-001"],
  "source_columns": ["revenue", "order_date"],
  "limitations": null
}
```

### Recommendation structure

```
{
  "recommendation_id": "REC-001",
  "title": "Increase Electronics marketing budget",
  "description": "Electronics shows highest growth...",
  "priority": "high",
  "based_on_insight": "INS-001",
  "evidence_ids": ["EV-001", "EV-006"]
}
```

### Validation step (Python, not LLM)

```
For each insight:
  - evidence_ids is not empty -> else REJECT
  - every evidence_id exists in evidence_registry -> else REJECT
  - every related_kpi exists in kpis.json -> else REJECT

For each recommendation:
  - based_on_insight exists in insights list -> else REJECT

If any REJECT -> remove that insight/recommendation and log it
```

---

## Agent 6: Report

### What it does
Takes all outputs and builds an HTML report. Python does all the rendering and sanitizes any user-originated string (§0.7). LLM only writes the Executive Summary text.

### Who does what

| Task | Done By | How |
|------|---------|-----|
| Load all output files | **Python** | `json.load()` for each file |
| Write Executive Summary | **LLM** | LLM writes 3-5 sentences based on top KPIs and insights |
| Sanitize formula-injection strings | **Python** | Applied to every rendered string field (§0.7) |
| Convert charts to base64 | **Python** | `base64.b64encode(open(img,'rb').read())` |
| Render HTML template | **Python** | `jinja2.Template.render()` |
| Save HTML file | **Python** | Write to `outputs/report.html` |
| Build Report Result | **Python** | Count sections, charts, insights |

### Report Sections

| Section | Data Source | Built By |
|---------|-----------|---------|
| Executive Summary | LLM text | LLM writes, Python inserts |
| Business Context | business_context.json | Python renders |
| Data Overview | data_profile.json + dataset_understanding.json | Python renders |
| KPIs | kpis.json | Python renders as cards |
| Statistical Analysis | statistical_results.json | Python renders as tables |
| Charts | charts/*.png | Python embeds as base64 |
| Insights | insights.json | Python renders |
| Recommendations | insights.json | Python renders |
| Limitations | insights.json | Python renders |
| Appendix | evidence_registry.json | Python renders |

### Input
- All files from `runs/{run_id}/outputs/` and `metadata/`
- `knowledge/business_context.json`
- `templates/report_template.html`

### Output

| Artifact | Format | Saved To |
|----------|--------|----------|
| Final HTML Report | HTML | `runs/{run_id}/outputs/report.html` |
| Report Result | JSON | `runs/{run_id}/outputs/metadata/report_result.json` |

### Report Result structure

```
{
  "report_path": "outputs/report.html",
  "executive_summary": "Analysis of 2,276 transactions...",
  "sections_count": 10,
  "charts_embedded": 5,
  "insights_included": 8,
  "recommendations_included": 5,
  "generated_at": "2025-07-01T12:30:00Z"
}
```

### Important
- LLM does NOT write HTML
- LLM only writes the Executive Summary paragraph
- Everything else is rendered by Jinja2 from structured JSON data, sanitized against formula injection

---

## Agent 7: QA

### What it does
Validates everything. Python re-checks the numbers **independently and completely**. LLM checks logic and readability.

### Who does what

| Task | Done By | How |
|------|---------|-----|
| Verify row/column counts | **Python** | `pd.read_csv(cleaned_data).shape` vs cleaning_result |
| Recompute **all** KPIs (Fix #8) | **Python** | Re-run every formula in kpis.json against cleaned_data.csv, compare with reported values |
| Check KPI discrepancy | **Python** | `abs(reported - recomputed) / recomputed < 0.0001` |
| Verify evidence references | **Python** | Check every evidence_id in insights exists in registry |
| Verify chart files exist | **Python** | `os.path.exists()` for each chart path |
| Verify report sections | **Python** | Parse HTML, check section headers exist |
| Check for fallbacks | **Python** | Read manifest, check agent statuses |
| Check insight logic | **LLM** | LLM reads insights + evidence, checks if conclusions follow |
| Check recommendation logic | **LLM** | LLM checks if recommendations make sense |
| Check report readability | **LLM** | LLM reads executive summary, checks clarity |
| Issue verdict | **Python rules** | Deterministic decision based on issue counts |

### Numerical Validation (Python) — now covers 100% of KPIs, not a sample

```
Step 1: Python reads cleaned_data.csv
Step 2: Python recomputes EVERY KPI in kpis.json independently
        (not "at least 3" — all of them; this file is small, the
        cost of checking all of it is negligible)
Step 3: Python compares each recomputed value with kpis.json
Step 4: If any discrepancy > 0.01% -> flag as CRITICAL issue
Step 5: If all match -> pass
```

### Verdict Rules (Python, deterministic)

| Condition | Verdict |
|-----------|---------|
| 0 critical issues, 0 warnings | APPROVED |
| 0 critical issues, some warnings | APPROVED with warnings |
| ≥ 1 critical issue | NEEDS_REVISION |
| Any agent used fallback | NEEDS_REVISION |
| Any insight has invalid evidence | NEEDS_REVISION |

### Input
- Everything. QA reads all files independently from disk within `runs/{run_id}/`.
- It does NOT trust context passed between agents.

### Output

| Artifact | Format | Saved To |
|----------|--------|----------|
| QA Verdict | JSON | `runs/{run_id}/outputs/metadata/qa_verdict.json` |

### QA Verdict structure

```
{
  "status": "approved",
  "critical_issues": 0,
  "warnings": 2,
  "info": 1,
  "issues": [
    {
      "severity": "warning",
      "category": "insight",
      "description": "INS-003 has only 1 evidence reference",
      "suggestion": "Consider adding more supporting data"
    }
  ],
  "numerical_checks": {
    "kpis_checked": 9,
    "kpis_passed": 9,
    "kpis_failed": 0
  },
  "overall_score": 92.5,
  "summary": "Report approved with 2 minor warnings"
}
```

### After QA

| Verdict | What Happens |
|---------|-------------|
| APPROVED | Pipeline done. Report delivered. Retention timer starts (§0.6). |
| APPROVED with warnings | Pipeline done. Warnings noted in report. |
| NEEDS_REVISION | Pipeline stops. User gets QA report with issues. Must re-run. |

---

## Data Flow Between Agents

### What passes as Context (small, structured — lives in the in-memory `PipelineState`, §0.1)

| From | To | What |
|------|----|------|
| Ingestion | Understanding | Data Profile, Business Context |
| Understanding | Cleaning | Column Roles, Dataset Understanding, Analysis Plan |
| Cleaning | Analyst | Plan Diff, Cleaning Result |
| Understanding | Analyst | Dataset Understanding |
| Ingestion | Analyst | Business Context |
| Analyst | Insight | KPIs, Stats, Evidence (summaries only) |
| Ingestion | Insight | Business Context |
| Understanding | Insight | Dataset Understanding |
| All | Report | All JSON summaries |
| All | QA | All JSON summaries |

### What passes as Files (large, on disk, namespaced under `runs/{run_id}/`)

| File | Written By | Read By |
|------|-----------|---------|
| `data/extracted/raw_data.csv` | Ingestion | Understanding, Cleaning |
| `data/processed/cleaned_data.csv` | Cleaning | Analyst, QA |
| `outputs/charts/*.png` | Analyst | Report, QA |
| `outputs/report.html` | Report | QA, User |

### Rule
> If it is bigger than 100 lines, it goes to a file.
> If it is a summary or metadata, it goes to context.
> Everything, files and context alike, lives under `runs/{run_id}/` — never a shared global path.

---

## Manifest

### What it tracks

```
{
  "run_id": "run_20250701_001",
  "pipeline_version": "3.1.0",
  "status": "completed",
  "created_at": "2025-07-01T12:00:00Z",
  "updated_at": "2025-07-01T12:31:00Z",
  "input_file": "sales_data.xlsx",
  "agents": [
    {
      "name": "ingestion_agent",
      "status": "completed",
      "started_at": "12:00:00",
      "finished_at": "12:00:08",
      "duration_ms": 8200,
      "attempts": 1,
      "fallback_triggered": false,
      "output_files": ["metadata/data_profile.json", "knowledge/business_context.json"]
    }
  ]
}
```

### Who updates it
- Orchestrator creates it at start (`runs/{run_id}/outputs/metadata/master_manifest.json`)
- Each agent updates its own section when done
- QA updates the final run status
- No agent modifies another agent's section

---

## Error Handling

### Retry Logic — retry semantics clarified (Fix #6)

Retries apply **only to LLM decision steps** (e.g., the LLM returned malformed JSON, or proposed a column name that doesn't exist, or a chart spec Python's validator rejected). They do **not** apply to Python execution bugs — a pandas `TypeError` from a code defect is a bug, not something a fourth LLM attempt will fix. Those are logged as errors immediately and go straight to fallback.

```
LLM decision step fails validation
  -> Retry 1 (feed the validation error back to the LLM, ask it to fix its output)
    -> Still fails -> Retry 2
      -> Still fails -> Retry 3
        -> Still fails -> Produce fallback output, mark agent "fallback" in manifest
              -> Pipeline continues -> QA detects fallback -> NEEDS_REVISION

Python execution error (code bug, missing file, etc.)
  -> Log immediately, no LLM retry
    -> Produce fallback output, mark agent "fallback" in manifest
      -> Pipeline continues -> QA detects fallback -> NEEDS_REVISION
```

### When pipeline stops completely

| Condition | Reason |
|-----------|--------|
| Unsupported file format | Nothing to analyze |
| File empty or < 5 rows | Not enough data |
| User cancels | User choice |
| Ingestion fails after all retries | No data for downstream agents |

### When pipeline continues with fallback

| Condition | What Happens |
|-----------|-------------|
| Cleaning fails | Fallback: pass raw data forward, flag issue |
| Analyst fails on one KPI | Fallback: skip that KPI, compute others |
| Chart generation fails | Fallback: skip chart, note in report |
| Insight generation fails | Fallback: produce empty insights, note limitation |

---

## LLM vs Python Summary

| Agent | LLM Does | Python Does |
|-------|---------|------------|
| Ingestion | Ask user questions, interpret answers | Read file, enforce limits, count rows, detect types, build profile, sanitize |
| Understanding | Classify domain, propose KPIs, write analysis plan | Compute column stats, unique counts, type detection, redact PII |
| Cleaning | Decide cleaning strategy | Execute all cleaning operations, re-validate Analysis Plan |
| Analyst | Select KPIs, select chart types, interpret results | Compute ALL numbers, validate columns exist, draw ALL charts, build evidence, enforce caps |
| Insight | Write insights, recommendations, assign confidence | Validate evidence references |
| Report | Write executive summary | Render entire HTML, sanitize strings, embed charts |
| QA | Check logic, readability, reasoning | Recompute ALL KPIs, verify files, check references, issue verdict |

### The Rule

> **If it involves a number, a file, a chart, or a computation → Python does it.**
> **If it involves a decision, an interpretation, or written text → LLM does it.**

---

## File Structure

```
insight-forge/
├── main.py
├── api/
│   └── routes.py          <- POST /analyze, GET /runs/{id}, POST /runs/{id}/answer
├── orchestrator.py        <- PipelineState + linear run loop (§0.1)
├── requirements.txt
├── agents/
│   ├── ingestion_agent.py
│   ├── data_understanding_agent.py
│   ├── cleaning_agent.py
│   ├── analyst_agent.py
│   ├── insight_agent.py
│   ├── report_agent.py
│   └── qa_agent.py
├── shared/
│   ├── tools.py            <- All Python tools (pandas, matplotlib, etc.)
│   ├── schemas.py          <- All Pydantic models
│   ├── redaction.py        <- PII redaction (§0.4)
│   ├── sanitize.py         <- Formula-injection sanitizer (§0.7)
│   ├── limits.py           <- Resource limit constants (§0.5)
│   └── utils.py            <- Helper functions
├── templates/
│   └── report_template.html
└── runs/
    └── {run_id}/
        ├── knowledge/
        ├── data/
        │   ├── raw/
        │   ├── extracted/
        │   └── processed/
        └── outputs/
            ├── report.html
            ├── kpis.json
            ├── statistical_results.json
            ├── insights.json
            ├── evidence_registry.json
            ├── charts/
            └── metadata/
                ├── master_manifest.json
                ├── data_profile.json
                ├── dataset_understanding.json
                ├── analysis_plan.json
                ├── cleaning_result.json
                ├── plan_diff.json
                ├── chart_metadata.json
                ├── report_result.json
                └── qa_verdict.json
```

A background cleanup job deletes any `runs/{run_id}/` older than `RUN_RETENTION_HOURS` (§0.6).

---