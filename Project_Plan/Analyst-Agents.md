# Insight Forge — Implementation Guide

**Version:** 4.4.0 · **For:** Development Team · **Status:** Production-oriented (not production-ready — §7)

---

## 1. Overview

### Golden Rule
> **The LLM decides WHAT to do. Python libraries DO the actual work.**

| | LLM does | Python does |
|---|---|---|
| Never | compute, draw, clean, read raw files | — |
| Always | decide what to run, interpret results, write insights & summary | every number, file, chart, cleaning op, validation |

Computation stack: **pandas, numpy, scipy, openpyxl, jinja2** (v4.7: statsmodels/babel/matplotlib/seaborn/weasyprint dropped — statistical suite runs on scipy, charts are the hand-rolled SVG renderer of v4.4+, and number formatting is shared/formatting.py).

### Pipeline

```
USER UPLOADS CSV / XLSX
        │
        ▼
[1] INGESTION        deep validation → extract → profile → business context
        ▼
[2] UNDERSTANDING    profile + sample → column roles → domain → analysis plan (DSL)
        ▼
[3] DATA QUALITY     schema/invalid/missingness → "passed" | "needs_repair"
        │                 └─ Repair: deterministic fix, internal to the Data Quality agent
        ▼
[4] CLEANING         LLM strategy + Python execution → cleaned data + log
        ▼
[5] ANALYSIS         full-data KPIs + statistical suite + charts + evidence
        ▼
[6] INSIGHTS         evidence-grounded insights + recommendations
        ▼
[7] REPORT           HTML render + LLM executive summary
        ▼
[8] QA               Python recompute (authoritative) + independent LLM
        ▼
   APPROVED / NEEDS_REVISION
```

Branches:
- DQ gate → `needs_repair`: **Repair is an internal step of the Data Quality agent** (not a separate stage) — Python fixes safe items deterministically, flags the rest for Cleaning
- Post-cleaning re-check **FAIL** → Cleaning re-runs (**max 3 re-runs**; at the cap the pipeline stops and emits **auto-verdict NEEDS_REVISION**, reason `cleaning_retry_limit_exceeded`)
- QA **NEEDS_REVISION** → stop, user gets QA report

**Hard limits (deterministic, prevents unbounded loops / resource blowups):**

| Limit | Value |
|-------|-------|
| LLM retries per agent task | 3 |
| Post-cleaning DQ re-checks | 3 (after that: fallback, flagged, **auto-verdict NEEDS_REVISION** with reason `cleaning_retry_limit_exceeded`) |
| HumanInputTool timeout | 5 min |
| Max file size | 200 MB (reject above) |
| Max rows | 5,000,000 (above → **chunked processing**, not rejection; see below) |
| Max columns | 10,000 |
| Max chart count | 20 (chart_planner ranks all candidates by evidence strength/LLM re-rank reason; **excess candidates beyond 20 are dropped, lowest-ranked first**, and logged as `charts_truncated: true` in `chart_metadata.json`) |
| Per-stage timeout | 600 s (default, `config.yaml`) — exceeded → retry/fail that stage |
| Per-run LLM cost cap | `config.yaml` `max_cost_usd` (default 5.00) — hard stop + fallback |
| Per-agent token cap | `config.yaml` `max_tokens_per_agent` (default 50k in/out) |
| Total run duration | `config.yaml` `max_run_seconds` (default 1800) — hard stop |

**Fallback semantics:** when a hard cap trips (cost / token / run-time / cleaning re-check), the pipeline stops cleanly, writes the partial-output run, and QA emits **auto-verdict NEEDS_REVISION** with a machine-readable reason code (`cleaning_retry_limit_exceeded`, `cost_limit_exceeded`, `token_limit_exceeded`, `run_time_limit_exceeded`).

**>5M rows:** Python parses via chunking (`pandas.read_csv(chunksize=50_000)`, `openpyxl.read_only`) so memory stays bounded; aggregations (sums, means, counts) are streamed chunk by chunk so final numbers are still **full-data**, never sampled. LLM/UX previews stay small.

### Crew (CrewAI — the only framework)

- Every LLM-deciding stage below = one CrewAI **Agent** (`role`/`goal`/`backstory`) with CrewAI **Tasks**
- All computation = CrewAI **Tools** (Python `@tool` wrappers)
- Branching = CrewAI **Flows** (`@flow`/`@router`)
- Deterministic stages (Data Quality) are **Flow steps** — Python only, no LLM call; implemented in `agents/data_quality.py` (kept alongside the other agent modules for a single agents/ folder, even though it's Engine-only — no LLM), invoked by `crew/flows.py`
- Models per agent set in the single root `config.yaml`
- **QA must use a different model than generation** (§2.8)

| Stage | Crew | Type |
|-------|------|------|
| 1 Ingestion | Agent | Engine + small LLM |
| 2 Understanding | Agent | LLM-heavy |
| 3 Data Quality | Flow step | Engine only (no LLM) |
| 4 Cleaning | Agent | LLM strategy + engine exec |
| 5 Analysis | Agent | LLM selects + engine computes |
| 6 Insights | Agent | LLM-heavy |
| 7 Report | Agent | Engine + LLM summary |
| 8 QA | Agent | Engine + independent LLM |

---

## 2. Agents

> All agents follow one template: **Mission → Who does what → Input → Output → Notes**. JSON examples appear only where the structure is load-bearing.

### 2.1 Agent 1: Ingestion

**Mission:** take the file, validate it deeply, extract data, ask business questions, emit profile + context.

| Who | Does |
|-----|------|
| Python | validate (extension + MIME + signature + parser), read CSV/XLSX, merge sheets, profile (`shape`, `dtypes`, nulls, duplicates), detect PII columns |
| LLM | ask which sheet + business questions (HumanInputTool), interpret answers |

**Input:** raw file · user answers
**Output:**
- `runs/<run_id>/data/extracted/` (CSV)
- `runs/<run_id>/metadata/data_profile.json`
- `runs/<run_id>/knowledge/business_context.json`

```json
{
  "file_name": "sales.xlsx", "file_hash": "sha256:...",
  "row_count": 2300, "column_count": 8,
  "columns": ["date","product","category","revenue","quantity","city"],
  "column_types": {"date": "datetime64", "revenue": "float64"},
  "missing_values": {"city": 45}, "duplicate_rows": 12,
  "pii_columns": ["customer_email"], "validation_status": "passed"
}
```

**Notes:** stop on unsupported format / empty or < 5 rows / user cancel / fail after **3 retries** (see §1 hard limits). Cell content = **UNTRUSTED DATA**. Question timeout 5 min → *Generic Analysis Mode* (§3.5).

**Tasks (`agents/ingestion_agent.py`):**

| Task | Does |
|------|------|
| `validate_and_extract` | runs deep validation, reads the file, merges sheets if needed |
| `profile_dataset` | builds `data_profile.json` (shape, dtypes, nulls, duplicates, PII flags) |
| `gather_business_context` | asks sheet-choice + business questions via HumanInputTool, interprets answers into `business_context.json` |

**Tools used:** `file_validator_tool` (extension + MIME + signature) · `file_reader_tool` (pandas/openpyxl, chunked above 5M rows) · `data_profiler_tool` · `pii_detector_tool` · `human_input_tool`

---

### 2.2 Agent 2: Data Understanding

**Mission:** from **profile + 20-row sample only**, classify column roles, detect domain, propose Analysis Plan in DSL.

| Who | Does |
|-----|------|
| Python | `nunique()`, `head()`, datetime/numeric detection, role rules |
| LLM | review roles, detect domain, entities, propose DSL KPIs, build plan |

**Column-role rules (Python):**

```
dtype datetime                → temporal
numeric & nunique == row_count → measure   (identifier if the name is id-like: order_id, zip_code)
numeric & nunique > 20        → measure
numeric & nunique ≤ 20        → measure or categorical
object & nunique == row_count → identifier
object & nunique ≤ 20         → dimension  (temporal if the name is date-like: date, month, year)
object & nunique > 50         → free_text
```

dtype rules win over the all-unique heuristic; id-like / date-like names refine it.
LLM may reclassify by name (e.g. numeric `zip_code` → identifier) and may flip an all-unique numeric between `measure` ↔ `identifier`.

> **Planning is a subtask of this same stage** — implemented as a second Crew Task inside `understanding_agent.py` (not a separate module), since it is **not** a 9th agent; it is a second Crew task of the Understanding Agent producing the DSL plan.

**Tasks (Understanding Agent, `agents/understanding_agent.py`):**

| Task | Does |
|------|------|
| `classify_column_roles` | reviews Python's rule-based role output, reclassifies by name/semantics where rules are ambiguous (e.g. numeric `zip_code` → identifier) |
| `detect_domain_and_entities` | infers `detected_domain`, `domain_confidence`, business entities from profile + sample |
| `build_analysis_plan` | proposes candidate KPIs (DSL only) + statistical tests → `analysis_plan.json` |

**Tools used:** `column_profiler_tool` (`nunique`, `head`, dtype/role rules) · `domain_classifier_tool` · `dsl_plan_builder_tool` (validates against `shared/dsl_validator.py` whitelist before returning)

**Input:** `data_profile.json` · `business_context.json` · 20-row sample
**Output:**
- `runs/<run_id>/metadata/dataset_understanding.json`
- `runs/<run_id>/metadata/analysis_plan.json` (DSL ops only — no freeform formulas)

```json
{
  "detected_domain": "sales", "domain_confidence": 0.87,
  "entities": ["Product","Customer","Order"],
  "temporal_columns": ["order_date"], "dimensions": ["product","category"],
  "measures": ["revenue","quantity"], "identifiers": ["order_id"],
  "candidate_kpis": [
    {"kpi_id":"KPI-001","name":"Total Revenue","operation":{"function":"sum","column":"revenue"}},
    {"kpi_id":"KPI-002","name":"AOV","operation":{"function":"ratio",
       "numerator":{"function":"sum","column":"revenue"},
       "denominator":{"function":"count","column":"order_id"}}}
  ],
  "statistical_tests":["descriptive","correlation","trend","anova"],
  "has_temporal_data": true, "limitations": []
}
```

---

### 2.3 Agent 3: Data Quality

**Mission:** deterministic pre-cleaning gate. Python only; catches what the cleaning strategy can't see.

Checks: schema · invalid values (impossible dates, negative revenue, % >100, `age=350`) · missingness **MCAR/MAR/MNAR** · duplicates · referential integrity · business rules · units/encoding.

**Input:** `dataset_understanding.json` · `data_profile.json` · `business_context.json` · raw sample
**Output:** `runs/<run_id>/metadata/data_quality_report.json`

```json
{
  "status": "passed | needs_repair",
  "invalid": {"revenue": ["negative"], "order_date": ["future_dates"]},
  "missingness": {"rate": 0.27, "pattern": "non_random", "assessment": "MAR_suspected"},
  "duplicates": 12,
  "issues": [{"severity":"high","category":"invalid_value"}]
}
```

**Gate:** `passed` → Cleaning. `needs_repair` → Repair, then Cleaning. Unresolved high-severity → QA must flag.

**Repair scope (what Python fixes automatically vs flags to Cleaning):**

| Issue | Auto-fix (deterministic) | Flagged to Cleaning |
|-------|--------------------------|---------------------|
| Negative values in measure | **No — never sign-flip.** Flagged (it's a data-source error) | Cleaning strategy decides (drop row / exclude) |
| Type mismatch (e.g. revenue as string) | Cast by role (`astype`, `to_datetime`) | — |
| Exact duplicates | `drop_duplicates()` | — |
| Impossible dates (e.g. `2099`, 30-Feb) | Row dropped + logged | — |
| Out-of-range percentages (>100%) | Flagged | Cleaning strategy decides |
| Unknown category values | Flagged | Cleaning strategy decides (map/keep/"Unknown") |
| MAR/MNAR missingness | **No imputation — preserve + flag** (§2.4 `flag_and_preserve`) | Cleaning builds flag features |

Rule: **Repair never invents data.** It only casts types, drops exact duplicates and impossible rows, and preserves/records everything else for Cleaning to decide.

**Tools used (`agents/data_quality.py`, no CrewAI Task — plain functions called from `crew/flows.py`):** `schema_checker_tool` · `invalid_value_checker_tool` · `missingness_analyzer_tool` (MCAR/MAR/MNAR) · `duplicate_detector_tool` · `referential_integrity_tool` · `deterministic_repair_tool`

---

### 2.4 Agent 4: Cleaning

**Mission:** clean by column roles **+ DQ report**. LLM picks strategy; Python executes and logs everything; Python re-checks the result.

| Who | Does |
|-----|------|
| LLM | decide strategy (JSON) |
| Python | fillna, `*_missing_flag` columns, cast types, drop dupes, IQR outliers, log + re-run DQ checks |

**Missing-value strategy by role + missingness type:**

| Role | MCAR <5% | MCAR 5-30% | MAR/MNAR (signal) | >70% |
|------|----------|------------|-------------------|------|
| measure | median fill | median + flag | **flag_and_preserve** + exclude | drop |
| dimension | mode fill | "Unknown" | **flag_and_preserve** | keep + flag |
| temporal | drop row | drop row | drop row | drop column |
| identifier | drop row | drop row | drop row | drop column |

> **flag_and_preserve** = keep missingness as a boolean feature (`revenue_missing_flag`). Never impute silently when the pattern is non-random — missingness is often a business signal.

**Input:** DQ report · understanding · profile · raw data
**Output:**
- `runs/<run_id>/data/processed/cleaned_data.csv` — always the **latest accepted attempt**; on a re-run the previous attempt is kept as `cleaned_data_attempt_<n>.csv` (not overwritten silently), so the lineage trail required by §3.3 stays intact even when Cleaning takes 2-3 tries to pass the DQ re-check
- `runs/<run_id>/metadata/cleaning_result.json` (pre/post rows, dupes removed, type casts, flags created, outliers, `attempt: <n>`)

**Notes:** result validation re-runs DQ on cleaned output → `passed` or re-run Cleaning (max 3 attempts, §1). Each attempt is numbered and logged; only the final passing attempt's `cleaned_data.csv` feeds Analysis.

**Tasks (`agents/cleaning_agent.py`):**

| Task | Does |
|------|------|
| `decide_cleaning_strategy` | picks the fill/drop/flag strategy per column, per the role×missingness table above (JSON output) |
| `execute_cleaning` | hands the strategy to Python for execution + logging |
| `recheck_data_quality` | re-runs the Data Quality agent's checks on the cleaned output; loops back on `FAIL` (max 3 re-runs, §1) |

**Tools used:** `cleaning_strategy_tool` (LLM) · `fillna_tool` · `flag_column_tool` (`*_missing_flag`) · `type_caster_tool` · `dedup_tool` · `iqr_outlier_tool` · `dq_recheck_tool` (calls `agents/data_quality.py`)

---

### 2.5 Agent 5: Analysis

**Mission:** compute **everything on the full dataset**. LLM chooses what/how; Python computes + draws + memorializes evidence.

| Who | Does |
|-----|------|
| LLM | select KPIs, interpret results, re-rank chart candidates (with reasons), **propose chart kinds** (`area`, `boxplot`, …) with a justification |
| Python | execute DSL (whitelist only), stats suite, **validate proposed kinds + pick the default chart shape**, draw, evidence registry |

**Statistical suite**

| Category | Tests |
|----------|-------|
| Descriptive | mean/median/std/quantiles/IQR/skew/kurtosis |
| Correlation | Pearson + **p-value + CI + effect size + n**; Spearman |
| Comparison 2 groups | t-test, Mann-Whitney U |
| Comparison 3+ | ANOVA, Kruskal-Wallis + post-hoc |
| Categorical | Chi-square, Cramér's V |
| Time series | YoY, MoM, WoW, rolling, seasonality, trend significance |

**KPI DSL (whitelist)** — the LLM never writes freeform formulas:
```
sum, mean, median, count, nunique, min, max, std, growth, correlation, ratio
```
Function signatures:

| Function | Parameters |
|----------|------------|
| `sum, mean, median, min, max, std, count, nunique` | `column` · `group_by` (opt) · `filter` (opt) |
| `correlation` | `column_a`, `column_b` · `method`: "pearson" \| "spearman" · `filter` (opt) |
| `ratio` | `numerator`, `denominator` (each a nested op) |
| `growth` | `column` · `over_column` (temporal) · `period`: "YoY" \| "MoM" \| "WoW" · `basis`: "previous_period" (default) \| "start_of_period" · `as_percent`: true |

```json
{
  "kpi_id": "KPI-001", "name": "Total Revenue",
  "operation": {"function": "sum", "column": "revenue", "group_by": null, "filter": null},
  "value": 2450000.50, "evidence_id": "EV-001", "computed_by": "pandas"
}
```
```json
{
  "kpi_id": "KPI-003", "name": "Revenue MoM Growth",
  "operation": {
    "function": "growth",
    "column": "revenue",
    "over_column": "order_date",
    "period": "MoM",
    "basis": "previous_period",
    "group_by": ["category"],
    "filter": null,
    "as_percent": 1
  },
  "value": 12.4, "evidence_id": "EV-003", "computed_by": "pandas"
}
```

`growth` semantic: `(current − baseline) / baseline`; if `over_column` alone → month-basis; for YoY use `period: "YoY"` (compares same calendar month/quarter vs prior year).

**Charts — data-driven planner (not a fixed menu) + LLM-proposed kinds (validated)**

The chart type is **decided by the shape of the data**, not by habit. Python inspects each candidate visual at `analysis/chart_planner.py` and picks the form from an ordered rule table. The LLM chooses *which facts deserve a chart*, may re-rank candidates with a justification, **and may propose a specific chart kind per KPI** — the shape stays Python's call:

| # | Data shape (measured by Python) | Chart |
|---|--------------------------------|-------|
| 1 | single dimension, ≤ 2 values swap | bar / donut |
| 2 | ordered axis (dates/months) with ≥ 3 points | line |
| 3 | single dimension, 3–12 values | vertical bar |
| 4 | single dimension, 13–50 values | horizontal bar |
| 5 | single dimension, > 50 values | barh **top-15 + "$rest"** rollup |
| 6 | numeric distribution / skew ask | histogram (bins: Freedman–Diaconis) |
| 7 | 2 numeric measures, asked together | scatter + trend line when r significant |
| 8 | ≥ 3 numeric measures | ranked correlation heatmap |
| 9 | share / "% of whole", parts ≈ 100% | doughnut (only then) |

**Chart-kind whitelist (12).** Python renders only these kinds; anything else a model proposes is rejected. `bar · barh · line · doughnut · histogram · scatter · heatmap · area · boxplot · stacked_bar · pie · lollipop` (the last five extend the rule table — they exist only via LLM proposal or explicit plan intent, never by accident).

**Hybrid proposal flow.** The LLM may submit `proposed_kinds: [{"kpi_id": "KPI-002", "kind": "boxplot", "reason": "…"}]` to `chart_planner_tool`. Python runs `validate_proposed_kinds`: (1) `kind` must be in the 12-kind whitelist; (2) the shape must fit the data — `line`/`area` need an ordered temporal axis with ≥ 3 points · `scatter` needs 2 numeric measures · `heatmap` ≥ 3 measures · `doughnut`/`pie` only for a share/"% of whole" KPI or a ≤ 2-value dimension · `histogram`/`boxplot` need a numeric measure · `stacked_bar` needs one dimension + ≥ 2 measures · `lollipop` needs a single dimension · unknown `kpi_id` rejected. Accepted proposals override the rule-table kind; rejected ones are **dropped with their reason and fall back to the rule table** — a bad proposal can never produce an unrenderable chart.

Wiring is: **`chart_planner`** takes the KPI list + `data_profile` + numeric/ordinal columns → returns one JSON per chart:
```json
{
  "chart_id": "CH-004", "kind": "line",
  "reason": "1 ordered dim (month) >= 3 points -> line",
  "columns": ["order_date", "revenue"],
  "data": [...], "evidence_id": "EV-007", "computed_by": "pandas"
}
```
Falling back if the data is too thin: planner downgrades to a simple bar or a table, and stamps `reliability: "low_n"`. If the LLM re-ranks, the plan must carry the reason — the final draw is still Python. `chart_metadata.json` records the planner decision + the shape rule hit (or the accepted proposal), so chart choice is reproducible (see §5).

**Input:** cleaned data · plan · understanding · business context · cleaning result
**Output:** `runs/<run_id>/outputs/kpis.json`, `runs/<run_id>/outputs/statistical_results.json`, `runs/<run_id>/outputs/charts/*.svg`, `runs/<run_id>/metadata/chart_metadata.json` (planner + shape rules + evidence refs), `runs/<run_id>/outputs/evidence_registry.json`

**Rule:** Python always aggregates on **all rows**; sampling is for LLM/UX inspection only. Every computed value gets an evidence_id.

**Tasks (`agents/analyst_agent.py`):**

| Task | Does |
|------|------|
| `select_kpis` | picks which candidate KPIs from the plan are worth computing given the cleaned data |
| `run_dsl_and_stats` | hands DSL ops + relevant statistical tests to Python for execution |
| `rank_chart_candidates` | reviews `chart_planner` output, may re-rank with a written reason and **propose chart kinds** (`proposed_kinds`, validated by Python) — shape/draw stays Python's call |

**Tools used:** `dsl_executor_tool` (whitelist only, via `shared/dsl_validator.py`) · `statistical_suite_tool` (scipy) · `chart_planner_tool` (`analysis/chart_planner.py` — deterministic rule table + `validate_proposed_kinds` for LLM proposals) · `chart_renderer_tool` (`analysis/chart_renderer.py` — hand-rolled SVG, deterministic, no plotting library) · `evidence_registry_tool` (`analysis/evidence.py` — the only writer)

**Chart accessibility:**
- **Color-blind safe palettes** — Okabe-Ito / seaborn colorblind; never red-green as the only encoding.
- **Pattern + label redundancies** — bar/pie/doughnut also label values; line/area charts get markers, not color-only.
- **Alt text** — each `<img>` in the HTML report carries a caption generated from chart metadata + associated insights (title + `reliability` + `evidence_id`).

**Localization:**
- All numeric/datetime formatting follows the report's `locale` (from business context; default `en`): decimal separators and date formats (`en-US` → `1,234.5`, `ar-EG` → `١٬٢٣٤٫٥`).
- Formatting is applied in the Report render step by Python (`shared/formatting.py` — hand-rolled, no babel dependency), not the LLM.
- `RTL` layout + Arabic font support for `ar` locales (see §7 roadmap).

---

### 2.6 Agent 6: Insights & Recommendations

**Mission:** write evidence-grounded insights + hedged recommendations. Most-LLM-heavy stage.

**Claim taxonomy**

| Type | Allowed when | Guard |
|------|--------------|-------|
| DESCRIPTIVE / COMPARATIVE | always | must quote EV-ids |
| CORRELATIONAL | p-value + CI present | report strength, not cause |
| PREDICTIVE | forecasting module ran | labeled forecast |
| CAUSAL | never without causal methodology | ❌ "discounts caused growth" → ✓ "revenue grew during discount periods" |

**Recommendation chain:** Observation → Finding → Implication → recommendation, always hedged ("consider testing…").

**Input:** kpis.json · stats.json · evidence_registry.json · business_context.json · dataset_understanding.json
**Output:** `runs/<run_id>/outputs/insights.json`

```json
{
  "insight_id": "INS-001", "claim_type": "COMPARATIVE",
  "title": "electronics fastest in Q3", "description": "…",
  "confidence": "high", "evidence_ids": ["EV-001","EV-006"],
  "required_evidence": ["group_comparison","growth_rate"],
  "related_kpis": ["KPI-001"]
}
```

**Lineage — every insight's evidence traces full chain** (raw → cleaning → filter → aggregation → value):
```json
{
  "evidence_id": "EV-006", "source": {"file_hash":"…","sheet":"Sales",
    "transformations":["removed_duplicates"],"filter":"category==Electronics",
    "aggregation":"monthly_sum","comparison":"Q4 vs Q3","result":27.4}
}
```

**Validation before saving:** non-empty evidence_ids · refs exist in registry · claim matches evidence types · recommendation → references only existing insights. Any failure → remove + log.

**Tasks (`agents/insight_agent.py`):**

| Task | Does |
|------|------|
| `generate_insights` | writes evidence-grounded insights per the claim taxonomy |
| `build_recommendations` | Observation → Finding → Implication → hedged recommendation chain |
| `validate_claims` | checks evidence_ids/refs/claim-type match before saving; strips failures |

**Tools used:** `evidence_lookup_tool` (reads `evidence_registry.json`) · `claim_validator_tool` · `human_input_tool` (optional review checkpoint, `review_required: true`)

**Human review checkpoint (optional, `config.yaml` `review_required: true`):** after `insights.json` is validated, the pipeline can pause and present the insights to a human analyst (HumanInputTool / web panel). The analyst can **approve**, **edit text** (edits must keep `evidence_ids` intact — the claim validator re-runs), or **request regeneration** (→ Insights re-run, bounded by retry caps). If approved or `review_required: false`, it proceeds to Report. This is a review gate, not a second LLM.

---

### 2.7 Agent 7: Report

**Mission:** render the full HTML report. Python renders; LLM writes **only** the 3-5 sentence executive summary.

| Section | Source |
|---------|--------|
| Summary, Business Context, DQ Summary, Data Overview, KPIs, Stats, Charts, Insights, Recommendations, Limitations, Evidence Appendix | LLM summary + all JSONs |

**Security (in render):** Jinja `autoescape=True`, HTML sanitizer, CSP header. Never render cells raw.

**Input:** all `runs/<run_id>/outputs/`, `runs/<run_id>/metadata/`, `runs/<run_id>/knowledge/` files · `resources/report_template.html`
**Output:** `runs/<run_id>/report.html`, `runs/<run_id>/metadata/report_result.json`

**Tasks (`agents/report_agent.py`):**

| Task | Does |
|------|------|
| `render_report` | Python renders every section from the run's JSONs into `resources/report_template.html` |
| `write_executive_summary` | LLM writes **only** the 3-5 sentence executive summary |

**Tools used:** `html_renderer_tool` (jinja2, `autoescape=True`) · `html_sanitizer_tool` · `locale_formatter_tool` (shared/formatting.py, per §2.5 localization) · `chart_embed_tool` (alt-text captions from chart metadata)

---

### 2.8 Agent 8: QA

**Mission:** last gate. **Python recomputation is authoritative**; an **independent model** (different LLM, different prompt) checks logic + readability.

| Who | Checks |
|-----|--------|
| Python | recompute **100% of KPIs** from cleaned data and compare (tolerance 0.01%) · refs valid · charts exist · HTML sections · manifest fallback |
| LLM | insight/recommendation logic alignment, summary readability |

**Score formula (deterministic):**
```
score = 100 – (critical×15) – (warnings×2.5) – (info×0.5)   [floor 0]
```

> **Note:** the score is **informational/reporting only**. The verdict below is decided purely by the logical conditions — a high score (e.g. 95+) does NOT override `NEEDS_REVISION` when a critical issue exists. There is no score threshold in the decision.

**Verdict (deterministic):**

| Condition | Verdict |
|---|---|
| any critical | NEEDS_REVISION |
| fallback used | NEEDS_REVISION |
| invalid evidence / unresolved DQ | NEEDS_REVISION |
| resource limit exceeded (`cleaning_retry_limit_exceeded`, cost/token/time caps) | NEEDS_REVISION (auto, no QA LLM needed) |
| else minor warnings | APPROVED_WITH_WARNINGS |
| else clean | APPROVED |

**Output:** `runs/<run_id>/metadata/qa_verdict.json`

**Tasks (`agents/qa_agent.py`):**

| Task | Does |
|------|------|
| `recompute_kpis` | Python recomputes 100% of KPIs from cleaned data, compares to reported values (tolerance 0.01%) |
| `validate_structure` | checks refs valid, charts exist, HTML sections complete, manifest fallback flags |
| `review_logic_and_readability` | independent LLM (different model than generation, §2.8) checks insight/recommendation alignment + summary readability |
| `compute_verdict` | applies the deterministic score formula + verdict table — no LLM in the decision itself |

**Tools used:** `kpi_recomputation_tool` · `reference_validator_tool` · `score_calculator_tool` · `verdict_tool`

---

## 3. Guardrails (rules that hold the whole system together)

### 3.1 Full data, always
Python aggregates the complete dataset. LLM/UX see samples/previews only.

### 3.2 Python is authoritative
Every numeric output is recomputed by QA from the cleaned data — **100% of KPIs**, not a sample — before "done".

### 3.3 Evidence is concrete
Every number, chart, insight, and claim carries an `evidence_id`; lineage = metadata for reproducible auditing.

### 3.4 Claims are conservative
Descriptive/comparative/correlational by default; predictive & causal gated (require methodology).

### 3.5 User is optional — Generic Analysis Mode
If business questions time out → Generic Mode with `context_confidence: 0`, report says "generation of context-specific recommendations was not possible."

---

## 4. File Structure

> **v4.2 cleanup — changes from v4.0:**
> 1. `planner_agent.py` merged into `understanding_agent.py` — §2.2 already states planning is a *second Task of the same Agent, not a 9th agent*; giving it its own file contradicted that and implied a module that doesn't independently exist.
> 2. `templates/` + `fixtures/` merged into one `resources/` — both were read-only static assets at the root; two folders for the same concern was unnecessary.
> 3. `data_quality_agent.py` renamed `data_quality.py` and **kept in `agents/`** (one folder for all pipeline stages, incl. the deterministic ones) — it's still called out as Engine-only/no-LLM everywhere it's referenced, so the distinction lives in the docs and in `crew/flows.py` wiring, not in a folder split.
> 4. Added `shared/logger.py` — the "structured log per stage" in §5 had no owning module.
>
> Net effect: `agents/` drops from 9 files to 8 (planner merged away), and top-level read-only dirs go from 2 to 1.
>
> **v4.3 — logical fixes (not structural):**
> 5. §5 caching rule rewritten — the old rule said Cleaning/Analysis were reusable whenever "only business context differs," which contradicted §1 (both call an LLM) and ignored that Analysis consumes the plan Understanding produces. Reuse is now keyed on **plan-hash equality**, not on "context changed y/n" (§5).
> 6. §6 golden datasets — the single `Expected: revenue=10,000...` line read as if it applied to all 8 fixtures; replaced with a per-fixture table so each fixture's assertion is unambiguous.
> 7. §1 `Max chart count` — added the truncation rule for when candidates exceed 20.
> 8. §2.4 Cleaning output — re-run attempts are now versioned (`cleaned_data_attempt_<n>.csv`) instead of silently overwritten, so lineage (§3.3) holds even when Cleaning takes multiple tries.
>
> **v4.4 — chart kinds + hybrid proposal (Stage 5b):**
> 9. §2.5 chart-kind whitelist extended to **12 kinds** (`area · boxplot · stacked_bar · pie · lollipop` added to the original 7); every kind has a deterministic hand-rolled SVG renderer in `analysis/chart_renderer.py` (no matplotlib dependency).
> 10. §2.5 hybrid proposal flow — the LLM may **propose chart kinds with reasons** (`proposed_kinds`); Python validates them (`validate_proposed_kinds`: whitelist + data-shape feasibility); rejected proposals fall back to the rule table with the reason recorded. Final draw is always Python.
> 11. §2.5 Stage 5b delivered inside `agents/analysis.py` (same agent as 5a — no separate `analyst_agent.py`); `chart_renderer_tool` (SVG preview) + `evidence_registry_tool` (registry write, only writer).
>
> **v4.5 — Stage 5b implemented (Task 7, 2026-08-16):**
> 12. `analysis/chart_renderer.py` ships the 12 hand-rolled SVG renderers (Okabe-Ito palette, value labels, XML-escaped, `<title>/<desc>` captions with evidence_id; Freedman–Diaconis bins; corr heatmap; trend-line scatter); `render_all` writes `runs/<run_id>/charts/<chart_id>.svg` and never raises on a single bad chart.
> 13. `ChartMetadata.kpi_id` links each chart to its base KPI candidate (shape-driven extras leave it null); the renderer recomputes series deterministically from **all rows** via new public `dsl_executor` wrappers (`growth_series`/`grouped_values`/`grouped_growth_values`) — grouped KPIs are stored expanded as scalars in `kpis.json`, so charts draw from the operation, never from a dict value.
> 14. `_rule_2_growth` now plans on the **actual period series** (≥3 points from `growth_series`) instead of the raw date cardinality — a YoY chart over a 10-day sample is no longer planned at all (was a dead "no data" SVG).
> 15. `chart_planner_tool` accepts `proposals_json` and returns `proposal_errors`; `chart_path` is filled into `chart_metadata.json` by the agent after `render_all`. Verified end-to-end on a dirty 14-row chain: 8 KPIs · 38 tests · 6 real SVGs · 52 evidence entries.
>
> **v4.6 — Stage 6 implemented (Task 8, 2026-08-16):**
> 16. `agents/insight_agent.py` ships the deterministic §2.6 claim pipeline: DESCRIPTIVE from computed KPIs · CORRELATIONAL gated on |r| ≥ 0.3 **and** p < 0.05 (CI quoted, "association, not cause") · COMPARATIVE gated on p < 0.05 (chi²/Cramér's V deduped per pair) · trend claims require ≥ 3 periods and are labeled "not a forecast" · PREDICTIVE/CAUSAL are never emitted.
> 17. Hedged recommendation chain per insight: Observation → Finding → Implication → "consider testing…" (bounded at 8). Every claim is validated before saving by `claim_validator_tool` (evidence refs exist · claim type matches evidence kinds derived from stage-5 artifacts · recommendations reference surviving insights) — failures are removed and logged.
> 18. `--crew` refines titles/descriptions only (ids/evidence stay byte-identical, re-validated, deterministic fallback on failure); `--review` enables the §2.6 human gate (approve / edit / regenerate), auto-approved in automated mode (`config.yaml review_required: false`).
> 19. Flow Review stage 6 is live: card lights up after analysis, `outputs/insights.json` previews in Artifacts with data-flow counts. Verified on `sales_demo.csv`: weak revenue×quantity correlation (r = −0.014) correctly gated out; significant product×category association kept.

```
insight-forge/
├── main.py                     # entry — builds Crew, runs Flow
├── config.yaml                 # single config: per-agent role/goal/backstory/model, hard limits (§1), retention (§5), review_required (§2.6)
├── .env.example                # API keys (loaded via shared/utils.py, never committed)
├── pyproject.toml              # deps: crewai + §1 stack (pandas, numpy, scipy, openpyxl, jinja2, pyyaml, python-dotenv)
├── crew/
│   ├── crew.py                 # CrewAI Agents + Tasks in order
│   └── flows.py                # DQ gate, Cleaning re-check, QA verdict branches; invokes deterministic stages
├── agents/                     # one module per §2 stage, all 8 stages together
│   ├── ingestion_agent.py      # stage 1
│   ├── understanding_agent.py  # stage 2 — roles + domain + DSL plan (planning is its 2nd Task, not a separate file)
│   ├── data_quality.py         # stage 3 — Engine only, no LLM (deterministic Flow step, invoked by flows.py — see §1)
│   ├── cleaning_agent.py       # stage 4
│   ├── analyst_agent.py        # stage 5
│   ├── insight_agent.py        # stage 6
│   ├── report_agent.py         # stage 7
│   └── qa_agent.py             # stage 8
├── analysis/                   # pure computation, no LLM
│   ├── chart_planner.py        # §2.5 data-shape rule table → chart kind + reason (deterministic)
│   ├── evidence.py             # evidence_id minting + evidence_registry read/write (the only writer)
│   ├── generic/ (descriptive✓ correlation✓ distribution✓ trend)
│   └── domains/ (sales•finance•marketing•hr•operations)
├── shared/
│   ├── tools.py                # CrewAI @tool wrappers
│   ├── schemas.py              # Pydantic
│   ├── dsl_validator.py        # whitelist / DSL — used by both understanding_agent (build) and analyst_agent (execute)
│   ├── logger.py               # writes runs/<run_id>/logs/ — LLM latency/tokens/cost + tool calls + retries (§5 structured log)
│   └── utils.py                # config load (config.yaml + env), run_id allocator
├── resources/                  # read-only static assets (merged templates/ + fixtures/)
│   ├── report_template.html    # (was templates/report_template.html)
│   └── business_context/       # static business-context templates (was fixtures/, itself renamed from knowledge/)
├── runs/                       # run isolation — the only writable surface per run
│   └── <run_id>/               # extracted + processed data, all outputs, logs, manifest
│       ├── data/               # raw/ extracted/ processed/
│       ├── knowledge/          # business_context.json (per-run Agent 1 output — distinct from resources/business_context/, which is the static template)
│       ├── metadata/           # data_profile · dataset_understanding · analysis_plan · data_quality_report · cleaning_result · chart_metadata · report_result · qa_verdict
│       ├── outputs/            # report.html · kpis.json · statistical_results.json · insights.json · evidence_registry.json · run_comparison.json · charts/
│       ├── logs/               # LLM/tool logs
│       └── master_manifest.json
├── cache/                      # key→run_id index (idempotency) + source_name→run_ids index (run comparison) (§5) — kept separate from runs/: different retention lifecycle (§5)
└── tests/      → unit/ integration/ regression/ security/ statistical/ agent/ e2e/ golden/ fixtures/
```

---

## 5. Reproducibility & Observability

Published for every run, side by side with the output files:

- **Manifest** (`runs/<run_id>/master_manifest.json`) — branch-agnostic **everything** to reproduce a run: `pipeline_version`, `code git_sha`, `data_hash`, `model`, `temperature`, `seed`, `python_version`, `packages`, `analysis_mode`.
- **Structured log per stage**: LLM call latency/tokens/cost + tool calls, errors, retries as audit trail.
- **Dashboard metrics**: runs, latency, tokens, cost, failures/fallbacks, QA score.

### Run isolation & concurrency

Each run gets a unique `run_id` (`run_<timestamp>_<seq>`) and writes **only** to `runs/<run_id>/` — extracted data, cleaned data, all outputs, logs, and its own manifest live there. Isolation model:

- **Scope:** one pipeline run = **its own sub-directory + own Crew instance**. Two runs never touch the same files.
- **Concurrency:** safe to launch many runs in parallel. Python-side (pandas) is process-local; no global temp dirs, no shared in-memory state.
- **Locking:** only the output root (`runs/`) needs a lightweight `run_id` allocator (atomic `mkdir`); reads of shared static assets (`resources/`) are read-only and safe concurrently.
- **Teardown:** `logs/` per run; retention policy (configurable) deletes old runs.

### Caching & idempotency

Re-running the **same input** is detectable and parts of it are skipped instead of paying full cost/time:

- **Cache key** = `sha256(file_bytes)` + `config_version` + `prompt_version` + model ids. Stored in a `cache/` index (key → run_id; plus `source_name` → run_ids for run comparison).
- **Full-file hit:** if the same `input_hash` + effective-config already produced an `APPROVED` run, the orchestrator returns that run's report/`run_comparison` directly (with a `cached: true` marker) — no LLM calls, no recompute.
- **Partial hits, precisely defined:** reuse is keyed on **what actually changed**, not on a blanket "context differs" rule — because Cleaning and Analysis both call an LLM (§1 table) and both consume the DSL plan, so "business context differs" alone doesn't make them safe to reuse:
  - **Ingestion (extraction + profile only, not PII/business-rule interpretation)** — always reusable on a full `input_hash` hit; it never reads business context.
  - **Data Quality's schema/type/duplicate checks** — reusable, since those don't depend on business context. Its **business-rules checks are re-run** whenever `business_context.json` changes, since that's a declared input (§2.3).
  - **Understanding/plan** — re-run whenever business context changes (unchanged from before), since domain/entities/candidate KPIs depend on it.
  - **Cleaning, Analysis** — reusable **only if the analysis_plan.json produced this run is byte-identical to the cached run's plan** (checked by hashing the plan, not by checking whether context changed) — since Analysis consumes the plan directly and Cleaning's strategy can reference plan-selected columns. If the plan hash differs, both re-run.
  - **Insights, Summary** — always re-run when upstream KPIs/evidence differ, or when `review_required` regeneration is requested.
- Idempotency guarantee: same key + same inputs → same outputs (deterministic stages deterministic; LLM stages reproducible via `temperature: 0` + `seed`, logged in manifest).
- Cache is **never** the source of truth for QA — QA always recomputes from current files when a run is delivered fresh.

### Run comparison / diffing

When a dataset evolves (same source, newer snapshot), a lightweight diff gives trend value across runs:

- Emitted as `runs/<run_id>/outputs/run_comparison.json` whenever a previous run with the same `source_name` exists (located via the `cache/` `source_name` index).
- Compare KPI-by-KPI: `{kpi_id, previous_value, current_value, abs_delta, pct_delta}` — matched by `kpi_id`, new/removed KPIs flagged.
- Also tracks: row-count, date-range, cleaned-row, QA-score deltas.
- Rendered as a "vs previous run" callout in the report. Informational only — never used in the verdict.

### Data retention & privacy policy

| Item | Default (overridable in `config.yaml`) |
|------|----------------------------------------|
| Raw uploads (`runs/<run_id>/data/raw/`) | kept **7 days**, then auto-deleted |
| Extracted + processed (`runs/<run_id>/data/extracted\|processed/`) | kept with the run (see below) |
| Run artifacts (`runs/<run_id>/`) | **30 days** retention |
| Logs (`logs/`) | **90 days** |
| PII columns | redacted from any LLM context, never written to logs; full PII in outputs only when explicitly consented |
| At-rest encryption | files under `runs/<run_id>/data/` + `runs/` stored encrypted (AES-256) when `config.encrypt_at_rest: true` |
| User deletion | `DELETE` of a run removes the run dir + its logs + cache entry immediately |
| Purging | background janitor job runs daily, deletes expired artifacts, updates manifest |

Policy is enforced by a `retention` block in `config.yaml` (days per category) + the janitor tool; QA does not depend on it.

---

## 6. Testing & Evaluation

**Test suites** (`tests/`):

| Suite | Covers |
|---|---|
| unit / integration | tools, DSL, handoffs |
| statistical | p-values / CI / effect-size vs known |
| agent | LLM JSON validity, plan viability |
| security | injection, XSS, malformed files |
| e2e on golden datasets | full pipeline matches ground truth |

**Golden datasets** (fixtures with precomputed truth, `tests/golden/fixtures/`):

| Fixture | Base data | What it tests | Expected (post-pipeline) |
|---|---|---|---|
| `sales_small` | clean, no issues | baseline correctness | revenue = 10,000 · orders = 100 · AOV = 100 |
| `sales_missing` | `sales_small` + injected MCAR/MAR gaps | missingness detection + Cleaning strategy | same totals as `sales_small` after `flag_and_preserve`/impute — flags column count also asserted |
| `sales_outliers` | `sales_small` + injected extreme values | IQR outlier handling | pre/post outlier counts asserted; totals must NOT match `sales_small` (outliers are excluded, not zeroed) |
| `sales_duplicates` | `sales_small` + exact duplicate rows | DQ `drop_duplicates()` | totals collapse back to `sales_small` values after dedup |
| `sales_injection` | `sales_small` + SQL/script strings in cells | security suite (cell content = untrusted data) | pipeline completes with content neutralized; no code execution, no unescaped HTML in report |
| `sales_pii` | `sales_small` + email/phone columns | PII detection + redaction | `pii_columns` populated in profile; PII absent from LLM context and logs |
| `hr` | independent domain fixture | domain detection + HR-specific KPIs | `detected_domain: "hr"`, domain-appropriate candidate_kpis |
| `finance` | independent domain fixture | domain detection + finance-specific KPIs | `detected_domain: "finance"`, domain-appropriate candidate_kpis |

Each derived fixture (`sales_missing`, `sales_outliers`, `sales_duplicates`, `sales_injection`, `sales_pii`) starts from `sales_small` with one problem class injected, so QA's "does the pipeline still recover the true totals" check has a fixed baseline to compare against.

**Safety checks:** cases actively poisoned with wrong numbers / missing evidence → QA must never approve (false-approval =0).

---

## 7. Status & Roadmap

**Production-oriented, not Production-ready.** Solid: isolation, PII handling, full-data compute, evidence chain, deterministic QA, guardrails, caching, retention policy. Still missing: strong evaluation harness, broad domain modules, hardened security, deeper observability.

**Roadmap (extensions, in order):**
1. Golden-dataset evaluation suite (see §6) — first P0 gate to "Production-ready".
2. **Domain modules** — folder stubs exist (`analysis/domains/`); fill each with its KPI templates + module validators. Until then the pipeline runs **generic-only** (detected domain → generic suite); `domain_confidence` decides whether domain modules apply.
3. Observability + cost panels (dashboard).
4. Interactive charts (Plotly / Chart.js) + **RTL + Arabic locale** reports.
5. Forecasting / anomaly / NLP / cohort advanced metrics.