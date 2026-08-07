# Insight Forge — Implementation Guide

**Version:** 4.0.0 · **For:** Development Team · **Status:** Production-oriented (not production-ready — §7)

---

## 1. Overview

### Golden Rule
> **The LLM decides WHAT to do. Python libraries DO the actual work.**

| | LLM does | Python does |
|---|---|---|
| Never | compute, draw, clean, read raw files | — |
| Always | decide what to run, interpret results, write insights & summary | every number, file, chart, cleaning op, validation |

Computation stack: **pandas, numpy, scipy, statsmodels, matplotlib, seaborn, openpyxl, jinja2, weasyprint**.

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
        │                 └─ Repair: deterministic fix, internal to Agent 3
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
- DQ gate → `needs_repair`: **Repair is an internal step of Agent 3** (not a separate stage) — Python fixes safe items deterministically, flags the rest for Cleaning
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
| Max chart count | 20 |
| Per-stage timeout | 600 s (default, `config.yaml`) — exceeded → retry/fail that stage |
| Per-run LLM cost cap | `config.yaml` `max_cost_usd` (default 5.00) — hard stop + fallback |
| Per-agent token cap | `config.yaml` `max_tokens_per_agent` (default 50k in/out) |
| Total run duration | `config.yaml` `max_run_seconds` (default 1800) — hard stop |

**>5M rows:** Python parses via chunking (`pandas.read_csv(chunksize=50_000)`, `openpyxl.read_only`) so memory stays bounded; aggregations (sums, means, counts) are streamed chunk by chunk so final numbers are still **full-data**, never sampled. LLM/UX previews stay small.

### Crew (CrewAI — the only framework)

- Every stage below = one CrewAI **Agent** (`role`/`goal`/`backstory`) with CrewAI **Tasks**
- All computation = CrewAI **Tools** (Python `@tool` wrappers)
- Branching = CrewAI **Flows** (`@flow`/`@router`)
- Models per agent set in `crew/config.yaml`
- **QA must use a different model than generation** (§2.8)

| Stage | Crew | Type |
|-------|------|------|
| 1 Ingestion | Agent | Engine + small LLM |
| 2 Understanding | Agent | LLM-heavy |
| 3 Data Quality | Agent | Engine only |
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
- `data/extracted/` (CSV)
- `metadata/data_profile.json`
- `knowledge/business_context.json`

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

**Notes:** stop on unsupported format / empty or < 5 rows / user cancel / fail after **3 retries** (see §1 hard limits). Cell content = **UNTRUSTED DATA**. Question timeout 5 min → *Generic Analysis Mode* (§3.4).

---

### 2.2 Agent 2: Data Understanding

**Mission:** from **profile + 20-row sample only**, classify column roles, detect domain, propose Analysis Plan in DSL.

| Who | Does |
|-----|------|
| Python | `nunique()`, `head()`, datetime/numeric detection, role rules |
| LLM | review roles, detect domain, entities, propose DSL KPIs, build plan |

**Column-role rules (Python):**

```
nunique == row_count          → identifier
dtype datetime                → temporal
numeric & nunique > 20        → measure
numeric & nunique ≤ 20        → measure or categorical
object & nunique ≤ 20         → dimension
object & nunique > 50         → free_text
```

LLM may reclassify by name (e.g. numeric `zip_code` → identifier).

> **Planning is a subtask of this same stage** — implemented by `planner_agent.py`, but it is **not** a 9th agent; it is a second Crew task of the Understanding Agent producing the DSL plan.

**Input:** `data_profile.json` · `business_context.json` · 20-row sample
**Output:**
- `metadata/dataset_understanding.json`
- `metadata/analysis_plan.json` (DSL ops only — no freeform formulas)

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
**Output:** `metadata/data_quality_report.json`

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
- `data/processed/cleaned_data.csv`
- `metadata/cleaning_result.json` (pre/post rows, dupes removed, type casts, flags created, outliers)

**Notes:** result validation re-runs DQ on cleaned output → `passed` or re-run Cleaning.

---

### 2.5 Agent 5: Analysis

**Mission:** compute **everything on the full dataset**. LLM chooses what/how; Python computes + draws + memorializes evidence.

| Who | Does |
|-----|------|
| LLM | select KPIs, interpret results, re-rank chart candidates (with reasons) |
| Python | execute DSL (whitelist only), stats suite, **planner picks chart shape**, draw, evidence registry |

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
  "kpi_id": "KPI-003", "name": "Revenue QoQ Growth",
  "operation": {
    "function": "growth",
    "column": "revenue",
    "over_column": "order_date",
    "period": "MoM",
    "basis": "one_period",
    "group_by": ["category"],
    "filter": null,
    "as_percent": 1
  },
  "value": 12.4, "evidence_id": "EV-003", "computed_by": "pandas"
}
```

`growth` semantic: `(current − baseline) / baseline`; if `over_column` alone → month-basis; for YoY use `period: "YoY"` (compares same calendar month/quarter vs prior year).

**Charts — data-driven planner (not a fixed menu)**

The chart type is **decided by the shape of the data**, not by habit. Python inspects each candidate visual at `analysis/chart_planner.py` and picks the form from an ordered rule table. The LLM only chooses *which facts deserve a chart* and may re-rank candidates with a justification; the shape itself is deterministic:

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

Wiring is: **`chart_planner`** takes the KPI list + `data_profile` + numeric/ordinal columns → returns one JSON per chart:
```json
{
  "chart_id": "CH-004", "kind": "line",
  "reason": "1 ordered dim (month) >= 3 points -> line",
  "columns": ["order_date", "revenue"],
  "data": [...], "evidence_id": "EV-007", "computed_by": "pandas"
}
```
Falling back if the data is too thin: planner downgrades to a simple bar or a table, and stamps `reliability: "low_n"`. If the LLM re-ranks, the plan must carry the reason — the final draw is still Python. `chart_metadata.json` records the planner decision + the shape rule hit, so chart choice is reproducible (see §5).

**Input:** cleaned data · plan · understanding · business context · cleaning result
**Output:** `outputs/kpis.json`, `outputs/statistical_results.json`, `outputs/charts/*.svg`, `metadata/chart_metadata.json` (planner + shape rules + evidence refs), `outputs/evidence_registry.json`

**Rule:** Python always aggregates on **all rows**; sampling is for LLM/UX inspection only. Every computed value gets an evidence_id.

**Chart accessibility:**
- **Color-blind safe palettes** — use `seaborn` colorblind / Okabe-Ito; never red-green as the only encoding.
- **Pattern + label redundancies** — bar/pie also label values; line charts get markers, not color-only.
- **Alt text** — each `<img>` in the HTML report carries a caption generated from chart metadata + associated insights.

**Localization:**
- All numeric/datetime formatting follows the report's `locale` (from business context; default `en`): decimal separators and date formats (`en-US` → `1,234.5`, `ar-EG` → `١٬٢٣٤٫٥`).
- Formatting is applied in the Report render step by Python (`babel`), not the LLM.
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
**Output:** `outputs/insights.json`

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

**Human review checkpoint (optional, `config.yaml` `review_required: true`):** after `insights.json` is validated, the pipeline can pause and present the insights to a human analyst (HumanInputTool / web panel). The analyst can **approve**, **edit text** (edits must keep `evidence_ids` intact — the claim validator re-runs), or **request regeneration** (→ Insights re-run, bounded by retry caps). If approved or `review_required: false`, it proceeds to Report. This is a review gate, not a second LLM.

---

### 2.7 Agent 7: Report

**Mission:** render the full HTML report. Python renders; LLM writes **only** the 3-5 sentence executive summary.

| Section | Source |
|---------|--------|
| Summary, Business Context, DQ Summary, Data Overview, KPIs, Stats, Charts, Insights, Recommendations, Limitations, Evidence Appendix | LLM summary + all JSONs |

**Security (in render):** Jinja `autoescape=True`, HTML sanitizer, CSP header. Never render cells raw.

**Input:** all `outputs/`, `metadata/`, `knowledge/` files · `templates/report_template.html`
**Output:** `outputs/report.html`, `metadata/report_result.json`

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

**Output:** `metadata/qa_verdict.json`

---

## 3. Guardrails (rules that hold the whole system together)

1. **Full data, always.** Python aggregates the complete dataset. LLM/UX see samples/previews only.
2. **Python is authoritative.** Every numeric output is recomputed by QA from the cleaned data — **100% of KPIs**, not a sample — before "done".
3. **Evidence is concrete.** Every number, chart, insight, and claim carries an `evidence_id`; lineage = metadata for reproducible auditing.
4. **Claims are conservative.** Descriptive/comparative/correlational by default; predictive & causal gated (require methodology).
5. **User is optional.** If business questions time out → Generic Mode with `context_confidence: 0`, report says "generation of context-specific recommendations was not possible."

---

## 4. File Structure

```
insight-forge/
├── main.py                     # entry — builds Crew, runs Flow
├── crew/
│   ├── crew.py                 # CrewAI Agents + Tasks in order
│   ├── flows.py                # DQ gate, Cleaning re-check, QA verdict branches
│   └── config.yaml             # per-agent role/goal/backstory/model
├── agents/                     # one module per §2 stage
│   ├── ingestion_agent.py
│   ├── understanding_agent.py      # stage 2 — roles + domain
│   ├── planner_agent.py            # stage 2 subtask — builds the DSL analysis plan (same Agent as understanding)
│   ├── data_quality_agent.py
│   ├── cleaning_agent.py
│   ├── analyst_agent.py
│   ├── insight_agent.py
│   ├── report_agent.py
│   └── qa_agent.py
├── analysis/                   # pure computation, no LLM
│   ├── generic/ (descriptive✓ correlation✓ distribution✓ trend)
│   └── domains/ (sales•finance•marketing•hr•operations)
├── shared/
│   ├── tools.py                # CrewAI @tool wrappers
│   ├── schemas.py              # Pydantic
│   ├── dsl_validator.py        # whitelist / DSL
│   └── utils.py
├── templates/  → report_template.html
├── knowledge/  → business_context.json
├── data/       → raw/ extracted/ processed/
├── outputs/    → report.html · kpis.json · statistical_results.json · insights.json · evidence_registry.json · charts/ · metadata/ · run_comparison.json
├── runs/       → <run_id>/   (run isolation)
├── logs/       → run_<id>/   (LLM/tool logs)
├── cache/      → key→run_id index for idempotency (§5)
└── tests/      → unit/ integration/ regression/ security/ statistical/ agent/ e2e/ golden/ fixtures/
```

---

## 5. Reproducibility & Observability

Published for every run, side by side with the output files:

- **Manifest** (`master_manifest.json`) — branch-agnostic **everything** to reproduce a run: `pipeline_version`, `code git_sha`, `data_hash`, `model`, `temperature`, `seed`, `python_version`, `packages`, `analysis_mode`.
- **Structured log per stage**: LLM call latency/tokens/cost + tool calls, errors, retries as audit trail.
- **Dashboard metrics**: runs, latency, tokens, cost, failures/fallbacks, QA score.

### Run isolation & concurrency

Each run gets a unique `run_id` (`run_<timestamp>_<seq>`) and writes **only** to `runs/<run_id>/` — extracted data, cleaned data, all outputs, logs, and its own manifest live there. Isolation model:

- **Scope:** one pipeline run = **its own sub-directory + own Crew instance**. Two runs never touch the same files.
- **Concurrency:** safe to launch many runs in parallel. Python-side (pandas/matplotlib) is process-local; no global temp dirs, no shared in-memory state.
- **Locking:** only the output root (`runs/`) needs a lightweight `run_id` allocator (atomic `mkdir`); read of shared fixtures (`knowledge/`, `templates/`) is read-only and safe concurrently.
- **Teardown:** `logs/` per run; retention policy (configurable) deletes old runs.

### Caching & idempotency

Re-running the **same input** is detectable and parts of it are skipped instead of paying full cost/time:

- **Cache key** = `sha256(file_bytes)` + `config_version` + `prompt_version` + model ids. Stored in a `cache/` index (key → run_id).
- **Full-file hit:** if the same `input_hash` + effective-config already produced an `APPROVED` run, the orchestrator returns that run's report/`run_comparison` directly (with a `cached: true` marker) — no LLM calls, no recompute.
- **Partial hits:** deterministic stages (Data Quality, Cleaning, Analysis) can be reused when only the business-context answers differ; LLM-heavy stages (Understanding/plan, Insights, Summary) are re-run.
- Idempotency guarantee: same key + same inputs → same outputs (deterministic stages deterministic; LLM stages reproducible via `temperature: 0` + `seed`, logged in manifest).
- Cache is **never** the source of truth for QA — QA always recomputes from current files when a run is delivered fresh.

### Run comparison / diffing

When a dataset evolves (same source, newer snapshot), a lightweight diff gives trend value across runs:

- Emitted as `outputs/run_comparison.json` whenever a previous run with the same `source_name` exists.
- Compare KPI-by-KPI: `{kpi_id, previous_value, current_value, abs_delta, pct_delta}` — matched by `kpi_id`, new/removed KPIs flagged.
- Also tracks: row-count, date-range, cleaned-row, QA-score deltas.
- Rendered as a "vs previous run" callout in the report. Informational only — never used in the verdict.

### Data retention & privacy policy

| Item | Default (overridable in `config.yaml`) |
|------|----------------------------------------|
| Raw uploads (`data/raw/`) | kept **7 days**, then auto-deleted |
| Extracted + processed (`data/extracted|processed/`) | kept with the run (see below) |
| Run artifacts (`runs/<run_id>/`) | **30 days** retention |
| Logs (`logs/`) | **90 days** |
| PII columns | redacted from any LLM context, never written to logs; full PII in outputs only when explicitly consented |
| At-rest encryption | files under `data/` + `runs/` stored encrypted (AES-256) when `config.encrypt_at_rest: true` |
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

**Golden datasets** (fixtures with precomputed truth):
```
sales_small · sales_missing · sales_outliers ·
sales_duplicates · sales_injection · sales_pii · hr · finance
Expected: revenue = 10,000 · orders = 100 · AOV = 100
```

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