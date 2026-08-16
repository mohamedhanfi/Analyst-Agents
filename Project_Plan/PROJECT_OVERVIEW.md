# Insight Forge — Project Overview (Plain-English Guide)

> A short, non-technical guide to the whole project: what it does, how it works,
> what is built, and what is left. Anyone can read this and explain the project.
> For full technical detail see [README.md](../README.md) and
> [Analyst-Agents.md](./Analyst-Agents.md) (the spec).

---

## 1. What this project is

**Insight Forge** turns a raw data file (CSV or XLSX) into a complete, checked
analytics report, automatically.

A user uploads a spreadsheet. The pipeline:

1. checks the file is valid and reads it,
2. figures out what kind of data it is and what each column means,
3. inspects the data for problems (missing values, negatives, duplicates…),
4. cleans it,
5. computes every KPI and statistical test on the **full** dataset and plans charts,
6. writes evidence-grounded insights and recommendations,
7. renders a final HTML report,
8. independently recomputes every number to check its own work.

The run finishes with a verdict: `APPROVED`, `APPROVED_WITH_WARNINGS`, or
`NEEDS_REVISION`.

The whole thing is built on **CrewAI** (a framework for "agent" teams) plus
pandas / scipy / statsmodels / matplotlib, with Pydantic for data contracts.

---

## 2. The golden rule

> **The LLM decides WHAT to do. Python does the WORK.**

- The LLM (a language model) is only allowed to **choose and describe**: which
  columns are measures, which KPIs matter, which cleaning strategy to use,
  which charts deserve attention, and the wording of insights.
- **Python** always does the actual computation: reading data, cleaning values,
  calculating numbers, drawing charts, and writing files.
- Two hard consequences:
  - **No raw cell content ever goes to the LLM.** Only profiles, summaries,
    and a small redacted sample.
  - **Every number and chart carries an `evidence_id`** (e.g. `EV-042`) with a
    lineage trail (source file -> cleaning steps -> calculation -> result), so
    nothing can appear in the report without a provable origin.

This makes the output trustworthy: the model cannot invent or miscalculate
numbers, because it never computes them.

---

## 3. The 8-stage pipeline

Flow: **Upload → 1 Ingestion → 2 Understanding → 3 Data Quality → 4 Cleaning →
5 Analysis → 6 Insights → 7 Report → 8 QA → Verdict**

| # | Stage | What it does | Who does it | Status |
|---|-------|--------------|-------------|--------|
| 1 | **Ingestion** | Validates the file, reads it, profiles it (shape, types, missing, PII), asks the user business questions | LLM (decisions) + Python (work) | [DONE] |
| 2 | **Understanding** | Classifies each column's role (measure / dimension / temporal / identifier…), detects the domain (sales, hr…), builds the analysis plan (a whitelist of safe KPI formulas) | LLM + Python | [DONE] |
| 3 | **Data Quality** | Deterministic checks: schema, invalid values (negative revenue, impossible dates…), missingness patterns (MCAR/MAR/MNAR), duplicates, referential integrity, business rules | Python only (no LLM) | [DONE] |
| 4 | **Cleaning** | Fixes the data per strategy: fill median/mode, flag non-random missingness, cast types, drop duplicates / negatives, IQR outliers; re-checks quality (max 3 retries) | LLM (strategy) + Python (execute) | [DONE] |
| 5 | **Analysis** | **5a – compute:** runs the whitelist KPIs, the statistical suite (descriptive, correlation, distribution, trend, comparison) and picks chart kinds from a data-shape rule table (12-kind whitelist); registers every value as evidence. **5b – draw:** renders the actual `*.svg` charts — the LLM may propose chart kinds with a reason, Python validates them (whitelist + data fit) and falls back to the rule table when rejected | LLM (select/rank/propose) + Python (validate/compute/draw) | 5a [DONE] · 5b [Task 7] |
| 6 | **Insights** | Writes evidence-grounded insights and hedged recommendations; validates every claim against the evidence registry | LLM (text) + Python (validate) | [STUB, Task 8] |
| 7 | **Report** | Renders the final HTML report (Python, from a template) and writes only the 3–5 sentence executive summary (LLM) | Python + LLM (summary only) | [STUB, Task 9] |
| 8 | **QA** | Recomputes 100% of the KPIs independently, validates structure, and issues the final verdict deterministically | Python (authoritative) + LLM (review) | [STUB, Task 10] |

Notes:
- Stages 3 is a **pure Python gate** inside the flow — not an LLM agent.
- Stage 4 re-checks quality after cleaning; if it still fails, it retries up to
  `limits.cleaning_max_rechecks` (3) then auto-fails the run.
- Anything that trips a hard limit (cost, tokens, time, chart count) stops the
  run cleanly and leads to `NEEDS_REVISION`.

### The KPI whitelist (DSL)

The LLM can only build formulas from these safe functions (no freeform math):

```
sum  mean  median  count  nunique  min  max  std  growth  correlation  ratio
```

`growth` supports YoY / MoM / WoW over a date column. This is enforced by
`shared/dsl_validator.py` — anything outside the list is rejected.

### The statistical suite (stage 5a)

- **Descriptive:** mean, median, std, quartiles, IQR, skew, kurtosis, min/max/count.
- **Correlation:** Pearson and Spearman, with p-value, 95% confidence interval,
  effect size, and sample size.
- **Distribution:** histogram bins (Freedman–Diaconis rule).
- **Trend:** YoY / MoM / WoW series, rolling means, seasonality.
- **Comparison:** t-test + Cohen's d, Mann-Whitney U, ANOVA + eta-squared,
  Kruskal-Wallis + post-hoc, chi-square + Cramér's V.

### The chart planner (stage 5a) + the renderer (stage 5b)

Python inspects each candidate's data shape and picks a chart kind from an
ordered 9-rule table (e.g. dates + ≥3 points → line; 2 measures → scatter;
≥3 measures → heatmap; share/"% of whole" → doughnut; distribution → histogram).
The **12-kind whitelist** also includes `area`, `boxplot`, `stacked_bar`, `pie`,
`lollipop` — the LLM may propose them per KPI (`proposed_kinds`); Python
validates each proposal against the whitelist and the data shape, rejecting
what cannot be drawn (with a reason) and falling back to the rule table.
If the data is too thin, the chart is downgraded to a simple bar stamped
`reliability: "low_n"`. A maximum of `max_chart_count` (20) charts are kept;
the rest are dropped and `charts_truncated` is set. Stage 5b then renders each
kept chart as a hand-rolled SVG (`analysis/chart_renderer.py`, no plotting
library): Okabe-Ito color-blind-safe palette, value labels, line markers, and
an XML-escaped caption (title + `reliability` + `evidence_id`) for alt text.

---

## 4. What happens in one run

Every run gets its own folder: `runs/<run_id>/` (e.g. `runs/demo_crew/`).
Artifacts flow stage by stage through it:

```
runs/<run_id>/
├── data/
│   ├── raw/                # the original uploaded file
│   ├── extracted/          # the readable CSV the pipeline works on
│   └── processed/          # cleaned_data.csv (+ versioned cleaned_data_attempt_*.csv)
├── knowledge/
│   └── business_context.json   # what the user told us (goals, domain)
├── metadata/               # the "brain" — JSON contracts between stages
│   ├── data_profile.json           (stage 1)
│   ├── dataset_understanding.json  (stage 2)
│   ├── analysis_plan.json          (stage 2 — the KPI plan)
│   ├── data_quality_report.json    (stage 3)
│   ├── cleaning_result.json        (stage 4)
│   ├── chart_metadata.json         (stage 5)
│   └── qa_verdict.json             (stage 8, future)
├── outputs/                # final results
│   ├── kpis.json           (stage 5)
│   ├── statistical_results.json    (stage 5)
│   ├── evidence_registry.json      (stage 5 — the lineage log)
│   ├── charts/             (stage 5b — the SVG files)
│   └── insights.json       (stage 6, future)
└── logs/
    └── run.jsonl           # audit trail: every stage + tool call
```

Each stage reads the files it needs from `metadata/` and writes its outputs
there, so stages can run standalone (you can re-run just stage 4 on an old run).

---

## 5. File structure of the repository

```
Insight Forge
├── main.py                        # [STUB] future entry point (Task 11) — empty
├── config.yaml                    # settings: agents, hard limits, retention, LLM key
├── .env / .env.example            # API key (OPENROUTER_API_KEY) — never committed
├── pyproject.toml                 # dependencies (crewai, pandas, scipy, ...)
├── crew/                          # [STUB] crew.py + flows.py (Task 11) — empty
│
├── agents/                        # one module per pipeline stage
│   ├── ingestion_agent.py         # [DONE] stage 1
│   ├── understanding_agent.py     # [DONE] stage 2
│   ├── data_quality.py            # [DONE] stage 3 (pure Python, no LLM)
│   ├── cleaning_agent.py          # [DONE] stage 4
│   ├── analysis.py                # [DONE] stage 5a (compute + crew runner) · 5b (charts) [Task 7]
│   ├── insight_agent.py           # [STUB] stage 6 (Task 8) — empty
│   ├── report_agent.py            # [STUB] stage 7 (Task 9) — empty
│   └── qa_agent.py                # [STUB] stage 8 (Task 10) — empty
│
├── analysis/                      # [DONE] pure math, no LLM (used by stage 5a)
│   ├── evidence.py                # evidence ids + evidence_registry.json (the only writer)
│   ├── dsl_executor.py            # runs the whitelist KPI formulas
│   ├── chart_planner.py           # 12-kind whitelist + rule table + proposal validation
│   ├── chart_renderer.py          # [Task 7] hand-rolled SVG renderers (12 kinds)
│   ├── generic/                   # statistical suite: descriptive, correlation,
│   │                              #   distribution, trend, comparison
│   └── domains/                   # [PLACEHOLDER] future domain KPIs
│                                  #   (sales, finance, marketing, hr, operations)
│
├── shared/                        # [DONE] shared infrastructure
│   ├── core/                      # pure logic (testable, no CrewAI)
│   │   ├── validation.py          #   file checks (extension, size, signature)
│   │   ├── reader.py              #   CSV/XLSX reading + sheet extraction
│   │   ├── profiler.py            #   data profiling + redacted sample
│   │   ├── pii.py                 #   PII column detection (rules, no LLM)
│   │   ├── business_context.py    #   business questions + Generic Mode
│   │   ├── understanding.py       #   column roles + domain + plan builder
│   │   ├── data_quality.py        #   stage-3 checks + deterministic repair
│   │   └── cleaning.py            #   stage-4 strategy table + executor
│   ├── tools/                     # thin CrewAI tool wrappers over core/
│   │   ├── file_io.py             #   file_validator / file_reader / sheet_extract
│   │   ├── profiling.py           #   pii_detector / data_profiler
│   │   ├── human.py               #   human_input (business questions)
│   │   ├── understanding.py       #   column_profiler / domain_classifier / dsl_plan_builder
│   │   ├── data_quality.py        #   7 stage-3 check/repair tools
│   │   ├── cleaning.py            #   7 stage-4 cleaning tools
│   │   └── analysis.py            #   5 stage-5 tools (dsl_executor / statistical_suite / chart_planner / chart_renderer / evidence_registry)
│   ├── schemas.py                 # Pydantic contracts for every artifact
│   ├── dsl_validator.py           # KPI whitelist rules
│   ├── llm.py                     # one LLM factory (used by all agents)
│   ├── logger.py                  # per-run structured logs
│   └── utils.py                   # config loading, run_id allocation
│
├── resources/
│   └── report_template.html       # HTML template for stage 7 (exists, wired in Task 9)
│
├── runs/                          # one folder per run (gitignored)
├── cache/                         # [PLACEHOLDER] idempotency cache (Task 11)
│
├── tests/
│   ├── unit/                      # [DONE] 34 files, 375 tests passing
│   ├── Flow_review/               # [DONE] live web viewer (app.py) for the pipeline
│   └── agent, integration, e2e, golden, fixtures, statistical, regression, security
│                                  # [EMPTY] future test suites (Task 12)
│
└── Project_Plan/                  # documentation
    ├── Analyst-Agents.md          #   the spec (v4.4) — full design
    ├── DAILY_TASKS.md             #   task tracker (12 tasks, who does what)
    ├── PROJECT_STRUCTURE.md       #   detailed structure reference
    ├── STATE.md                   #   status notes
    └── PROJECT_OVERVIEW.md        #   this file
```

---

## 6. What is done vs what is remaining

### Done (Tasks 1–6) — green baseline

| Task | Area | What exists |
|------|------|-------------|
| 1 | Contracts | `shared/dsl_validator.py` (KPI whitelist) + stage 5–8 schemas |
| 2 | Stage 1 Ingestion | `agents/ingestion_agent.py` + extraction/profile/context |
| 3 | Stage 2 Understanding | column roles, domain detection, DSL plan builder |
| 4 | Stage 3 Data Quality | full check suite + deterministic repair (pure Python) |
| 5 | Stage 4 Cleaning | strategy table, executor, versioned attempts, recheck cap |
| 6 | Stage 5a Analysis | evidence registry, DSL executor, statistical suite, chart planner, 3 tools, `agents/analysis.py` |

**Tests: 375 passing** in `tests/unit/`. Each stage is live-verified on
`runs/demo_crew` — both the deterministic path and the real CrewAI (`--crew`)
path produce identical artifacts (6 KPIs · 16 statistical tests · 4 charts ·
26 evidence entries).

### Remaining (Tasks 7–12)

| Task | Area | What's left |
|------|------|-------------|
| 7 | Stage 5b Charts | render real `*.svg` files — `analysis/chart_renderer.py` (12-kind whitelist), hybrid `validate_proposed_kinds`, `chart_renderer_tool` + `evidence_registry_tool`, `chart_path` in `chart_metadata.json` |
| 8 | Stage 6 Insights | evidence-grounded insights + recommendations + claim validator |
| 9 | Stage 7 Report | HTML report from the template + executive summary |
| 10 | Stage 8 QA | independent recomputation + final verdict (`agents.qa.model` is still a TODO) |
| 11 | Orchestration | `crew/crew.py`, `crew/flows.py`, `main.py` — wire all stages end-to-end |
| 12 | Tests & golden datasets | integration / e2e / security / golden fixtures |

Notes:
- The agent files for tasks 7–10 and the whole `crew/` + `main.py` exist as
  **empty stubs** — the placeholders are in place, the logic is not.
- **Latest commits:** Task 5 (Stage 4 Cleaning) `944b13e`, Task 6 (Stage 5a
  Analysis) `2e02c88` — pulled into the working tree.
- The report template (`resources/report_template.html`) already exists and
  will be used by Task 9.

---

## 7. How to run and verify

Everything runs in the project's conda environment (`py312_env`). The system
Python has no dependencies.

```
C:\Users\Malik\miniconda3\envs\py312_env\python.exe
```

**Run the full unit suite (must stay green):**

```
& "C:\Users\Malik\miniconda3\envs\py312_env\python.exe" -m pytest tests/unit -q -p no:cacheprovider
```

**Run each finished stage standalone** (deterministic — no API key needed):

| Stage | Command |
|-------|---------|
| 1 Ingestion | `python -m agents.ingestion_agent <file.csv/xlsx>` |
| 2 Understanding | `python -m agents.understanding_agent <run_dir>` |
| 3 Data Quality | `python -m agents.data_quality <run_dir>` |
| 4 Cleaning | `python -m agents.cleaning_agent <run_dir>` |
| 5a Analysis | `python -m agents.analysis <run_dir>` |

Add `--crew` to run a stage through the real CrewAI agent (requires the
`OPENROUTER_API_KEY` in `.env`). Example:

```
python -m agents.analysis runs\demo_crew --crew
```

**Watch the pipeline live** (web viewer for stages 1–3):

```
python tests\Flow_review\app.py --demo
```

---

## 8. Cheat-sheet glossary

| Term | Meaning |
|------|---------|
| **LLM** | The language model (via OpenRouter) that chooses and writes text. |
| **CrewAI** | The framework that runs the agents and their tasks. |
| **DSL** | The safe KPI formula language — a fixed whitelist of functions. |
| **KPI** | A single calculated value, e.g. "Total Revenue". Carries an `evidence_id`. |
| **KPI-### / EV-### / CH-###** | IDs for KPIs, evidence entries, and charts. |
| **evidence_id** | A unique id proving where a number/chart came from (lineage trail). |
| **Evidence registry** | `outputs/evidence_registry.json` — the log of all evidence. |
| **Role** | What a column is: measure / dimension / temporal / identifier / free_text. |
| **Generic Mode** | If the user can't answer business questions in time, the run continues with generic defaults instead of blocking. |
| **MCAR / MAR / MNAR** | Missingness patterns (completely random / related to other columns / related to the value itself). Cleaning treats them differently. |
| **low_n** | A reliability stamp on a chart whose data is too thin (downgraded to a simple bar). |
| **12-kind whitelist** | The only chart kinds Python can render: `bar · barh · line · doughnut · histogram · scatter · heatmap · area · boxplot · stacked_bar · pie · lollipop`. |
| **proposed_kinds** | The LLM's chart-kind suggestions `[{kpi_id, kind, reason}]`; Python validates each (whitelist + data fit) and falls back to the rule table when rejected. |
| **charts_truncated** | Flag that says some charts were dropped to respect the 20-chart cap. |
| **Verdict** | Final result: `APPROVED`, `APPROVED_WITH_WARNINGS`, or `NEEDS_REVISION`. |

---

## 9. Where to look for more detail

| For this… | See this file |
|-----------|---------------|
| The full design spec (v4.4) — every stage in depth | `Project_Plan/Analyst-Agents.md` |
| The task tracker — 12 tasks, owners, handoff log | `Project_Plan/DAILY_TASKS.md` |
| Detailed file-by-file structure | `Project_Plan/PROJECT_STRUCTURE.md` |
| The design-level README with the same info expanded | `README.md` |
| Live configuration (agents, limits, retention) | `config.yaml` |
