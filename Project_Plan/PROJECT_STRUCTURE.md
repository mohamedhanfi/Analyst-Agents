# Insight Forge — Project Structure

> Quick reference for the `insight-forge/` layout and what lives inside each file, based on `Project_Plan/Analyst-Agents.md` (v4.3).
> Full implementation guide: [`Analyst-Agents.md`](./Analyst-Agents.md)

---

## Quick reference

```
insight-forge/
├── main.py                     # entry point
├── config.yaml                 # single source of config
├── .env.example                # API keys sample (never committed)
├── pyproject.toml              # dependencies
├── crew/
│   ├── crew.py                 # CrewAI Agents + Tasks in order
│   └── flows.py                # flow branches (DQ gate, Cleaning re-check, QA verdict)
├── agents/                     # one module per §2 stage — all 8 stages together
│   ├── ingestion_agent.py      # stage 1
│   ├── understanding_agent.py  # stage 2 — roles + domain + DSL plan (planning = its 2nd Task, not a separate file)
│   ├── data_quality.py         # stage 3 — Engine only, no LLM (deterministic Flow step)
│   ├── cleaning_agent.py       # stage 4
│   ├── analyst_agent.py        # stage 5
│   ├── insight_agent.py        # stage 6
│   ├── report_agent.py         # stage 7
│   └── qa_agent.py             # stage 8
├── analysis/                   # pure computation, no LLM
│   ├── chart_planner.py        # data-shape rule table → chart kind + reason (deterministic)
│   ├── evidence.py             # evidence_id minting + registry read/write (the only writer)
│   ├── generic/                # descriptive · correlation · distribution · trend
│   └── domains/                # sales · finance · marketing · hr · operations
├── shared/
│   ├── tools.py                # CrewAI @tool wrappers
│   ├── schemas.py              # Pydantic models
│   ├── dsl_validator.py        # whitelist / DSL — used by both understanding (build) and analyst (execute)
│   ├── logger.py               # structured logs to runs/<run_id>/logs/ — latency/tokens/cost + tool calls + retries
│   └── utils.py                # config load (config.yaml + env) + run_id allocator
├── resources/                  # read-only static assets
│   ├── report_template.html    # HTML report template
│   └── business_context/       # static business-context templates
├── runs/                       # run isolation — the only writable surface per run
│   └── <run_id>/               # data/ · knowledge/ · metadata/ · outputs/ · logs/ · master_manifest.json
├── cache/                      # key→run_id index (idempotency) + source_name→run_ids index (run comparison)
└── tests/                      # unit/ integration/ regression/ security/ statistical/ agent/ e2e/ golden/ fixtures/
```

---

## Root

| File | Contents | Who reads/writes |
|---|---|---|
| `main.py` | Entry point — loads `config.yaml`, builds the Crew, runs the Flow, allocates `run_id` (via `shared/utils.py`) | `python main.py <file>` |
| `config.yaml` | Single config: per-agent role/goal/backstory/model, hard limits (§1), retention (§5), review_required (§2.6) | `shared/utils.py` |
| `.env.example` | Sample API keys — copy to `.env` (the live one, gitignored) | `shared/utils.py` |
| `pyproject.toml` | Deps: crewai + pandas, numpy, scipy, statsmodels, matplotlib, seaborn, openpyxl, jinja2, weasyprint, babel | pip |

---

## crew/

| File | Contents |
|---|---|
| `crew.py` | Represents each LLM stage (1,2,4,5,6,7,8) as a CrewAI **Agent** with `role`/`goal`/`backstory` and wires its **Tasks** in pipeline order. Models per agent read from `config.yaml` — QA **must** use a different model than generation (§2.8). |
| `flows.py` | Deterministic branches: DQ gate (`passed`→Cleaning, `needs_repair`→Repair then Cleaning) · post-cleaning DQ re-check (max 3 attempts, then auto-verdict `NEEDS_REVISION`) · QA verdict incoming. Implements the Engine-only stages as plain Python steps (stage 3 → calls `agents/data_quality.py`) — no LLM. |

---

## agents/ — one module per stage

| File | Stage | Who does what | Tasks | Required tools | Outputs |
|---|---|---|---|---|---|
| `ingestion_agent.py` | 1 — Ingestion | Python: deep validation + read + sheet merge + profile + PII detection. LLM: business questions via HumanInputTool + interpretation | `validate_and_extract` · `profile_dataset` · `gather_business_context` | `file_validator_tool` (extension+MIME+signature) · `file_reader_tool` (chunked >5M rows) · `data_profiler_tool` · `pii_detector_tool` · `human_input_tool` | `data/extracted/` · `metadata/data_profile.json` · `knowledge/business_context.json` |
| `understanding_agent.py` | 2 — Understanding**+ planning (2nd Task of the same Agent)** | Python: `nunique/head/kinds` + role rules. LLM: review roles (e.g. numeric `zip_code`→identifier), detect domain/entities, propose DSL KPIs | `classify_column_roles` · `detect_domain_and_entities` · `build_analysis_plan` | `column_profiler_tool` · `domain_classifier_tool` · `dsl_plan_builder_tool` (validated against `shared/dsl_validator.py` whitelist) | `metadata/understanding.json` · `metadata/analysis_plan.json` |
| `data_quality.py` | 3 — Engine only, **no LLM** | Deterministic: schema, invalid values, missingness MCAR/MAR/MNAR, duplicates, referential integrity, business rules, units/encoding | plain functions invoked by `flows.py` (no CrewAI Task) | `schema_checker_tool` · `invalid_value_checker_tool` · `missingness_analyzer_tool` · `duplicate_detector_tool` · `referential_integrity_tool` · `deterministic_repair_tool` | `metadata/data_quality_report.json` |
| `cleaning_agent.py` | 4 — Cleaning | LLM: strategy decision (JSON). Python: fillna, `*_missing_flag`, type cast, dedup, IQR, logging + DQ re-check | `decide_cleaning_strategy` · `execute_cleaning` · `recheck_data_quality` (max 3) | `cleaning_strategy_tool` · `fillna_tool` · `flag_column_tool` · `type_caster_tool` · `dedup_tool` · `iqr_outlier_tool` · `dq_recheck_tool` | `outputs/processed/cleaned_data.csv` (+`cleaned_data_attempt_<n>.csv`) · `metadata/cleaning_result.json` |
| `analyst_agent.py` | 5 — Analysis | LLM: KPI selection, interpretation, chart re-rank. Python: DSL execution, stats suite, chart drawing, evidence registry | `select_kpis` · `run_dsl_and_stats` · `rank_chart_candidates` | `dsl_executor_tool` (whitelist via `shared/dsl_validator.py`) · `statistical_suite_tool` · `chart_planner_tool` · `chart_renderer_tool` · `evidence_registry_tool` | `outputs/kpis.json` · `statistical_results.json` · `charts/*.svg` · `metadata/chart_metadata.json` · `evidence_registry.json` |
| `insight_agent.py` | 6 — Insights | LLM-heavy: evidence-grounded insights + hedged recommendations + claim validation | `generate_insights` · `build_recommendations` · `validate_claims` | `evidence_lookup_tool` · `claim_validator_tool` · `human_input_tool` (if `review_required: true`) | `outputs/insights.json` |
| `report_agent.py` | 7 — Report | Python renders full report from JSONs via template; LLM writes **only** 3–5 sentence executive summary | `render_report` · `write_executive_summary` | `html_renderer_tool` (jinja2 autoescape) · `html_sanitizer_tool` · `locale_formatter_tool` (babel) · `chart_embed_tool` | `<run_id>/report.html` · `metadata/report_result.json` |
| `qa_agent.py` | 8 — QA | Python recomputes **100% of KPIs** (tolerance 0.01%), checks refs/charts/HTML. Independent LLM (different model) reviews logic + readability | `recompute_kpis` · `validate_structure` · `review_logic_and_readability` · `compute_verdict` (deterministic, no LLM) | `kpi_recomputation_tool` · `reference_validator_tool` · `score_calculator_tool` · `verdict_tool` | `metadata/qa_verdict.json` (APPROVED / APPROVED_WITH_WARNINGS / NEEDS_REVISION) |

---

## analysis/ — pure computation, no LLM

| File/Dir | Contents |
|---|---|
| `chart_planner.py` | Ordered rule table (§2.5): dimension/n-point shape → chart kind (line/bar/barh/donut/histogram/scatter/heatmap) + `reason` + `reliability` (`low_n` on thin data). No fixed menu — rules only. |
| `evidence.py` | evidence-ids mint + evidence_registry read/write — **the only writer** of the registry. Every value/chart/insight passes through it. |
| `generic/` | General computation: `descriptive` · `correlation` (Pearson + p-value + CI + effect) · `distribution` (histograms) · `trend` (rolling/seasonality). |
| `domains/` | Per-domain KPI templates + validators (roadmap §7): `sales` · `finance` · `marketing` · `hr` · `operations`. Until filled, pipeline runs **generic-only**. |

---

## shared/ — common infrastructure

| File | Contents |
|---|---|
| `tools.py` | All CrewAI `@tool` wrappers for the stages. Every computation/file op lives here (Golden Rule: LLM decides, Python executes). Cell content = **UNTRUSTED** — never passes to the model. |
| `schemas.py` | Pydantic models for every JSON artifact (data_profile, dataset_understanding, analysis_plan, cleaning_result, …). |
| `dsl_validator.py` | DSL whitelist + validator — **dual use**: Understanding builds, Analyst executes. Defines: `sum mean median count nunique min max std growth correlation ratio` + their parameters. |
| `logger.py` | Owner of the structured per-stage log: writes `runs/<run_id>/logs/` — LLM latency/tokens/cost, tool calls, retries (§5). |
| `utils.py` | `load_config()` (config.yaml + env) · `allocate_run_id()` — atomic `mkdir` under `runs/` (concurrency-safe). |

---

## resources/ — read-only static assets

| File/Dir | Contents |
|---|---|
| `report_template.html` | Base template for the report (all sections: Summary · Business Context · DQ Summary · Data Overview · KPIs · Stats · Charts · Insights · Recommendations · Limitations · Evidence Appendix), rendered with Jinja `autoescape=True`. |
| `business_context/` | Static business-context templates (formerly `fixtures/` — renamed from `knowledge/`). Read-only. Per-run context written to `runs/<run_id>/knowledge/business_context.json` (different thing). |

---

## runs/ — run isolation (the only writable surface)

```
runs/<run_id>/
├── data/
│   ├── raw/          # uploaded file (retention 7 days)
│   ├── extracted/    # CSV from stage 1
│   └── processed/    # cleaned_data.csv (+ cleaned_data_attempt_<n>.csv)
├── knowledge/        # business_context.json (stage 1 output, per-run)
├── metadata/         # data_profile · dataset_understanding · analysis_plan · data_quality_report · cleaning_result · chart_metadata · report_result · qa_verdict
├── outputs/          # report.html · kpis.json · statistical_results.json · insights.json · evidence_registry.json · run_comparison.json · charts/
├── logs/             # per-stage LLM/tool logs (retention 90 days)
└── master_manifest.json  # reproducibility: pipeline_version · git_sha · data_hash · model · temperature · seed · python_version · packages · analysis_mode
```

- `run_id` = `run_<timestamp>_<seq>`, minted by `shared/utils.py`.
- Parallel runs never touch the same files — each has its own dir + Crew instance.
- Retention: raw 7d · run 30d · logs 90d (configurable via `config.retention`; janitor job enforces it, QA independent).

---

## cache/ — reuse indexes

| Index | Contents |
|---|---|
| by key | `sha256(file_bytes) + config_version + prompt_version + model_ids` → `run_id` (idempotency: same input → no full re-cost). |
| by source | `source_name` → `run_ids` (for `run_comparison` across runs of the same source). |
| partial-reuse rule | Cleaning/Analysis reuse **only** when `analysis_plan.json` is byte-identical (plan-hash equality), not on blanket "context changed" (§5). |

> QA is never a caching consumer — it always recomputes from the current files.

---

## tests/ — test suites (§6)

| Dir | Covers |
|---|---|
| `unit/` | tools, DSL, small units |
| `integration/` | inter-stage hand-offs |
| `regression/` | previously fixed behavior stays fixed |
| `security/` | injection/XSS, malformed files, untrusted cell content |
| `statistical/` | p-values / CI / effect sizes vs known values |
| `agent/` | LLM JSON validity, plan viability |
| `e2e/` | full pipeline on golden datasets vs precomputed truth |
| `golden/` | golden datasets with expected values (sales_small · sales_missing · sales_outliers · sales_duplicates · sales_injection · sales_pii · hr · finance) |
| `fixtures/` | generated test files used by e2e/golden |

---

## Golden rules that keep the structure law-abiding

1. **LLM decides WHAT — Python libraries DO the work.** Python is the sole authority for every number/file/chart.
2. **Cell content = UNTRUSTED DATA** — never sent raw to the model or the template; escaped + treated dangerously.
3. Every value/chart/insight carries an `evidence_id` from `analysis/evidence.py` — full lineage in `evidence_registry.json`.
4. **Full data always** — LLM sees samples only for display/UX, but aggregations run on pre all rows; QA recomputes **100%** of KPIs.
5. Any deviation from this layout is a violation — the canonical reference is `Project_Plan/Analyst-Agents.md`.