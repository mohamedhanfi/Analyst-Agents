# Insight Forge — Task Plan (MK & MM)

> Coordination + handoff document. **One task at a time, alternating owners.**
> Task 1 → **MK**, Task 2 → **MM**, Task 3 → **MK**, ... *(swap the names if you want MM to start instead)*
>
> Spec: [`Analyst-Agents.md`](./Analyst-Agents.md) (v4.3) · Structure: [`PROJECT_STRUCTURE.md`](./PROJECT_STRUCTURE.md)
> This file is **the source of truth for who does what and what is done**. Update it every time you finish a task.

---

## 📏 Rules of the game

1. **One task at a time.** The owner finishes the task, then hands off.
2. **Before touching code:**
   - `git pull` (get your partner's work from the previous task).
   - Read the **Handoff Log** below — anything that affects you is written there.
   - Run `pytest tests/unit -q` — baseline must be green before you start.
3. **Mid-task:** if you change anything that affects a **future** task (a shared signature, schema field, path, config key, tool name) → append a Handoff Log row **immediately**, not at the end of the task.
4. **Before finishing a task:**
   - Finish the task → mark `Done when` checklist ✓ in the task table.
   - Run `pytest tests/unit -q` once more (and any new tests you wrote).
   - Append a Handoff Log row for the task.
   - Update **Current state snapshot**.
   - `git add` + commit with a clear message + `git push`.
5. **Commit message format (no more `V1`, `5`, `00`, `**`):**
   ```
   feat(stage-1): validate + extract + profile wiring
   fix(shared/dsl_validator): reject unknown function names
   docs(DAILY_TASKS): task 3 handoff notes
   test(stage-5a): correlation p-value vs scipy reference
   ```
6. **Never commit:** `.env`, `runs/*` (except `runs/_template/`), `cache/*` — already in `.gitignore`.

---

## 🚀 Daily workflow — follow this exactly (keeps the project conflict-free)

### 0. Environment — use ONLY this interpreter

The system Python has no deps. Everything runs in the `py312_env` conda env:

```
& "C:\Users\Malik\miniconda3\envs\py312_env\python.exe" -m pytest tests/unit -q
```

Installed: `crewai 1.15.11 · pandas 3.0.3 · pydantic 2.12.5 · pytest 8.4.2`. The API key lives in `.env` (gitignored, never committed) — if you pull a fresh checkout, copy `.env.example` → `.env` and paste your `OPENROUTER_API_KEY`.

### 1. Before touching code

1. `git pull` — get your partner's work from the previous task. If it says *behind*, pull again before committing.
2. Read the **Handoff Log** — anything affecting your task is written there.
3. Run the test command above — baseline must be green before you start.

### 2. Reproduce the current code (Task 2, Stage 1)

| Command | What it does |
|---|---|
| `& "<py312>\python.exe" -m agents.ingestion_agent <file.csv/xlsx>` | Deterministic run — no LLM, no key. Same outputs. |
| `& "<py312>\python.exe" -m agents.ingestion_agent <file> --crew` | Real CrewAI agent run (needs the key). |
| `--run-dir <path>` | Write into an existing run dir instead of allocating a new one. |

Every run creates a fresh `runs/run_<YYYYmmdd_HHMMSS>_<seq>/` with:
`data/extracted/` (CSV) · `metadata/data_profile.json` · `knowledge/business_context.json` · `logs/run.jsonl` (audit trail).

**Run the `--crew` version in a real terminal** (not through tool output) so the user can answer the sheet / business questions interactively. If no terminal input is available, the run degrades cleanly to **Generic Mode** (`generic_mode: true`, multi-sheet XLSX falls back to the **largest sheet**) — never blocks.

### 3. Conflict-free rules — how we never collide

1. **One task at a time**, owner in the Task table. Don't edit files owned by another task unless the Handoff Log says it's required.
2. `git pull` at the start, `git commit` + `git push` at the end. Never push stale work; never force-push.
3. **Commit only the files you changed.** Never stage `.env`, `runs/*`, `cache/*` — they are gitignored.
4. Shared contracts (schema fields, tool names, config keys, paths) change **only** with a Handoff Log note the same task — mid-task, not at the end.
5. Baseline stays green: run `pytest tests/unit -q` **before starting** and **again before committing**.
6. Commit message format from **Rules §5** (e.g. `feat(stage-2): understanding agent + DSL plan builder + tests`).

### 4. Handoff contract — Task 3 (Stage 2 Understanding)

- **Inputs you consume:** `runs/<run_id>/metadata/data_profile.json` + `runs/<run_id>/knowledge/business_context.json` + the **20-row PII-redacted** `profile.sample` (never raw cells — golden rule).
- **Follow the same pattern as Stage 1:**
  1. Build pure logic + tools (`column_profiler_tool`, `domain_classifier_tool`, `dsl_plan_builder_tool` → validate against `shared/dsl_validator.py`).
  2. Implement `agents/understanding_agent.py` (`classify_column_roles` → `detect_domain_and_entities` → `build_analysis_plan`).
  3. Unit tests in `tests/unit/` → suite green.
  4. Standalone deterministic run + live `--crew` run on a sample CSV/XLSX.
  5. Append a Handoff Log row + update the snapshot, commit, push.

---

## 🗓️ Task table

> `Owner` follows strict alternation. Adjust the table if you change the starting owner or the number of tasks.

| # | Owner | Task | What to do | Done when |
|---|-------|------|-----------|-----------|
| 1 | MK | ✅ **Contracts — DSL + schemas 5–8** *(DONE)* | Implement `shared/dsl_validator.py` (whitelist: `sum mean median count nunique min max std growth correlation ratio` + parameter validation per §2.5 function signatures; reject anything outside the whitelist — no freeform formulas). Extend `shared/schemas.py` with stage 5–8 models: KPI result, statistical results, chart metadata, evidence entry + registry, insight + recommendation, report result, QA verdict (Pydantic v2, same style as existing file). Write `tests/unit/test_dsl_validator.py` + a schema round-trip test. | ✅ Done — `pytest -q` green (94 passed); non-whitelisted ops rejected; schemas round-trip cleanly. |
| 2 | MM | ✅ **Stage 1 — `agents/ingestion_agent.py`** *(DONE)* | Build the first CrewAI agent (role/goal/backstory from `config.yaml` `agents.ingestion`). Tasks: `validate_and_extract` (file_validator_tool → file_reader_tool; XLSX multi-sheet → pick via `business_context.sheet_used` → file_sheet_extract_tool) · `profile_dataset` (pii_detector_tool → data_profiler_tool) · `gather_business_context` (human_input_tool / BusinessContextGatherer, Generic Mode on timeout). Allocate `run_id`, wire `RunLogger`, write outputs under `runs/<run_id>/`. **Never pass raw cell content to the LLM.** | ✅ Done — deterministic + `--crew` runs on CSV & XLSX produce all 3 outputs; multi-sheet asks user (fallback: largest sheet); 8 unit tests green. |
| 3 | MK | ✅ **Stage 2 — Understanding + DSL tools** *(DONE)* | Add `column_profiler_tool` (per-column facts: dtype / nunique / nullable — role rules from §2.2) to `shared/tools/`. Add `domain_classifier_tool` (profiled facts the LLM uses to name domain/entities) and `dsl_plan_builder_tool` (validates the LLM's proposed KPIs against `shared/dsl_validator.py`, normalizes into `AnalysisPlan`). Implement `agents/understanding_agent.py`: `classify_column_roles` → `detect_domain_and_entities` → `build_analysis_plan` (2nd task, same agent — **not** a 9th agent). Outputs `dataset_understanding.json` + `analysis_plan.json`. | ✅ Done — outputs match the §2.2 example shape (live deterministic run on CSV); plan emits whitelist ops only (rejected `evil`/bad ratios in tests); role rules unit-tested; `_finalize_understanding` Python-authoritative with deterministic fallbacks; suite green (157 passed). |
| 4 | MM | ✅ **Stage 3 — `agents/data_quality.py`** *(DONE)* | Plain functions (no CrewAI) invoked later by `flows.py`. Checks per §2.3: schema · invalid values (impossible dates, negative revenue, % >100, age=350) · missingness MCAR/MAR/MNAR · duplicates · referential integrity · business rules · units/encoding. Deterministic repair per the §2.3 table (cast types, drop exact dupes, drop impossible rows — **never sign-flip, never invent data**). Wrap each check as a tool in `shared/tools/` (schema_checker, invalid_value_checker, missingness_analyzer, duplicate_detector, referential_integrity, deterministic_repair). Output `data_quality_report.json`. | ✅ Done — **192 passed**; live run on `sales_demo.csv` → `needs_repair` (invalid revenue=[negative], 2 dups, missingness MCAR 0.0014, repair drops both dupes); Flow Review lights up stage 3 with all 7 tool events + report/repair-log panels; `python -m agents.data_quality <run_dir>` CLI works. |
| 5 | MK | **Stage 4 — `agents/cleaning_agent.py`** | LLM picks a strategy JSON (`decide_cleaning_strategy`); Python executes. Tools: cleaning_strategy, fillna, flag_column (`*_missing_flag`), type_caster, dedup, iqr_outlier, dq_recheck. Implement the missing-value strategy table §2.4 (role × missingness type) incl. `flag_and_preserve`. Version re-run attempts as `cleaned_data_attempt_<n>.csv` (v4.3). Re-check DQ on output; re-run logic up to `limits.cleaning_max_rechecks` (3). Outputs `data/processed/cleaned_data.csv` + `metadata/cleaning_result.json`. | Cleaned data + `cleaning_result.json`; strategy matches §2.4; recheck loop works; unit tests. |
| 6 | MM | **Stage 5a — Analysis compute layer** | `analysis/evidence.py` — evidence_id minting + registry read/write (**the only writer**). `analysis/generic/` — descriptive (mean/median/std/quantiles/IQR/skew/kurtosis), correlation (Pearson + **p-value + CI + effect size + n**, Spearman), distribution (histograms, Freedman–Diaconis bins), trend (YoY/MoM/WoW, rolling, seasonality). `analysis/chart_planner.py` — §2.5 rule table → chart kind + `reason` (+ `reliability: low_n` fallback). Tools: dsl_executor_tool (executes whitelist ops over **all rows**), statistical_suite_tool, chart_planner_tool. | Every DSL op unit-tested against known values; correlation numbers match scipy/statsmodels; planner picks kinds per the table. |
| 7 | MK | **Stage 5b — `agents/analyst_agent.py` + charts** | Agent tasks: `select_kpis` → `run_dsl_and_stats` → `rank_chart_candidates` (re-rank carries a reason; final draw is Python). Tools: chart_renderer_tool (SVG, color-blind-safe palette, value labels / line markers, caption for alt text), evidence_registry_tool (writes `evidence_registry.json` via `analysis/evidence.py`). Enforce `max_chart_count` (20) + `charts_truncated` flag. Outputs: `kpis.json` · `statistical_results.json` · `charts/*.svg` · `chart_metadata.json` · `evidence_registry.json`. | Full stage-5 outputs from a cleaned dataset; **every value/chart has an `evidence_id`**; ≤20 charts. |
| 8 | MM | **Stage 6 — `agents/insight_agent.py`** | Tasks: `generate_insights` → `build_recommendations` → `validate_claims`. Claim taxonomy §2.6 (DESCRIPTIVE/COMPARATIVE/CORRELATIONAL/PREDICTIVE/CAUSAL gating). Recommendation chain Observation → Finding → Implication → hedged recommendation. Tools: evidence_lookup_tool, claim_validator_tool (evidence_ids non-empty · refs exist · claim matches evidence types · recommendations reference existing insights; failure → remove + log). Human review gate if `config.review_required: true` (approve / edit / regenerate). Output `insights.json`. | Valid `insights.json`; validator rejects ungrounded claims; review gate works when enabled. |
| 9 | MK | **Stage 7 — `agents/report_agent.py`** | Tasks: `render_report` (Python, Jinja) + `write_executive_summary` (LLM writes **only** the 3–5 sentence summary). Tools: html_renderer_tool (`autoescape=True`), html_sanitizer_tool, locale_formatter_tool (babel — en default, decimals/dates per `locale`), chart_embed_tool (each `<img>` gets a caption). Finalize `resources/report_template.html` (Summary · Business Context · DQ Summary · Data Overview · KPIs · Stats · Charts · Insights · Recommendations · Limitations · Evidence Appendix). Outputs `runs/<run_id>/report.html` + `metadata/report_result.json`. | Full HTML renders from all JSONs; XSS-safe (escape + sanitize + CSP); no raw cells in the page. |
| 10 | MM | **Stage 8 — `agents/qa_agent.py`** | Tasks: `recompute_kpis` (recompute **100%** of KPIs from cleaned data, tolerance 0.01%) → `validate_structure` (refs, charts exist, HTML sections) → `review_logic_and_readability` (independent LLM) → `compute_verdict` (deterministic, no LLM). Tools: kpi_recomputation_tool, reference_validator_tool, score_calculator_tool, verdict_tool. Score formula + verdict table §2.8 (score is informational only). Set a **distinct QA model** in `config.yaml` (`agents.qa.model` — currently `TODO`). Output `qa_verdict.json`. | Verdict correct per the §2.8 table; poisoned evidence / fallback / cap-exceeded → `NEEDS_REVISION`; score never overrides logic. |
| 11 | MK | **Orchestration — crew + flows + main** | `crew/crew.py` — wire all agents + tasks in pipeline order (§1). `crew/flows.py` — deterministic branches: DQ gate (`passed`/`needs_repair` → repair then cleaning) · post-cleaning re-check loop (max 3 → auto-verdict `cleaning_retry_limit_exceeded`) · QA verdict branch · caps → fallback reasons (`cost_limit_exceeded` / `token_limit_exceeded` / `run_time_limit_exceeded`). Stage 3 invoked here as a plain Python step. `main.py` — entry: load config, allocate `run_id`, build Crew, run Flow, write `master_manifest.json`. | `python main.py <file>` runs the **full 8-stage pipeline** end-to-end on a sample CSV and writes a run dir with all artifacts + verdict. |
| 12 | MM | **Tests & golden datasets** | `tests/agent/` (LLM JSON validity, plan viability) · `tests/integration/` (inter-stage hand-offs) · `tests/e2e/` (full pipeline on golden datasets vs precomputed truth). `tests/golden/` — per-fixture expected values table (§6: sales_small · sales_missing · sales_outliers · sales_duplicates · sales_injection · sales_pii · hr · finance) + `tests/fixtures/` generated files. **Safety cases:** datasets poisoned with wrong numbers / missing evidence → QA must **never** approve (false-approval = 0). | e2e passes on golden sets; poisoned cases always `NEEDS_REVISION`; `pytest -q` fully green. |

---

## 📋 Handoff log

> Fill one row at the **end of every task** (and mid-task when you change something that affects a future task).
> Format: `# | owner | task | what changed | affects next task | tests green?`

| # | Owner | Task | What changed (file → what) | Affects next task | Tests green? |
|---|-------|------|----------------------------|-------------------|--------------|
| 1 | MK | Contracts | `shared/dsl_validator.py` → whitelist + semantic validator (`validate_operation` / `validate_plan`); `shared/schemas.py` → added stage 5–8 models (KPI result, statistical results, chart metadata, evidence, insight, recommendation, report result, QA verdict); **`DslOperation.basis` default moved to `None`** (was `"previous_period"` — polluted every op's `model_dump`; executor applies the default at runtime); `tests/unit/test_dsl_validator.py` (27) + `tests/unit/test_schemas.py` (16) added | Task 2+ import these; **Task 6 growth executor must default `basis` → `previous_period` when absent** | ✅ 94 passed |
| 2 | MM | Stage 1 | `agents/ingestion_agent.py` → `build_ingestion_agent()` / `build_ingestion_tasks()` (3 CrewAI Tasks) + `run_ingestion(file, use_crew=True|False)` + CLI (`python -m agents.ingestion_agent <file> [--crew]`); run_id + `RunLogger` wired; **`shared/core/business_context.py` → EOFError on stdin now returns Generic Mode** (was ugly thread traceback); crew path `_finalize_profile` **recomputes file_hash/file_name in Python** (LLM tool args untrusted — golden rule); multi-sheet crew description falls back to **largest sheet** on empty/timeout answer | Task 3+ imports these; **Understanding (Task 3) consumes `data_profile.json` + `business_context.json` + 20-row sample** | ✅ 102 passed |
| 3 | MK | Stage 2 | **new** `shared/core/understanding.py` → `ColumnProfiler` + §2.2 role rules, `build_domain_facts`, `build_analysis_plan` (gates every KPI through `dsl_validator.validate_operation`, drops invalid + logs reasons, whitelist statistical tests), `apply_role_overrides` (LLM may reclassify into alternates/identifier only — Python authoritative), `detect_domain_heuristic` (keyword scan of context answers; generic → `("generic", 0.0)`), `default_plan` (deterministic whitelist plan from profile), `assemble_understanding`; **new** `shared/tools/understanding.py` → `column_profiler_tool` · `domain_classifier_tool` · `dsl_plan_builder_tool` (all JSON-metadata only, never raw cells); **new** `agents/understanding_agent.py` → 3 Tasks in ONE agent + `run_understanding(run_dir, use_crew=...)` + CLI (`python -m agents.understanding_agent <run_dir> [--crew]`) + `_finalize_understanding` (Python-authoritative, LLM failure → deterministic fallback + log warning); **new** `shared/llm.py` → single `build_llm(cfg, agent_name)` factory; **`agents/ingestion_agent.py` refactored** to import it (local `build_llm` removed) — Task-2 file touched; **`shared/core/business_context.py` → input returning `None` now = Generic Mode** (was `NoneType.strip` crash when stdin unavailable); **new `tests/Flow_review/app.py` + `index.html` → live web flow viewer** (stdlib only, `python tests/Flow_review/app.py [--demo] [--port]`; uploads a file, runs ingestion → understanding live, renders stage cards + live event trail from run.jsonl + artifact previews; stages 3-8 appear as dimmed "Task N" cards and light up automatically as they land); tests: +55 unit (role rules, domain facts, plan builder, understanding core, agent deterministic + finalize, llm factory, business_context None) | Task 4 consumes `dataset_understanding.json`; **future agents use `shared.llm.build_llm(cfg, "<agent>")` — do NOT copy a local build_llm**; **Task 6 growth executor: default `basis` → `previous_period` when absent**; **Flow_review shows new stages automatically once they write to run.jsonl — no changes needed** | ✅ 157 passed |
| 4 | MM | Stage 3 | **new** `shared/core/data_quality.py` → all §2.3 logic, pure + deterministic: `check_schema` (missing/unknown columns, role-type mismatch), `check_invalid_values` (negative measure · % >100 · age out-of-range · impossible dates (unparseable / year ≥2100) · future dates beyond today+365d), `check_business_rules` (Generic Mode skips; parses declared ranges from goal/answers: between/from-to and >=; case-duplicate column → encoding), `analyze_missingness` (per-column + overall rate; MCAR / MAR_suspected (group-rate gap >0.2) / MNAR_suspected (numeric, index-thirds gap >0.3)), `detect_duplicates`, `check_referential_integrity` (identifier nulls + name-based orphan refs — avoids false positives from unrelated overlap), `deterministic_repair` (§2.3 table: cast measure→numeric + temporal→datetime, drop exact dupes, drop impossible rows + log indices — **no sign flips, no imputation**; string identifiers kept), `assemble_report` → `DataQualityReport` (status gate: high-severity OR repair-applied → `needs_repair`); **new** `agents/data_quality.py` → `run_data_quality(run_dir)` + CLI (`python -m agents.data_quality <run_dir>`), logs all 7 tool calls via RunLogger; **new** `shared/tools/data_quality.py` → 6+1 `@tool` wrappers (schema_checker · invalid_value_checker · business_rules_checker · missingness_analyzer · duplicate_detector · referential_integrity · deterministic_repair) exported from `shared/tools/__init__`; **`tests/Flow_review/app.py` + `index.html`** → stage-3 wired into the live pipeline (runs after understanding, `needs_repair` shows on the card, report/repair-log artifact panels render — `ARTIFACTS` map extended on both sides); tests: +35 unit (checks · missingness MCAR/MAR/MNAR · repair golden table · fixture report) | Task 5 (Cleaning) consumes `data_quality_report.json` + `repair_log.json`; **repair runs in-memory only — it never rewrites the extracted CSV (Cleaning owns writing `cleaned_data.csv`)**; `run_data_quality(run_dir, cfg=...)` is the call contract `flows.py` must use (Task 11) | ✅ 192 passed |
| 4b | MM | Stage 2 role rules fix | `shared/core/understanding.py` → `infer_role` **reordered**: dtype-datetime → `temporal` + numeric → `measure` now fire **before** `nunique == row_count → identifier`; new id-name heuristic (`_id`/`id`/`code`/`key`/`sku` → identifier) + date-name heuristic (`date`/`time`/`year`/… → temporal for str columns); `apply_role_overrides` widened to dtype-plausible roles so the crew LLM may flip an all-unique numeric between `measure` ↔ `identifier` (e.g. `zip_code` → identifier per §2.2); `tests/unit/test_column_profiler.py` +6 regression tests; **spec §2.2 rule table amended to match** | Task 5 (Cleaning) consumes `dataset_understanding.json` roles — all-unique measures (`revenue`) & temporal (`date`) are now classified correctly (verified: 7-row run went 1 → 6 KPIs, `has_temporal_data: true`) | ✅ 205 passed |
| 5 | MK | Stage 4 | *(fill at end of task)* | | |
| 6 | MM | Stage 5a | *(fill at end of task)* | | |
| 7 | MK | Stage 5b | *(fill at end of task)* | | |
| 8 | MM | Stage 6 | *(fill at end of task)* | | |
| 9 | MK | Stage 7 | *(fill at end of task)* | | |
| 10 | MM | Stage 8 | *(fill at end of task)* | | |
| 11 | MK | Orchestration | *(fill at end of task)* | | |
| 12 | MM | Tests & golden | *(fill at end of task)* | | |

---

## 📌 Current state snapshot

> Update this every time you finish a task.

- **Last finished task:** Task 4 — Stage 3 Data Quality — ✅ done; **today's Stage 2 role-rules fix applied** (`infer_role` reorder + id/date name heuristics + widened `apply_role_overrides`; spec §2.2 table amended to match). Verified end-to-end on the 7-row CSV run: roles correct (`date`→temporal, `revenue`/`quantity`→measures), plan went **1 → 6 whitelist KPIs** (sum/mean/growth/correlation), Stage 3 re-run clean (`needs_repair` only for the expected temporal cast `date` object→datetime64); **`205 passed in ~11s`**
- **Last commit:** `b0168ef` (MK, "Update", pushed) — Tasks 3 + 4 committed + pushed by MK across `56c6c1d` → `9ca1e3d` → `b0168ef`
- **Baseline tests:** 205 passing in `tests/unit/` (94 contracts + 8 ingestion + 55 understanding + 35 data quality + 7 data-quality tools + 6 role-rules regression)
- **Working tree:** 4 files modified, **uncommitted** — `shared/core/understanding.py` · `tests/unit/test_column_profiler.py` (today's Stage 2 role-rules fix) + `Project_Plan/Analyst-Agents.md` (§2.2) + `Project_Plan/DAILY_TASKS.md` (this) — **commit before Task 5**; `.env` is local + gitignored (OpenRouter key)
- **Open items / decisions:**
  - Task 5 Cleaning (**next task, owner MK**): consume `data_quality_report.json` + `repair_log.json` + `dataset_understanding.json` roles; deterministic repair stays in-memory — Cleaning owns writing `data/processed/cleaned_data.csv`; re-check loop up to `limits.cleaning_max_rechecks`; start protocol: `git pull` → read handoff row 4b → pytest baseline 205 → implement `agents/cleaning_agent.py`
  - Task 6 growth executor: default `basis` → `previous_period` when absent
  - `config.yaml` `agents.qa.model` still `TODO` — needs a distinct QA model before Task 10
  - Use `py312_env` (pydantic 2.12.5, pandas 3.0.3, pytest 8.4.2, crewai 1.15.11); system Python lacks deps
  - `runs/` demo runs from today are gitignored (retention policy will sweep them)
