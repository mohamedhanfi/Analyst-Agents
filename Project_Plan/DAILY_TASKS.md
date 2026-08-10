# Insight Forge — Daily Task Plan (MK & MM)

> Coordination + handoff document. **One task per day, alternating owners.**
> Day 1 → **MK**, Day 2 → **MM**, Day 3 → **MK**, ... *(swap the names if you want MM to start instead)*
>
> Spec: [`Analyst-Agents.md`](./Analyst-Agents.md) (v4.3) · Structure: [`PROJECT_STRUCTURE.md`](./PROJECT_STRUCTURE.md)
> This file is **the source of truth for who does what and what is done**. Update it every day before you commit.

---

## 📏 Rules of the game

1. **One task per day.** Each of us works a full day, finishes the task, then hands off.
2. **Start of your day (before touching code):**
   - `git pull` (get your partner's work from yesterday).
   - Read the **Handoff Log** below — anything that affects you is written there.
   - Run `pytest tests/unit -q` — baseline must be green before you start.
3. **Mid-day:** if you change anything that affects a **future** task (a shared signature, schema field, path, config key, tool name) → append a Handoff Log row **immediately**, not at the end of the day.
4. **End of your day (before leaving):**
   - Finish the day's task → mark `Done when` checklist ✓ in the task table.
   - Run `pytest tests/unit -q` once more (and any new tests you wrote).
   - Append a Handoff Log row for the day.
   - Update **Current state snapshot**.
   - `git add` + commit with a clear message + `git push`.
5. **Commit message format (no more `V1`, `5`, `00`, `**`):**
   ```
   feat(stage-1): validate + extract + profile wiring
   fix(shared/dsl_validator): reject unknown function names
   docs(DAILY_TASKS): day 3 handoff notes
   test(stage-5a): correlation p-value vs scipy reference
   ```
6. **Never commit:** `.env`, `runs/*` (except `runs/_template/`), `cache/*` — already in `.gitignore`.

---

## 🗓️ Task table

> `Owner` follows strict alternation. Adjust the table if you change the starting owner or the number of tasks.

| Day | Owner | Task | What to do today | Done when |
|-----|-------|------|------------------|-----------|
| 1 | MK | ✅ **Contracts — DSL + schemas 5–8** *(DONE)* | Implement `shared/dsl_validator.py` (whitelist: `sum mean median count nunique min max std growth correlation ratio` + parameter validation per §2.5 function signatures; reject anything outside the whitelist — no freeform formulas). Extend `shared/schemas.py` with stage 5–8 models: KPI result, statistical results, chart metadata, evidence entry + registry, insight + recommendation, report result, QA verdict (Pydantic v2, same style as existing file). Write `tests/unit/test_dsl_validator.py` + a schema round-trip test. | ✅ Done — `pytest -q` green (94 passed); non-whitelisted ops rejected; schemas round-trip cleanly. |
| 2 | MM | **Stage 1 — `agents/ingestion_agent.py`** | Build the first CrewAI agent (role/goal/backstory from `config.yaml` `agents.ingestion`). Tasks: `validate_and_extract` (file_validator_tool → file_reader_tool; XLSX multi-sheet → pick via `business_context.sheet_used` → file_sheet_extract_tool) · `profile_dataset` (pii_detector_tool → data_profiler_tool) · `gather_business_context` (human_input_tool / BusinessContextGatherer, Generic Mode on timeout). Allocate `run_id`, wire `RunLogger`, write outputs under `runs/<run_id>/`. **Never pass raw cell content to the LLM.** | Standalone run on a CSV **and** an XLSX produces `data/extracted/` + `metadata/data_profile.json` + `knowledge/business_context.json`; multi-sheet asks the user; unit test for the orchestration path. |
| 3 | MK | **Stage 2 — Understanding + DSL tools** | Add `column_profiler_tool` (per-column facts: dtype / nunique / nullable — role rules from §2.2) to `shared/tools/`. Add `domain_classifier_tool` (profiled facts the LLM uses to name domain/entities) and `dsl_plan_builder_tool` (validates the LLM's proposed KPIs against `shared/dsl_validator.py`, normalizes into `AnalysisPlan`). Implement `agents/understanding_agent.py`: `classify_column_roles` → `detect_domain_and_entities` → `build_analysis_plan` (2nd task, same agent — **not** a 9th agent). Outputs `dataset_understanding.json` + `analysis_plan.json`. | Outputs match the §2.2 examples; plan emits whitelist ops only; unit tests for role rules. |
| 4 | MM | **Stage 3 — `agents/data_quality.py`** (deterministic, no LLM) | Plain functions (no CrewAI) invoked later by `flows.py`. Checks per §2.3: schema · invalid values (impossible dates, negative revenue, % >100, age=350) · missingness MCAR/MAR/MNAR · duplicates · referential integrity · business rules · units/encoding. Deterministic repair per the §2.3 table (cast types, drop exact dupes, drop impossible rows — **never sign-flip, never invent data**). Wrap each check as a tool in `shared/tools/` (schema_checker, invalid_value_checker, missingness_analyzer, duplicate_detector, referential_integrity, deterministic_repair). Output `data_quality_report.json`. | Report JSON per §2.3 (`passed` / `needs_repair`); repair obeys the table; unit tests for every check + repair. |
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

> Fill one row at the **end of every day** (and mid-day when you change something that affects a future task).
> Format: `day | owner | task | what changed | affects next task | tests green?`

| Day | Owner | Task | What changed (file → what) | Affects next task | Tests green? |
|-----|-------|------|----------------------------|-------------------|--------------|
| 1 | MK | Contracts | `shared/dsl_validator.py` → whitelist + semantic validator (`validate_operation` / `validate_plan`); `shared/schemas.py` → added stage 5–8 models (KPI result, statistical results, chart metadata, evidence, insight, recommendation, report result, QA verdict); **`DslOperation.basis` default moved to `None`** (was `"previous_period"` — polluted every op's `model_dump`; executor applies the default at runtime); `tests/unit/test_dsl_validator.py` (27) + `tests/unit/test_schemas.py` (16) added | Day 2+ import these; **Day 6 growth executor must default `basis` → `previous_period` when absent** | ✅ 94 passed |
| 2 | MM | Stage 1 | *(fill at end of day)* | | |
| 3 | MK | Stage 2 | *(fill at end of day)* | | |
| 4 | MM | Stage 3 | *(fill at end of day)* | | |
| 5 | MK | Stage 4 | *(fill at end of day)* | | |
| 6 | MM | Stage 5a | *(fill at end of day)* | | |
| 7 | MK | Stage 5b | *(fill at end of day)* | | |
| 8 | MM | Stage 6 | *(fill at end of day)* | | |
| 9 | MK | Stage 7 | *(fill at end of day)* | | |
| 10 | MM | Stage 8 | *(fill at end of day)* | | |
| 11 | MK | Orchestration | *(fill at end of day)* | | |
| 12 | MM | Tests & golden | *(fill at end of day)* | | |

---

## 📌 Current state snapshot

> Update this every evening before pushing.

- **Last finished task:** Day 1 — Contracts (DSL + schemas 5–8) — ✅ done, `94 passed in ~2.1s` via `py312_env`
- **Last commit:** `d2e6daa V1` (old style — switch to `feat/fix/docs/test(...)` from Day 1 on)
- **Baseline tests:** 94 passing in `tests/unit/` (32 baseline + 62 new) — verify with `& "C:\Users\Malik\miniconda3\envs\py312_env\python.exe" -m pytest tests/unit -q`
- **Working tree:** Day 1 changes uncommitted — **MK must commit before Day 2** (msg: `feat(contracts): DSL whitelist validator + stages 5–8 schemas + tests`); `Project_Plan/DAILY_TASKS.md` is untracked (mentor-created)
- **Open items / decisions:**
  - Day 6 growth executor: default `basis` → `previous_period` when absent (validator no longer fills it)
  - `config.yaml` `agents.qa.model` still `TODO` — needs a distinct QA model before Day 10
  - Use `py312_env` (pydantic 2.12.5, pandas 3.0.3, pytest 8.4.2); system Python lacks deps
