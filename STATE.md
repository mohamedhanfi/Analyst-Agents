# STATE — حالة المشروع الإجمالية

> آخر تحديث: أغسطس 2026 — بعد مراجعة شاملة في 09/08/2026
> تقدير الاكتمال الكلي: **≈ 30%** (البنية التحتية جاهزة، أما الوكلاء والتحليل فلم يبدأوا بعد)

---

## 📊 نظرة عامة على اكتمال المشروع

| المنطقة | الحالة | النسبة |
|---|---|---|
| `shared/core/` (المنطق النقي) | ✅ مكتمل | 100% |
| `shared/tools/` (أدوات CrewAI) | ✅ مكتمل | 100% |
| `shared/` (schemas · utils · logger) | ✅ مكتمل | 100% |
| `shared/dsl_validator.py` | ⚠️ **فارغ (0 سطر) — يُنفَّذ مع المرحلة 2** | 0% |
| الاختبارات الوحدوية | ✅ 32 اختبارًا مجتازًا | 100% |
| `config.yaml` + `PROJECT_STRUCTURE.md` | ✅ محدَّث | 90% |
| `agents/` (6 وكلاء) + `crew/` + `main.py` | ❌ ملفات فارغة (stubs) | 0% |
| `analysis/` (إحصاءات · رسوم · مجالات) | ❌ ملفات فارغة (stubs) | 0% |
| التقرير (`report_template.html`) | ⚠️ قالب فقط | 10% |
| `cache/` (idempotency) + `runs/` | ✅ هيكل جاهز، لا منطق | 20% |
| اختبارات agent/e2e/integration | ❌ لم تبدأ | 0% |
| **الإجمالي** | | **≈ 30%** |

---

## ✅ ما تم تنفيذه فعليًا (مكتمل)

### طبقة `shared/` — البنية التحتية (100%)
```
shared/
├── core/                    # منطق نقي، بدون CrewAI — قابل للاختبار بالكامل
│   ├── validation.py        # FileValidator: امتداد ← حجم ← توقيع ← تحليل ← صفوف
│   ├── reader.py            # FileReader: اكتشاف CSV/XLSX + استخراج الأوراق (تدفقي)
│   ├── profiler.py          # DataProfiler: ملف كامل + نموذج 100 صف (PII مخفي) + data_profile.json
│   ├── pii.py               # PiiDetector: قواعد بدون LLM (أسماء أعمدة + أنماط قيم)
│   └── business_context.py  # حوار أسئلة الأعمال + Generic Mode عند المهلة
├── tools/                   # أغلفة رفيعة فوق core/ — مجمَّعة في __init__.py
│   ├── file_io.py           # file_validator_tool · file_reader_tool · file_sheet_extract_tool
│   ├── profiling.py         # pii_detector_tool · data_profiler_tool
│   └── human.py             # human_input_tool
├── schemas.py               # نماذج Pydantic لكل المخرجات
├── dsl_validator.py         # ⚠️ فارغ — القائمة البيضاء / DSL (مع المرحلة 2)
├── logger.py                # سجلات منظمة لكل run
└── utils.py                 # config + run_id + حساب SHA
```

**مبادئ ثابتة وأُنجزت:** golden rule (الـ LLM يقرر، Python ينفذ) · حجم الملف يُحسب من القرص · لا ثقة بالمدخلات · PII يُخفى من كل العينات · الأعمدة الرقمية/الزمنية ليست PII.

### الاختبارات — ✅ 32 مجتازًا (الأرقام الفعلية بعد المراجعة)
`test_file_validator` (10) · `test_reader` (7) · `test_pii` (6) · `test_business_context` (5) · `test_profiler` (4)
> لا يوجد `test_dsl_validator` — الـ DSL غير مطبَّق بعد.

---

## ⚠️ ما لم يُنفَّذ بعد (الباقي في المشروع ككل)

### أ. الوكلاء — 0% (كلها stubs فارغة)
| الوكيل | المرحلة | المطلوب تنفيذه |
|---|---|---|
| `ingestion_agent.py` | 1 — الاستقبال | تحقق ← قراءة ← أسئلة أعمال ← ملف البيانات → `metadata/` + `knowledge/` |
| `understanding_agent.py` | 2 — الفهم | أدوار الأعمدة + كشف المجال/الكيانات + خطة تحليل DSL |
| `cleaning_agent.py` | 3 — التنظيف | معالجة القيم الناقصة/الشاذة/المكررة + مخرجات تنظيف موثقة |
| `analyst_agent.py` | 5 — التحليل | تنفيذ DSL + إحصاءات + ترتيب الرسوم + سجل أدلة |
| `insight_agent.py` | 6 — الرؤى | رؤى مبنية على أدلة + توصيات + تحقق من الادعاءات |
| `report_agent.py` | 7 — التقرير | توليد تقرير HTML نهائي بمستوى ثقة |

### ب. أدوات المراحـل الأخرى — 0%
`column_profiler_tool` · `domain_classifier_tool` · `dsl_plan_builder_tool` · `dsl_executor_tool` · `statistical_suite_tool` (correlation/distribution/trend في `analysis/generic/`) · `chart_planner_tool` · `chart_renderer_tool` · `evidence_registry_tool` · `claim_validator_tool`

### ج. حلقات التشغيل — 0%
`crew/crew.py` + `crew/flows.py` + `main.py` (نقطة الدخول) + `runs/` isolation + `cache/` idempotency + الربط بـ `logger.py`

### د. الباقي
- `analysis/evidence.py` + `chart_planner.py` (فارغان)
- قوالب `analysis/domains/` (sales · finance · marketing · hr · operations) — خطة مستقبلية
- `report_template.html` (قالب فقط)
- اختبارات agent/integration/e2e/golden + تجربة Fixtures حقيقية
- ❗ **Commit** التغييرات الحالية (كلها في working tree بدون commit)

---

## 🔍 نتائج مراجعة 09/08/2026

### أُصلح خلال المراجعة (كلها في working tree، لم تُرتكب بعد)
1. `shared/logger.py` — كان يستخدم `os.PathLike` بدون `import os` → أضيف الاستيراد
2. أدوات فحص الملفات كانت تفشل بدون مفتاح LLM → `load_config(require_key=False)` لأدوات الحساب الخالص
3. `pyproject.toml` كان يستبعد `shared.core` و `shared.tools` من الحزمة → أُدرجا ضمن `packages`
4. عدّ صفوف XLSX في `FileValidator` كان عبر `max_row` غير الموثوق → عدّ صريح مطابق لـ `FileReader`

### ملاحظات مؤجلة عمدًا (سجّل للتنفيذ لاحقًا)
- `shared/dsl_validator.py` فارغ تمامًا (0 سطر) — يُنفَّذ مع المرحلة 2 (أدوات الفهم)
- `business_context._recompute_deadline` دالة لا تفعل شيئًا (تعيد نفس القيمة) — تنظيف اختياري
- قراءة >5M سطر (تقطيع chunked) غير مطبَّقة بعد — الحدود معرّفة في `config.yaml` فقط
- `schemas.py` يغطي المراحل 1–4 فقط — نماذج المراحل 5–6 (KPIs · إحصاءات · رؤى · توصيات · تقرير) تُضاف مع وكلائها
- حساب SHA للملف مكرر داخل `FileReader._calculate_hash` ولا حاجة لدالة عامة في `utils` — ترك كما هو عمدًا

---

## 🧭 اقتراحات للخطوات القادمة (بالترتيب المقترح)

| الأولوية | الخطوة | لماذا أولًا؟ |
|---|---|---|
| 1 | `agents/ingestion_agent.py` → `agents/understanding_agent.py` | يكملان المرحلتين 1+2 باستخدام `shared/` الجاهز تمامًا — أول تشغيلة حقيقية |
| 2 | أدوات المرحلتين (column/domain/dsl tools) | لا بديل عنها وقابلة للاختبار الوحدوي فورًا |
| 3 | `main.py` + `crew/flows.py` | يربط كل شيء في مسار واحد قابل للتشغيل |
| 4 | `cleaning_agent.py` | يعتمد على فهم المرحلتين السابقتين |
| 5 | `analysis/` + `dsl_executor` + `evidence_registry` | قلب التحليل وتتبُّع الأدلة (المرحلة 5) |
| 6 | `insight_agent.py` + `report_agent.py` | إغلاق الدورة بتقرير نهائي |
| 7 | Commit أولي + اختبارات agent/e2e | تثبيت التقدم وتغطية التشغيل الفعلي |

> **توصية عاجلة:** عمل commit للتغييرات الحالية قبل بدء الوكلاء — حتى نقر في العودة لأي تقدم.

---

## 🗂️ مراجع
- `Project_Plan/PROJECT_STRUCTURE.md` — البنية الكاملة
- `Project_Plan/PROJECT_PLAN.md` — الخطة والمنهجية (المرجعية الأصلية)
- `config.yaml` — الحدود والإعدادات
- `tests/unit/` — الاختبارات الحالية