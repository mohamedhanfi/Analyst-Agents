# حالتنا الحالية — ما تم تنفيذه وخطواتنا القادمة

> آخر تحديث: أغسطس 2026 — وثيقة تتبع سير تنفيذ مشروع **Insight-Forge** (خط أنابيب تحليل البيانات بـ 6 وكلاء)

---

## ✅ ما تم تنفيذه

### 1) إعادة هيكلة `shared/` — فصل المنطق عن الوكلاء
المبدأ الحاكم (Golden Rule): **الذكاء الاصطناعي يقرر، والـ Python ينفذ** — كل الحسابات وقراءة الملفات في كود نقي قابل للاختبار، والوكلاء طبقة رفيعة تستدعي الأدوات فقط.

```
shared/
├── core/                    # المنطق النقي — بدون CrewAI
│   ├── validation.py        # فحص الملفات العميق (امتداد → حجم → توقيع محتوى → تحليل + صفوف)
│   ├── reader.py            # قراءة/اكتشاف CSV وXLSX + استخراج الأوراق (تدفقي)
│   ├── profiler.py          # الملف الكامل + نموذج 100 صف (PII مخفي) + كتابة data_profile.json
│   ├── pii.py               # كشف PII بقواعد (أسماء أعمدة + أنماط قيم) بدون LLM
│   └── business_context.py  # حوار أسئلة الأعمال + Generic Mode عند انتهاء المهلة
├── tools/                   # أغلفة CrewAI @tool — مجمّعة في __init__.py
│   ├── file_io.py           # file_validator_tool · file_reader_tool · file_sheet_extract_tool
│   ├── profiling.py         # pii_detector_tool · data_profiler_tool
│   └── human.py             # human_input_tool
├── schemas.py               # نماذج Pydantic لكل المخرجات
├── dsl_validator.py         # القائمة البيضاء / DSL
├── logger.py                # سجلات منظمة لكل run
└── utils.py                 # تحميل config + تخصيص run_id + حساب SHA ملفات
```

### 2) تحسينات فنية
- حساب حجم الملف **من القرص دائمًا** (لا نثق بالمدخلات)
- كشف PII لا يلتقط الأعمدة الرقمية (مثل `order_id`) ولا الأعمدة الزمنية
- `is_string_dtype` للتوافق مع pandas الجديدة (dtype `str` بدل `object`)
- `ask_one()` مضاف لـ `BusinessContextGatherer`، ومهلة كل سؤال من `config.yaml`

### 3) الاختبارات — ✅ 32 اختبارًا مجتازًا
```
tests/unit/test_file_validator.py    (7)
tests/unit/test_reader.py            (6)
tests/unit/test_pii.py               (5)
tests/unit/test_profiler.py          (4)
tests/unit/test_business_context.py  (5)
tests/unit/test_dsl_validator.py     (5)
===========================================
32 passed
```

### 4) التوثيق
- `Project_Plan/PROJECT_STRUCTURE.md` محدّث بالبنية الجديدة (core/ + tools/)

---

## ⚠️ ملاحظة مهمة

التغييرات الحالية **غير ملتزمة commit** بعد — موجودة في working tree فقط.

---

## 🔜 الخطوات القادمة

| # | الخطوة | الوصف |
|---|---|---|
| 1 | `agents/ingestion_agent.py` | أول وكيل فعلي: يستقبل الملف → يستدعي أدوات المرحلة 1 (تحقق → قراءة → أسئلة الأعمال → ملف البيانات) → يكتب `metadata/` و`knowledge/` |
| 2 | `agents/understanding_agent.py` | المرحلة 2: تصنيف أدوار الأعمدة، كشف المجال/الكيانات، بناء خطة تحليل DSL |
| 3 | أدوات المرحلة 2 | `column_profiler_tool` · `domain_classifier_tool` · `dsl_plan_builder_tool` |
| 4 | `main.py` | نقطة الدخول: تحميل config → بناء الـ Crew → تشغيل الـ Flow → تخصيص `run_id` |
| 5 | المراحل 3–4 | `cleaning_agent.py` (تنظيف البيانات) + `analyst_agent.py` (تحليل: DSL + إحصاءات + رسوم) |
| 6 | المراحل 5–6 | `insight_agent.py` (رؤى وتوصيات) + `report_agent.py` (تقرير HTML) |
| 7 | `evidence_registry` | سجل الأدلة لكل قيمة/رسم/رؤية |
| 8 | `cache/` | مؤشر idempotency لمنع إعادة التشغيل |

---

## 🗂️ الملفات المهمة للمرجعية
- `Project_Plan/PROJECT_STRUCTURE.md` — البنية الكاملة
- `Project_Plan/PROJECT_PLAN.md` — خطة المشروع ومنهجيته
- `config.yaml` — الإعدادات والحدود
- `tests/unit/` — اختبارات الطبقة النقية