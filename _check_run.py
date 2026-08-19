import json
from pathlib import Path

d = Path("runs/run_20260819_184532_1/outputs")
insights = json.loads((d/"insights.json").read_text(encoding="utf-8"))
print(f"Insights: {len(insights['insights'])}")
print(f"Recommendations: {len(insights['recommendations'])}")
evidence = json.loads((d/"evidence_registry.json").read_text(encoding="utf-8"))
print(f"Evidence entries: {len(evidence)}")

rec = insights["recommendations"][0]
print(f"\nSample recommendation:")
print(rec["description"][:400])

u = json.loads(Path("runs/run_20260819_184532_1/metadata/dataset_understanding.json").read_text(encoding="utf-8"))
for c in u["columns"]:
    if c["name"] == "employee_id":
        print(f"\nemployee_id role: {c['role']}")

dq = json.loads(Path("runs/run_20260819_184532_1/metadata/data_quality_report.json").read_text(encoding="utf-8"))
print(f"\nDQ issues: {len(dq['issues'])}")
for i in dq["issues"]:
    print(f"  {i['severity']}: {i['detail']} ({i['column']})")

plan = json.loads(Path("runs/run_20260819_184532_1/metadata/analysis_plan.json").read_text(encoding="utf-8"))
kpi_names = [k["name"] for k in plan.get("candidate_kpis", [])]
growth = [k for k in kpi_names if "growth" in k.lower() or "yoy" in k.lower()]
print(f"\nGrowth KPIs: {growth}")
print(f"Total KPIs: {len(kpi_names)}")

import pandas as pd
cleaned = pd.read_csv("runs/run_20260819_184532_1/data/processed/cleaned_data.csv")
print(f"\nCleaned rows: {len(cleaned)}")
print(f"salary neg: {(cleaned['salary'] < 0).sum()}")
print(f"hours neg: {(cleaned['hours_worked'] < 0).sum()}")
print(f"bonus neg: {(cleaned['bonus_amount'] < 0).sum()}")
sql_vals = cleaned[cleaned["department"].str.contains("DROP", case=False, na=False)]
print(f"SQL injection in dept: {len(sql_vals)}")
