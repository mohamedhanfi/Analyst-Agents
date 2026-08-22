"""Tests for the data-contract + normalization layer (§4.2): amount/percent/
currency parsing, heuristic contract generation + validation, and the
normalization of categories/dates/units — plus the DQ-repair integration that
keeps currency strings as real numbers."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from agents.cleaning_agent import run_cleaning
from agents.data_quality import run_data_quality
from agents.ingestion_agent import run_ingestion
from agents.understanding_agent import run_understanding
from shared.core.contracts import (SENTINELS, build_contracts, load_contracts,
                                   normalize_columns, parse_amount,
                                   validate_contracts)
from shared.core.data_quality import deterministic_repair
from shared.schemas import (ColumnUnderstanding, DatasetUnderstanding)
from shared.utils import load_config


@pytest.fixture
def cfg():
    return load_config(require_key=False)

# ---------------------------------------------------------------------------
# parse_amount
# ---------------------------------------------------------------------------


def test_parse_amount_currencies():
    assert parse_amount("$1,200") == 1200.0
    assert parse_amount("EGP 1,200") == 1200.0
    assert parse_amount("1200 EGP") == 1200.0
    assert parse_amount("€9.99") == 9.99
    assert parse_amount("1.2k") == 1200.0
    assert parse_amount("12%") == 12.0
    assert parse_amount("1 200") == 1200.0
    assert parse_amount(42) == 42.0
    assert parse_amount("abc") is None
    assert parse_amount("") is None
    assert parse_amount(None) is None


# ---------------------------------------------------------------------------
# Contracts: generation + validation
# ---------------------------------------------------------------------------


def _understanding(roles):
    columns = [ColumnUnderstanding(name=n, role=r, dtype="object",
                                   nunique=0, nullable=False)
               for n, r in roles.items()]
    return DatasetUnderstanding(detected_domain="sales",
                                domain_confidence=0.8, columns=columns)


def test_build_contracts_captures_shape():
    df = pd.DataFrame({
        "revenue_usd": ["$100", "$200", "$300"],
        "date": ["2024-01-01", "2024-01-02", "2024-01-03"],
        "order_id": ["a", "b", "c"],
        "product": ["x", "y", "z"],
    })
    understanding = _understanding({
        "revenue_usd": "measure", "date": "temporal",
        "order_id": "identifier", "product": "dimension"})
    contracts = {c.column: c for c in build_contracts(understanding, df)}
    assert contracts["revenue_usd"].expected_type == "numeric"
    assert contracts["revenue_usd"].unit == "USD"
    assert contracts["revenue_usd"].allowed_min is not None
    assert contracts["date"].expected_type == "date"
    assert contracts["order_id"].expected_type == "id"
    assert contracts["order_id"].unique is True
    assert contracts["product"].expected_type == "categorical"


def test_validate_contracts_reports_violations():
    df = pd.DataFrame({
        "revenue": [100.0, 5000.0, 100.0],
        "order_id": ["a", "b", "a"],
        "note": ["ok", None, "?"],
    })
    understanding = _understanding({"revenue": "measure",
                                   "order_id": "identifier",
                                   "note": "free_text"})
    contracts = build_contracts(understanding, df)
    from shared.core.contracts import ColumnContract
    contracts[0] = ColumnContract(column="revenue", expected_type="numeric",
                                  allowed_min=0, allowed_max=1000)
    contracts[1] = ColumnContract(column="order_id", expected_type="id",
                                  unique=True)
    violations = validate_contracts(contracts, df)
    kinds = {v["kind"] for v in violations}
    assert "above_contract_range" in kinds
    assert any(v["column"] == "order_id" and v["kind"] == "uniqueness"
               for v in violations)


def test_sentinels_are_catalogued():
    assert "n/a" in SENTINELS
    assert "unknown" in SENTINELS
    assert "null" in SENTINELS


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------


def test_normalize_currency_strings_to_numbers():
    df = pd.DataFrame({
        "revenue": ["$1,200", "$2,000", "$1,500"],
        "product": ["A", "B", "A"],
    })
    understanding = _understanding({"revenue": "measure",
                                   "product": "dimension"})
    out, ops = normalize_columns(df, understanding)
    assert pd.api.types.is_numeric_dtype(out["revenue"])
    assert out["revenue"].tolist() == [1200.0, 2000.0, 1500.0]
    assert any(op["op"] == "parse_amount" and op["column"] == "revenue"
               for op in ops)


def test_normalize_percent_stays_percent_points():
    df = pd.DataFrame({"growth_pct": ["12%", "8%", "15%"],
                       "product": ["A", "B", "A"]})
    understanding = _understanding({"growth_pct": "measure",
                                   "product": "dimension"})
    out, ops = normalize_columns(df, understanding)
    assert out["growth_pct"].tolist() == [12.0, 8.0, 15.0]


def test_normalize_categories_strip_case_unicode():
    df = pd.DataFrame({
        "product": ["  Apple ", "apple", " banana\t", "🍎", " 🍎 "],
        "note": ["  hello  ", "HELLO", "  world", None, "x"],
    })
    understanding = _understanding({"product": "dimension",
                                   "note": "free_text"})
    out, ops = normalize_columns(df, understanding)
    assert out["product"].tolist() == ["apple", "apple", "banana", "🍎", "🍎"]
    note = out["note"].tolist()
    assert note[:3] == ["hello", "hello", "world"]
    assert pd.isna(note[3])
    assert note[4] == "x"
    assert out["note"].isna().sum() == 1
    ops_cols = {op["column"] for op in ops}
    assert "product" in ops_cols and "note" in ops_cols


def test_normalize_dates_multiformat():
    df = pd.DataFrame({
        "date": ["2024-01-01", "01/15/2024", "2024-03-01"],
        "product": ["A", "B", "A"],
    })
    understanding = _understanding({"date": "temporal",
                                   "product": "dimension"})
    out, ops = normalize_columns(df, understanding)
    assert pd.api.types.is_datetime64_any_dtype(out["date"])
    assert any(op["op"] == "parse_datetime" for op in ops)


def test_normalize_unit_conversion_scale_only():
    df = pd.DataFrame({
        "weight_kg": ["500 g", "1 kg", "2 kg", "250 g"],
        "product": ["A", "B", "A", "B"],
    })
    understanding = _understanding({"weight_kg": "measure",
                                   "product": "dimension"})
    out, ops = normalize_columns(df, understanding)
    assert out["weight_kg"].tolist() == [0.5, 1.0, 2.0, 0.25]
    assert any(op["op"] == "convert_unit" for op in ops)


# ---------------------------------------------------------------------------
# DQ-repair integration: currency strings survive the repair
# ---------------------------------------------------------------------------


def test_deterministic_repair_keeps_currency_values(tmp_path):
    df = pd.DataFrame({
        "revenue": ["$1,200", "$2,000", "$1,500"],
        "date": ["2024-01-01", "2024-01-02", "2024-01-03"],
    })
    understanding = _understanding({"revenue": "measure",
                                   "date": "temporal"})
    repaired, log = deterministic_repair(understanding, df)
    assert repaired["revenue"].tolist() == [1200.0, 2000.0, 1500.0]
    assert repaired["revenue"].isna().sum() == 0
    assert "object->amount_float64" in log["type_casts"].values()


def test_pipeline_saves_contracts_and_normalizes_currency(tmp_path, cfg):
    rows = [
        {"date": "2024-01-01", "product": " A ", "revenue": "$1,000",
         "quantity": 2},
        {"date": "2024-01-02", "product": "B", "revenue": "EGP 2,500",
         "quantity": 3},
        {"date": "2024-01-03", "product": " A ", "revenue": "$1,500",
         "quantity": 4},
        {"date": "2024-01-04", "product": "B", "revenue": "$2,000",
         "quantity": 5},
        {"date": "2024-01-05", "product": " A ", "revenue": "EGP 3,000",
         "quantity": 6},
    ]
    csv = tmp_path / "sales.csv"
    pd.DataFrame(rows).to_csv(csv, index=False)
    run_dir = tmp_path / "r1"
    provider = lambda _p: "Track revenue"  # single answer -> generic-ish
    assert run_ingestion(str(csv), run_dir=run_dir, cfg=cfg,
                         answer_provider=provider)["status"] == "passed"
    assert run_understanding(run_dir, cfg=cfg)["status"] == "passed"
    s3 = run_data_quality(run_dir, cfg=cfg)
    assert s3["status"] in ("passed", "needs_repair")

    contracts = load_contracts(run_dir)
    assert any(c.column == "revenue" and c.expected_type == "numeric"
               for c in contracts)

    validated = pd.read_csv(run_dir / "data" / "processed"
                            / "validated_data.csv", encoding="utf-8-sig")
    assert validated["revenue"].tolist() == [1000.0, 2500.0, 1500.0,
                                             2000.0, 3000.0]

    assert run_cleaning(run_dir, cfg=cfg)["status"] == "passed"
    cleaned = pd.read_csv(run_dir / "data" / "processed"
                          / "cleaned_data.csv", encoding="utf-8-sig")
    assert cleaned["product"].tolist() == ["A", "B", "A", "B", "A"]
    lineage = json.loads((run_dir / "metadata" / "lineage.json")
                         .read_text(encoding="utf-8"))
    cleaned_step = next(s for s in lineage["steps"]
                        if s["stage"] == "cleaned")
    assert any("normalize" in str(op.get("op", "")) or "parse" in str(op.get("op", ""))
               for op in cleaned_step["ops"])