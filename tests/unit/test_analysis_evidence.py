"""Unit tests for analysis/evidence.py — evidence id minting + registry IO."""
from __future__ import annotations

import json

import pytest

from analysis.evidence import EvidenceRegistry
from shared.schemas import EvidenceSource


def test_mint_returns_unique_ids():
    registry = EvidenceRegistry()
    ids = [registry.mint() for _ in range(3)]
    assert ids == ["EV-001", "EV-002", "EV-003"]
    assert len(set(ids)) == 3


def test_add_and_get():
    registry = EvidenceRegistry()
    eid = registry.add_value(12.5, aggregation="sum",
                             filter_str="category==X")
    assert eid == "EV-001"
    entry = registry.get(eid)
    assert entry is not None
    assert entry.source.aggregation == "sum"
    assert entry.source.filter == "category==X"
    assert entry.source.result == 12.5
    assert eid in registry
    assert len(registry) == 1


def test_duplicate_add_raises():
    registry = EvidenceRegistry()
    eid = registry.mint()
    registry.add(eid, EvidenceSource(result=1, aggregation="sum"))
    with pytest.raises(ValueError):
        registry.add(eid, EvidenceSource(result=2, aggregation="mean"))


def test_save_load_round_trip_and_no_id_collision(tmp_path):
    registry = EvidenceRegistry(file_hash="sha256:abc",
                                transformations=["removed_duplicates"])
    first = registry.add_value(10, aggregation="sum", comparison="YoY")
    second = registry.add_value(20, aggregation="mean")
    path = registry.save(tmp_path / "evidence_registry.json")

    assert path.is_file()
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert len(payload) == 2
    assert payload[0]["evidence_id"] == first

    reloaded = EvidenceRegistry.load(path)
    assert len(reloaded) == 2
    assert reloaded.get(first).source.comparison == "YoY"
    assert reloaded.get(first).source.transformations == ["removed_duplicates"]
    next_id = reloaded.mint()
    assert next_id not in (first, second)


def test_load_missing_file_is_empty(tmp_path):
    registry = EvidenceRegistry.load(tmp_path / "nope.json")
    assert len(registry) == 0
    assert registry.mint() == "EV-001"


def test_add_value_records_lineage_fields():
    registry = EvidenceRegistry(file_hash="sha256:h", sheet=None,
                                transformations=["cleaning_median_fill"])
    eid = registry.add_value(0.5, aggregation="growth_MoM",
                             comparison="MoM", filter_str="product in [A]")
    entry = registry.get(eid)
    assert entry.source.file_hash == "sha256:h"
    assert entry.source.transformations == ["cleaning_median_fill"]
    assert entry.source.comparison == "MoM"
    assert entry.source.result == 0.5


def test_entries_sorted_by_id():
    registry = EvidenceRegistry()
    registry.add_value(1, aggregation="sum")
    registry.add_value(2, aggregation="mean")
    registry.add_value(3, aggregation="count")
    assert [e.evidence_id for e in registry.entries()] == \
        ["EV-001", "EV-002", "EV-003"]
