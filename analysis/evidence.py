"""Evidence registry (§2.5/§2.6) — THE ONLY writer of evidence_registry.json.

Every computed value in stage 5+ carries an evidence_id; this module mints the
ids and owns the registry file (outputs/evidence_registry.json). Lineage is a
full chain per EvidenceSource: raw -> cleaning transformations -> filter ->
aggregation -> comparison -> result.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from shared.schemas import EvidenceEntry, EvidenceSource

_EVIDENCE_PREFIX = "EV"


class EvidenceRegistry:
    """In-memory registry + file IO. Single owner of the JSON artifact.

    Minting is monotonically increasing within the run and never repeats
    (IDs survive even when the registry is reloaded from disk).
    """

    def __init__(self, run_id: Optional[str] = None,
                 file_hash: Optional[str] = None,
                 sheet: Optional[str] = None,
                 transformations: Optional[List[str]] = None,
                 prefix: str = _EVIDENCE_PREFIX) -> None:
        self.run_id = run_id
        self.file_hash = file_hash
        self.sheet = sheet
        self.transformations = list(transformations or [])
        self.prefix = prefix
        self._entries: Dict[str, EvidenceEntry] = {}
        self._counter = 0
        self._rescan()

    def _rescan(self) -> None:
        """Restore the counter so minting never collides with existing ids."""
        for entry in self._entries.values():
            try:
                number = int(entry.evidence_id.rsplit("-", 1)[-1])
                self._counter = max(self._counter, number)
            except ValueError:
                continue

    # ------------------------------------------------------------------
    # Minting
    # ------------------------------------------------------------------

    def mint(self) -> str:
        """Allocate a fresh, unique evidence id (e.g. 'EV-001')."""
        self._counter += 1
        return f"{self.prefix}-{self._counter:03d}"

    # ------------------------------------------------------------------
    # Adding
    # ------------------------------------------------------------------

    def add(self, evidence_id: str,
            source: EvidenceSource | Dict[str, Any]) -> EvidenceEntry:
        """Register one evidence entry; a duplicated id raises."""
        if evidence_id in self._entries:
            raise ValueError(f"evidence id {evidence_id} already registered")
        if isinstance(source, dict):
            source = EvidenceSource(**source)
        entry = EvidenceEntry(evidence_id=evidence_id, source=source)
        self._entries[evidence_id] = entry
        return entry

    def add_value(self, value: Any, *, aggregation: str,
                  comparison: Optional[str] = None,
                  filter_str: Optional[str] = None,
                  extra: Optional[Dict[str, Any]] = None) -> str:
        """Convenience: mint + register a value's lineage in one call.

        Returns the minted evidence id.
        """
        source: Dict[str, Any] = {
            "file_hash": self.file_hash,
            "sheet": self.sheet,
            "transformations": self.transformations,
            "aggregation": aggregation,
            "result": value,
        }
        if comparison:
            source["comparison"] = comparison
        if filter_str:
            source["filter"] = filter_str
        if extra:
            source.update(extra)
        evidence_id = self.mint()
        self.add(evidence_id, source)
        return evidence_id

    def get(self, evidence_id: str) -> Optional[EvidenceEntry]:
        return self._entries.get(evidence_id)

    def entries(self) -> List[EvidenceEntry]:
        return [self._entries[k] for k in sorted(self._entries)]

    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, evidence_id: str) -> bool:
        return evidence_id in self._entries

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> Path:
        """Write outputs/evidence_registry.json (plain list of entries)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = [e.model_dump() for e in self.entries()]
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                        encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: str | Path) -> "EvidenceRegistry":
        """Reload a previously saved registry (ids preserved on next mint)."""
        path = Path(path)
        registry = cls()
        if path.exists():
            payload = json.loads(path.read_text(encoding="utf-8"))
            for row in payload:
                registry.add(row["evidence_id"], EvidenceSource(**row["source"]))
            registry._rescan()
        return registry


def evidence_id_count(registry: EvidenceRegistry) -> int:
    """Number of minted ids so far (including gaps from collisions)."""
    return registry._counter
