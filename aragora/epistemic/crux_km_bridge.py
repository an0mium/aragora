"""CruxReceipt → Knowledge Mound ingestion bridge (DIC-16 / #6026).

Flag-gated (``ARAGORA_CRUX_RECEIPT_ENABLED``): returns an empty result
when unset — no KM writes, no queue mutations, no network calls.
Advances issue #6026.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from aragora.epistemic.crux_receipt import CruxReceipt
    from aragora.knowledge.unified.types import ConfidenceLevel, KnowledgeItem


@dataclass
class CruxKMIngestionResult:
    """Typed outcome of a CruxReceipt → KM conversion. Empty when flag is off."""

    receipt_id: str
    debate_id: str
    crux_count: int
    items: list["KnowledgeItem"] = field(default_factory=list)

    @property
    def success(self) -> bool:
        return len(self.items) > 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "receipt_id": self.receipt_id,
            "debate_id": self.debate_id,
            "crux_count": self.crux_count,
            "items_produced": len(self.items),
            "success": self.success,
        }


def crux_receipt_to_knowledge_items(receipt: "CruxReceipt") -> "CruxKMIngestionResult":
    """Convert *receipt* cruxes into KnowledgeItems (empty when flag is off)."""
    from aragora.epistemic.crux_receipt import crux_receipt_enabled

    empty = CruxKMIngestionResult(
        receipt_id=receipt.receipt_id,
        debate_id=receipt.debate_id,
        crux_count=len(receipt.cruxes),
    )

    if not crux_receipt_enabled():
        return empty

    if not receipt.cruxes:
        return empty

    from aragora.knowledge.unified.types import (
        KnowledgeItem,
        KnowledgeSource,
    )

    now = datetime.now(tz=timezone.utc)
    items: list[KnowledgeItem] = []

    for entry in receipt.cruxes:
        confidence = _score_to_confidence(entry.load_bearing_score)
        item = KnowledgeItem(
            id=f"crux_km_{entry.crux_id}",
            content=entry.statement,
            source=KnowledgeSource.BELIEF,
            source_id=entry.crux_id,
            confidence=confidence,
            created_at=now,
            updated_at=now,
            importance=round(entry.load_bearing_score, 4),
            metadata={
                "receipt_id": receipt.receipt_id,
                "debate_id": receipt.debate_id,
                "question": receipt.question,
                "affected_claims": list(entry.affected_claims),
                "contesting_agents": list(entry.contesting_agents),
                "load_bearing_score": round(entry.load_bearing_score, 4),
                "uncertainty_score": round(entry.uncertainty_score, 4),
                "resolution_impact": round(entry.resolution_impact, 4),
                "dic_track": "DIC-16",
            },
        )
        items.append(item)

    return CruxKMIngestionResult(
        receipt_id=receipt.receipt_id,
        debate_id=receipt.debate_id,
        crux_count=len(receipt.cruxes),
        items=items,
    )


def _score_to_confidence(score: float) -> "ConfidenceLevel":
    # >= 0.75 → HIGH, >= 0.45 → MEDIUM, otherwise → LOW
    from aragora.knowledge.unified.types import ConfidenceLevel

    if score >= 0.75:
        return ConfidenceLevel.HIGH
    if score >= 0.45:
        return ConfidenceLevel.MEDIUM
    return ConfidenceLevel.LOW


__all__ = [
    "CruxKMIngestionResult",
    "crux_receipt_to_knowledge_items",
]
