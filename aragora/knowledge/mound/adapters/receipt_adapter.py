"""
ReceiptAdapter - Bridges Decision Receipts to the Knowledge Mound.

This adapter enables automatic persistence of decision evidence:

- Data flow IN: Verified claims from receipts are stored as knowledge items
- Data flow IN: Findings are stored with severity and provenance
- Data flow IN: Dissenting views are preserved for future context
- Reverse flow: KM can retrieve past decisions for similar queries

The adapter provides:
- Automatic extraction of verified claims to knowledge items
- Finding persistence with severity classification
- Bidirectional linking between receipt and knowledge items
- Audit trail integration for compliance

"Every decision leaves a trace in institutional memory."
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

if TYPE_CHECKING:
    from aragora.export.decision_receipt import (
        DecisionReceipt,
        ReceiptFinding,
        ReceiptVerification,
    )

from aragora.knowledge.unified.types import (
    ConfidenceLevel,
    KnowledgeItem,
    KnowledgeSource,
    RelationshipType,
)

# Map receipt sources to knowledge sources
RECEIPT_SOURCE = KnowledgeSource.DEBATE  # Use DEBATE for all receipt-derived items

logger = logging.getLogger(__name__)

# Type alias for event callback
EventCallback = Callable[[str, Dict[str, Any]], None]


class ReceiptAdapterError(Exception):
    """Base exception for receipt adapter errors."""

    pass


class ReceiptNotFoundError(ReceiptAdapterError):
    """Raised when a receipt is not found in the store."""

    pass


@dataclass
class ReceiptIngestionResult:
    """Result of ingesting a decision receipt into Knowledge Mound."""

    receipt_id: str
    claims_ingested: int
    findings_ingested: int
    relationships_created: int
    knowledge_item_ids: List[str]
    errors: List[str]

    @property
    def success(self) -> bool:
        """Check if ingestion was successful."""
        return len(self.errors) == 0 and (self.claims_ingested > 0 or self.findings_ingested > 0)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "receipt_id": self.receipt_id,
            "claims_ingested": self.claims_ingested,
            "findings_ingested": self.findings_ingested,
            "relationships_created": self.relationships_created,
            "knowledge_item_ids": self.knowledge_item_ids,
            "errors": self.errors,
            "success": self.success,
        }


class ReceiptAdapter:
    """
    Adapter that bridges Decision Receipts to the Knowledge Mound.

    Provides methods to:
    - Ingest verified claims as knowledge items
    - Store findings with provenance tracking
    - Create relationships between receipt components
    - Retrieve past decisions for context

    Usage:
        from aragora.export.decision_receipt import DecisionReceipt
        from aragora.knowledge.mound.adapters import ReceiptAdapter
        from aragora.knowledge.mound.core import KnowledgeMound

        mound = KnowledgeMound()
        adapter = ReceiptAdapter(mound)

        # Ingest a decision receipt
        result = await adapter.ingest_receipt(receipt, workspace_id="ws-123")

        # Query related decisions
        related = await adapter.find_related_decisions("contract terms", limit=5)
    """

    ID_PREFIX = "rcpt_"
    CLAIM_PREFIX = "claim_"
    FINDING_PREFIX = "find_"
    MIN_CONFIDENCE_FOR_CLAIM = 0.7

    def __init__(
        self,
        mound: Optional[Any] = None,
        enable_dual_write: bool = False,
        event_callback: Optional[EventCallback] = None,
        auto_ingest: bool = True,
    ):
        """
        Initialize the adapter.

        Args:
            mound: Optional KnowledgeMound instance to use
            enable_dual_write: If True, writes go to both receipt store and KM
            event_callback: Optional callback for emitting events
            auto_ingest: If True, automatically ingest receipts on creation
        """
        self._mound = mound
        self._enable_dual_write = enable_dual_write
        self._event_callback = event_callback
        self._auto_ingest = auto_ingest
        self._ingested_receipts: Dict[str, ReceiptIngestionResult] = {}

    def set_event_callback(self, callback: EventCallback) -> None:
        """Set the event callback for WebSocket notifications."""
        self._event_callback = callback

    def set_mound(self, mound: Any) -> None:
        """Set the Knowledge Mound instance."""
        self._mound = mound

    def _emit_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Emit an event if callback is configured."""
        if self._event_callback:
            try:
                self._event_callback(event_type, data)
            except Exception as e:
                logger.warning(f"Failed to emit event {event_type}: {e}")

    async def ingest_receipt(
        self,
        receipt: "DecisionReceipt",
        workspace_id: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> ReceiptIngestionResult:
        """
        Ingest a decision receipt into the Knowledge Mound.

        Extracts verified claims, findings, and dissenting views,
        storing them as knowledge items with provenance tracking.

        Args:
            receipt: The DecisionReceipt to ingest
            workspace_id: Optional workspace for scoping
            tags: Optional tags to apply to all items

        Returns:
            ReceiptIngestionResult with counts and any errors
        """
        errors: List[str] = []
        knowledge_item_ids: List[str] = []
        claims_ingested = 0
        findings_ingested = 0
        relationships_created = 0

        if not self._mound:
            errors.append("Knowledge Mound not configured")
            return ReceiptIngestionResult(
                receipt_id=receipt.receipt_id,
                claims_ingested=0,
                findings_ingested=0,
                relationships_created=0,
                knowledge_item_ids=[],
                errors=errors,
            )

        base_tags = tags or []
        base_tags.extend(
            [
                f"receipt:{receipt.receipt_id}",
                f"verdict:{receipt.verdict}",
                f"risk:{receipt.risk_level}",
            ]
        )

        # 1. Ingest verified claims
        for verification in receipt.verified_claims:
            if not verification.verified:
                continue  # Skip unverified claims

            try:
                item = self._verification_to_knowledge_item(
                    verification,
                    receipt,
                    workspace_id,
                    base_tags,
                )
                item_id = await self._store_item(item)
                if item_id:
                    knowledge_item_ids.append(item_id)
                    claims_ingested += 1
            except Exception as e:
                errors.append(f"Failed to ingest claim: {str(e)[:100]}")
                logger.warning(f"Claim ingestion failed: {e}")

        # 2. Ingest critical and high severity findings
        for finding in receipt.findings:
            if finding.severity not in ("CRITICAL", "HIGH"):
                continue  # Only persist high-severity findings

            try:
                item = self._finding_to_knowledge_item(
                    finding,
                    receipt,
                    workspace_id,
                    base_tags,
                )
                item_id = await self._store_item(item)
                if item_id:
                    knowledge_item_ids.append(item_id)
                    findings_ingested += 1
            except Exception as e:
                errors.append(f"Failed to ingest finding: {str(e)[:100]}")
                logger.warning(f"Finding ingestion failed: {e}")

        # 3. Create receipt summary item
        try:
            summary_item = self._receipt_to_summary_item(receipt, workspace_id, base_tags)
            summary_id = await self._store_item(summary_item)
            if summary_id:
                knowledge_item_ids.append(summary_id)

                # Create relationships from summary to claims/findings
                for item_id in knowledge_item_ids[:-1]:  # Exclude summary itself
                    try:
                        await self._create_relationship(
                            source_id=summary_id,
                            target_id=item_id,
                            relationship_type=RelationshipType.SUPPORTS,
                        )
                        relationships_created += 1
                    except Exception:
                        pass  # Non-critical
        except Exception as e:
            errors.append(f"Failed to create receipt summary: {str(e)[:100]}")

        result = ReceiptIngestionResult(
            receipt_id=receipt.receipt_id,
            claims_ingested=claims_ingested,
            findings_ingested=findings_ingested,
            relationships_created=relationships_created,
            knowledge_item_ids=knowledge_item_ids,
            errors=errors,
        )

        # Cache result
        self._ingested_receipts[receipt.receipt_id] = result

        # Emit event
        self._emit_event(
            "receipt_ingested",
            {
                "receipt_id": receipt.receipt_id,
                "verdict": receipt.verdict,
                "claims_ingested": claims_ingested,
                "findings_ingested": findings_ingested,
            },
        )

        logger.info(
            "receipt_ingested",
            extra={
                "receipt_id": receipt.receipt_id,
                "claims": claims_ingested,
                "findings": findings_ingested,
                "relationships": relationships_created,
            },
        )

        return result

    def _verification_to_knowledge_item(
        self,
        verification: "ReceiptVerification",
        receipt: "DecisionReceipt",
        workspace_id: Optional[str],
        tags: List[str],
    ) -> KnowledgeItem:
        """Convert a verified claim to a knowledge item."""
        # Generate deterministic ID from claim content
        claim_hash = hashlib.sha256(verification.claim.encode()).hexdigest()[:12]
        item_id = f"{self.CLAIM_PREFIX}{claim_hash}"

        confidence = ConfidenceLevel.HIGH if verification.verified else ConfidenceLevel.LOW
        now = datetime.now(timezone.utc)

        return KnowledgeItem(
            id=item_id,
            content=verification.claim,
            source=RECEIPT_SOURCE,
            source_id=receipt.receipt_id,
            confidence=confidence,
            created_at=now,
            updated_at=now,
            metadata={
                "receipt_id": receipt.receipt_id,
                "gauntlet_id": receipt.gauntlet_id,
                "verification_method": verification.method,
                "proof_hash": verification.proof_hash,
                "receipt_verdict": receipt.verdict,
                "receipt_confidence": receipt.confidence,
                "workspace_id": workspace_id or "",
                "tags": tags + ["verified_claim", f"method:{verification.method}"],
                "item_type": "verified_claim",
            },
        )

    def _finding_to_knowledge_item(
        self,
        finding: "ReceiptFinding",
        receipt: "DecisionReceipt",
        workspace_id: Optional[str],
        tags: List[str],
    ) -> KnowledgeItem:
        """Convert a finding to a knowledge item."""
        # Generate deterministic ID from finding
        finding_hash = hashlib.sha256(f"{finding.id}:{finding.title}".encode()).hexdigest()[:12]
        item_id = f"{self.FINDING_PREFIX}{finding_hash}"

        # Map severity to confidence (inverse - high severity = important but uncertain)
        confidence_map = {
            "CRITICAL": ConfidenceLevel.HIGH,
            "HIGH": ConfidenceLevel.HIGH,
            "MEDIUM": ConfidenceLevel.MEDIUM,
            "LOW": ConfidenceLevel.LOW,
        }
        confidence = confidence_map.get(finding.severity, ConfidenceLevel.MEDIUM)

        content = f"[{finding.severity}] {finding.title}: {finding.description}"
        if finding.mitigation:
            content += f"\n\nMitigation: {finding.mitigation}"

        now = datetime.now(timezone.utc)

        return KnowledgeItem(
            id=item_id,
            content=content,
            source=RECEIPT_SOURCE,
            source_id=receipt.receipt_id,
            confidence=confidence,
            created_at=now,
            updated_at=now,
            metadata={
                "receipt_id": receipt.receipt_id,
                "finding_id": finding.id,
                "severity": finding.severity,
                "category": finding.category,
                "finding_source": finding.source,
                "verified": finding.verified,
                "mitigation": finding.mitigation,
                "workspace_id": workspace_id or "",
                "tags": tags
                + [
                    "finding",
                    f"severity:{finding.severity.lower()}",
                    f"category:{finding.category}",
                ],
                "item_type": "finding",
            },
        )

    def _receipt_to_summary_item(
        self,
        receipt: "DecisionReceipt",
        workspace_id: Optional[str],
        tags: List[str],
    ) -> KnowledgeItem:
        """Create a summary knowledge item for the receipt."""
        item_id = f"{self.ID_PREFIX}{receipt.receipt_id}"

        # Map verdict to confidence
        confidence_map = {
            "APPROVED": ConfidenceLevel.HIGH,
            "APPROVED_WITH_CONDITIONS": ConfidenceLevel.MEDIUM,
            "NEEDS_REVIEW": ConfidenceLevel.LOW,
            "REJECTED": ConfidenceLevel.LOW,
        }
        confidence = confidence_map.get(receipt.verdict, ConfidenceLevel.MEDIUM)

        summary = (
            f"Decision Receipt: {receipt.verdict}\n\n"
            f"Input: {receipt.input_summary[:500]}\n\n"
            f"Confidence: {receipt.confidence:.0%}\n"
            f"Risk Level: {receipt.risk_level}\n"
            f"Findings: {len(receipt.findings)} "
            f"(Critical: {receipt.critical_count}, High: {receipt.high_count})\n"
            f"Verified Claims: {len(receipt.verified_claims)}"
        )

        now = datetime.now(timezone.utc)

        return KnowledgeItem(
            id=item_id,
            content=summary,
            source=RECEIPT_SOURCE,
            source_id=receipt.receipt_id,
            confidence=confidence,
            created_at=now,
            updated_at=now,
            metadata={
                "receipt_id": receipt.receipt_id,
                "gauntlet_id": receipt.gauntlet_id,
                "verdict": receipt.verdict,
                "confidence": receipt.confidence,
                "risk_level": receipt.risk_level,
                "risk_score": receipt.risk_score,
                "critical_count": receipt.critical_count,
                "high_count": receipt.high_count,
                "agents_involved": receipt.agents_involved,
                "duration_seconds": receipt.duration_seconds,
                "checksum": receipt.checksum,
                "audit_trail_id": receipt.audit_trail_id,
                "workspace_id": workspace_id or "",
                "tags": tags + ["decision_receipt", "summary"],
                "item_type": "decision_summary",
            },
        )

    async def _store_item(self, item: KnowledgeItem) -> Optional[str]:
        """Store a knowledge item in the mound."""
        if not self._mound:
            return None

        try:
            # Try direct store method first
            if hasattr(self._mound, "store"):
                result = await self._mound.store(item)
                return result.id if hasattr(result, "id") else item.id
            # Fall back to ingest
            elif hasattr(self._mound, "ingest"):
                await self._mound.ingest(item)
                return item.id
            return item.id
        except Exception as e:
            logger.warning(f"Failed to store item {item.id}: {e}")
            return None

    async def _create_relationship(
        self,
        source_id: str,
        target_id: str,
        relationship_type: RelationshipType,
    ) -> bool:
        """Create a relationship between knowledge items."""
        if not self._mound:
            return False

        try:
            if hasattr(self._mound, "link"):
                await self._mound.link(
                    source_id=source_id,
                    target_id=target_id,
                    relationship_type=relationship_type,
                )
                return True
        except Exception as e:
            logger.debug(f"Failed to create relationship: {e}")
        return False

    async def find_related_decisions(
        self,
        query: str,
        workspace_id: Optional[str] = None,
        limit: int = 5,
    ) -> List[KnowledgeItem]:
        """
        Find decisions related to a query.

        Args:
            query: Search query
            workspace_id: Optional workspace filter
            limit: Maximum results

        Returns:
            List of related decision knowledge items
        """
        if not self._mound:
            return []

        try:
            if hasattr(self._mound, "query"):
                results = await self._mound.query(
                    query=query,
                    tags=["decision_receipt"],
                    workspace_id=workspace_id,
                    limit=limit,
                )
                return results.items if hasattr(results, "items") else []
        except Exception as e:
            logger.warning(f"Failed to find related decisions: {e}")

        return []

    def get_ingestion_result(self, receipt_id: str) -> Optional[ReceiptIngestionResult]:
        """Get the ingestion result for a receipt."""
        return self._ingested_receipts.get(receipt_id)

    def get_stats(self) -> Dict[str, Any]:
        """Get adapter statistics."""
        total_claims = sum(r.claims_ingested for r in self._ingested_receipts.values())
        total_findings = sum(r.findings_ingested for r in self._ingested_receipts.values())
        total_errors = sum(len(r.errors) for r in self._ingested_receipts.values())

        return {
            "receipts_processed": len(self._ingested_receipts),
            "total_claims_ingested": total_claims,
            "total_findings_ingested": total_findings,
            "total_errors": total_errors,
            "mound_connected": self._mound is not None,
            "auto_ingest_enabled": self._auto_ingest,
        }


__all__ = [
    "ReceiptAdapter",
    "ReceiptAdapterError",
    "ReceiptNotFoundError",
    "ReceiptIngestionResult",
]
