"""
Tests for ReceiptAdapter - Decision Receipt to Knowledge Mound bridge.

Tests cover:
- Receipt ingestion to Knowledge Mound
- Verified claim extraction
- Finding persistence
- Relationship creation
- Error handling
"""

import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

from aragora.knowledge.mound.adapters.receipt_adapter import (
    ReceiptAdapter,
    ReceiptAdapterError,
    ReceiptIngestionResult,
)
from aragora.knowledge.unified.types import ConfidenceLevel, KnowledgeSource


class MockReceiptVerification:
    """Mock verification for testing."""

    def __init__(
        self,
        claim: str = "Test claim",
        verified: bool = True,
        method: str = "formal",
        proof_hash: str = "abc123",
    ):
        self.claim = claim
        self.verified = verified
        self.method = method
        self.proof_hash = proof_hash


class MockReceiptFinding:
    """Mock finding for testing."""

    def __init__(
        self,
        id: str = "finding-1",
        severity: str = "HIGH",
        category: str = "security",
        title: str = "Test Finding",
        description: str = "Test description",
        mitigation: str = "Test mitigation",
        source: str = "agent-1",
        verified: bool = True,
    ):
        self.id = id
        self.severity = severity
        self.category = category
        self.title = title
        self.description = description
        self.mitigation = mitigation
        self.source = source
        self.verified = verified


class MockDecisionReceipt:
    """Mock decision receipt for testing."""

    def __init__(
        self,
        receipt_id: str = "receipt-test-123",
        gauntlet_id: str = "gauntlet-test",
        verdict: str = "APPROVED",
        confidence: float = 0.85,
        risk_level: str = "MEDIUM",
        risk_score: float = 0.35,
    ):
        self.receipt_id = receipt_id
        self.gauntlet_id = gauntlet_id
        self.timestamp = datetime.now(timezone.utc).isoformat()
        self.input_summary = "Test input for decision"
        self.input_type = "spec"
        self.verdict = verdict
        self.confidence = confidence
        self.risk_level = risk_level
        self.risk_score = risk_score
        self.critical_count = 0
        self.high_count = 1
        self.medium_count = 2
        self.low_count = 0
        self.checksum = "abc123checksum"
        self.audit_trail_id = "trail-123"
        self.agents_involved = ["claude", "gpt-4"]
        self.duration_seconds = 45.5

        # Add some verified claims
        self.verified_claims = [
            MockReceiptVerification(
                claim="The API is secure against SQL injection",
                verified=True,
                method="static_analysis",
            ),
            MockReceiptVerification(
                claim="Rate limiting is properly configured",
                verified=True,
                method="integration_test",
            ),
            MockReceiptVerification(
                claim="Unverified claim",
                verified=False,
                method="manual_review",
            ),
        ]

        # Add some findings
        self.findings = [
            MockReceiptFinding(
                id="f-1",
                severity="CRITICAL",
                category="security",
                title="Authentication bypass",
                description="Critical vulnerability in auth flow",
            ),
            MockReceiptFinding(
                id="f-2",
                severity="HIGH",
                category="performance",
                title="Memory leak",
                description="Memory leak in request handler",
            ),
            MockReceiptFinding(
                id="f-3",
                severity="MEDIUM",
                category="documentation",
                title="Missing docs",
                description="API documentation incomplete",
            ),
        ]


class TestReceiptAdapter:
    """Tests for ReceiptAdapter."""

    @pytest.fixture
    def adapter(self):
        """Create an adapter for testing."""
        return ReceiptAdapter()

    @pytest.fixture
    def mock_mound(self):
        """Create a mock Knowledge Mound."""
        mound = MagicMock()
        mound.store = AsyncMock(return_value=MagicMock(id="stored-id"))
        mound.link = AsyncMock(return_value=True)
        mound.query = AsyncMock(return_value=MagicMock(items=[]))
        return mound

    @pytest.fixture
    def receipt(self):
        """Create a mock receipt."""
        return MockDecisionReceipt()

    def test_adapter_initialization(self, adapter):
        """Test adapter initialization."""
        assert adapter._mound is None
        assert adapter._auto_ingest is True
        assert len(adapter._ingested_receipts) == 0

    def test_set_mound(self, adapter, mock_mound):
        """Test setting the mound."""
        adapter.set_mound(mock_mound)
        assert adapter._mound is mock_mound

    def test_get_stats_empty(self, adapter):
        """Test stats when empty."""
        stats = adapter.get_stats()

        assert stats["receipts_processed"] == 0
        assert stats["total_claims_ingested"] == 0
        assert stats["total_findings_ingested"] == 0
        assert stats["mound_connected"] is False

    @pytest.mark.asyncio
    async def test_ingest_receipt_no_mound(self, adapter, receipt):
        """Test ingestion fails gracefully without mound."""
        result = await adapter.ingest_receipt(receipt)

        assert not result.success
        assert "Knowledge Mound not configured" in result.errors
        assert result.claims_ingested == 0

    @pytest.mark.asyncio
    async def test_ingest_receipt_success(self, adapter, mock_mound, receipt):
        """Test successful receipt ingestion."""
        adapter.set_mound(mock_mound)

        result = await adapter.ingest_receipt(receipt, workspace_id="test-ws")

        assert result.success
        assert result.receipt_id == receipt.receipt_id
        # 2 verified claims (one is not verified)
        assert result.claims_ingested == 2
        # 2 high-severity findings (CRITICAL + HIGH, not MEDIUM)
        assert result.findings_ingested == 2
        assert result.relationships_created > 0
        assert len(result.knowledge_item_ids) > 0

    @pytest.mark.asyncio
    async def test_ingest_only_verified_claims(self, adapter, mock_mound, receipt):
        """Test that only verified claims are ingested."""
        adapter.set_mound(mock_mound)

        result = await adapter.ingest_receipt(receipt)

        # Should have 2 verified claims, not 3
        assert result.claims_ingested == 2

    @pytest.mark.asyncio
    async def test_ingest_only_high_severity_findings(self, adapter, mock_mound, receipt):
        """Test that only high-severity findings are ingested."""
        adapter.set_mound(mock_mound)

        result = await adapter.ingest_receipt(receipt)

        # Should have CRITICAL + HIGH, not MEDIUM
        assert result.findings_ingested == 2

    @pytest.mark.asyncio
    async def test_verification_to_knowledge_item(self, adapter, receipt):
        """Test conversion of verification to knowledge item."""
        verification = receipt.verified_claims[0]

        item = adapter._verification_to_knowledge_item(
            verification,
            receipt,
            "test-ws",
            ["base-tag"],
        )

        assert item.content == verification.claim
        assert item.source == KnowledgeSource.DEBATE
        assert item.confidence == ConfidenceLevel.HIGH
        assert item.source_id == receipt.receipt_id
        assert item.metadata["workspace_id"] == "test-ws"
        assert "verified_claim" in item.metadata["tags"]
        assert item.metadata["receipt_id"] == receipt.receipt_id

    @pytest.mark.asyncio
    async def test_finding_to_knowledge_item(self, adapter, receipt):
        """Test conversion of finding to knowledge item."""
        finding = receipt.findings[0]

        item = adapter._finding_to_knowledge_item(
            finding,
            receipt,
            "test-ws",
            ["base-tag"],
        )

        assert finding.title in item.content
        assert finding.description in item.content
        assert item.source == KnowledgeSource.DEBATE
        assert "finding" in item.metadata["tags"]
        assert f"severity:{finding.severity.lower()}" in item.metadata["tags"]
        assert item.metadata["severity"] == finding.severity

    @pytest.mark.asyncio
    async def test_receipt_to_summary_item(self, adapter, receipt):
        """Test conversion of receipt to summary item."""
        item = adapter._receipt_to_summary_item(
            receipt,
            "test-ws",
            ["base-tag"],
        )

        assert "Decision Receipt" in item.content
        assert receipt.verdict in item.content
        assert "decision_receipt" in item.metadata["tags"]
        assert "summary" in item.metadata["tags"]
        assert item.metadata["verdict"] == receipt.verdict
        assert item.metadata["checksum"] == receipt.checksum

    @pytest.mark.asyncio
    async def test_get_ingestion_result(self, adapter, mock_mound, receipt):
        """Test retrieving ingestion result."""
        adapter.set_mound(mock_mound)

        # Before ingestion
        assert adapter.get_ingestion_result(receipt.receipt_id) is None

        # After ingestion
        await adapter.ingest_receipt(receipt)
        result = adapter.get_ingestion_result(receipt.receipt_id)

        assert result is not None
        assert result.receipt_id == receipt.receipt_id

    @pytest.mark.asyncio
    async def test_event_callback_called(self, adapter, mock_mound, receipt):
        """Test that event callback is invoked."""
        events_received = []

        def callback(event_type, data):
            events_received.append((event_type, data))

        adapter.set_mound(mock_mound)
        adapter.set_event_callback(callback)

        await adapter.ingest_receipt(receipt)

        assert len(events_received) == 1
        event_type, data = events_received[0]
        assert event_type == "receipt_ingested"
        assert data["receipt_id"] == receipt.receipt_id

    @pytest.mark.asyncio
    async def test_find_related_decisions(self, adapter, mock_mound):
        """Test finding related decisions."""
        adapter.set_mound(mock_mound)

        # Should not raise
        results = await adapter.find_related_decisions("test query", limit=5)

        mock_mound.query.assert_called_once()
        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_stats_after_ingestion(self, adapter, mock_mound, receipt):
        """Test stats after ingestion."""
        adapter.set_mound(mock_mound)

        await adapter.ingest_receipt(receipt)

        stats = adapter.get_stats()
        assert stats["receipts_processed"] == 1
        assert stats["total_claims_ingested"] == 2
        assert stats["total_findings_ingested"] == 2
        assert stats["mound_connected"] is True


class TestReceiptIngestionResult:
    """Tests for ReceiptIngestionResult dataclass."""

    def test_success_with_ingestions(self):
        """Test success property with ingestions."""
        result = ReceiptIngestionResult(
            receipt_id="test",
            claims_ingested=2,
            findings_ingested=1,
            relationships_created=3,
            knowledge_item_ids=["a", "b", "c"],
            errors=[],
        )

        assert result.success is True

    def test_success_with_errors(self):
        """Test success property with errors."""
        result = ReceiptIngestionResult(
            receipt_id="test",
            claims_ingested=2,
            findings_ingested=1,
            relationships_created=3,
            knowledge_item_ids=["a", "b", "c"],
            errors=["Some error"],
        )

        assert result.success is False

    def test_success_with_no_ingestions(self):
        """Test success property with no ingestions."""
        result = ReceiptIngestionResult(
            receipt_id="test",
            claims_ingested=0,
            findings_ingested=0,
            relationships_created=0,
            knowledge_item_ids=[],
            errors=[],
        )

        assert result.success is False

    def test_to_dict(self):
        """Test serialization to dict."""
        result = ReceiptIngestionResult(
            receipt_id="test-123",
            claims_ingested=2,
            findings_ingested=1,
            relationships_created=3,
            knowledge_item_ids=["a", "b"],
            errors=[],
        )

        data = result.to_dict()

        assert data["receipt_id"] == "test-123"
        assert data["claims_ingested"] == 2
        assert data["success"] is True
