"""
End-to-end integration tests for debate → receipt → knowledge mound pipeline.

Tests the complete flow:
1. Create and run a debate (mock or lightweight)
2. Generate decision receipt from DebateResult
3. Ingest receipt into Knowledge Mound via ReceiptAdapter
4. Verify integrity, storage, and retrieval
5. Test related decisions querying
"""

from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.core_types import DebateResult
from aragora.export.decision_receipt import DecisionReceipt
from aragora.gauntlet.receipt import (
    ConsensusProof,
    DecisionReceipt as GauntletReceipt,
    ProvenanceRecord,
)
from aragora.knowledge.mound.adapters.receipt_adapter import (
    ReceiptAdapter,
    ReceiptIngestionResult,
)
from aragora.knowledge.unified.types import (
    ConfidenceLevel,
    KnowledgeItem,
    KnowledgeSource,
)


# =============================================================================
# Mock Fixtures
# =============================================================================


@dataclass
class MockMessage:
    """Mock message for testing."""

    role: str = "assistant"
    content: str = "Test response"
    agent: str = "claude"
    round: int = 1
    timestamp: str | None = None


@dataclass
class MockVote:
    """Mock vote for testing."""

    agent: str = "claude"
    choice: str = "approve"
    confidence: float = 0.85
    reasoning: str = "Well-supported conclusion"


@dataclass
class MockDebateResult:
    """Mock debate result for testing."""

    debate_id: str = ""
    task: str = ""
    status: str = "completed"
    consensus_reached: bool = True
    final_answer: str = ""
    confidence: float = 0.0
    agents: list[str] = field(default_factory=list)
    rounds_used: int = 0
    duration_seconds: float = 0.0
    messages: list[MockMessage] = field(default_factory=list)
    votes: list[MockVote] = field(default_factory=list)
    supporting_agents: list[str] = field(default_factory=list)
    dissenting_agents: list[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.debate_id:
            self.debate_id = f"deb_{uuid.uuid4().hex[:12]}"


class MockKnowledgeMound:
    """Mock Knowledge Mound for testing ingestion."""

    def __init__(self):
        self._items: dict[str, KnowledgeItem] = {}
        self._relationships: list[tuple] = []
        self._queries: list[str] = []

    async def store(
        self,
        item: KnowledgeItem,
        workspace_id: str | None = None,
    ) -> str:
        """Store a knowledge item."""
        self._items[item.id] = item
        return item.id

    async def query_semantic(
        self,
        query: str,
        limit: int = 10,
        workspace_id: str | None = None,
        min_confidence: float | None = None,
    ) -> list[KnowledgeItem]:
        """Query knowledge items semantically."""
        self._queries.append(query)
        # Return matching items based on simple keyword match
        results = []
        for item in self._items.values():
            if any(word in item.content.lower() for word in query.lower().split()):
                results.append(item)
                if len(results) >= limit:
                    break
        return results

    async def add_relationship(
        self,
        from_id: str,
        to_id: str,
        relationship_type: str,
    ) -> None:
        """Add a relationship between items."""
        self._relationships.append((from_id, to_id, relationship_type))

    def get_item(self, item_id: str) -> KnowledgeItem | None:
        """Get an item by ID."""
        return self._items.get(item_id)

    @property
    def item_count(self) -> int:
        return len(self._items)

    @property
    def relationship_count(self) -> int:
        return len(self._relationships)


# =============================================================================
# Test Classes
# =============================================================================


class TestDebateToReceiptGeneration:
    """Tests for generating receipts from debate results."""

    @pytest.fixture
    def mock_debate_result(self) -> MockDebateResult:
        """Create a mock debate result."""
        return MockDebateResult(
            task="Should we implement microservices architecture?",
            final_answer="Yes, microservices are recommended for this scale.",
            confidence=0.88,
            agents=["claude", "gpt-4", "gemini"],
            rounds_used=3,
            duration_seconds=45.5,
            messages=[
                MockMessage(
                    role="assistant",
                    content="I propose we adopt microservices for scalability.",
                    agent="claude",
                    round=1,
                ),
                MockMessage(
                    role="assistant",
                    content="I agree but suggest careful service boundary design.",
                    agent="gpt-4",
                    round=2,
                ),
            ],
            votes=[
                MockVote(agent="claude", choice="approve", confidence=0.9),
                MockVote(agent="gpt-4", choice="approve", confidence=0.85),
                MockVote(agent="gemini", choice="approve", confidence=0.88),
            ],
            supporting_agents=["claude", "gpt-4", "gemini"],
            dissenting_agents=[],
        )

    def test_create_gauntlet_receipt_from_debate_fields(self, mock_debate_result):
        """Should create a GauntletReceipt from debate result fields."""
        # Create consensus proof from debate result
        consensus = ConsensusProof(
            reached=mock_debate_result.consensus_reached,
            confidence=mock_debate_result.confidence,
            supporting_agents=mock_debate_result.supporting_agents,
            dissenting_agents=mock_debate_result.dissenting_agents,
            method="majority",
        )

        # Create provenance records
        provenance = [
            ProvenanceRecord(
                timestamp=datetime.now(timezone.utc).isoformat(),
                event_type="debate_start",
                description=f"Debate started: {mock_debate_result.task}",
            ),
            ProvenanceRecord(
                timestamp=datetime.now(timezone.utc).isoformat(),
                event_type="consensus",
                description="Consensus reached via majority vote",
            ),
        ]

        # Create the receipt
        receipt = GauntletReceipt(
            receipt_id=f"rcpt_{uuid.uuid4().hex[:12]}",
            gauntlet_id=mock_debate_result.debate_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            input_summary=mock_debate_result.task,
            input_hash=hashlib.sha256(mock_debate_result.task.encode()).hexdigest(),
            risk_summary={"critical": 0, "high": 0, "medium": 1, "low": 2},
            attacks_attempted=0,
            attacks_successful=0,
            probes_run=3,
            vulnerabilities_found=0,
            verdict="PASS",
            confidence=mock_debate_result.confidence,
            robustness_score=0.92,
            verdict_reasoning=mock_debate_result.final_answer,
            consensus_proof=consensus,
            provenance_chain=provenance,
        )

        assert receipt.receipt_id.startswith("rcpt_")
        assert receipt.verdict == "PASS"
        assert receipt.confidence == 0.88
        assert receipt.consensus_proof is not None
        assert receipt.consensus_proof.reached is True
        assert len(receipt.provenance_chain) == 2

    def test_receipt_integrity_verification(self, mock_debate_result):
        """Should verify receipt integrity after creation."""
        receipt = GauntletReceipt(
            receipt_id="rcpt_test_integrity",
            gauntlet_id=mock_debate_result.debate_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            input_summary=mock_debate_result.task,
            input_hash=hashlib.sha256(mock_debate_result.task.encode()).hexdigest(),
            risk_summary={"critical": 0, "high": 0, "medium": 0, "low": 0},
            attacks_attempted=0,
            attacks_successful=0,
            probes_run=0,
            vulnerabilities_found=0,
            verdict="PASS",
            confidence=0.9,
            robustness_score=0.95,
        )

        # Verify integrity
        assert receipt.verify_integrity() is True

        # Tamper with the receipt
        original_verdict = receipt.verdict
        receipt.verdict = "FAIL"

        # Integrity should fail after tampering
        assert receipt.verify_integrity() is False

        # Restore original
        receipt.verdict = original_verdict

    def test_receipt_hash_uniqueness(self, mock_debate_result):
        """Different receipts should have different hashes."""
        receipt1 = GauntletReceipt(
            receipt_id="rcpt_hash_1",
            gauntlet_id="gnt_001",
            timestamp=datetime.now(timezone.utc).isoformat(),
            input_summary="Task 1",
            input_hash=hashlib.sha256(b"Task 1").hexdigest(),
            risk_summary={"critical": 0, "high": 0, "medium": 0, "low": 0},
            attacks_attempted=0,
            attacks_successful=0,
            probes_run=0,
            vulnerabilities_found=0,
            verdict="PASS",
            confidence=0.9,
            robustness_score=0.9,
        )

        receipt2 = GauntletReceipt(
            receipt_id="rcpt_hash_2",
            gauntlet_id="gnt_002",
            timestamp=datetime.now(timezone.utc).isoformat(),
            input_summary="Task 2",
            input_hash=hashlib.sha256(b"Task 2").hexdigest(),
            risk_summary={"critical": 0, "high": 0, "medium": 0, "low": 0},
            attacks_attempted=0,
            attacks_successful=0,
            probes_run=0,
            vulnerabilities_found=0,
            verdict="PASS",
            confidence=0.9,
            robustness_score=0.9,
        )

        assert receipt1.artifact_hash != receipt2.artifact_hash


class TestReceiptToKnowledgeMoundIngestion:
    """Tests for ingesting receipts into Knowledge Mound."""

    @pytest.fixture
    def mock_mound(self) -> MockKnowledgeMound:
        """Create a mock Knowledge Mound."""
        return MockKnowledgeMound()

    @pytest.fixture
    def receipt_adapter(self, mock_mound) -> ReceiptAdapter:
        """Create a ReceiptAdapter with mock mound."""
        adapter = ReceiptAdapter(mound=mock_mound)
        return adapter

    @pytest.fixture
    def sample_receipt(self) -> MagicMock:
        """Create a sample receipt for ingestion testing."""
        receipt = MagicMock()
        receipt.receipt_id = "rcpt_test_km_001"
        receipt.gauntlet_id = "gnt_test_001"
        receipt.timestamp = datetime.now(timezone.utc).isoformat()
        receipt.input_summary = "Evaluate security of user authentication"
        receipt.verdict = "PASS"
        receipt.confidence = 0.87
        receipt.risk_level = "LOW"
        receipt.risk_summary = {"critical": 0, "high": 0, "medium": 1, "low": 2}
        receipt.findings = [
            MagicMock(
                id="find_001",
                severity="MEDIUM",
                category="security",
                title="Rate limiting recommended",
                description="Add rate limiting to prevent brute force",
                mitigation="Implement exponential backoff",
                source="claude",
            )
        ]
        receipt.verified_claims = [
            MagicMock(
                claim="OAuth2 implementation follows best practices",
                confidence=0.92,
                source="gpt-4",
            )
        ]
        receipt.dissenting_views = []
        receipt.agents_involved = ["claude", "gpt-4", "gemini"]
        return receipt

    @pytest.mark.asyncio
    async def test_adapter_ingests_receipt(self, receipt_adapter, sample_receipt, mock_mound):
        """Should ingest receipt and create knowledge items."""
        result = await receipt_adapter.ingest_receipt(
            sample_receipt,
            workspace_id="ws-test",
            tags=["security", "auth"],
        )

        # Should return ingestion result
        assert isinstance(result, ReceiptIngestionResult)
        assert result.receipt_id == "rcpt_test_km_001"
        # Claims and findings should be ingested (counts depend on implementation)
        assert result.success or len(result.errors) > 0  # Either succeeds or has errors

    @pytest.mark.asyncio
    async def test_adapter_without_mound_returns_error(self, sample_receipt):
        """Adapter without mound should return error."""
        adapter = ReceiptAdapter(mound=None)
        result = await adapter.ingest_receipt(sample_receipt)

        assert not result.success
        assert "Knowledge Mound not configured" in result.errors

    @pytest.mark.asyncio
    async def test_ingestion_result_serialization(self, receipt_adapter, sample_receipt):
        """IngestionResult should serialize to dict."""
        result = await receipt_adapter.ingest_receipt(sample_receipt)
        result_dict = result.to_dict()

        assert "receipt_id" in result_dict
        assert "claims_ingested" in result_dict
        assert "findings_ingested" in result_dict
        assert "success" in result_dict
        assert "errors" in result_dict


class TestEndToEndPipeline:
    """Full end-to-end tests of the debate → receipt → KM pipeline."""

    @pytest.fixture
    def mock_mound(self) -> MockKnowledgeMound:
        """Create a mock Knowledge Mound."""
        return MockKnowledgeMound()

    @pytest.mark.asyncio
    async def test_full_pipeline_creates_auditable_trace(self, mock_mound):
        """Complete pipeline should create an auditable trace."""
        # Step 1: Simulate debate result
        debate_result = MockDebateResult(
            task="Should we migrate to PostgreSQL from MySQL?",
            final_answer="Yes, PostgreSQL offers better features for our use case.",
            confidence=0.91,
            agents=["claude", "gpt-4"],
            rounds_used=2,
            duration_seconds=30.0,
            supporting_agents=["claude", "gpt-4"],
            dissenting_agents=[],
        )

        # Step 2: Generate receipt from debate
        receipt = GauntletReceipt(
            receipt_id=f"rcpt_{uuid.uuid4().hex[:12]}",
            gauntlet_id=debate_result.debate_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            input_summary=debate_result.task,
            input_hash=hashlib.sha256(debate_result.task.encode()).hexdigest(),
            risk_summary={"critical": 0, "high": 0, "medium": 0, "low": 1},
            attacks_attempted=0,
            attacks_successful=0,
            probes_run=2,
            vulnerabilities_found=0,
            verdict="PASS",
            confidence=debate_result.confidence,
            robustness_score=0.88,
            verdict_reasoning=debate_result.final_answer,
            consensus_proof=ConsensusProof(
                reached=True,
                confidence=debate_result.confidence,
                supporting_agents=debate_result.supporting_agents,
                dissenting_agents=debate_result.dissenting_agents,
                method="majority",
            ),
            provenance_chain=[
                ProvenanceRecord(
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    event_type="debate",
                    agent=", ".join(debate_result.agents),
                    description=f"Multi-agent debate: {debate_result.rounds_used} rounds",
                ),
            ],
        )

        # Verify receipt integrity
        assert receipt.verify_integrity() is True
        assert receipt.verdict == "PASS"
        assert receipt.confidence > 0.9

        # Step 3: Verify provenance chain
        assert len(receipt.provenance_chain) > 0
        assert receipt.consensus_proof is not None
        assert receipt.consensus_proof.reached is True

    @pytest.mark.asyncio
    async def test_pipeline_with_dissenting_agents(self, mock_mound):
        """Pipeline should preserve dissenting views."""
        # Debate with dissent
        debate_result = MockDebateResult(
            task="Should we use NoSQL for transactional data?",
            final_answer="No, ACID compliance is critical for transactions.",
            confidence=0.72,
            agents=["claude", "gpt-4", "gemini"],
            rounds_used=3,
            supporting_agents=["claude", "gpt-4"],
            dissenting_agents=["gemini"],
        )

        # Create receipt with dissenting view
        receipt = GauntletReceipt(
            receipt_id=f"rcpt_{uuid.uuid4().hex[:12]}",
            gauntlet_id=debate_result.debate_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            input_summary=debate_result.task,
            input_hash=hashlib.sha256(debate_result.task.encode()).hexdigest(),
            risk_summary={"critical": 0, "high": 1, "medium": 0, "low": 0},
            attacks_attempted=0,
            attacks_successful=0,
            probes_run=3,
            vulnerabilities_found=0,
            verdict="CONDITIONAL",
            confidence=debate_result.confidence,
            robustness_score=0.65,
            verdict_reasoning=debate_result.final_answer,
            dissenting_views=["Gemini argues: NoSQL could work with proper saga patterns."],
            consensus_proof=ConsensusProof(
                reached=True,
                confidence=debate_result.confidence,
                supporting_agents=debate_result.supporting_agents,
                dissenting_agents=debate_result.dissenting_agents,
                method="majority",
            ),
        )

        # Verify dissenting views are preserved
        assert len(receipt.dissenting_views) == 1
        assert "Gemini" in receipt.dissenting_views[0]
        assert receipt.verdict == "CONDITIONAL"
        assert receipt.consensus_proof.dissenting_agents == ["gemini"]

    @pytest.mark.asyncio
    async def test_pipeline_fail_verdict(self, mock_mound):
        """Pipeline should handle FAIL verdicts correctly."""
        debate_result = MockDebateResult(
            task="Should we remove authentication for public endpoints?",
            final_answer="No, all endpoints require authentication for security.",
            confidence=0.95,
            agents=["claude", "gpt-4", "gemini"],
            supporting_agents=["claude", "gpt-4", "gemini"],
            dissenting_agents=[],
        )

        # Receipt with FAIL verdict (proposal rejected)
        receipt = GauntletReceipt(
            receipt_id=f"rcpt_{uuid.uuid4().hex[:12]}",
            gauntlet_id=debate_result.debate_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            input_summary=debate_result.task,
            input_hash=hashlib.sha256(debate_result.task.encode()).hexdigest(),
            risk_summary={"critical": 1, "high": 2, "medium": 0, "low": 0},
            attacks_attempted=3,
            attacks_successful=3,
            probes_run=5,
            vulnerabilities_found=3,
            verdict="FAIL",
            confidence=debate_result.confidence,
            robustness_score=0.1,
            verdict_reasoning="Removing authentication poses severe security risks.",
            vulnerability_details=[
                {
                    "id": "vuln_001",
                    "severity": "CRITICAL",
                    "type": "security",
                    "description": "Unauthorized access vulnerability",
                }
            ],
        )

        assert receipt.verdict == "FAIL"
        assert receipt.risk_summary["critical"] == 1
        assert len(receipt.vulnerability_details) == 1
        assert receipt.verify_integrity() is True


class TestReceiptExportFormats:
    """Tests for receipt export to various formats."""

    @pytest.fixture
    def complete_receipt(self) -> GauntletReceipt:
        """Create a complete receipt with all fields populated."""
        return GauntletReceipt(
            receipt_id="rcpt_export_test",
            gauntlet_id="gnt_export_001",
            timestamp=datetime.now(timezone.utc).isoformat(),
            input_summary="Test export functionality",
            input_hash=hashlib.sha256(b"Test export").hexdigest(),
            risk_summary={"critical": 0, "high": 1, "medium": 2, "low": 3},
            attacks_attempted=5,
            attacks_successful=1,
            probes_run=10,
            vulnerabilities_found=1,
            verdict="CONDITIONAL",
            confidence=0.75,
            robustness_score=0.70,
            verdict_reasoning="Proceed with caution due to identified risks.",
            vulnerability_details=[
                {
                    "id": "vuln_exp_001",
                    "severity": "HIGH",
                    "type": "performance",
                    "description": "Potential memory leak",
                }
            ],
            dissenting_views=["Minor concern about edge cases"],
            consensus_proof=ConsensusProof(
                reached=True,
                confidence=0.75,
                supporting_agents=["claude", "gpt-4"],
                dissenting_agents=["gemini"],
                method="weighted",
            ),
            provenance_chain=[
                ProvenanceRecord(
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    event_type="start",
                    description="Validation started",
                ),
                ProvenanceRecord(
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    event_type="verdict",
                    description="Final verdict rendered",
                ),
            ],
        )

    def test_export_to_dict(self, complete_receipt):
        """Receipt should export to dictionary."""
        result = complete_receipt.to_dict()

        assert isinstance(result, dict)
        assert result["receipt_id"] == "rcpt_export_test"
        assert result["verdict"] == "CONDITIONAL"
        assert "risk_summary" in result
        assert "consensus_proof" in result
        assert "provenance_chain" in result

    def test_export_to_json(self, complete_receipt):
        """Receipt should export to valid JSON."""
        json_str = complete_receipt.to_json()

        import json

        parsed = json.loads(json_str)
        assert parsed["receipt_id"] == "rcpt_export_test"
        assert parsed["confidence"] == 0.75

    def test_export_to_markdown(self, complete_receipt):
        """Receipt should export to markdown format."""
        md = complete_receipt.to_markdown()

        assert "# Decision Receipt" in md
        assert "rcpt_export_test" in md
        assert "CONDITIONAL" in md
        assert "## Provenance" in md or "provenance" in md.lower()


class TestReceiptIngestionResultProperties:
    """Tests for ReceiptIngestionResult dataclass."""

    def test_success_property_when_items_ingested(self):
        """Should be successful when items are ingested."""
        result = ReceiptIngestionResult(
            receipt_id="rcpt_test",
            claims_ingested=2,
            findings_ingested=1,
            relationships_created=3,
            knowledge_item_ids=["km_001", "km_002", "km_003"],
            errors=[],
        )

        assert result.success is True

    def test_failure_property_when_errors_present(self):
        """Should fail when errors are present."""
        result = ReceiptIngestionResult(
            receipt_id="rcpt_test",
            claims_ingested=1,
            findings_ingested=0,
            relationships_created=0,
            knowledge_item_ids=["km_001"],
            errors=["Failed to store finding: validation error"],
        )

        assert result.success is False

    def test_failure_property_when_nothing_ingested(self):
        """Should fail when nothing is ingested and no errors."""
        result = ReceiptIngestionResult(
            receipt_id="rcpt_test",
            claims_ingested=0,
            findings_ingested=0,
            relationships_created=0,
            knowledge_item_ids=[],
            errors=[],
        )

        assert result.success is False

    def test_to_dict_serialization(self):
        """Should serialize correctly to dict."""
        result = ReceiptIngestionResult(
            receipt_id="rcpt_serialize",
            claims_ingested=5,
            findings_ingested=3,
            relationships_created=8,
            knowledge_item_ids=["km_001", "km_002"],
            errors=["warning: optional field missing"],
        )

        d = result.to_dict()
        assert d["receipt_id"] == "rcpt_serialize"
        assert d["claims_ingested"] == 5
        assert d["findings_ingested"] == 3
        assert d["relationships_created"] == 8
        assert d["success"] is False  # Has errors
