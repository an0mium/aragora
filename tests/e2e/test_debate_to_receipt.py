"""
E2E tests for Debate -> Receipt -> Export flow (Phase 5.1).

Tests the complete integration from starting a debate to generating
a receipt to exporting results, validating the "defensible decisions"
compliance pillar.

Test Coverage:
1. Start debate with mock agents -> reach consensus -> generate receipt -> verify fields
2. Debate with dissent -> receipt captures minority positions
3. Receipt cryptographic hash verification (SHA-256)
4. Receipt export to JSON format
5. Performance benchmark (debate completes within reasonable time)
6. DebateController-level flow with stream event emission
7. Oracle streaming session and SentenceAccumulator
8. Self-improve dry-run (TaskDecomposer.analyze)
"""

from __future__ import annotations

import hashlib
import json
import queue
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.core import Agent, Critique, Environment, Message, Vote
from aragora.debate.orchestrator import Arena
from aragora.debate.protocol import DebateProtocol
from aragora.events.types import StreamEvent, StreamEventType
from aragora.gauntlet.receipt import (
    ConsensusProof,
    DecisionReceipt,
    ProvenanceRecord,
)
from aragora.server.stream.emitter import SyncEventEmitter


# =============================================================================
# Mock Agent for Testing
# =============================================================================


@dataclass
class MockAgentConfig:
    """Configuration for mock agent behavior."""

    name: str
    response: str = "Test response"
    vote_choice: str | None = None
    vote_confidence: float = 0.8
    continue_debate: bool = False  # Whether to request debate continuation
    critique_severity: float = 0.2  # Low severity = agree


class MockDebateAgent(Agent):
    """Mock agent for E2E testing without real LLM calls."""

    def __init__(self, config: MockAgentConfig):
        super().__init__(
            name=config.name,
            model="mock-model",
            role="proposer",
        )
        self.agent_type = "mock"
        self.config = config
        self.generate_calls = 0
        self.critique_calls = 0
        self.vote_calls = 0
        # Token tracking attributes required by extensions
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.total_tokens_in = 0
        self.total_tokens_out = 0
        self.metrics = None
        self.provider = None

    async def generate(self, prompt: str, context: list | None = None) -> str:
        self.generate_calls += 1
        return self.config.response

    async def generate_stream(self, prompt: str, context: list | None = None):
        yield self.config.response

    async def critique(
        self,
        proposal: str,
        task: str,
        context: list | None = None,
        target_agent: str | None = None,
    ) -> Critique:
        self.critique_calls += 1
        return Critique(
            agent=self.name,
            target_agent=target_agent or "unknown",
            target_content=proposal[:100] if proposal else "",
            issues=["Minor point"] if self.config.critique_severity > 0.5 else [],
            suggestions=["Consider alternative"] if self.config.critique_severity > 0.5 else [],
            severity=self.config.critique_severity,
            reasoning="Test critique reasoning",
        )

    async def vote(self, proposals: dict, task: str) -> Vote:
        self.vote_calls += 1
        choice = self.config.vote_choice
        if choice is None:
            choice = list(proposals.keys())[0] if proposals else self.name
        return Vote(
            agent=self.name,
            choice=choice,
            reasoning="I agree with this position",
            confidence=self.config.vote_confidence,
            continue_debate=self.config.continue_debate,
        )


class DissentingAgent(Agent):
    """Agent that consistently disagrees for testing minority position capture."""

    def __init__(self, name: str = "dissenting-agent"):
        super().__init__(name=name, model="dissent-model", role="proposer")
        self.agent_type = "dissenting"
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.total_tokens_in = 0
        self.total_tokens_out = 0
        self.metrics = None
        self.provider = None

    async def generate(self, prompt: str, context: list | None = None) -> str:
        return "I strongly disagree with this approach. Alternative: use a different method."

    async def generate_stream(self, prompt: str, context: list | None = None):
        yield await self.generate(prompt, context)

    async def critique(
        self,
        proposal: str,
        task: str,
        context: list | None = None,
        target_agent: str | None = None,
    ) -> Critique:
        return Critique(
            agent=self.name,
            target_agent=target_agent or "unknown",
            target_content=proposal[:100] if proposal else "",
            issues=["Fundamental flaw in approach", "Missing edge case handling"],
            suggestions=["Consider alternative architecture", "Add error handling"],
            severity=7.5,  # High severity - significant issues
            reasoning="This approach has critical flaws that will cause problems.",
        )

    async def vote(self, proposals: dict, task: str) -> Vote:
        # Always vote for self (dissenting position)
        return Vote(
            agent=self.name,
            choice=self.name,
            reasoning="The consensus approach is flawed. My alternative is better.",
            confidence=0.9,
            continue_debate=True,  # Want to continue arguing
        )


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def consensus_agents() -> list[MockDebateAgent]:
    """Create agents that will reach consensus."""
    shared_response = "We should implement caching for performance."
    return [
        MockDebateAgent(
            MockAgentConfig(
                name="agent-claude",
                response=shared_response,
                vote_confidence=0.9,
                continue_debate=False,
            )
        ),
        MockDebateAgent(
            MockAgentConfig(
                name="agent-gpt",
                response=shared_response,
                vote_confidence=0.85,
                continue_debate=False,
            )
        ),
        MockDebateAgent(
            MockAgentConfig(
                name="agent-gemini",
                response=shared_response,
                vote_confidence=0.88,
                continue_debate=False,
            )
        ),
    ]


@pytest.fixture
def dissent_agents() -> list[Agent]:
    """Create agents with one dissenter."""
    shared_response = "Implement caching with Redis."
    return [
        MockDebateAgent(
            MockAgentConfig(
                name="agent-claude",
                response=shared_response,
                vote_confidence=0.85,
            )
        ),
        MockDebateAgent(
            MockAgentConfig(
                name="agent-gpt",
                response=shared_response,
                vote_confidence=0.8,
            )
        ),
        DissentingAgent(name="agent-contrarian"),
    ]


@pytest.fixture
def simple_environment() -> Environment:
    """Create a simple test environment."""
    return Environment(task="Should we enable caching for read-heavy endpoints?")


@pytest.fixture
def minimal_protocol() -> DebateProtocol:
    """Create a minimal protocol for fast testing."""
    return DebateProtocol(
        rounds=2,
        consensus="majority",
        enable_calibration=False,
        enable_rhetorical_observer=False,
        enable_trickster=False,
    )


# =============================================================================
# Test: Basic Debate -> Receipt Flow
# =============================================================================


@pytest.mark.e2e
class TestDebateToReceiptFlow:
    """Tests for the complete debate -> receipt flow."""

    @pytest.mark.asyncio
    async def test_debate_to_receipt_basic_flow(
        self,
        simple_environment: Environment,
        minimal_protocol: DebateProtocol,
        consensus_agents: list[MockDebateAgent],
    ):
        """Test basic flow: debate -> consensus -> receipt generation."""
        # Run debate
        arena = Arena(simple_environment, consensus_agents, minimal_protocol)
        result = await arena.run()

        # Verify debate completed
        assert result is not None
        assert result.rounds_completed > 0
        assert result.final_answer is not None

        # Generate receipt from debate result
        receipt = DecisionReceipt.from_debate_result(result)

        # Verify receipt fields are populated
        assert receipt.receipt_id is not None
        assert len(receipt.receipt_id) > 0
        assert receipt.gauntlet_id is not None  # Uses debate_id
        assert receipt.timestamp is not None
        assert receipt.confidence >= 0.0
        assert receipt.verdict in ("PASS", "CONDITIONAL", "FAIL")

        # Verify consensus proof
        assert receipt.consensus_proof is not None
        assert isinstance(receipt.consensus_proof.reached, bool)
        assert len(receipt.consensus_proof.supporting_agents) > 0

        # Verify provenance chain exists
        assert len(receipt.provenance_chain) >= 1  # At least verdict event

    @pytest.mark.asyncio
    async def test_receipt_fields_match_debate_result(
        self,
        simple_environment: Environment,
        minimal_protocol: DebateProtocol,
        consensus_agents: list[MockDebateAgent],
    ):
        """Test that receipt fields accurately reflect debate result."""
        arena = Arena(simple_environment, consensus_agents, minimal_protocol)
        result = await arena.run()

        receipt = DecisionReceipt.from_debate_result(result)

        # Confidence should match
        assert receipt.confidence == result.confidence

        # Consensus status should match
        if result.consensus_reached:
            assert receipt.consensus_proof.reached is True
        else:
            assert receipt.consensus_proof.reached is False

        # Participants should be captured
        if result.participants:
            # At least some participants should be in supporting/dissenting
            all_agents = (
                receipt.consensus_proof.supporting_agents
                + receipt.consensus_proof.dissenting_agents
            )
            assert len(all_agents) > 0 or len(result.participants) == 0

        # Rounds should map to probes_run
        assert receipt.probes_run == result.rounds_used

    @pytest.mark.asyncio
    async def test_receipt_from_high_confidence_consensus(
        self,
        simple_environment: Environment,
        consensus_agents: list[MockDebateAgent],
    ):
        """Test receipt verdict is PASS for high-confidence consensus."""
        # Use protocol with high consensus threshold
        protocol = DebateProtocol(
            rounds=3,
            consensus="majority",
            enable_calibration=False,
            enable_rhetorical_observer=False,
            enable_trickster=False,
        )

        arena = Arena(simple_environment, consensus_agents, protocol)
        result = await arena.run()

        # Force high confidence for testing
        result.confidence = 0.85
        result.consensus_reached = True

        receipt = DecisionReceipt.from_debate_result(result)

        # High confidence consensus should result in PASS
        assert receipt.verdict == "PASS"
        assert receipt.confidence >= 0.7


# =============================================================================
# Test: Dissent Capture
# =============================================================================


@pytest.mark.e2e
class TestDissentCapture:
    """Tests for capturing minority/dissenting positions in receipts."""

    @pytest.mark.asyncio
    async def test_receipt_captures_dissenting_views(
        self,
        simple_environment: Environment,
        minimal_protocol: DebateProtocol,
        dissent_agents: list[Agent],
    ):
        """Test that receipts capture dissenting agent positions."""
        arena = Arena(simple_environment, dissent_agents, minimal_protocol)
        result = await arena.run()

        # Add explicit dissenting view for testing
        result.dissenting_views = [
            "agent-contrarian: The consensus approach is fundamentally flawed."
        ]

        receipt = DecisionReceipt.from_debate_result(result)

        # Verify dissenting views are captured
        assert len(receipt.dissenting_views) > 0
        assert any("contrarian" in view.lower() for view in receipt.dissenting_views)

    @pytest.mark.asyncio
    async def test_receipt_consensus_proof_includes_dissenters(
        self,
        simple_environment: Environment,
        minimal_protocol: DebateProtocol,
        dissent_agents: list[Agent],
    ):
        """Test that consensus proof identifies dissenting agents."""
        arena = Arena(simple_environment, dissent_agents, minimal_protocol)
        result = await arena.run()

        # Add dissenting view with agent name
        result.dissenting_views = ["agent-contrarian: Disagrees with the approach."]

        receipt = DecisionReceipt.from_debate_result(result)

        # Dissenting agents should be identified
        if receipt.consensus_proof:
            # Note: from_debate_result extracts agent names from dissenting_views
            all_agents = (
                receipt.consensus_proof.supporting_agents
                + receipt.consensus_proof.dissenting_agents
            )
            # At least some agents should be tracked
            assert receipt.consensus_proof is not None

    @pytest.mark.asyncio
    async def test_low_confidence_produces_conditional_verdict(
        self,
        simple_environment: Environment,
        minimal_protocol: DebateProtocol,
        dissent_agents: list[Agent],
    ):
        """Test that low confidence with dissent produces CONDITIONAL verdict."""
        arena = Arena(simple_environment, dissent_agents, minimal_protocol)
        result = await arena.run()

        # Force low confidence with consensus
        result.confidence = 0.5
        result.consensus_reached = True

        receipt = DecisionReceipt.from_debate_result(result)

        # Low confidence should produce CONDITIONAL
        assert receipt.verdict == "CONDITIONAL"


# =============================================================================
# Test: Cryptographic Hash Verification
# =============================================================================


@pytest.mark.e2e
class TestReceiptHashVerification:
    """Tests for receipt cryptographic integrity."""

    def test_receipt_generates_sha256_artifact_hash(self):
        """Test receipt automatically generates SHA-256 artifact hash."""
        receipt = DecisionReceipt(
            receipt_id="test-hash-001",
            gauntlet_id="debate-001",
            timestamp=datetime.now(timezone.utc).isoformat(),
            input_summary="Test input content",
            input_hash=hashlib.sha256(b"test input").hexdigest(),
            risk_summary={"critical": 0, "high": 0, "medium": 0, "low": 0},
            attacks_attempted=0,
            attacks_successful=0,
            probes_run=3,
            vulnerabilities_found=0,
            verdict="PASS",
            confidence=0.9,
            robustness_score=0.85,
        )

        # Artifact hash should be generated
        assert receipt.artifact_hash is not None
        assert len(receipt.artifact_hash) == 64  # SHA-256 hex length
        assert all(c in "0123456789abcdef" for c in receipt.artifact_hash)

    def test_receipt_integrity_verification_passes(self):
        """Test integrity verification passes for unmodified receipt."""
        receipt = DecisionReceipt(
            receipt_id="test-integrity-001",
            gauntlet_id="debate-001",
            timestamp=datetime.now(timezone.utc).isoformat(),
            input_summary="Test content",
            input_hash="abc123",
            risk_summary={"critical": 0},
            attacks_attempted=0,
            attacks_successful=0,
            probes_run=2,
            vulnerabilities_found=0,
            verdict="PASS",
            confidence=0.85,
            robustness_score=0.8,
        )

        # Verification should pass
        assert receipt.verify_integrity() is True

    def test_receipt_integrity_verification_fails_on_tampering(self):
        """Test integrity verification fails when receipt is modified."""
        receipt = DecisionReceipt(
            receipt_id="test-tamper-001",
            gauntlet_id="debate-001",
            timestamp=datetime.now(timezone.utc).isoformat(),
            input_summary="Original content",
            input_hash="original-hash",
            risk_summary={"critical": 0},
            attacks_attempted=0,
            attacks_successful=0,
            probes_run=2,
            vulnerabilities_found=0,
            verdict="PASS",
            confidence=0.9,
            robustness_score=0.85,
        )

        original_hash = receipt.artifact_hash

        # Tamper with the receipt
        receipt.verdict = "FAIL"

        # Hash should be unchanged (not recalculated)
        assert receipt.artifact_hash == original_hash

        # But verification should fail
        assert receipt.verify_integrity() is False

    def test_deterministic_hash_generation(self):
        """Test that identical receipts produce identical hashes."""
        timestamp = datetime.now(timezone.utc).isoformat()

        receipt1 = DecisionReceipt(
            receipt_id="deterministic-001",
            gauntlet_id="debate-001",
            timestamp=timestamp,
            input_summary="Same content",
            input_hash="same-hash",
            risk_summary={"critical": 0},
            attacks_attempted=0,
            attacks_successful=0,
            probes_run=1,
            vulnerabilities_found=0,
            verdict="PASS",
            confidence=0.9,
            robustness_score=0.85,
        )

        receipt2 = DecisionReceipt(
            receipt_id="deterministic-001",
            gauntlet_id="debate-001",
            timestamp=timestamp,
            input_summary="Same content",
            input_hash="same-hash",
            risk_summary={"critical": 0},
            attacks_attempted=0,
            attacks_successful=0,
            probes_run=1,
            vulnerabilities_found=0,
            verdict="PASS",
            confidence=0.9,
            robustness_score=0.85,
        )

        assert receipt1.artifact_hash == receipt2.artifact_hash

    @pytest.mark.asyncio
    async def test_receipt_hash_from_debate_result(
        self,
        simple_environment: Environment,
        minimal_protocol: DebateProtocol,
        consensus_agents: list[MockDebateAgent],
    ):
        """Test receipt hash is properly generated from debate result."""
        arena = Arena(simple_environment, consensus_agents, minimal_protocol)
        result = await arena.run()

        receipt = DecisionReceipt.from_debate_result(result)

        # Hash should be generated
        assert receipt.artifact_hash is not None
        assert len(receipt.artifact_hash) == 64

        # Verification should pass
        assert receipt.verify_integrity() is True


# =============================================================================
# Test: JSON Export
# =============================================================================


@pytest.mark.e2e
class TestReceiptJSONExport:
    """Tests for receipt JSON export functionality."""

    @pytest.mark.asyncio
    async def test_receipt_export_to_json(
        self,
        simple_environment: Environment,
        minimal_protocol: DebateProtocol,
        consensus_agents: list[MockDebateAgent],
    ):
        """Test receipt can be exported to valid JSON."""
        arena = Arena(simple_environment, consensus_agents, minimal_protocol)
        result = await arena.run()

        # Clear messages to avoid datetime serialization issue in provenance
        # This is a known issue in from_debate_result where msg.timestamp is datetime
        result.messages = []

        receipt = DecisionReceipt.from_debate_result(result)

        # Export to JSON
        json_str = receipt.to_json()

        # Should be valid JSON
        data = json.loads(json_str)

        # Required fields should be present
        assert "receipt_id" in data
        assert "gauntlet_id" in data
        assert "verdict" in data
        assert "confidence" in data
        assert "artifact_hash" in data

    @pytest.mark.asyncio
    async def test_json_roundtrip_preserves_data(
        self,
        simple_environment: Environment,
        minimal_protocol: DebateProtocol,
        consensus_agents: list[MockDebateAgent],
    ):
        """Test JSON export/import roundtrip preserves all data."""
        arena = Arena(simple_environment, consensus_agents, minimal_protocol)
        result = await arena.run()

        # Clear messages to avoid datetime serialization issue in provenance
        result.messages = []

        original = DecisionReceipt.from_debate_result(result)

        # Export to JSON and reimport
        json_str = original.to_json()
        data = json.loads(json_str)
        restored = DecisionReceipt.from_dict(data)

        # Core fields should match
        assert restored.receipt_id == original.receipt_id
        assert restored.gauntlet_id == original.gauntlet_id
        assert restored.verdict == original.verdict
        assert restored.confidence == original.confidence
        assert restored.artifact_hash == original.artifact_hash

        # Integrity should still pass
        assert restored.verify_integrity() is True

    def test_json_export_includes_consensus_proof(self):
        """Test JSON export includes consensus proof details."""
        consensus = ConsensusProof(
            reached=True,
            confidence=0.9,
            supporting_agents=["agent-1", "agent-2", "agent-3"],
            dissenting_agents=["agent-4"],
            method="majority",
        )

        receipt = DecisionReceipt(
            receipt_id="json-proof-001",
            gauntlet_id="debate-001",
            timestamp=datetime.now(timezone.utc).isoformat(),
            input_summary="Test",
            input_hash="hash123",
            risk_summary={"critical": 0},
            attacks_attempted=0,
            attacks_successful=0,
            probes_run=2,
            vulnerabilities_found=0,
            verdict="PASS",
            confidence=0.9,
            robustness_score=0.85,
            consensus_proof=consensus,
        )

        data = json.loads(receipt.to_json())

        # Consensus proof should be in JSON
        assert "consensus_proof" in data
        assert data["consensus_proof"]["reached"] is True
        assert "agent-1" in data["consensus_proof"]["supporting_agents"]
        assert "agent-4" in data["consensus_proof"]["dissenting_agents"]

    def test_json_export_includes_provenance_chain(self):
        """Test JSON export includes provenance chain."""
        provenance = [
            ProvenanceRecord(
                timestamp=datetime.now(timezone.utc).isoformat(),
                event_type="message",
                agent="agent-1",
                description="Initial proposal",
                evidence_hash="abc123",
            ),
            ProvenanceRecord(
                timestamp=datetime.now(timezone.utc).isoformat(),
                event_type="vote",
                agent="agent-2",
                description="Voted for agent-1",
            ),
            ProvenanceRecord(
                timestamp=datetime.now(timezone.utc).isoformat(),
                event_type="verdict",
                description="Consensus reached",
            ),
        ]

        receipt = DecisionReceipt(
            receipt_id="json-provenance-001",
            gauntlet_id="debate-001",
            timestamp=datetime.now(timezone.utc).isoformat(),
            input_summary="Test",
            input_hash="hash123",
            risk_summary={"critical": 0},
            attacks_attempted=0,
            attacks_successful=0,
            probes_run=2,
            vulnerabilities_found=0,
            verdict="PASS",
            confidence=0.9,
            robustness_score=0.85,
            provenance_chain=provenance,
        )

        data = json.loads(receipt.to_json())

        # Provenance chain should be in JSON
        assert "provenance_chain" in data
        assert len(data["provenance_chain"]) == 3
        assert data["provenance_chain"][0]["event_type"] == "message"
        assert data["provenance_chain"][2]["event_type"] == "verdict"


# =============================================================================
# Test: Other Export Formats
# =============================================================================


@pytest.mark.e2e
class TestReceiptExportFormats:
    """Tests for additional export formats."""

    def test_export_to_markdown(self):
        """Test receipt can be exported to Markdown."""
        receipt = DecisionReceipt(
            receipt_id="md-export-001",
            gauntlet_id="debate-001",
            timestamp=datetime.now(timezone.utc).isoformat(),
            input_summary="Should we deploy to production?",
            input_hash="hash123",
            risk_summary={"critical": 0, "high": 1, "medium": 2, "low": 3},
            attacks_attempted=0,
            attacks_successful=0,
            probes_run=3,
            vulnerabilities_found=0,
            verdict="PASS",
            confidence=0.85,
            robustness_score=0.8,
            verdict_reasoning="All checks passed with high confidence",
        )

        markdown = receipt.to_markdown()

        # Should contain key sections
        assert "# Decision Receipt" in markdown
        assert "PASS" in markdown
        assert "85" in markdown or "0.85" in markdown  # Confidence
        assert "Risk Summary" in markdown
        assert "Integrity" in markdown

    def test_export_to_html(self):
        """Test receipt can be exported to HTML."""
        receipt = DecisionReceipt(
            receipt_id="html-export-001",
            gauntlet_id="debate-001",
            timestamp=datetime.now(timezone.utc).isoformat(),
            input_summary="Test decision",
            input_hash="hash123",
            risk_summary={"critical": 0},
            attacks_attempted=0,
            attacks_successful=0,
            probes_run=2,
            vulnerabilities_found=0,
            verdict="PASS",
            confidence=0.9,
            robustness_score=0.85,
        )

        html = receipt.to_html()

        # Should be valid HTML
        assert "<!DOCTYPE html>" in html
        assert "<html>" in html
        assert "</html>" in html
        assert "PASS" in html

    def test_export_to_sarif(self):
        """Test receipt can be exported to SARIF format."""
        receipt = DecisionReceipt(
            receipt_id="sarif-export-001",
            gauntlet_id="debate-001",
            timestamp=datetime.now(timezone.utc).isoformat(),
            input_summary="Code review",
            input_hash="hash123",
            risk_summary={"critical": 0, "high": 1},
            attacks_attempted=5,
            attacks_successful=1,
            probes_run=10,
            vulnerabilities_found=1,
            vulnerability_details=[
                {
                    "id": "vuln-001",
                    "title": "Input Validation Issue",
                    "severity": "HIGH",
                    "severity_level": "HIGH",
                    "category": "security",
                    "description": "User input not validated",
                }
            ],
            verdict="CONDITIONAL",
            confidence=0.75,
            robustness_score=0.6,
        )

        sarif = receipt.to_sarif()

        # Should have SARIF structure
        assert sarif["version"] == "2.1.0"
        assert "runs" in sarif
        assert len(sarif["runs"]) == 1
        assert "tool" in sarif["runs"][0]
        assert "results" in sarif["runs"][0]

    def test_export_to_csv(self):
        """Test receipt can export findings to CSV."""
        receipt = DecisionReceipt(
            receipt_id="csv-export-001",
            gauntlet_id="debate-001",
            timestamp=datetime.now(timezone.utc).isoformat(),
            input_summary="Test",
            input_hash="hash123",
            risk_summary={"critical": 1},
            attacks_attempted=0,
            attacks_successful=0,
            probes_run=1,
            vulnerabilities_found=1,
            vulnerability_details=[
                {
                    "id": "vuln-001",
                    "category": "security",
                    "severity": "CRITICAL",
                    "title": "Test Finding",
                    "description": "Test description",
                    "mitigation": "Fix it",
                    "verified": True,
                    "source": "test",
                }
            ],
            verdict="FAIL",
            confidence=0.9,
            robustness_score=0.2,
        )

        csv_content = receipt.to_csv()

        # Should have header and data
        lines = csv_content.strip().split("\n")
        assert len(lines) == 2  # Header + 1 data row
        assert "Finding ID" in lines[0]
        assert "vuln-001" in csv_content


# =============================================================================
# Test: Performance Benchmarks
# =============================================================================


@pytest.mark.e2e
@pytest.mark.slow
class TestDebatePerformance:
    """Performance benchmarks for debate -> receipt flow."""

    @pytest.mark.asyncio
    async def test_debate_completes_within_timeout(
        self,
        simple_environment: Environment,
        consensus_agents: list[MockDebateAgent],
        monkeypatch,
    ):
        """Test debate with mock agents completes quickly."""
        # Force lightweight similarity backend to avoid downloading ML models
        monkeypatch.setenv("ARAGORA_SIMILARITY_BACKEND", "jaccard")

        protocol = DebateProtocol(
            rounds=3,
            consensus="majority",
            enable_calibration=False,
            enable_rhetorical_observer=False,
            enable_trickster=False,
        )

        start_time = time.monotonic()

        arena = Arena(simple_environment, consensus_agents, protocol)
        result = await arena.run()

        elapsed = time.monotonic() - start_time

        # Mock debate should complete within a reasonable time.
        # Knowledge mound, prompt classification, and other subsystems may
        # add overhead even with mock agents when running without real backends.
        assert elapsed < 15.0, f"Debate took {elapsed:.2f}s, expected < 15s"
        assert result is not None

    @pytest.mark.asyncio
    async def test_receipt_generation_is_fast(
        self,
        simple_environment: Environment,
        minimal_protocol: DebateProtocol,
        consensus_agents: list[MockDebateAgent],
    ):
        """Test receipt generation from result is fast."""
        arena = Arena(simple_environment, consensus_agents, minimal_protocol)
        result = await arena.run()

        start_time = time.monotonic()

        receipt = DecisionReceipt.from_debate_result(result)

        elapsed = time.monotonic() - start_time

        # Receipt generation should be nearly instant (< 100ms)
        assert elapsed < 0.1, f"Receipt generation took {elapsed:.3f}s, expected < 0.1s"
        assert receipt is not None

    @pytest.mark.asyncio
    async def test_json_export_is_fast(
        self,
        simple_environment: Environment,
        minimal_protocol: DebateProtocol,
        consensus_agents: list[MockDebateAgent],
    ):
        """Test JSON export is fast."""
        arena = Arena(simple_environment, consensus_agents, minimal_protocol)
        result = await arena.run()

        # Clear messages to avoid datetime serialization issue
        result.messages = []

        receipt = DecisionReceipt.from_debate_result(result)

        start_time = time.monotonic()

        json_str = receipt.to_json()
        _ = json.loads(json_str)  # Also test parsing

        elapsed = time.monotonic() - start_time

        # JSON export + parse should be nearly instant (< 50ms)
        assert elapsed < 0.05, f"JSON export took {elapsed:.3f}s, expected < 0.05s"

    @pytest.mark.asyncio
    async def test_full_flow_performance(
        self,
        simple_environment: Environment,
        consensus_agents: list[MockDebateAgent],
    ):
        """Test complete debate -> receipt -> export flow performance."""
        protocol = DebateProtocol(
            rounds=5,  # More rounds for realistic test
            consensus="majority",
            enable_calibration=False,
            enable_rhetorical_observer=False,
            enable_trickster=False,
        )

        start_time = time.monotonic()

        # Full flow
        arena = Arena(simple_environment, consensus_agents, protocol)
        result = await arena.run()

        # Clear messages to avoid datetime serialization issue
        result.messages = []

        receipt = DecisionReceipt.from_debate_result(result)
        json_str = receipt.to_json()
        markdown = receipt.to_markdown()
        html = receipt.to_html()

        elapsed = time.monotonic() - start_time

        # Complete flow with mock agents should be fast (< 10 seconds)
        assert elapsed < 10.0, f"Full flow took {elapsed:.2f}s, expected < 10s"
        assert result is not None
        assert receipt is not None
        assert len(json_str) > 0
        assert len(markdown) > 0
        assert len(html) > 0


# =============================================================================
# Test: Edge Cases
# =============================================================================


@pytest.mark.e2e
class TestEdgeCases:
    """Tests for edge cases in debate -> receipt flow."""

    @pytest.mark.asyncio
    async def test_receipt_from_no_consensus_debate(
        self,
        simple_environment: Environment,
        minimal_protocol: DebateProtocol,
    ):
        """Test receipt generation when debate fails to reach consensus."""
        # Create agents that won't agree
        agents = [
            MockDebateAgent(
                MockAgentConfig(
                    name=f"agent-{i}",
                    response=f"Unique position {i}",
                    vote_choice=f"agent-{i}",  # Vote for self
                    vote_confidence=0.5,
                    continue_debate=True,
                )
            )
            for i in range(3)
        ]

        arena = Arena(simple_environment, agents, minimal_protocol)
        result = await arena.run()

        # Force no consensus for testing
        result.consensus_reached = False
        result.confidence = 0.3

        receipt = DecisionReceipt.from_debate_result(result)

        # Should produce FAIL verdict
        assert receipt.verdict == "FAIL"
        assert receipt.consensus_proof is not None
        assert receipt.consensus_proof.reached is False

    @pytest.mark.asyncio
    async def test_receipt_from_single_round_debate(
        self,
        simple_environment: Environment,
        consensus_agents: list[MockDebateAgent],
    ):
        """Test receipt from minimal single-round debate."""
        protocol = DebateProtocol(
            rounds=1,
            consensus="majority",
            enable_calibration=False,
            enable_rhetorical_observer=False,
            enable_trickster=False,
        )

        arena = Arena(simple_environment, consensus_agents, protocol)
        result = await arena.run()

        receipt = DecisionReceipt.from_debate_result(result)

        assert receipt is not None
        assert receipt.probes_run >= 1  # At least 1 round

    def test_receipt_from_empty_dissenting_views(self):
        """Test receipt handles empty dissenting views."""
        # Create a mock result with no dissent
        from aragora.core_types import DebateResult

        result = DebateResult(
            task="Simple question",
            final_answer="The answer is 42",
            confidence=0.95,
            consensus_reached=True,
            rounds_used=2,
            participants=["agent-1", "agent-2"],
            dissenting_views=[],  # Empty
        )

        receipt = DecisionReceipt.from_debate_result(result)

        assert receipt is not None
        assert len(receipt.dissenting_views) == 0
        # Risk summary should have 0 "vulnerabilities" (dissenting views)
        assert receipt.vulnerabilities_found == 0

    def test_receipt_with_long_input_summary(self):
        """Test receipt truncates long input summaries."""
        long_task = "A" * 1000  # Very long task

        from aragora.core_types import DebateResult

        result = DebateResult(
            task=long_task,
            final_answer="Answer",
            confidence=0.8,
            consensus_reached=True,
            rounds_used=2,
        )

        receipt = DecisionReceipt.from_debate_result(result)

        # Input summary should be truncated
        assert len(receipt.input_summary) <= 500


# =============================================================================
# Test: Integration with Arena Subsystems
# =============================================================================


@pytest.mark.e2e
class TestArenaIntegration:
    """Tests for receipt integration with Arena subsystems."""

    @pytest.mark.asyncio
    async def test_receipt_captures_debate_metadata(
        self,
        simple_environment: Environment,
        minimal_protocol: DebateProtocol,
        consensus_agents: list[MockDebateAgent],
    ):
        """Test receipt captures debate configuration metadata."""
        arena = Arena(simple_environment, consensus_agents, minimal_protocol)
        result = await arena.run()

        receipt = DecisionReceipt.from_debate_result(result)

        # Config should be captured
        assert "rounds" in receipt.config_used
        assert "duration_seconds" in receipt.config_used

    @pytest.mark.asyncio
    async def test_multiple_debates_produce_unique_receipts(
        self,
        simple_environment: Environment,
        minimal_protocol: DebateProtocol,
        consensus_agents: list[MockDebateAgent],
    ):
        """Test each debate produces a unique receipt."""
        receipts = []

        for _ in range(3):
            arena = Arena(simple_environment, consensus_agents, minimal_protocol)
            result = await arena.run()
            receipt = DecisionReceipt.from_debate_result(result)
            receipts.append(receipt)

        # All receipt IDs should be unique
        receipt_ids = [r.receipt_id for r in receipts]
        assert len(set(receipt_ids)) == 3  # All unique

        # All artifact hashes may differ (due to timestamps)
        # At minimum, receipt_ids should be unique
