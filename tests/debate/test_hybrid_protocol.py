"""
Tests for the HybridDebateProtocol module.

Tests cover:
- HybridDebateConfig initialization
- HybridDebateProtocol initialization
- run_with_external() happy path (consensus reached)
- run_with_external() with refinement loop
- run_with_external() max rounds exceeded
- Consensus calculation logic
- Receipt generation
- VerificationResult serialization
- Error handling
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock, PropertyMock, patch

import pytest

from aragora.core_types import Agent, Critique, Message
from aragora.debate.hybrid_protocol import (
    HybridDebateConfig,
    HybridDebateProtocol,
    VerificationResult,
    create_hybrid_protocol,
)


class MockAgent(Agent):
    """Mock agent for testing."""

    def __init__(
        self,
        name: str = "mock-agent",
        generate_response: str = "Mock response",
        critique_content: str = "This is a valid and correct proposal. I agree.",
    ) -> None:
        super().__init__(name=name, model="mock", role="proposer")
        self._generate_response = generate_response
        self._critique_content = critique_content

    async def generate(self, prompt: str, context: list[Message] | None = None) -> str:
        return self._generate_response

    async def critique(
        self,
        proposal: str,
        task: str,
        context: list[Message] | None = None,
        target_agent: str | None = None,
    ) -> Critique:
        return Critique(
            agent=self.name,
            target_agent=target_agent or "external",
            target_content=proposal,
            issues=[],
            suggestions=[],
            severity=2.0,
            reasoning=self._critique_content,
        )


class MockExternalAgent:
    """Mock external framework agent for testing."""

    def __init__(
        self,
        name: str = "external-framework",
        initial_response: str = "Initial proposal from external agent",
        refined_response: str = "Refined proposal addressing feedback",
    ) -> None:
        self.name = name
        self._initial_response = initial_response
        self._refined_response = refined_response
        self._call_count = 0

    async def generate(self, prompt: str, context: list[Message] | None = None) -> str:
        self._call_count += 1
        # First call is initial proposal, subsequent are refinements
        if self._call_count == 1:
            return self._initial_response
        return self._refined_response


class TestHybridDebateConfig:
    """Tests for HybridDebateConfig dataclass."""

    def test_config_with_required_fields(self):
        """Config can be created with required fields."""
        external = MockExternalAgent()
        verifiers = [MockAgent("verifier-1")]

        config = HybridDebateConfig(
            external_agent=external,
            verification_agents=verifiers,
        )

        assert config.external_agent == external
        assert config.verification_agents == verifiers
        assert config.consensus_threshold == 0.7  # default
        assert config.max_refinement_rounds == 3  # default

    def test_config_with_custom_values(self):
        """Config accepts custom threshold and rounds."""
        external = MockExternalAgent()
        verifiers = [MockAgent("verifier-1"), MockAgent("verifier-2")]

        config = HybridDebateConfig(
            external_agent=external,
            verification_agents=verifiers,
            consensus_threshold=0.9,
            max_refinement_rounds=5,
            require_receipt=False,
            auto_execute_on_consensus=True,
        )

        assert config.consensus_threshold == 0.9
        assert config.max_refinement_rounds == 5
        assert config.require_receipt is False
        assert config.auto_execute_on_consensus is True

    def test_config_default_concurrency(self):
        """Config has default concurrency settings."""
        external = MockExternalAgent()
        verifiers = [MockAgent()]

        config = HybridDebateConfig(
            external_agent=external,
            verification_agents=verifiers,
        )

        assert config.critique_concurrency == 5
        assert config.refinement_feedback_limit == 5


class TestHybridDebateProtocolInit:
    """Tests for HybridDebateProtocol initialization."""

    def test_init_with_valid_config(self):
        """Protocol initializes with valid config."""
        external = MockExternalAgent()
        verifiers = [MockAgent("verifier-1")]

        config = HybridDebateConfig(
            external_agent=external,
            verification_agents=verifiers,
        )
        protocol = HybridDebateProtocol(config)

        assert protocol.config == config
        assert protocol.external_agent == external
        assert protocol.verification_agents == verifiers

    def test_init_requires_verification_agents(self):
        """Protocol raises error if no verification agents."""
        external = MockExternalAgent()

        config = HybridDebateConfig(
            external_agent=external,
            verification_agents=[],
        )

        with pytest.raises(ValueError, match="At least one verification agent"):
            HybridDebateProtocol(config)


class TestRunWithExternal:
    """Tests for run_with_external() method."""

    @pytest.mark.asyncio
    async def test_happy_path_consensus_reached(self):
        """Consensus reached on first round returns verified result."""
        external = MockExternalAgent(initial_response="A well-designed rate limiter proposal")
        verifiers = [
            MockAgent("verifier-1", critique_content="I agree this is correct and valid."),
            MockAgent("verifier-2", critique_content="This is good and I approve."),
        ]

        config = HybridDebateConfig(
            external_agent=external,
            verification_agents=verifiers,
            consensus_threshold=0.7,
        )
        protocol = HybridDebateProtocol(config)

        result = await protocol.run_with_external("Design a rate limiter")

        assert result.verified is True
        assert result.consensus_score >= 0.7
        assert result.proposal == "A well-designed rate limiter proposal"
        assert result.rounds_used == 1
        assert len(result.critiques) == 2
        assert result.receipt_hash is not None
        assert len(result.receipt_hash) == 64  # SHA-256 hex

    @pytest.mark.asyncio
    async def test_consensus_with_custom_decision_id(self):
        """Custom decision_id is used in result."""
        external = MockExternalAgent()
        verifiers = [MockAgent(critique_content="Agree and approve this valid solution.")]

        config = HybridDebateConfig(
            external_agent=external,
            verification_agents=verifiers,
        )
        protocol = HybridDebateProtocol(config)

        result = await protocol.run_with_external(
            "Test task",
            decision_id="custom-id-123",
        )

        assert result.debate_id == "custom-id-123"

    @pytest.mark.asyncio
    async def test_refinement_loop_reaches_consensus(self):
        """Proposal is refined until consensus is reached."""
        external = MockExternalAgent(
            initial_response="Initial weak proposal",
            refined_response="Improved proposal after feedback",
        )

        # First verifier is critical, second is supportive
        # After refinement, both become supportive
        call_count = [0]

        class AdaptiveVerifier(MockAgent):
            async def critique(
                self,
                proposal: str,
                task: str,
                context: list[Message] | None = None,
                target_agent: str | None = None,
            ) -> Critique:
                call_count[0] += 1
                # Be critical on first round, supportive after
                if "Initial" in proposal:
                    content = "This proposal has issues and is problematic."
                else:
                    content = "This improved proposal is correct and I agree."

                return Critique(
                    agent=self.name,
                    target_agent=target_agent or "external",
                    target_content=proposal,
                    issues=[],
                    suggestions=[],
                    severity=2.0,
                    reasoning=content,
                )

        verifiers = [AdaptiveVerifier("adaptive-1"), AdaptiveVerifier("adaptive-2")]

        config = HybridDebateConfig(
            external_agent=external,
            verification_agents=verifiers,
            consensus_threshold=0.7,
            max_refinement_rounds=3,
        )
        protocol = HybridDebateProtocol(config)

        result = await protocol.run_with_external("Test task")

        assert result.verified is True
        assert result.rounds_used == 2
        assert len(result.refinements) == 1
        assert "Improved" in result.proposal

    @pytest.mark.asyncio
    async def test_max_rounds_exceeded_returns_unverified(self):
        """Returns unverified result when max rounds exceeded."""
        external = MockExternalAgent()
        # Always critical verifiers
        verifiers = [
            MockAgent("critic-1", critique_content="I disagree, this is wrong and flawed."),
            MockAgent("critic-2", critique_content="This has issues and errors."),
        ]

        config = HybridDebateConfig(
            external_agent=external,
            verification_agents=verifiers,
            consensus_threshold=0.9,
            max_refinement_rounds=2,
        )
        protocol = HybridDebateProtocol(config)

        result = await protocol.run_with_external("Difficult task")

        assert result.verified is False
        assert result.rounds_used == 2
        assert result.consensus_score < 0.9
        assert result.receipt_hash is None  # No receipt for unverified

    @pytest.mark.asyncio
    async def test_no_receipt_when_disabled(self):
        """No receipt generated when require_receipt=False."""
        external = MockExternalAgent()
        verifiers = [MockAgent(critique_content="Agree and valid.")]

        config = HybridDebateConfig(
            external_agent=external,
            verification_agents=verifiers,
            require_receipt=False,
        )
        protocol = HybridDebateProtocol(config)

        result = await protocol.run_with_external("Test task")

        assert result.verified is True
        assert result.receipt_hash is None

    @pytest.mark.asyncio
    async def test_external_agent_failure_propagates(self):
        """External agent failure raises exception."""

        class FailingExternal:
            name = "failing-external"

            async def generate(self, prompt: str, context=None) -> str:
                raise RuntimeError("External service unavailable")

        external = FailingExternal()
        verifiers = [MockAgent()]

        config = HybridDebateConfig(
            external_agent=external,
            verification_agents=verifiers,
        )
        protocol = HybridDebateProtocol(config)

        with pytest.raises(RuntimeError, match="External service unavailable"):
            await protocol.run_with_external("Test task")

    @pytest.mark.asyncio
    async def test_verifier_failure_handled_gracefully(self):
        """Single verifier failure does not crash the protocol."""

        class FailingVerifier(MockAgent):
            async def critique(self, *args, **kwargs) -> Critique:
                raise RuntimeError("Verifier crashed")

        external = MockExternalAgent()
        verifiers = [
            FailingVerifier("failing"),
            MockAgent("working", critique_content="Agree and valid."),
        ]

        config = HybridDebateConfig(
            external_agent=external,
            verification_agents=verifiers,
            consensus_threshold=0.5,  # Lower threshold since one fails
        )
        protocol = HybridDebateProtocol(config)

        # Should complete despite one verifier failing
        result = await protocol.run_with_external("Test task")

        # Only one critique collected (from working verifier)
        assert len(result.critiques) == 1

    @pytest.mark.asyncio
    async def test_context_passed_to_external_agent(self):
        """Context is passed through to external agent."""
        received_context = []

        class ContextCapturingExternal:
            name = "context-capturer"

            async def generate(self, prompt: str, context: list[Message] | None = None) -> str:
                received_context.append(context)
                return "Response"

        external = ContextCapturingExternal()
        verifiers = [MockAgent(critique_content="Agree.")]

        config = HybridDebateConfig(
            external_agent=external,
            verification_agents=verifiers,
        )
        protocol = HybridDebateProtocol(config)

        test_context = [Message(role="user", agent="user", content="Previous message")]
        await protocol.run_with_external("Test task", context=test_context)

        assert received_context[0] == test_context


class TestConsensusCalculation:
    """Tests for _calculate_consensus() method."""

    @pytest.fixture
    def protocol(self) -> HybridDebateProtocol:
        """Create a protocol for testing."""
        external = MockExternalAgent()
        verifiers = [MockAgent()]
        config = HybridDebateConfig(
            external_agent=external,
            verification_agents=verifiers,
        )
        return HybridDebateProtocol(config)

    @pytest.mark.asyncio
    async def test_full_agreement(self, protocol: HybridDebateProtocol):
        """All supportive critiques give 1.0 score."""
        critiques = [
            "I agree this is correct and valid.",
            "This is good and I approve the solution.",
            "Excellent work, this is sound and thorough.",
        ]

        score = await protocol._calculate_consensus("proposal", critiques)

        assert score == 1.0

    @pytest.mark.asyncio
    async def test_full_disagreement(self, protocol: HybridDebateProtocol):
        """All critical critiques give 0.0 score."""
        critiques = [
            "I disagree, this is wrong and flawed.",
            "This has critical issues and errors.",
            "The proposal is inadequate and problematic.",
        ]

        score = await protocol._calculate_consensus("proposal", critiques)

        assert score == 0.0

    @pytest.mark.asyncio
    async def test_mixed_opinions(self, protocol: HybridDebateProtocol):
        """Mixed critiques give intermediate score."""
        critiques = [
            "I agree this is correct.",
            "This has issues and is problematic.",
            "Good approach, I approve.",
            "I disagree with this flawed solution.",
        ]

        score = await protocol._calculate_consensus("proposal", critiques)

        # 2 supportive, 2 critical = 0.5
        assert score == 0.5

    @pytest.mark.asyncio
    async def test_empty_critiques(self, protocol: HybridDebateProtocol):
        """Empty critique list returns 0.0 (fail-safe)."""
        score = await protocol._calculate_consensus("proposal", [])

        assert score == 0.0

    @pytest.mark.asyncio
    async def test_neutral_critiques(self, protocol: HybridDebateProtocol):
        """Neutral critiques (no keywords) count as non-supportive."""
        critiques = [
            "The proposal addresses the requirements.",
            "This solution uses a database.",
        ]

        score = await protocol._calculate_consensus("proposal", critiques)

        # No supportive keywords = 0
        assert score == 0.0


class TestReceiptGeneration:
    """Tests for _generate_receipt() method."""

    @pytest.fixture
    def protocol(self) -> HybridDebateProtocol:
        """Create a protocol for testing."""
        external = MockExternalAgent(name="test-external")
        verifiers = [MockAgent("verifier-1"), MockAgent("verifier-2")]
        config = HybridDebateConfig(
            external_agent=external,
            verification_agents=verifiers,
        )
        return HybridDebateProtocol(config)

    def test_receipt_is_sha256_hex(self, protocol: HybridDebateProtocol):
        """Receipt hash is valid SHA-256 hexadecimal."""
        receipt = protocol._generate_receipt(
            proposal="Test proposal",
            task="Test task",
            consensus_score=0.8,
            critiques=["Critique 1", "Critique 2"],
            decision_id="test-123",
        )

        # SHA-256 produces 64 hex characters
        assert len(receipt) == 64
        assert all(c in "0123456789abcdef" for c in receipt)

    def test_receipt_deterministic(self, protocol: HybridDebateProtocol):
        """Same inputs produce same receipt hash."""
        args = {
            "proposal": "Test proposal",
            "task": "Test task",
            "consensus_score": 0.8,
            "critiques": ["Critique 1"],
            "decision_id": "test-123",
        }

        with patch("aragora.debate.hybrid_protocol.datetime") as mock_dt:
            # Fix timestamp for deterministic test
            mock_dt.now.return_value = datetime(2024, 1, 1, tzinfo=timezone.utc)
            mock_dt.side_effect = lambda *args, **kw: datetime(*args, **kw)

            receipt1 = protocol._generate_receipt(**args)
            receipt2 = protocol._generate_receipt(**args)

        assert receipt1 == receipt2

    def test_receipt_changes_with_input(self, protocol: HybridDebateProtocol):
        """Different inputs produce different receipts."""
        base_args = {
            "proposal": "Test proposal",
            "task": "Test task",
            "consensus_score": 0.8,
            "critiques": ["Critique 1"],
            "decision_id": "test-123",
        }

        with patch("aragora.debate.hybrid_protocol.datetime") as mock_dt:
            mock_dt.now.return_value = datetime(2024, 1, 1, tzinfo=timezone.utc)
            mock_dt.side_effect = lambda *args, **kw: datetime(*args, **kw)

            receipt1 = protocol._generate_receipt(**base_args)

            # Change proposal
            modified_args = {**base_args, "proposal": "Different proposal"}
            receipt2 = protocol._generate_receipt(**modified_args)

        assert receipt1 != receipt2


class TestVerificationResult:
    """Tests for VerificationResult dataclass."""

    def test_to_dict_serialization(self):
        """VerificationResult.to_dict() produces valid dictionary."""
        result = VerificationResult(
            proposal="Test proposal",
            verified=True,
            consensus_score=0.85,
            critiques=["Critique 1", "Critique 2"],
            refinements=["Refinement 1"],
            receipt_hash="abc123",
            debate_id="debate-456",
            rounds_used=2,
            external_agent="external-agent",
            verification_agent_names=["verifier-1", "verifier-2"],
        )

        data = result.to_dict()

        assert data["proposal"] == "Test proposal"
        assert data["verified"] is True
        assert data["consensus_score"] == 0.85
        assert data["critiques"] == ["Critique 1", "Critique 2"]
        assert data["refinements"] == ["Refinement 1"]
        assert data["receipt_hash"] == "abc123"
        assert data["debate_id"] == "debate-456"
        assert data["rounds_used"] == 2
        assert data["external_agent"] == "external-agent"
        assert data["verification_agent_names"] == ["verifier-1", "verifier-2"]

    def test_to_dict_timestamp_format(self):
        """Timestamp is serialized as ISO format string."""
        ts = datetime(2024, 6, 15, 12, 30, 45, tzinfo=timezone.utc)
        result = VerificationResult(
            proposal="Test",
            verified=True,
            consensus_score=0.8,
            critiques=[],
            refinements=[],
            timestamp=ts,
        )

        data = result.to_dict()

        assert data["timestamp"] == "2024-06-15T12:30:45+00:00"

    def test_to_dict_json_serializable(self):
        """to_dict() output is JSON serializable."""
        result = VerificationResult(
            proposal="Test proposal",
            verified=True,
            consensus_score=0.85,
            critiques=["Critique 1"],
            refinements=[],
        )

        data = result.to_dict()

        # Should not raise
        json_str = json.dumps(data)
        assert isinstance(json_str, str)


class TestCreateHybridProtocol:
    """Tests for create_hybrid_protocol() factory function."""

    def test_creates_protocol_with_defaults(self):
        """Factory creates protocol with default values."""
        external = MockExternalAgent()
        verifiers = [MockAgent()]

        protocol = create_hybrid_protocol(
            external_agent=external,
            verification_agents=verifiers,
        )

        assert isinstance(protocol, HybridDebateProtocol)
        assert protocol.config.consensus_threshold == 0.7
        assert protocol.config.max_refinement_rounds == 3
        assert protocol.config.require_receipt is True

    def test_creates_protocol_with_custom_values(self):
        """Factory accepts custom configuration values."""
        external = MockExternalAgent()
        verifiers = [MockAgent()]

        protocol = create_hybrid_protocol(
            external_agent=external,
            verification_agents=verifiers,
            consensus_threshold=0.9,
            max_refinement_rounds=5,
            require_receipt=False,
            auto_execute_on_consensus=True,
        )

        assert protocol.config.consensus_threshold == 0.9
        assert protocol.config.max_refinement_rounds == 5
        assert protocol.config.require_receipt is False
        assert protocol.config.auto_execute_on_consensus is True


class TestConcurrencyAndLimits:
    """Tests for concurrency and limit handling."""

    @pytest.mark.asyncio
    async def test_critique_concurrency_limit(self):
        """Critiques respect concurrency limit."""
        active_count = [0]
        max_active = [0]

        class SlowVerifier(MockAgent):
            async def critique(self, *args, **kwargs) -> Critique:
                import asyncio

                active_count[0] += 1
                max_active[0] = max(max_active[0], active_count[0])
                await asyncio.sleep(0.01)  # Simulate work
                active_count[0] -= 1

                return Critique(
                    agent=self.name,
                    target_agent="external",
                    target_content="",
                    issues=[],
                    suggestions=[],
                    severity=1.0,
                    reasoning="Agree.",
                )

        external = MockExternalAgent()
        # 10 verifiers but only 3 concurrent
        verifiers = [SlowVerifier(f"v-{i}") for i in range(10)]

        config = HybridDebateConfig(
            external_agent=external,
            verification_agents=verifiers,
            critique_concurrency=3,
        )
        protocol = HybridDebateProtocol(config)

        await protocol.run_with_external("Test task")

        # Should never exceed concurrency limit
        assert max_active[0] <= 3

    @pytest.mark.asyncio
    async def test_refinement_feedback_limit(self):
        """Refinement prompt respects critique limit."""
        received_prompts = []

        class CapturingExternal:
            name = "capturing"
            _call_count = 0

            async def generate(self, prompt: str, context=None) -> str:
                self._call_count += 1
                received_prompts.append(prompt)
                # First call returns weak proposal, second returns refined
                if self._call_count == 1:
                    return "Initial proposal"
                return "Refined proposal"

        # Create many critical verifiers to generate many critiques
        external = CapturingExternal()
        verifiers = [
            MockAgent(f"critic-{i}", critique_content="I disagree, this is wrong.")
            for i in range(10)
        ]

        config = HybridDebateConfig(
            external_agent=external,
            verification_agents=verifiers,
            consensus_threshold=0.99,  # High threshold to force refinement
            max_refinement_rounds=2,
            refinement_feedback_limit=3,  # Only include 3 critiques
        )
        protocol = HybridDebateProtocol(config)

        await protocol.run_with_external("Test task")

        # Check refinement prompt (second call)
        if len(received_prompts) > 1:
            refinement_prompt = received_prompts[1]
            # Count critique bullets in prompt
            critique_lines = [
                line for line in refinement_prompt.split("\n") if line.startswith("- ")
            ]
            # Should have at most refinement_feedback_limit critiques
            assert len(critique_lines) <= 3


class TestConsensusFailSafe:
    """Tests for consensus fail-safe behavior."""

    @pytest.mark.asyncio
    async def test_empty_critiques_returns_zero(self):
        """Empty critiques should return 0.0, not 1.0."""
        external = MockExternalAgent()
        verifiers = [MockAgent()]
        config = HybridDebateConfig(
            external_agent=external,
            verification_agents=verifiers,
        )
        protocol = HybridDebateProtocol(config)

        score = await protocol._calculate_consensus("proposal", [])
        assert score == 0.0

    @pytest.mark.asyncio
    async def test_quorum_not_met_returns_unverified(self):
        """If min_verification_quorum not met, result should be unverified."""

        class FailingVerifier(MockAgent):
            async def critique(self, *args, **kwargs) -> Critique:
                raise RuntimeError("Verifier unreachable")

        external = MockExternalAgent()
        # All verifiers fail, so 0 critiques collected
        verifiers = [FailingVerifier("fail-1"), FailingVerifier("fail-2")]

        config = HybridDebateConfig(
            external_agent=external,
            verification_agents=verifiers,
            min_verification_quorum=1,
            max_refinement_rounds=2,
        )
        protocol = HybridDebateProtocol(config)

        result = await protocol.run_with_external("Test task")

        assert result.verified is False

    @pytest.mark.asyncio
    async def test_quorum_zero_allows_no_critiques(self):
        """Setting min_verification_quorum=0 should allow empty critiques."""

        class FailingVerifier(MockAgent):
            async def critique(self, *args, **kwargs) -> Critique:
                raise RuntimeError("Verifier unreachable")

        external = MockExternalAgent()
        verifiers = [FailingVerifier("fail-1")]

        config = HybridDebateConfig(
            external_agent=external,
            verification_agents=verifiers,
            min_verification_quorum=0,
            max_refinement_rounds=1,
            consensus_threshold=0.0,  # 0.0 threshold so empty critiques (score 0.0) meets it
        )
        protocol = HybridDebateProtocol(config)

        result = await protocol.run_with_external("Test task")

        # With quorum=0, empty critiques are allowed; consensus of 0.0 meets threshold 0.0
        assert result.verified is True

    @pytest.mark.asyncio
    async def test_single_critique_meets_default_quorum(self):
        """One critique should meet the default quorum of 1."""
        external = MockExternalAgent()
        verifiers = [MockAgent("v1", critique_content="I agree this is correct and valid.")]

        config = HybridDebateConfig(
            external_agent=external,
            verification_agents=verifiers,
            min_verification_quorum=1,
        )
        protocol = HybridDebateProtocol(config)

        result = await protocol.run_with_external("Test task")

        assert result.verified is True
        assert result.consensus_score >= 0.7

    @pytest.mark.asyncio
    async def test_insufficient_critiques_across_rounds(self):
        """If critiques insufficient across all rounds, debate returns unverified."""

        class FailingVerifier(MockAgent):
            async def critique(self, *args, **kwargs) -> Critique:
                raise RuntimeError("Verifier unreachable")

        external = MockExternalAgent()
        verifiers = [FailingVerifier("fail-1")]

        config = HybridDebateConfig(
            external_agent=external,
            verification_agents=verifiers,
            min_verification_quorum=1,
            max_refinement_rounds=3,
        )
        protocol = HybridDebateProtocol(config)

        result = await protocol.run_with_external("Test task")

        assert result.verified is False
        assert result.rounds_used == 3

    def test_min_verification_quorum_config_default(self):
        """Default min_verification_quorum should be 1."""
        external = MockExternalAgent()
        verifiers = [MockAgent()]

        config = HybridDebateConfig(
            external_agent=external,
            verification_agents=verifiers,
        )

        assert config.min_verification_quorum == 1

    def test_min_verification_quorum_config_custom(self):
        """Custom quorum value should be respected."""
        external = MockExternalAgent()
        verifiers = [MockAgent()]

        config = HybridDebateConfig(
            external_agent=external,
            verification_agents=verifiers,
            min_verification_quorum=3,
        )

        assert config.min_verification_quorum == 3


class TestURLValidation:
    """Tests for URL validation in HybridDebateProtocol."""

    def test_rejects_unsafe_external_agent_url(self):
        """Should reject external agents with unsafe URLs."""
        from aragora.security.ssrf_protection import SSRFValidationError

        class UnsafeExternalAgent:
            name = "unsafe-agent"
            base_url = "http://169.254.169.254/latest/meta-data/"

            async def generate(self, prompt, context=None):
                return "response"

        external = UnsafeExternalAgent()
        verifiers = [MockAgent()]

        config = HybridDebateConfig(
            external_agent=external,
            verification_agents=verifiers,
        )

        with pytest.raises(SSRFValidationError):
            HybridDebateProtocol(config)

    def test_accepts_safe_external_agent_url(self):
        """Should accept external agents with safe URLs."""

        class SafeExternalAgent:
            name = "safe-agent"
            base_url = "https://api.example.com/v1/agent"

            async def generate(self, prompt, context=None):
                return "response"

        external = SafeExternalAgent()
        verifiers = [MockAgent()]

        config = HybridDebateConfig(
            external_agent=external,
            verification_agents=verifiers,
        )

        # Should not raise
        protocol = HybridDebateProtocol(config)
        assert protocol.external_agent is external

    def test_consensus_threshold_validation_too_high(self):
        """Should reject consensus_threshold > 1.0."""
        external = MockExternalAgent()
        verifiers = [MockAgent()]

        config = HybridDebateConfig(
            external_agent=external,
            verification_agents=verifiers,
            consensus_threshold=1.5,
        )

        with pytest.raises(ValueError, match="consensus_threshold must be between 0.0 and 1.0"):
            HybridDebateProtocol(config)

    def test_consensus_threshold_validation_too_low(self):
        """Should reject consensus_threshold < 0.0."""
        external = MockExternalAgent()
        verifiers = [MockAgent()]

        config = HybridDebateConfig(
            external_agent=external,
            verification_agents=verifiers,
            consensus_threshold=-0.1,
        )

        with pytest.raises(ValueError, match="consensus_threshold must be between 0.0 and 1.0"):
            HybridDebateProtocol(config)
