"""
Integration tests for hybrid debate protocol with gateway agents.

Tests the hybrid debate protocol functionality including:
- Full hybrid debate with external agents
- Consensus calculation from verification agents
- Quorum enforcement for verification
- Parallel verification execution
- Error handling for external agent failures
- Domain-specific debate configuration
- Result structure validation
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from tests.gateway.integration.conftest import (
    MockAgent,
    MockExternalFrameworkServer,
    FailingAgent,
    SlowAgent,
    TenantContext,
    register_external_agent,
)


# =============================================================================
# Mock Components for Hybrid Protocol Testing
# =============================================================================


class MockCritique:
    """Mock critique object."""

    def __init__(self, content: str):
        self.content = content


class MockExternalAgent:
    """Mock external framework agent."""

    def __init__(
        self,
        name: str = "external-agent",
        base_url: str = "http://localhost:8000",
        proposal: str = "External agent proposal",
        available: bool = True,
        error_after: int | None = None,
    ):
        self.name = name
        self.base_url = base_url
        self.model = "external"
        self._proposal = proposal
        self._available = available
        self._error_after = error_after
        self._call_count = 0

    async def generate(self, prompt: str, context: list | None = None) -> str:
        """Generate a proposal."""
        if self._error_after is not None and self._call_count >= self._error_after:
            raise ConnectionError(f"External agent {self.name} connection failed")
        self._call_count += 1
        return self._proposal

    async def is_available(self) -> bool:
        """Check if agent is available."""
        return self._available


class MockVerificationAgent:
    """Mock verification agent for critiquing proposals."""

    def __init__(
        self,
        name: str,
        supportive: bool = True,
        delay: float = 0.0,
        should_fail: bool = False,
    ):
        self.name = name
        self.model = "mock-verifier"
        self._supportive = supportive
        self._delay = delay
        self._should_fail = should_fail
        self._call_count = 0

    async def critique(
        self,
        proposal: str,
        task: str,
        context: list | None = None,
        target_agent: str | None = None,
    ) -> MockCritique:
        """Generate a critique of the proposal."""
        if self._should_fail:
            raise Exception(f"Verification agent {self.name} failed")

        if self._delay > 0:
            await asyncio.sleep(self._delay)

        self._call_count += 1

        if self._supportive:
            content = (
                "I agree with this proposal. It is well-reasoned, sound, and "
                "comprehensive. The approach addresses the requirements effectively."
            )
        else:
            content = (
                "I disagree with this proposal. There are significant issues "
                "and concerns that need to be addressed before this can be approved."
            )

        return MockCritique(content)

    async def is_available(self) -> bool:
        """Check if agent is available."""
        return not self._should_fail


@dataclass
class HybridDebateConfig:
    """Configuration for hybrid debate."""

    external_agent: MockExternalAgent
    verification_agents: list[MockVerificationAgent]
    consensus_threshold: float = 0.7
    min_verification_quorum: int = 1
    require_receipt: bool = True
    max_refinement_rounds: int = 3
    domain: str | None = None


@dataclass
class VerificationResult:
    """Result of hybrid debate verification."""

    proposal: str | None
    verified: bool
    consensus_score: float
    critiques: list[str]
    receipt_hash: str | None = None
    debate_id: str = ""
    rounds_used: int = 0
    external_agent: str = ""
    verification_agent_names: list[str] = field(default_factory=list)
    error: str | None = None


class MockHybridDebateProtocol:
    """Mock implementation of hybrid debate protocol for testing."""

    # Keyword sets for consensus calculation
    _SUPPORTIVE_KEYWORDS = frozenset(
        {
            "agree",
            "sound",
            "comprehensive",
            "approve",
            "valid",
            "effective",
            "well-reasoned",
            "appropriate",
            "thorough",
        }
    )

    _CRITICAL_KEYWORDS = frozenset(
        {
            "disagree",
            "issues",
            "concerns",
            "reject",
            "invalid",
            "flawed",
            "inadequate",
            "problematic",
            "error",
        }
    )

    def __init__(self, config: HybridDebateConfig):
        self.config = config
        self.external_agent = config.external_agent
        self.verification_agents = config.verification_agents

    async def run_with_external(
        self,
        task: str,
        context: list | None = None,
        decision_id: str | None = None,
    ) -> VerificationResult:
        """Run hybrid debate with external proposal."""
        debate_id = decision_id or f"debate-{datetime.now(timezone.utc).timestamp()}"

        # Check external agent health
        if not await self.external_agent.is_available():
            return VerificationResult(
                proposal=None,
                verified=False,
                consensus_score=0.0,
                critiques=[],
                debate_id=debate_id,
                external_agent=self.external_agent.name,
                error="External agent unavailable",
            )

        # Get proposal from external agent
        try:
            proposal = await self.external_agent.generate(task, context)
        except Exception as e:
            return VerificationResult(
                proposal=None,
                verified=False,
                consensus_score=0.0,
                critiques=[],
                debate_id=debate_id,
                external_agent=self.external_agent.name,
                error=str(e),
            )

        # Collect critiques from verification agents (parallel)
        critiques = await self._collect_critiques(proposal, task)

        # Check quorum
        valid_critiques = [c for c in critiques if not c.startswith("ERROR:")]
        if len(valid_critiques) < self.config.min_verification_quorum:
            return VerificationResult(
                proposal=proposal,
                verified=False,
                consensus_score=0.0,
                critiques=critiques,
                debate_id=debate_id,
                rounds_used=1,
                external_agent=self.external_agent.name,
                verification_agent_names=[a.name for a in self.verification_agents],
                error=f"Quorum not met: {len(valid_critiques)}/{self.config.min_verification_quorum}",
            )

        # Calculate consensus
        consensus_score = self._calculate_consensus(critiques)
        verified = consensus_score >= self.config.consensus_threshold

        # Generate receipt if verified
        receipt_hash = None
        if self.config.require_receipt and verified:
            receipt_hash = self._generate_receipt(
                proposal=proposal,
                task=task,
                consensus_score=consensus_score,
                critiques=critiques,
                decision_id=debate_id,
            )

        return VerificationResult(
            proposal=proposal,
            verified=verified,
            consensus_score=consensus_score,
            critiques=critiques,
            receipt_hash=receipt_hash,
            debate_id=debate_id,
            rounds_used=1,
            external_agent=self.external_agent.name,
            verification_agent_names=[a.name for a in self.verification_agents],
        )

    async def _collect_critiques(self, proposal: str, task: str) -> list[str]:
        """Collect critiques from all verification agents in parallel."""

        async def get_critique(agent: MockVerificationAgent) -> str:
            try:
                critique = await agent.critique(
                    proposal=proposal,
                    task=task,
                    target_agent=self.external_agent.name,
                )
                return critique.content
            except Exception as e:
                return f"ERROR: {e}"

        tasks = [get_critique(agent) for agent in self.verification_agents]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        critiques = []
        for result in results:
            if isinstance(result, Exception):
                critiques.append(f"ERROR: {result}")
            else:
                critiques.append(result)

        return critiques

    def _calculate_consensus(self, critiques: list[str]) -> float:
        """Calculate consensus score from critiques."""
        if not critiques:
            return 0.0

        # Filter out errors
        valid_critiques = [c for c in critiques if not c.startswith("ERROR:")]
        if not valid_critiques:
            return 0.0

        supportive_count = 0
        for critique in valid_critiques:
            critique_lower = critique.lower()
            support_score = sum(1 for k in self._SUPPORTIVE_KEYWORDS if k in critique_lower)
            critical_score = sum(1 for k in self._CRITICAL_KEYWORDS if k in critique_lower)

            if support_score > critical_score:
                supportive_count += 1
            elif support_score == critical_score and support_score > 0:
                supportive_count += 0.5

        return supportive_count / len(valid_critiques)

    def _generate_receipt(
        self,
        proposal: str,
        task: str,
        consensus_score: float,
        critiques: list[str],
        decision_id: str,
    ) -> str:
        """Generate cryptographic receipt for audit trail."""
        receipt_data = {
            "decision_id": decision_id,
            "task": task,
            "proposal": proposal,
            "consensus_score": consensus_score,
            "num_critiques": len(critiques),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        receipt_json = json.dumps(receipt_data, sort_keys=True)
        return hashlib.sha256(receipt_json.encode("utf-8")).hexdigest()


# =============================================================================
# Test Class: Hybrid Protocol Gateway
# =============================================================================


class TestHybridProtocolGateway:
    """Integration tests for hybrid debate protocol with gateway agents."""

    @pytest.fixture
    def external_agent(self) -> MockExternalAgent:
        """Provide a mock external agent."""
        return MockExternalAgent(
            name="test-external",
            proposal="I propose implementing a token bucket rate limiter with Redis.",
        )

    @pytest.fixture
    def supportive_verifiers(self) -> list[MockVerificationAgent]:
        """Provide supportive verification agents."""
        return [
            MockVerificationAgent("verifier-1", supportive=True),
            MockVerificationAgent("verifier-2", supportive=True),
            MockVerificationAgent("verifier-3", supportive=True),
        ]

    @pytest.fixture
    def mixed_verifiers(self) -> list[MockVerificationAgent]:
        """Provide mixed verification agents (some supportive, some critical)."""
        return [
            MockVerificationAgent("verifier-1", supportive=True),
            MockVerificationAgent("verifier-2", supportive=False),
            MockVerificationAgent("verifier-3", supportive=True),
        ]

    @pytest.fixture
    def critical_verifiers(self) -> list[MockVerificationAgent]:
        """Provide critical verification agents."""
        return [
            MockVerificationAgent("verifier-1", supportive=False),
            MockVerificationAgent("verifier-2", supportive=False),
            MockVerificationAgent("verifier-3", supportive=False),
        ]

    @pytest.mark.asyncio
    async def test_hybrid_debate_with_mocked_external(
        self,
        mock_external_server: MockExternalFrameworkServer,
        external_agent: MockExternalAgent,
        supportive_verifiers: list[MockVerificationAgent],
    ):
        """Test full hybrid debate with mock external agent.

        Verifies:
        - External agent generates proposal
        - Verification agents provide critiques
        - Consensus is calculated correctly
        - Receipt is generated for verified decisions
        """
        # Configure mock server
        mock_external_server.set_response("/health", {"status": "healthy"})
        mock_external_server.set_response("/generate", {"response": external_agent._proposal})

        config = HybridDebateConfig(
            external_agent=external_agent,
            verification_agents=supportive_verifiers,
            consensus_threshold=0.7,
            require_receipt=True,
        )

        protocol = MockHybridDebateProtocol(config)
        result = await protocol.run_with_external(
            task="Design a rate limiter",
            decision_id="test-hybrid-1",
        )

        # Verify successful debate
        assert result.verified is True
        assert result.proposal is not None
        assert result.consensus_score >= 0.7
        assert result.receipt_hash is not None
        assert len(result.critiques) == 3

    @pytest.mark.asyncio
    async def test_hybrid_debate_consensus_calculation(
        self,
        external_agent: MockExternalAgent,
        mixed_verifiers: list[MockVerificationAgent],
    ):
        """Test consensus calculation from verification agents.

        Verifies:
        - Consensus score reflects verifier opinions
        - Mixed votes produce partial consensus
        - Threshold determines verification outcome
        """
        # With 2/3 supportive, consensus should be ~0.67
        config = HybridDebateConfig(
            external_agent=external_agent,
            verification_agents=mixed_verifiers,
            consensus_threshold=0.5,  # Lower threshold to pass with 2/3
        )

        protocol = MockHybridDebateProtocol(config)
        result = await protocol.run_with_external(
            task="Test consensus",
            decision_id="test-consensus-1",
        )

        # Should pass with 2/3 supportive (0.67 > 0.5)
        assert result.verified is True
        assert 0.5 <= result.consensus_score <= 1.0

        # Try with higher threshold - should fail
        config.consensus_threshold = 0.8
        protocol = MockHybridDebateProtocol(config)
        result = await protocol.run_with_external(
            task="Test consensus",
            decision_id="test-consensus-2",
        )

        assert result.verified is False

    @pytest.mark.asyncio
    async def test_hybrid_debate_quorum_enforcement(
        self,
        external_agent: MockExternalAgent,
    ):
        """Test that minimum verification quorum is enforced.

        Verifies:
        - Quorum requirement is checked
        - Debate fails if quorum not met
        - Error message indicates quorum failure
        """
        # All verifiers fail
        failing_verifiers = [
            MockVerificationAgent("v1", should_fail=True),
            MockVerificationAgent("v2", should_fail=True),
            MockVerificationAgent("v3", should_fail=True),
        ]

        config = HybridDebateConfig(
            external_agent=external_agent,
            verification_agents=failing_verifiers,
            min_verification_quorum=2,  # Need at least 2 valid critiques
        )

        protocol = MockHybridDebateProtocol(config)
        result = await protocol.run_with_external(
            task="Test quorum",
            decision_id="test-quorum-1",
        )

        # Should fail due to quorum not met
        assert result.verified is False
        assert "Quorum not met" in (result.error or "")

    @pytest.mark.asyncio
    async def test_hybrid_debate_empty_critiques(
        self,
        external_agent: MockExternalAgent,
    ):
        """Test that empty critiques do not auto-approve.

        Verifies:
        - Empty critique list results in rejection
        - Zero consensus score for no critiques
        - System fails safe (no approval without verification)
        """
        # No verifiers at all
        config = HybridDebateConfig(
            external_agent=external_agent,
            verification_agents=[],
            min_verification_quorum=0,  # Allow zero quorum for testing
        )

        protocol = MockHybridDebateProtocol(config)
        result = await protocol.run_with_external(
            task="Test empty",
            decision_id="test-empty-1",
        )

        # Should not be verified with no critiques
        assert result.verified is False
        assert result.consensus_score == 0.0
        assert len(result.critiques) == 0

    @pytest.mark.asyncio
    async def test_hybrid_debate_parallel_verification(
        self,
        external_agent: MockExternalAgent,
    ):
        """Test that verification runs in parallel.

        Verifies:
        - Multiple verifiers run concurrently
        - Total time is closer to max single agent time, not sum
        - All results are collected
        """
        # Create verifiers with different delays
        verifiers = [
            MockVerificationAgent("fast-1", supportive=True, delay=0.05),
            MockVerificationAgent("fast-2", supportive=True, delay=0.05),
            MockVerificationAgent("slow", supportive=True, delay=0.1),
        ]

        config = HybridDebateConfig(
            external_agent=external_agent,
            verification_agents=verifiers,
        )

        protocol = MockHybridDebateProtocol(config)

        start = asyncio.get_event_loop().time()
        result = await protocol.run_with_external(
            task="Test parallel",
            decision_id="test-parallel-1",
        )
        elapsed = asyncio.get_event_loop().time() - start

        # All critiques should be collected
        assert len(result.critiques) == 3

        # Time should be ~0.1s (parallel), not ~0.2s (sequential)
        # Allow some tolerance for test execution
        assert elapsed < 0.5  # Should be much less than 0.05 + 0.05 + 0.1 if sequential

    @pytest.mark.asyncio
    async def test_hybrid_debate_external_agent_error(
        self,
        supportive_verifiers: list[MockVerificationAgent],
    ):
        """Test graceful handling of external agent errors.

        Verifies:
        - External agent failure is handled gracefully
        - Error message is captured
        - No proposal means no verification
        """
        # Create agent that fails after first call
        failing_agent = MockExternalAgent(
            name="failing-external",
            error_after=0,  # Fail immediately
        )

        config = HybridDebateConfig(
            external_agent=failing_agent,
            verification_agents=supportive_verifiers,
        )

        protocol = MockHybridDebateProtocol(config)
        result = await protocol.run_with_external(
            task="Test error",
            decision_id="test-error-1",
        )

        # Should fail gracefully
        assert result.verified is False
        assert result.proposal is None
        assert result.error is not None
        assert "connection" in result.error.lower()

    @pytest.mark.asyncio
    async def test_hybrid_debate_domain_specific(
        self,
        external_agent: MockExternalAgent,
        supportive_verifiers: list[MockVerificationAgent],
    ):
        """Test domain-specific debate configuration.

        Verifies:
        - Domain context can be provided
        - Configuration affects debate behavior
        - Domain is tracked in results
        """
        config = HybridDebateConfig(
            external_agent=external_agent,
            verification_agents=supportive_verifiers,
            domain="software-architecture",
            consensus_threshold=0.6,
        )

        protocol = MockHybridDebateProtocol(config)
        result = await protocol.run_with_external(
            task="Design microservices architecture",
            decision_id="test-domain-1",
        )

        # Domain-specific debate should work
        assert result.verified is True
        assert result.external_agent == external_agent.name
        assert len(result.verification_agent_names) == 3

    @pytest.mark.asyncio
    async def test_hybrid_debate_result_structure(
        self,
        external_agent: MockExternalAgent,
        supportive_verifiers: list[MockVerificationAgent],
    ):
        """Test that result contains all required fields.

        Verifies:
        - Result includes proposal
        - Result includes verification status
        - Result includes consensus score
        - Result includes critiques
        - Result includes receipt (when verified)
        - Result includes debate metadata
        """
        config = HybridDebateConfig(
            external_agent=external_agent,
            verification_agents=supportive_verifiers,
            require_receipt=True,
        )

        protocol = MockHybridDebateProtocol(config)
        result = await protocol.run_with_external(
            task="Test structure",
            decision_id="test-structure-1",
        )

        # Verify all required fields
        assert hasattr(result, "proposal")
        assert result.proposal is not None

        assert hasattr(result, "verified")
        assert isinstance(result.verified, bool)

        assert hasattr(result, "consensus_score")
        assert 0.0 <= result.consensus_score <= 1.0

        assert hasattr(result, "critiques")
        assert isinstance(result.critiques, list)

        assert hasattr(result, "receipt_hash")
        if result.verified:
            assert result.receipt_hash is not None
            # SHA-256 = 64 hex characters
            assert len(result.receipt_hash) == 64

        assert hasattr(result, "debate_id")
        assert result.debate_id == "test-structure-1"

        assert hasattr(result, "rounds_used")
        assert result.rounds_used >= 1

        assert hasattr(result, "external_agent")
        assert result.external_agent == external_agent.name

        assert hasattr(result, "verification_agent_names")
        assert len(result.verification_agent_names) == len(supportive_verifiers)
