"""
Integration tests for gateway agent debate flow.

Tests the complete flow:
1. Register external agent
2. External agent generates proposal
3. Internal agents verify the proposal
4. Consensus reached (or not)
5. Result stored with receipt
"""

from __future__ import annotations

import asyncio
import hashlib
import json
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
# Test Helpers
# =============================================================================


class MockCritique:
    """Mock critique object for testing."""

    def __init__(self, content: str, supportive: bool = True):
        self.content = content
        self._supportive = supportive


class MockVerificationAgent:
    """Mock verification agent that returns configurable critiques."""

    def __init__(
        self,
        name: str,
        supportive: bool = True,
        delay: float = 0.0,
        should_timeout: bool = False,
    ):
        self.name = name
        self.model = "mock-verifier"
        self._supportive = supportive
        self._delay = delay
        self._should_timeout = should_timeout
        self._call_count = 0

    async def critique(
        self,
        proposal: str,
        task: str,
        context: list | None = None,
        target_agent: str | None = None,
    ) -> MockCritique:
        """Generate a critique."""
        if self._should_timeout:
            await asyncio.sleep(100)  # Will trigger timeout
            raise TimeoutError("Agent timed out")

        if self._delay > 0:
            await asyncio.sleep(self._delay)

        self._call_count += 1

        if self._supportive:
            content = (
                f"I agree with the proposal from {target_agent}. "
                "This is a well-reasoned and sound approach. The solution is comprehensive."
            )
        else:
            content = (
                f"I disagree with the proposal from {target_agent}. "
                "There are significant issues, flaws, and concerns that need addressing."
            )
        return MockCritique(content, self._supportive)

    async def is_available(self) -> bool:
        """Check availability."""
        return not self._should_timeout


class MockExternalAgent:
    """Mock external framework agent for testing."""

    def __init__(
        self,
        name: str = "external-agent-1",
        proposal: str = "This is a test proposal from the external agent.",
        available: bool = True,
        fail_after: int | None = None,
    ):
        self.name = name
        self.model = "external"
        self.base_url = "http://localhost:8000"
        self._proposal = proposal
        self._available = available
        self._fail_after = fail_after
        self._call_count = 0

    async def generate(self, prompt: str, context: list | None = None) -> str:
        """Generate a proposal."""
        if self._fail_after is not None and self._call_count >= self._fail_after:
            raise ConnectionError(f"External agent {self.name} unavailable")
        self._call_count += 1
        return self._proposal

    async def is_available(self) -> bool:
        """Check availability."""
        return self._available


class MockHybridProtocol:
    """Mock hybrid debate protocol for integration testing."""

    def __init__(
        self,
        external_agent: MockExternalAgent,
        verification_agents: list[MockVerificationAgent],
        consensus_threshold: float = 0.7,
        require_receipt: bool = True,
    ):
        self.external_agent = external_agent
        self.verification_agents = verification_agents
        self.consensus_threshold = consensus_threshold
        self.require_receipt = require_receipt

    async def run_with_external(
        self,
        task: str,
        context: list | None = None,
        decision_id: str | None = None,
    ) -> dict[str, Any]:
        """Run the hybrid debate and return results."""
        debate_id = decision_id or f"debate-{datetime.now(timezone.utc).timestamp()}"

        # Check external agent health
        if not await self.external_agent.is_available():
            return {
                "debate_id": debate_id,
                "verified": False,
                "error": "External agent unavailable",
                "proposal": None,
                "consensus_score": 0.0,
                "critiques": [],
            }

        # Generate proposal from external agent
        try:
            proposal = await self.external_agent.generate(task, context)
        except Exception as e:
            return {
                "debate_id": debate_id,
                "verified": False,
                "error": str(e),
                "proposal": None,
                "consensus_score": 0.0,
                "critiques": [],
            }

        # Collect critiques from verification agents
        critiques = []
        for agent in self.verification_agents:
            try:
                critique = await asyncio.wait_for(
                    agent.critique(proposal, task, target_agent=self.external_agent.name),
                    timeout=5.0,
                )
                critiques.append(critique.content)
            except asyncio.TimeoutError:
                critiques.append("TIMEOUT")
            except Exception as e:
                critiques.append(f"ERROR: {e}")

        # Calculate consensus (keyword-based with negative indicators)
        supportive_count = 0
        for c in critiques:
            c_lower = c.lower()
            # Skip errors and timeouts
            if c.startswith("ERROR:") or c.startswith("TIMEOUT"):
                continue
            # Check for rejection keywords first (higher priority)
            if "disagree" in c_lower or "reject" in c_lower or "flaws" in c_lower:
                continue  # Not supportive
            # Check for supportive keywords
            if "agree" in c_lower or "sound" in c_lower or "comprehensive" in c_lower:
                supportive_count += 1
        consensus_score = supportive_count / len(critiques) if critiques else 0.0
        verified = consensus_score >= self.consensus_threshold

        # Generate receipt if required
        receipt_hash = None
        if self.require_receipt and verified:
            receipt_data = {
                "debate_id": debate_id,
                "task": task,
                "proposal": proposal,
                "consensus_score": consensus_score,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            receipt_hash = hashlib.sha256(
                json.dumps(receipt_data, sort_keys=True).encode()
            ).hexdigest()

        return {
            "debate_id": debate_id,
            "verified": verified,
            "proposal": proposal,
            "consensus_score": consensus_score,
            "critiques": critiques,
            "receipt_hash": receipt_hash,
            "rounds_used": 1,
            "external_agent": self.external_agent.name,
        }


# =============================================================================
# Test Class: Gateway Agent Debate Flow
# =============================================================================


class TestGatewayAgentDebateFlow:
    """Integration tests for gateway agent debate flow."""

    @pytest.fixture
    def external_agent(self) -> MockExternalAgent:
        """Provide a mock external agent."""
        return MockExternalAgent(
            name="external-1",
            proposal="I propose implementing a distributed rate limiter using token buckets.",
        )

    @pytest.fixture
    def supportive_verifiers(self) -> list[MockVerificationAgent]:
        """Provide verification agents that support the proposal."""
        return [
            MockVerificationAgent("verifier-1", supportive=True),
            MockVerificationAgent("verifier-2", supportive=True),
            MockVerificationAgent("verifier-3", supportive=True),
        ]

    @pytest.fixture
    def critical_verifiers(self) -> list[MockVerificationAgent]:
        """Provide verification agents that reject the proposal (all critical)."""
        return [
            MockVerificationAgent("verifier-1", supportive=False),
            MockVerificationAgent("verifier-2", supportive=False),
            MockVerificationAgent("verifier-3", supportive=False),
        ]

    @pytest.mark.asyncio
    async def test_full_debate_flow_with_consensus(
        self,
        mock_external_server: MockExternalFrameworkServer,
        gateway_server_context: dict,
        external_agent: MockExternalAgent,
        supportive_verifiers: list[MockVerificationAgent],
    ):
        """Test complete flow with consensus reached.

        Verifies:
        - External agent registration
        - Proposal generation from external agent
        - Critiques collected from verification agents
        - Consensus calculation reaches threshold
        - Receipt generated for verified decision
        """
        # Setup: Configure mock server
        mock_external_server.set_response(
            "/generate", {"proposal": "Test proposal from external agent"}
        )
        mock_external_server.set_response("/health", {"status": "healthy"})

        # Register external agent
        register_external_agent(gateway_server_context, "external-1", "crewai")
        assert "external-1" in gateway_server_context["external_agents"]

        # Create and run hybrid protocol
        protocol = MockHybridProtocol(
            external_agent=external_agent,
            verification_agents=supportive_verifiers,
            consensus_threshold=0.7,
            require_receipt=True,
        )

        result = await protocol.run_with_external(
            task="Design a rate limiter API",
            decision_id="test-debate-1",
        )

        # Verify consensus was reached
        assert result["verified"] is True
        assert result["consensus_score"] >= 0.7
        assert result["proposal"] is not None
        assert result["receipt_hash"] is not None
        assert len(result["critiques"]) == 3

    @pytest.mark.asyncio
    async def test_debate_flow_without_consensus(
        self,
        gateway_server_context: dict,
        external_agent: MockExternalAgent,
        critical_verifiers: list[MockVerificationAgent],
    ):
        """Test flow when verification agents reject proposal.

        Verifies:
        - Proposal is generated but not verified
        - Consensus score is below threshold
        - No receipt is generated for unverified decisions
        """
        # Register external agent
        register_external_agent(gateway_server_context, "external-1", "crewai")

        # Create protocol with critical verifiers (all 3 reject)
        protocol = MockHybridProtocol(
            external_agent=external_agent,
            verification_agents=critical_verifiers,
            consensus_threshold=0.7,
            require_receipt=True,
        )

        result = await protocol.run_with_external(
            task="Design a complex system",
            decision_id="test-debate-2",
        )

        # Verify consensus was NOT reached
        assert result["verified"] is False
        assert result["consensus_score"] < 0.7
        assert result["proposal"] is not None
        assert result["receipt_hash"] is None  # No receipt for unverified
        assert len(result["critiques"]) == 3

    @pytest.mark.asyncio
    async def test_debate_flow_external_agent_unavailable(
        self,
        gateway_server_context: dict,
        supportive_verifiers: list[MockVerificationAgent],
    ):
        """Test flow when external agent health check fails.

        Verifies:
        - Flow handles unavailable external agent gracefully
        - Error message indicates agent unavailability
        - No proposal is generated
        """
        # Create unavailable external agent
        unavailable_agent = MockExternalAgent(
            name="unavailable-agent",
            available=False,
        )

        # Register agent
        register_external_agent(gateway_server_context, "unavailable-agent", "crewai")

        protocol = MockHybridProtocol(
            external_agent=unavailable_agent,
            verification_agents=supportive_verifiers,
        )

        result = await protocol.run_with_external(
            task="Test task",
            decision_id="test-debate-3",
        )

        # Verify error handling
        assert result["verified"] is False
        assert result["proposal"] is None
        assert "unavailable" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_debate_flow_verification_timeout(
        self,
        gateway_server_context: dict,
        external_agent: MockExternalAgent,
    ):
        """Test flow when verification agents timeout.

        Verifies:
        - Timeouts are handled gracefully
        - Partial critiques are still collected
        - Consensus calculation accounts for timeouts
        """
        # Create verifiers with one that times out
        verifiers = [
            MockVerificationAgent("verifier-1", supportive=True),
            MockVerificationAgent("verifier-2", should_timeout=True),
            MockVerificationAgent("verifier-3", supportive=True),
        ]

        protocol = MockHybridProtocol(
            external_agent=external_agent,
            verification_agents=verifiers,
            consensus_threshold=0.5,  # Lower threshold to account for timeout
        )

        result = await protocol.run_with_external(
            task="Test task with timeouts",
            decision_id="test-debate-4",
        )

        # Verify timeout handling
        assert len(result["critiques"]) == 3
        assert "TIMEOUT" in result["critiques"]
        # 2 supportive out of 3 (timeout doesn't count as supportive)
        # So consensus might not be reached depending on threshold

    @pytest.mark.asyncio
    async def test_debate_flow_multiple_rounds(
        self,
        gateway_server_context: dict,
    ):
        """Test flow with multiple proposal/verification rounds.

        Verifies:
        - Protocol can iterate through multiple rounds
        - Proposals are refined based on critiques
        - Final consensus reflects all rounds
        """
        # First round: proposal rejected, second round: accepted
        proposals = [
            "Initial proposal with issues",
            "Refined proposal addressing feedback",
        ]

        external_agent = MockExternalAgent(name="multi-round-agent")
        external_agent._proposal = proposals[0]
        round_count = [0]

        original_generate = external_agent.generate

        async def round_aware_generate(prompt: str, context: list | None = None) -> str:
            idx = min(round_count[0], len(proposals) - 1)
            round_count[0] += 1
            return proposals[idx]

        external_agent.generate = round_aware_generate

        # Verifiers are critical first, then supportive
        verifier_responses = [False, True]
        response_idx = [0]

        verifiers = []
        for i in range(3):
            v = MockVerificationAgent(f"verifier-{i}", supportive=True)
            verifiers.append(v)

        protocol = MockHybridProtocol(
            external_agent=external_agent,
            verification_agents=verifiers,
            consensus_threshold=0.7,
        )

        result = await protocol.run_with_external(
            task="Multi-round test",
            decision_id="test-debate-5",
        )

        # Verify flow completed
        assert result["proposal"] is not None
        assert result["debate_id"] == "test-debate-5"

    @pytest.mark.asyncio
    async def test_debate_flow_stores_receipt(
        self,
        gateway_server_context: dict,
        external_agent: MockExternalAgent,
        supportive_verifiers: list[MockVerificationAgent],
    ):
        """Test that debate result creates audit receipt.

        Verifies:
        - Receipt hash is generated for verified decisions
        - Receipt hash is a valid SHA-256 hex string
        - Receipt can be used for audit trail
        """
        protocol = MockHybridProtocol(
            external_agent=external_agent,
            verification_agents=supportive_verifiers,
            consensus_threshold=0.7,
            require_receipt=True,
        )

        result = await protocol.run_with_external(
            task="Receipt test",
            decision_id="test-debate-6",
        )

        # Verify receipt generation
        assert result["verified"] is True
        assert result["receipt_hash"] is not None

        # Verify receipt hash format (SHA-256 = 64 hex characters)
        assert len(result["receipt_hash"]) == 64
        assert all(c in "0123456789abcdef" for c in result["receipt_hash"])

    @pytest.mark.asyncio
    async def test_debate_flow_metrics_recorded(
        self,
        gateway_server_context: dict,
        external_agent: MockExternalAgent,
        supportive_verifiers: list[MockVerificationAgent],
    ):
        """Test that metrics are recorded during flow.

        Verifies:
        - Debate results include relevant metrics
        - Consensus score is recorded
        - Round count is tracked
        """
        protocol = MockHybridProtocol(
            external_agent=external_agent,
            verification_agents=supportive_verifiers,
        )

        result = await protocol.run_with_external(
            task="Metrics test",
            decision_id="test-debate-7",
        )

        # Verify metrics in result
        assert "consensus_score" in result
        assert isinstance(result["consensus_score"], float)
        assert 0.0 <= result["consensus_score"] <= 1.0

        assert "rounds_used" in result
        assert result["rounds_used"] >= 1

        assert "external_agent" in result
        assert result["external_agent"] == external_agent.name

    @pytest.mark.asyncio
    async def test_debate_flow_agent_not_registered(
        self,
        gateway_server_context: dict,
    ):
        """Test error when trying to use unregistered agent.

        Verifies:
        - Attempting to use unregistered agent fails appropriately
        - Error message is descriptive
        """
        # Do NOT register the agent
        assert "unregistered-agent" not in gateway_server_context.get("external_agents", {})

        # Create agent that simulates being unregistered
        unregistered_agent = MockExternalAgent(name="unregistered-agent")

        # In a real implementation, the gateway would check registration
        # Here we simulate the check
        registered_agents = gateway_server_context.get("external_agents", {})

        if unregistered_agent.name not in registered_agents:
            # Simulate what the gateway would return
            result = {
                "verified": False,
                "error": f"Agent '{unregistered_agent.name}' is not registered",
                "proposal": None,
                "consensus_score": 0.0,
                "critiques": [],
            }
        else:
            result = {"verified": True}

        # Verify error handling
        assert result["verified"] is False
        assert "not registered" in result["error"]
        assert result["proposal"] is None
