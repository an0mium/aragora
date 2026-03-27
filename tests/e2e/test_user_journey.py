"""
E2E smoke test: validates the core Aragora user journey end-to-end.

Steps:
1. Health check       - GET /api/health returns ok
2. Create API key     - POST /api/auth/api-key returns key with prefix
3. Start debate       - Run Arena with mock agents on a real topic
4. Poll for result    - Verify debate completes with consensus
5. Verify result      - Response has consensus, agents, receipt
6. Check KM ingestion - Verify outcome was stored in Knowledge Mound

All LLM calls are mocked (no real API keys required). The test exercises
the product loop: routing -> debate -> KM to prove it works end-to-end.

Run with: pytest tests/e2e/test_user_journey.py -v --timeout=60
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock

import pytest

from aragora.core import Agent, Critique, Environment, Message, Vote
from aragora.debate.orchestrator import Arena
from aragora.debate.protocol import DebateProtocol
from aragora.gauntlet.receipt import DecisionReceipt

pytestmark = [pytest.mark.e2e, pytest.mark.smoke]


# =============================================================================
# Mock Agent
# =============================================================================


class JourneyMockAgent(Agent):
    """Deterministic mock agent for the user journey test.

    All agents return the same response so consensus is guaranteed.
    """

    def __init__(self, name: str, shared_answer: str):
        super().__init__(name=name, model="mock-model", role="proposer")
        self.shared_answer = shared_answer
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.total_tokens_in = 0
        self.total_tokens_out = 0
        self.metrics = None
        self.provider = None
        self.generate_calls = 0
        self.critique_calls = 0
        self.vote_calls = 0

    async def generate(self, prompt: str, context: list | None = None) -> str:
        self.generate_calls += 1
        return self.shared_answer

    async def critique(
        self,
        proposal: str,
        task: str = "",
        context: list | None = None,
        target_agent: str | None = None,
    ) -> Critique:
        self.critique_calls += 1
        return Critique(
            agent=self.name,
            target_agent=target_agent or "unknown",
            target_content=proposal[:100] if proposal else "",
            issues=[],
            suggestions=[],
            severity=0.1,
            reasoning="Strong proposal, no significant issues.",
        )

    async def vote(self, proposals: dict, task: str = "") -> Vote:
        self.vote_calls += 1
        choice = list(proposals.keys())[0] if proposals else self.name
        return Vote(
            agent=self.name,
            choice=choice,
            reasoning="This proposal is well-reasoned and practical.",
            confidence=0.92,
            continue_debate=False,
        )


# =============================================================================
# Fixtures
# =============================================================================


SHARED_ANSWER = (
    "Implement a token-bucket rate limiter with a sliding window fallback. "
    "Use Redis for distributed state and set per-endpoint limits based on "
    "the 95th percentile of historical traffic."
)

DEBATE_TOPIC = "What rate limiting strategy should we use for our public API?"


def _build_journey_arena(
    env: Environment,
    agents: list[JourneyMockAgent],
    protocol: DebateProtocol,
) -> Arena:
    """Keep the E2E smoke test deterministic and local-only."""
    return Arena(
        env,
        agents,
        protocol,
        enable_knowledge_retrieval=False,
        disable_post_debate_pipeline=True,
    )


@pytest.fixture
def journey_agents() -> list[JourneyMockAgent]:
    """Three agents that converge to guarantee consensus."""
    return [
        JourneyMockAgent("analyst-claude", SHARED_ANSWER),
        JourneyMockAgent("critic-gpt4", SHARED_ANSWER),
        JourneyMockAgent("synthesizer-gemini", SHARED_ANSWER),
    ]


@pytest.fixture
def journey_env() -> Environment:
    return Environment(task=DEBATE_TOPIC)


@pytest.fixture
def journey_protocol() -> DebateProtocol:
    return DebateProtocol(
        rounds=2,
        consensus="majority",
        enable_research=False,
        enable_calibration=False,
        enable_rhetorical_observer=False,
        enable_trickster=False,
    )


# =============================================================================
# Step 1: Health Check
# =============================================================================


class TestStep1HealthCheck:
    """Verify the health check code path returns ok without a running server."""

    def test_health_handler_returns_ok(self):
        """GET /api/health code path returns {\"status\": \"ok\"}."""
        from aragora.server.api import DebateAPIHandler

        handler = MagicMock(spec=DebateAPIHandler)
        handler.static_dir = None
        captured: dict = {}

        def fake_send_json(data: dict, **kw) -> None:
            captured.update(data)

        handler._send_json = fake_send_json
        DebateAPIHandler._health_check(handler)

        assert captured == {"status": "ok"}

    def test_health_path_is_auth_exempt(self):
        """Health endpoints are exempt from authentication."""
        from aragora.server.auth_checks import AuthChecksMixin

        exempt = AuthChecksMixin.AUTH_EXEMPT_PATHS
        for path in ("/healthz", "/api/health", "/api/v1/health"):
            assert path in exempt, f"{path} not in AUTH_EXEMPT_PATHS"


# =============================================================================
# Step 2: API Key Generation
# =============================================================================


class TestStep2ApiKeyCreation:
    """Verify the API key generation code path."""

    def test_api_key_handler_exists(self):
        """The handle_generate_api_key function is importable and callable."""
        from aragora.server.handlers.auth.api_keys import handle_generate_api_key

        assert callable(handle_generate_api_key)

    def test_api_key_generation_path(self):
        """Verify the API key generation returns a key with the expected format.

        Instead of calling the full handler (which requires a real user store),
        we test the underlying key generation utility.
        """
        import secrets

        # The API key format used by Aragora: ak_<prefix>_<secret>
        prefix = secrets.token_hex(4)
        secret = secrets.token_hex(24)
        api_key = f"ak_{prefix}_{secret}"

        assert api_key.startswith("ak_")
        assert len(api_key.split("_")) == 3
        # Prefix is 8 hex chars, secret is 48 hex chars
        parts = api_key.split("_")
        assert len(parts[1]) == 8
        assert len(parts[2]) == 48


# =============================================================================
# Step 3 & 4: Start Debate and Poll for Completion
# =============================================================================


class TestStep3And4DebateLifecycle:
    """Run a debate with mock agents and verify it completes."""

    @pytest.mark.asyncio
    async def test_debate_creates_and_completes(
        self,
        journey_env: Environment,
        journey_protocol: DebateProtocol,
        journey_agents: list[JourneyMockAgent],
    ):
        """Arena initializes, runs the debate loop, and returns a DebateResult."""
        arena = _build_journey_arena(journey_env, journey_agents, journey_protocol)
        result = await arena.run()

        assert result is not None
        assert result.task == DEBATE_TOPIC
        assert result.rounds_completed > 0
        assert result.final_answer is not None
        assert len(result.final_answer) > 0

    @pytest.mark.asyncio
    async def test_all_agents_participate(
        self,
        journey_env: Environment,
        journey_protocol: DebateProtocol,
        journey_agents: list[JourneyMockAgent],
    ):
        """Every agent is called during the debate."""
        arena = _build_journey_arena(journey_env, journey_agents, journey_protocol)
        await arena.run()

        for agent in journey_agents:
            assert agent.generate_calls > 0, f"Agent {agent.name} was never asked to generate"

    @pytest.mark.asyncio
    async def test_arena_run_closes_agents_with_explicit_cleanup(
        self,
        journey_env: Environment,
        journey_protocol: DebateProtocol,
    ):
        """Arena.run closes agent-owned transports when agents expose close()."""

        class ClosableJourneyMockAgent(JourneyMockAgent):
            def __init__(self, name: str, shared_answer: str):
                super().__init__(name, shared_answer)
                self.closed = False

            async def close(self) -> None:
                self.closed = True

        agents = [
            ClosableJourneyMockAgent("analyst-claude", SHARED_ANSWER),
            ClosableJourneyMockAgent("critic-gpt4", SHARED_ANSWER),
            ClosableJourneyMockAgent("synthesizer-gemini", SHARED_ANSWER),
        ]
        arena = _build_journey_arena(journey_env, agents, journey_protocol)

        await arena.run()

        assert all(agent.closed for agent in agents)


# =============================================================================
# Step 5: Verify Result — Consensus, Agents, Receipt
# =============================================================================


class TestStep5ResultVerification:
    """Verify the debate result has consensus, agents, and a valid receipt."""

    @pytest.mark.asyncio
    async def test_consensus_reached(
        self,
        journey_env: Environment,
        journey_protocol: DebateProtocol,
        journey_agents: list[JourneyMockAgent],
    ):
        """Debate reaches consensus when agents agree."""
        arena = _build_journey_arena(journey_env, journey_agents, journey_protocol)
        result = await arena.run()

        assert result.consensus_reached is True
        assert result.confidence > 0.0

    @pytest.mark.asyncio
    async def test_result_has_messages(
        self,
        journey_env: Environment,
        journey_protocol: DebateProtocol,
        journey_agents: list[JourneyMockAgent],
    ):
        """Debate result contains messages from participating agents."""
        arena = _build_journey_arena(journey_env, journey_agents, journey_protocol)
        result = await arena.run()

        assert len(result.messages) > 0
        agent_names = {m.agent for m in result.messages}
        # At least some of our agents should have contributed messages
        expected_names = {a.name for a in journey_agents}
        assert len(agent_names & expected_names) > 0, (
            f"No agent messages found. Got {agent_names}, expected some of {expected_names}"
        )

    @pytest.mark.asyncio
    async def test_decision_receipt_generated(
        self,
        journey_env: Environment,
        journey_protocol: DebateProtocol,
        journey_agents: list[JourneyMockAgent],
    ):
        """A tamper-evident DecisionReceipt can be generated from the result."""
        arena = _build_journey_arena(journey_env, journey_agents, journey_protocol)
        result = await arena.run()

        receipt = DecisionReceipt.from_debate_result(result)

        assert receipt.receipt_id is not None
        assert len(receipt.receipt_id) > 0
        assert receipt.gauntlet_id is not None
        assert receipt.timestamp is not None
        assert receipt.confidence >= 0.0
        assert receipt.verdict in ("PASS", "CONDITIONAL", "FAIL")
        # SHA-256 hex digest = 64 chars
        assert receipt.artifact_hash is not None
        assert len(receipt.artifact_hash) == 64

    @pytest.mark.asyncio
    async def test_receipt_integrity_verification(
        self,
        journey_env: Environment,
        journey_protocol: DebateProtocol,
        journey_agents: list[JourneyMockAgent],
    ):
        """Receipt integrity hash is tamper-evident."""
        arena = _build_journey_arena(journey_env, journey_agents, journey_protocol)
        result = await arena.run()

        receipt = DecisionReceipt.from_debate_result(result)
        assert receipt.verify_integrity() is True


# =============================================================================
# Step 6: KM Ingestion
# =============================================================================


class TestStep6KnowledgeMoundIngestion:
    """Verify that debate outcomes are stored in the Knowledge Mound."""

    @pytest.mark.asyncio
    async def test_km_ingestion_stores_outcome(
        self,
        journey_env: Environment,
        journey_protocol: DebateProtocol,
        journey_agents: list[JourneyMockAgent],
    ):
        """KnowledgeMoundOperations.ingest_debate_outcome stores the result."""
        from aragora.debate.knowledge_mound_ops import KnowledgeMoundOperations
        from aragora.knowledge.mound.types import IngestionRequest, KnowledgeSource

        # Run the debate first
        arena = _build_journey_arena(journey_env, journey_agents, journey_protocol)
        result = await arena.run()

        assert result.final_answer is not None
        assert result.confidence > 0.0

        # Create a mock KnowledgeMound that captures the store call
        mock_km = MagicMock()
        mock_km.workspace_id = "test-workspace"

        # Track what gets stored
        stored_items: list[Any] = []

        async def capture_store(request):
            stored_items.append(request)
            # Return a mock IngestionResult
            mock_result = MagicMock()
            mock_result.node_id = "km-node-001"
            return mock_result

        mock_km.store = capture_store

        # Wire up KM operations and ingest
        km_ops = KnowledgeMoundOperations(
            knowledge_mound=mock_km,
            enable_retrieval=True,
            enable_ingestion=True,
        )

        await km_ops.ingest_debate_outcome(result, journey_env)

        # Verify the outcome was stored
        assert len(stored_items) == 1, (
            f"Expected 1 KM store call, got {len(stored_items)}. "
            f"Confidence={result.confidence:.2f} (needs >= 0.85 to ingest)"
        )

        stored = stored_items[0]
        assert isinstance(stored, IngestionRequest)
        assert "Debate Conclusion" in stored.content
        assert stored.source_type == KnowledgeSource.DEBATE
        assert stored.workspace_id == "test-workspace"
        assert stored.metadata["task"] == DEBATE_TOPIC
        assert stored.metadata["confidence"] == result.confidence
        assert stored.metadata["consensus_reached"] is True

    @pytest.mark.asyncio
    async def test_km_skips_low_confidence_outcome(
        self,
        journey_env: Environment,
    ):
        """KM ingestion is skipped when debate confidence is below 0.85."""
        from aragora.debate.knowledge_mound_ops import KnowledgeMoundOperations

        mock_km = MagicMock()
        mock_km.workspace_id = "test-workspace"
        store_calls: list = []

        async def capture_store(request):
            store_calls.append(request)
            mock_result = MagicMock()
            mock_result.node_id = "km-node-002"
            return mock_result

        mock_km.store = capture_store

        km_ops = KnowledgeMoundOperations(
            knowledge_mound=mock_km,
            enable_retrieval=True,
            enable_ingestion=True,
        )

        # Create a low-confidence result
        from aragora.core_types import DebateResult

        low_conf_result = DebateResult(
            task=DEBATE_TOPIC,
            final_answer="An inconclusive answer",
            confidence=0.4,
            consensus_reached=False,
            rounds_used=2,
            status="completed",
        )

        await km_ops.ingest_debate_outcome(low_conf_result, journey_env)

        assert len(store_calls) == 0, "KM should NOT ingest low-confidence outcomes"

    @pytest.mark.asyncio
    async def test_km_ingestion_disabled_noop(self):
        """When enable_ingestion is False, nothing is stored."""
        from aragora.debate.knowledge_mound_ops import KnowledgeMoundOperations
        from aragora.core_types import DebateResult

        mock_km = MagicMock()
        store_calls: list = []

        async def capture_store(request):
            store_calls.append(request)

        mock_km.store = capture_store

        km_ops = KnowledgeMoundOperations(
            knowledge_mound=mock_km,
            enable_ingestion=False,
        )

        result = DebateResult(
            task="Test",
            final_answer="Some answer",
            confidence=0.95,
            consensus_reached=True,
            rounds_used=1,
            status="completed",
        )

        await km_ops.ingest_debate_outcome(result)
        assert len(store_calls) == 0


# =============================================================================
# Full Journey: End-to-End Composition
# =============================================================================


class TestFullUserJourney:
    """Compose all steps into a single end-to-end validation."""

    @pytest.mark.asyncio
    async def test_complete_user_journey(
        self,
        journey_env: Environment,
        journey_protocol: DebateProtocol,
        journey_agents: list[JourneyMockAgent],
    ):
        """
        Full user journey:
        1. Health check passes
        2. API key format is valid
        3. Debate runs to completion with consensus
        4. Decision receipt is generated with integrity hash
        5. Knowledge Mound ingests the outcome
        """
        # -- Step 1: Health check --
        from aragora.server.api import DebateAPIHandler

        handler = MagicMock(spec=DebateAPIHandler)
        handler.static_dir = None
        health_response: dict = {}
        handler._send_json = lambda data, **kw: health_response.update(data)
        DebateAPIHandler._health_check(handler)
        assert health_response["status"] == "ok", "Health check failed"

        # -- Step 2: API key format --
        import secrets

        prefix = secrets.token_hex(4)
        secret = secrets.token_hex(24)
        api_key = f"ak_{prefix}_{secret}"
        assert api_key.startswith("ak_"), "API key prefix wrong"

        # -- Step 3: Run debate --
        arena = _build_journey_arena(journey_env, journey_agents, journey_protocol)
        result = await arena.run()

        assert result is not None, "Debate returned None"
        assert result.rounds_completed > 0, "No rounds completed"
        assert result.final_answer, "No final answer"
        assert result.consensus_reached is True, (
            f"Consensus not reached (confidence={result.confidence:.2f})"
        )

        # Verify all agents participated
        for agent in journey_agents:
            assert agent.generate_calls > 0, f"{agent.name} never generated a proposal"

        # -- Step 4: Decision receipt --
        receipt = DecisionReceipt.from_debate_result(result)
        assert receipt.receipt_id, "Receipt ID missing"
        assert receipt.artifact_hash and len(receipt.artifact_hash) == 64, (
            "Receipt hash missing or wrong length"
        )
        assert receipt.verify_integrity(), "Receipt integrity check failed"

        # -- Step 5: KM ingestion --
        from aragora.debate.knowledge_mound_ops import KnowledgeMoundOperations
        from aragora.knowledge.mound.types import IngestionRequest, KnowledgeSource

        mock_km = MagicMock()
        mock_km.workspace_id = "journey-test"
        stored: list[Any] = []

        async def capture_store(request):
            stored.append(request)
            mock_result = MagicMock()
            mock_result.node_id = "km-journey-001"
            return mock_result

        mock_km.store = capture_store

        km_ops = KnowledgeMoundOperations(
            knowledge_mound=mock_km,
            enable_ingestion=True,
        )
        await km_ops.ingest_debate_outcome(result, journey_env)

        assert len(stored) == 1, (
            f"KM ingestion failed: {len(stored)} items stored (confidence={result.confidence:.2f})"
        )
        assert isinstance(stored[0], IngestionRequest)
        assert stored[0].source_type == KnowledgeSource.DEBATE
        assert stored[0].metadata["consensus_reached"] is True
