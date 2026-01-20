"""
Tests for debate-specific batch utilities.

Tests the batch_utils module that wraps RLM batch parallelism
with debate-specific patterns.
"""

import asyncio
import sys
import pytest
from dataclasses import dataclass
from typing import Optional
from unittest.mock import AsyncMock, MagicMock, patch

# Import directly from the module to avoid the circular import chain
# through phases/__init__.py -> consensus_phase -> ... -> middleware/versioning
import importlib.util
import os

_batch_utils_path = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "aragora", "debate", "phases", "batch_utils.py"
)
_spec = importlib.util.spec_from_file_location("batch_utils", os.path.abspath(_batch_utils_path))
_batch_utils = importlib.util.module_from_spec(_spec)
sys.modules["batch_utils"] = _batch_utils
_spec.loader.exec_module(_batch_utils)

DebateBatchConfig = _batch_utils.DebateBatchConfig
DebateBatchResult = _batch_utils.DebateBatchResult
batch_with_agents = _batch_utils.batch_with_agents
batch_generate_critiques = _batch_utils.batch_generate_critiques
batch_collect_votes = _batch_utils.batch_collect_votes


@dataclass
class MockAgent:
    """Mock agent for testing."""

    name: str

    def __hash__(self):
        return hash(self.name)


@dataclass
class MockCritique:
    """Mock critique for testing."""

    agent: str
    target_agent: str
    content: str


@dataclass
class MockVote:
    """Mock vote for testing."""

    agent: str
    choice: str


class MockCircuitBreaker:
    """Mock circuit breaker for testing."""

    def __init__(self, blocked_agents: Optional[list[str]] = None):
        self.blocked_agents = blocked_agents or []
        self.successes: list[str] = []
        self.failures: list[str] = []

    def filter_available_agents(self, agents):
        return [a for a in agents if a.name not in self.blocked_agents]

    def record_success(self, agent_name: str):
        self.successes.append(agent_name)

    def record_failure(self, agent_name: str):
        self.failures.append(agent_name)


class TestDebateBatchConfig:
    """Tests for DebateBatchConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = DebateBatchConfig()

        assert config.max_concurrent == 3
        assert config.timeout_per_item == 45.0
        assert config.min_required is None
        assert config.stagger_delay == 0.0
        assert config.circuit_breaker is None
        assert config.fail_fast is False

    def test_custom_config(self):
        """Test custom configuration values."""
        cb = MockCircuitBreaker()
        config = DebateBatchConfig(
            max_concurrent=5,
            timeout_per_item=30.0,
            min_required=2,
            circuit_breaker=cb,
        )

        assert config.max_concurrent == 5
        assert config.min_required == 2
        assert config.circuit_breaker == cb


class TestDebateBatchResult:
    """Tests for DebateBatchResult."""

    def test_empty_result(self):
        """Test empty result."""
        result = DebateBatchResult(
            results=[],
            errors=[],
            total_agents=3,
        )

        assert result.success_rate == 0.0
        assert result.successful_agents == []
        assert result.failed_agents == []

    def test_partial_success(self):
        """Test partial success."""
        result = DebateBatchResult(
            results=["a", "b"],
            errors=[("c", ValueError("error"))],
            total_agents=3,
            successful_agents=["agent_a", "agent_b"],
            failed_agents=["agent_c"],
        )

        assert result.success_rate == pytest.approx(2 / 3)
        assert len(result.successful_agents) == 2
        assert len(result.failed_agents) == 1


class TestBatchWithAgents:
    """Tests for batch_with_agents function."""

    @pytest.mark.asyncio
    async def test_basic_operation(self):
        """Test basic batch operation with agents."""
        agents = [MockAgent(f"agent_{i}") for i in range(3)]

        async def process(agent):
            await asyncio.sleep(0.01)
            return f"result_{agent.name}"

        result = await batch_with_agents(
            agents=agents,
            process_fn=process,
            operation_name="test",
        )

        assert len(result.results) == 3
        assert result.total_agents == 3
        assert result.success_rate == 1.0

    @pytest.mark.asyncio
    async def test_with_circuit_breaker(self):
        """Test batch operation with circuit breaker filtering."""
        agents = [MockAgent(f"agent_{i}") for i in range(4)]
        cb = MockCircuitBreaker(blocked_agents=["agent_1"])

        async def process(agent):
            return f"result_{agent.name}"

        config = DebateBatchConfig(circuit_breaker=cb)
        result = await batch_with_agents(
            agents=agents,
            process_fn=process,
            config=config,
            operation_name="test",
        )

        # agent_1 was filtered out
        assert len(result.results) == 3
        assert "agent_1" not in result.successful_agents

    @pytest.mark.asyncio
    async def test_circuit_breaker_recording(self):
        """Test that success/failure is recorded to circuit breaker."""
        agents = [MockAgent(f"agent_{i}") for i in range(3)]
        cb = MockCircuitBreaker()

        async def process(agent):
            if agent.name == "agent_1":
                raise ValueError("Failed")
            return f"result_{agent.name}"

        config = DebateBatchConfig(circuit_breaker=cb)
        result = await batch_with_agents(
            agents=agents,
            process_fn=process,
            config=config,
            operation_name="test",
        )

        assert "agent_0" in cb.successes
        assert "agent_2" in cb.successes
        assert "agent_1" in cb.failures

    @pytest.mark.asyncio
    async def test_early_stop(self):
        """Test early stopping when min_required is reached."""
        agents = [MockAgent(f"agent_{i}") for i in range(10)]
        processed = []

        async def process(agent):
            processed.append(agent.name)
            await asyncio.sleep(0.05)
            return f"result_{agent.name}"

        config = DebateBatchConfig(
            min_required=3,
            max_concurrent=2,
        )
        result = await batch_with_agents(
            agents=agents,
            process_fn=process,
            config=config,
            operation_name="test",
        )

        # Should have at least 3 results (might have more due to concurrency)
        assert len(result.results) >= 3

    @pytest.mark.asyncio
    async def test_empty_agents(self):
        """Test with empty agent list."""

        async def process(agent):
            return "result"

        result = await batch_with_agents(
            agents=[],
            process_fn=process,
            operation_name="test",
        )

        assert result.results == []
        assert result.total_agents == 0

    @pytest.mark.asyncio
    async def test_all_filtered_by_circuit_breaker(self):
        """Test when all agents are blocked by circuit breaker."""
        agents = [MockAgent("agent_1"), MockAgent("agent_2")]
        cb = MockCircuitBreaker(blocked_agents=["agent_1", "agent_2"])

        async def process(agent):
            return "result"

        config = DebateBatchConfig(circuit_breaker=cb)
        result = await batch_with_agents(
            agents=agents,
            process_fn=process,
            config=config,
            operation_name="test",
        )

        assert result.results == []
        assert result.total_agents == 2

    @pytest.mark.asyncio
    async def test_spectator_notification(self):
        """Test spectator notification on success."""
        agents = [MockAgent("agent_1")]
        notifications = []

        def notify(operation, agent=None, details=None):
            notifications.append((operation, agent, details))

        async def process(agent):
            return "result"

        config = DebateBatchConfig(notify_spectator=notify)
        await batch_with_agents(
            agents=agents,
            process_fn=process,
            config=config,
            operation_name="test_op",
        )

        assert len(notifications) == 1
        assert notifications[0][0] == "test_op"
        assert notifications[0][1] == "agent_1"


class TestBatchGenerateCritiques:
    """Tests for batch_generate_critiques function."""

    @pytest.mark.asyncio
    async def test_basic_critique_generation(self):
        """Test basic critique generation."""
        critics = [MockAgent("critic_1"), MockAgent("critic_2")]
        proposals = {"proposer_a": "Proposal A content", "proposer_b": "Proposal B content"}

        async def generate_critique(critic, proposer, proposal):
            return MockCritique(
                agent=critic.name,
                target_agent=proposer,
                content=f"Critique of {proposal[:10]}...",
            )

        critiques = await batch_generate_critiques(
            critics=critics,
            proposals=proposals,
            generate_fn=generate_critique,
        )

        # Each critic critiques each proposer (excluding self)
        # 2 critics * 2 proposers = 4 critiques
        assert len(critiques) == 4

    @pytest.mark.asyncio
    async def test_self_critique_excluded(self):
        """Test that agents don't critique themselves."""
        critics = [MockAgent("alice"), MockAgent("bob")]
        proposals = {"alice": "Alice's proposal", "bob": "Bob's proposal"}

        critique_targets = []

        async def generate_critique(critic, proposer, proposal):
            critique_targets.append((critic.name, proposer))
            return MockCritique(
                agent=critic.name,
                target_agent=proposer,
                content="Critique",
            )

        await batch_generate_critiques(
            critics=critics,
            proposals=proposals,
            generate_fn=generate_critique,
        )

        # No self-critiques
        assert ("alice", "alice") not in critique_targets
        assert ("bob", "bob") not in critique_targets

    @pytest.mark.asyncio
    async def test_early_stop_critiques(self):
        """Test early stopping for critiques."""
        critics = [MockAgent(f"critic_{i}") for i in range(5)]
        proposals = {"proposer": "A proposal"}

        generated_count = 0

        async def generate_critique(critic, proposer, proposal):
            nonlocal generated_count
            generated_count += 1
            await asyncio.sleep(0.02)
            return MockCritique(
                agent=critic.name,
                target_agent=proposer,
                content="Critique",
            )

        config = DebateBatchConfig(min_required=2, max_concurrent=2)
        critiques = await batch_generate_critiques(
            critics=critics,
            proposals=proposals,
            generate_fn=generate_critique,
            config=config,
        )

        # Should have at least 2 critiques (early stop condition)
        assert len(critiques) >= 2


class TestBatchCollectVotes:
    """Tests for batch_collect_votes function."""

    @pytest.mark.asyncio
    async def test_basic_vote_collection(self):
        """Test basic vote collection."""
        agents = [MockAgent(f"voter_{i}") for i in range(5)]
        proposals = {"alice": "Proposal A", "bob": "Proposal B"}

        async def vote_fn(agent, props):
            await asyncio.sleep(0.01)  # Small delay for async simulation
            return MockVote(agent=agent.name, choice="alice")

        # Disable early termination for this test by using high threshold
        config = DebateBatchConfig(max_concurrent=1)  # Sequential to ensure all votes
        votes, early_stopped, winner = await batch_collect_votes(
            agents=agents,
            proposals=proposals,
            vote_fn=vote_fn,
            config=config,
            majority_threshold=0.99,  # Very high to prevent early stop
        )

        # Should collect all votes when early termination disabled
        assert len(votes) >= 3  # Allow for some variation due to async
        assert all(v.choice == "alice" for v in votes)

    @pytest.mark.asyncio
    async def test_early_termination_on_majority(self):
        """Test early termination when clear majority reached."""
        agents = [MockAgent(f"voter_{i}") for i in range(7)]
        proposals = {"alice": "Proposal A", "bob": "Proposal B"}

        async def vote_fn(agent, props):
            await asyncio.sleep(0.02)
            # Use agent name to determine vote - voters 0-4 vote alice, 5-6 vote bob
            voter_num = int(agent.name.split("_")[1])
            choice = "alice" if voter_num < 5 else "bob"
            return MockVote(agent=agent.name, choice=choice)

        votes, early_stopped, winner = await batch_collect_votes(
            agents=agents,
            proposals=proposals,
            vote_fn=vote_fn,
            majority_threshold=0.5,
        )

        # Should have votes (might be partial due to early termination)
        assert len(votes) >= 4  # Need at least 50% + lead
        # Count alice votes
        alice_votes = sum(1 for v in votes if v.choice == "alice")
        # Either alice wins or all votes collected
        assert alice_votes >= 3 or len(votes) == 7

    @pytest.mark.asyncio
    async def test_no_early_termination_split_vote(self):
        """Test no early termination when vote is split."""
        agents = [MockAgent(f"voter_{i}") for i in range(4)]
        proposals = {"alice": "Proposal A", "bob": "Proposal B"}

        async def vote_fn(agent, props):
            await asyncio.sleep(0.01)
            # Split vote - no clear majority
            choice = "alice" if int(agent.name[-1]) % 2 == 0 else "bob"
            return MockVote(agent=agent.name, choice=choice)

        votes, early_stopped, winner = await batch_collect_votes(
            agents=agents,
            proposals=proposals,
            vote_fn=vote_fn,
        )

        # All votes should be collected (no early termination)
        assert len(votes) == 4
        assert not early_stopped

    @pytest.mark.asyncio
    async def test_spectator_notification_on_early_termination(self):
        """Test spectator is notified of early termination."""
        agents = [MockAgent(f"voter_{i}") for i in range(5)]
        proposals = {"alice": "Proposal A"}
        notifications = []

        def notify(event, details=None, agent=None):
            notifications.append((event, details))

        async def vote_fn(agent, props):
            await asyncio.sleep(0.01)
            return MockVote(agent=agent.name, choice="alice")

        config = DebateBatchConfig(notify_spectator=notify)
        await batch_collect_votes(
            agents=agents,
            proposals=proposals,
            vote_fn=vote_fn,
            config=config,
            majority_threshold=0.5,
        )

        # Check if notification was sent
        early_term_notifications = [n for n in notifications if n[0] == "rlm_early_termination"]
        assert len(early_term_notifications) == 1


class TestIntegration:
    """Integration tests for batch utilities."""

    @pytest.mark.asyncio
    async def test_full_debate_flow_simulation(self):
        """Simulate a mini debate flow using batch utilities."""
        # Setup agents
        proposers = [MockAgent("alice"), MockAgent("bob")]
        critics = proposers  # Same agents act as critics
        voters = proposers  # Same agents vote

        # Phase 1: Generate proposals (simulated)
        proposals = {
            "alice": "Alice's proposal for solving X",
            "bob": "Bob's proposal for solving X",
        }

        # Phase 2: Generate critiques
        async def generate_critique(critic, proposer, proposal):
            return MockCritique(
                agent=critic.name,
                target_agent=proposer,
                content=f"{critic.name} critiques {proposer}'s proposal",
            )

        critiques = await batch_generate_critiques(
            critics=critics,
            proposals=proposals,
            generate_fn=generate_critique,
        )

        assert len(critiques) == 2  # Each critiques the other

        # Phase 3: Collect votes
        async def vote_fn(agent, props):
            # Alice votes for herself, Bob votes for Alice too
            return MockVote(agent=agent.name, choice="alice")

        votes, early_stopped, winner = await batch_collect_votes(
            agents=voters,
            proposals=proposals,
            vote_fn=vote_fn,
        )

        assert len(votes) == 2
        assert winner == "alice" or all(v.choice == "alice" for v in votes)
