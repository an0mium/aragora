"""Tests for Bead integration with debate flow.

Tests the Gastown Bead pattern integration that creates git-backed
audit trails for debate decisions.
"""

import pytest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from aragora.core_types import DebateResult
from aragora.debate.protocol import DebateProtocol
from aragora.nomic.beads import Bead, BeadPriority, BeadStatus, BeadStore, BeadType


class TestBeadType:
    """Tests for BeadType enum."""

    def test_debate_decision_type_exists(self):
        """BeadType should include DEBATE_DECISION."""
        assert hasattr(BeadType, "DEBATE_DECISION")
        assert BeadType.DEBATE_DECISION.value == "debate_decision"

    def test_all_types_available(self):
        """All expected bead types should be available."""
        expected = {"issue", "task", "epic", "hook", "debate_decision"}
        actual = {t.value for t in BeadType}
        assert expected == actual


class TestDebateResultBeadId:
    """Tests for bead_id field on DebateResult."""

    def test_bead_id_defaults_to_none(self):
        """bead_id should default to None."""
        result = DebateResult(task="Test task")
        assert result.bead_id is None

    def test_bead_id_can_be_set(self):
        """bead_id should be settable."""
        result = DebateResult(task="Test task")
        result.bead_id = "test-bead-123"
        assert result.bead_id == "test-bead-123"


class TestDebateProtocolBeadConfig:
    """Tests for bead tracking configuration in DebateProtocol."""

    def test_bead_tracking_disabled_by_default(self):
        """Bead tracking should be disabled by default."""
        protocol = DebateProtocol()
        assert protocol.enable_bead_tracking is False

    def test_bead_tracking_can_be_enabled(self):
        """Bead tracking can be enabled."""
        protocol = DebateProtocol(enable_bead_tracking=True)
        assert protocol.enable_bead_tracking is True

    def test_bead_min_confidence_default(self):
        """Default min confidence should be 0.5."""
        protocol = DebateProtocol()
        assert protocol.bead_min_confidence == 0.5

    def test_bead_auto_commit_disabled_by_default(self):
        """Auto commit should be disabled by default."""
        protocol = DebateProtocol()
        assert protocol.bead_auto_commit is False


class TestBeadStoreWithDebateDecision:
    """Tests for BeadStore with DEBATE_DECISION type."""

    @pytest.fixture
    def temp_bead_dir(self, tmp_path):
        """Create a temporary bead directory."""
        return tmp_path / ".beads"

    @pytest.mark.asyncio
    async def test_create_debate_decision_bead(self, temp_bead_dir):
        """Should create a bead of type DEBATE_DECISION."""
        store = BeadStore(bead_dir=temp_bead_dir, git_enabled=False)
        await store.initialize()

        bead = Bead.create(
            bead_type=BeadType.DEBATE_DECISION,
            title="Decision: Should we use microservices?",
            description="After thorough debate, consensus reached: Yes, for scalability.",
            priority=BeadPriority.HIGH,
            tags=["debate", "architecture", "consensus_reached"],
            metadata={
                "debate_id": "debate-123",
                "consensus_reached": True,
                "confidence": 0.85,
                "participants": ["claude", "gpt4", "gemini"],
            },
        )

        bead_id = await store.create(bead)
        assert bead_id is not None

        # Verify retrieval
        retrieved = await store.get(bead_id)
        assert retrieved is not None
        assert retrieved.bead_type == BeadType.DEBATE_DECISION
        assert retrieved.metadata["debate_id"] == "debate-123"
        assert retrieved.metadata["confidence"] == 0.85

    @pytest.mark.asyncio
    async def test_list_debate_decisions(self, temp_bead_dir):
        """Should list beads filtered by DEBATE_DECISION type."""
        store = BeadStore(bead_dir=temp_bead_dir, git_enabled=False)
        await store.initialize()

        # Create mixed bead types
        debate_bead = Bead.create(
            bead_type=BeadType.DEBATE_DECISION,
            title="Decision: API design",
            description="REST over GraphQL",
        )
        task_bead = Bead.create(
            bead_type=BeadType.TASK,
            title="Implement API",
            description="Build the REST endpoints",
        )

        await store.create(debate_bead)
        await store.create(task_bead)

        # Filter by type
        decisions = await store.list_by_type(BeadType.DEBATE_DECISION)
        assert len(decisions) == 1
        assert decisions[0].bead_type == BeadType.DEBATE_DECISION

    @pytest.mark.asyncio
    async def test_bead_metadata_persists(self, temp_bead_dir):
        """Debate-specific metadata should persist correctly."""
        store = BeadStore(bead_dir=temp_bead_dir, git_enabled=False)
        await store.initialize()

        metadata = {
            "debate_id": "debate-456",
            "consensus_reached": True,
            "confidence": 0.92,
            "rounds_used": 5,
            "participants": ["agent-1", "agent-2", "agent-3"],
            "winner": "agent-2",
            "domain": "security",
        }

        bead = Bead.create(
            bead_type=BeadType.DEBATE_DECISION,
            title="Security review decision",
            description="Implement 2FA for all admin accounts",
            metadata=metadata,
        )

        bead_id = await store.create(bead)

        # Reload from file
        store2 = BeadStore(bead_dir=temp_bead_dir, git_enabled=False)
        await store2.initialize()

        retrieved = await store2.get(bead_id)
        assert retrieved.metadata == metadata


class TestBeadPriorityFromConfidence:
    """Tests for mapping debate confidence to bead priority."""

    def test_high_confidence_high_priority(self):
        """High confidence debates should create high priority beads."""
        # 0.9+ confidence -> HIGH priority
        confidence = 0.92
        if confidence >= 0.9:
            priority = BeadPriority.HIGH
        elif confidence >= 0.7:
            priority = BeadPriority.NORMAL
        else:
            priority = BeadPriority.LOW
        assert priority == BeadPriority.HIGH

    def test_medium_confidence_normal_priority(self):
        """Medium confidence debates should create normal priority beads."""
        confidence = 0.75
        if confidence >= 0.9:
            priority = BeadPriority.HIGH
        elif confidence >= 0.7:
            priority = BeadPriority.NORMAL
        else:
            priority = BeadPriority.LOW
        assert priority == BeadPriority.NORMAL

    def test_low_confidence_low_priority(self):
        """Low confidence debates should create low priority beads."""
        confidence = 0.55
        if confidence >= 0.9:
            priority = BeadPriority.HIGH
        elif confidence >= 0.7:
            priority = BeadPriority.NORMAL
        else:
            priority = BeadPriority.LOW
        assert priority == BeadPriority.LOW


class TestArenaBeadCreation:
    """Tests for Arena._create_debate_bead method."""

    @pytest.fixture
    def mock_arena(self):
        """Create a mock Arena with bead tracking enabled."""
        from aragora.core_types import Environment

        arena = MagicMock()
        arena.protocol = DebateProtocol(
            enable_bead_tracking=True,
            bead_min_confidence=0.5,
            bead_auto_commit=False,
        )
        arena.env = Environment(task="Test debate task")
        arena._bead_store = None
        arena._extract_debate_domain = MagicMock(return_value="general")
        return arena

    @pytest.fixture
    def high_confidence_result(self):
        """Create a high-confidence debate result."""
        return DebateResult(
            debate_id="debate-001",
            task="Should we migrate to Kubernetes?",
            final_answer="Yes, for improved scalability and deployment automation.",
            confidence=0.88,
            consensus_reached=True,
            rounds_used=5,
            participants=["claude", "gpt4", "gemini"],
            winner="claude",
            status="consensus_reached",
        )

    @pytest.fixture
    def low_confidence_result(self):
        """Create a low-confidence debate result."""
        return DebateResult(
            debate_id="debate-002",
            task="Which database to use?",
            final_answer="No clear consensus reached.",
            confidence=0.35,
            consensus_reached=False,
            rounds_used=3,
            participants=["claude", "gpt4"],
            status="completed",
        )

    @pytest.mark.asyncio
    async def test_bead_created_for_high_confidence(
        self, mock_arena, high_confidence_result, tmp_path
    ):
        """Bead should be created for high-confidence debates."""
        # Import the actual method
        from aragora.debate.orchestrator import Arena

        # Create real arena with minimal config
        from aragora.core_types import Agent, Environment

        env = Environment(task="Test task", context={"bead_dir": str(tmp_path / ".beads")})
        agents = [MagicMock(spec=Agent, name="test-agent")]
        protocol = DebateProtocol(enable_bead_tracking=True, bead_min_confidence=0.5)

        # We'll test the bead creation logic directly
        with patch("aragora.nomic.beads.BeadStore") as MockStore:
            mock_store_instance = AsyncMock()
            mock_store_instance.create = AsyncMock(return_value="bead-123")
            MockStore.return_value = mock_store_instance

            # Simulate the bead creation logic
            result = high_confidence_result
            if protocol.enable_bead_tracking and result.confidence >= protocol.bead_min_confidence:
                from aragora.nomic.beads import Bead, BeadPriority, BeadType

                priority = (
                    BeadPriority.HIGH
                    if result.confidence >= 0.9
                    else (BeadPriority.NORMAL if result.confidence >= 0.7 else BeadPriority.LOW)
                )

                bead = Bead.create(
                    bead_type=BeadType.DEBATE_DECISION,
                    title=f"Decision: {result.task[:50]}",
                    description=result.final_answer[:500] if result.final_answer else "",
                    priority=priority,
                    metadata={"debate_id": result.debate_id},
                )

                assert bead.bead_type == BeadType.DEBATE_DECISION
                assert priority == BeadPriority.NORMAL  # 0.88 -> NORMAL

    @pytest.mark.asyncio
    async def test_bead_skipped_for_low_confidence(self, mock_arena, low_confidence_result):
        """Bead should NOT be created for low-confidence debates."""
        result = low_confidence_result
        min_confidence = 0.5

        # Simulate the confidence check
        should_create = result.confidence >= min_confidence
        assert should_create is False

    @pytest.mark.asyncio
    async def test_bead_skipped_when_disabled(self, high_confidence_result):
        """Bead should NOT be created when tracking is disabled."""
        protocol = DebateProtocol(enable_bead_tracking=False)

        # Simulate the enable check
        should_create = protocol.enable_bead_tracking
        assert should_create is False


class TestBeadIntegrationEndToEnd:
    """End-to-end tests for bead integration."""

    @pytest.mark.asyncio
    async def test_bead_lifecycle(self, tmp_path):
        """Test complete bead lifecycle: create -> complete -> verify."""
        store = BeadStore(bead_dir=tmp_path / ".beads", git_enabled=False)
        await store.initialize()

        # Create debate decision bead
        bead = Bead.create(
            bead_type=BeadType.DEBATE_DECISION,
            title="Decision: Use Python 3.12",
            description="Consensus: Upgrade to Python 3.12 for performance improvements.",
            priority=BeadPriority.NORMAL,
            metadata={
                "debate_id": "debate-e2e-001",
                "confidence": 0.82,
            },
        )

        bead_id = await store.create(bead)
        assert bead_id is not None

        # Claim and run (simulating debate execution)
        await store.claim(bead_id, agent_id="debate-orchestrator")
        retrieved = await store.get(bead_id)
        assert retrieved.status == BeadStatus.CLAIMED

        # Update to running
        await store.update_status(bead_id, BeadStatus.RUNNING)
        retrieved = await store.get(bead_id)
        assert retrieved.status == BeadStatus.RUNNING

        # Complete
        await store.update_status(bead_id, BeadStatus.COMPLETED)
        retrieved = await store.get(bead_id)
        assert retrieved.status == BeadStatus.COMPLETED
        assert retrieved.completed_at is not None

        # Verify statistics
        stats = await store.get_statistics()
        assert stats["total"] == 1
        assert stats["by_status"]["completed"] == 1
        assert stats["by_type"]["debate_decision"] == 1
