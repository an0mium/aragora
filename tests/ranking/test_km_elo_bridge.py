"""
Tests for KMEloBridge - Knowledge Mound to ELO ranking bridge.

Tests the orchestration layer that syncs KM patterns to ELO adjustments.
"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

from aragora.ranking.km_elo_bridge import (
    KMEloBridge,
    KMEloBridgeConfig,
    KMEloBridgeSyncResult,
    create_km_elo_bridge,
)


@dataclass
class MockAgentRating:
    """Mock agent rating for testing."""

    agent_name: str
    elo: float = 1000.0


@dataclass
class MockKMEloPattern:
    """Mock KM ELO pattern for testing."""

    agent_name: str
    pattern_type: str
    confidence: float = 0.8
    observation_count: int = 5


@dataclass
class MockEloSyncResult:
    """Mock sync result for testing."""

    adjustments_recommended: int = 0
    adjustments_applied: int = 0
    adjustments_skipped: int = 0
    total_elo_change: float = 0.0
    agents_affected: List[str] = field(default_factory=list)


class MockEloSystem:
    """Mock ELO system for testing."""

    def __init__(self):
        self.ratings = {
            "claude": MockAgentRating("claude", 1200.0),
            "gpt4": MockAgentRating("gpt4", 1150.0),
            "gemini": MockAgentRating("gemini", 1100.0),
        }

    def get_all_ratings(self) -> List[MockAgentRating]:
        return list(self.ratings.values())

    def get_rating(self, agent_name: str) -> Optional[MockAgentRating]:
        return self.ratings.get(agent_name)


class MockEloAdapter:
    """Mock ELO adapter for testing."""

    def __init__(self):
        self.patterns: Dict[str, List[MockKMEloPattern]] = {}
        self.pending_adjustments = []
        self.applied_adjustments = []
        self.analyze_calls = []
        self.sync_calls = []

    def set_elo_system(self, elo_system):
        pass

    async def analyze_km_patterns_for_agent(
        self,
        agent_name: str,
        km_items: List[Dict],
        min_confidence: float = 0.7,
    ) -> List[MockKMEloPattern]:
        self.analyze_calls.append(
            {"agent_name": agent_name, "km_items": km_items, "min_confidence": min_confidence}
        )
        # Return mock patterns if we have enough items
        if len(km_items) >= 5:
            pattern = MockKMEloPattern(
                agent_name=agent_name,
                pattern_type="success_contributor",
                confidence=0.85,
            )
            self.patterns[agent_name] = [pattern]
            return [pattern]
        return []

    async def sync_km_to_elo(
        self,
        agent_patterns: Dict[str, List],
        max_adjustment: float = 50.0,
        min_confidence: float = 0.7,
        auto_apply: bool = False,
    ) -> MockEloSyncResult:
        self.sync_calls.append(
            {
                "agent_patterns": agent_patterns,
                "max_adjustment": max_adjustment,
                "min_confidence": min_confidence,
                "auto_apply": auto_apply,
            }
        )
        return MockEloSyncResult(
            adjustments_recommended=len(agent_patterns),
            adjustments_applied=len(agent_patterns) if auto_apply else 0,
            total_elo_change=15.0 * len(agent_patterns) if auto_apply else 0.0,
            agents_affected=list(agent_patterns.keys()) if auto_apply else [],
        )

    def get_agent_km_patterns(self, agent_name: str) -> List[MockKMEloPattern]:
        return self.patterns.get(agent_name, [])

    def get_pending_adjustments(self) -> List:
        return self.pending_adjustments

    def clear_pending_adjustments(self) -> int:
        count = len(self.pending_adjustments)
        self.pending_adjustments = []
        return count


class MockKnowledgeMound:
    """Mock Knowledge Mound for testing."""

    def __init__(self):
        self.items = {
            "claude": [
                {"id": "km_1", "agent": "claude", "confidence": 0.9},
                {"id": "km_2", "agent": "claude", "confidence": 0.85},
                {"id": "km_3", "agent": "claude", "confidence": 0.8},
                {"id": "km_4", "agent": "claude", "confidence": 0.75},
                {"id": "km_5", "agent": "claude", "confidence": 0.7},
                {"id": "km_6", "agent": "claude", "confidence": 0.65},
            ],
            "gpt4": [
                {"id": "km_10", "agent": "gpt4", "confidence": 0.8},
                {"id": "km_11", "agent": "gpt4", "confidence": 0.75},
            ],
        }
        self.query_calls = []

    async def query_by_agent(self, agent_name: str, limit: int = 100) -> List[Dict]:
        self.query_calls.append({"agent_name": agent_name, "limit": limit})
        return self.items.get(agent_name, [])


class TestKMEloBridgeConfig:
    """Tests for KMEloBridgeConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = KMEloBridgeConfig()

        assert config.sync_interval_hours == 24
        assert config.min_pattern_confidence == 0.7
        assert config.max_adjustment_per_sync == 50.0
        assert config.auto_apply is False
        assert config.track_history is True
        assert config.batch_size == 20

    def test_custom_config(self):
        """Test custom configuration values."""
        config = KMEloBridgeConfig(
            sync_interval_hours=12,
            min_pattern_confidence=0.8,
            max_adjustment_per_sync=30.0,
            auto_apply=True,
        )

        assert config.sync_interval_hours == 12
        assert config.min_pattern_confidence == 0.8
        assert config.max_adjustment_per_sync == 30.0
        assert config.auto_apply is True


class TestKMEloBridgeInitialization:
    """Tests for KMEloBridge initialization."""

    def test_basic_initialization(self):
        """Test basic bridge initialization."""
        bridge = KMEloBridge()

        assert bridge.elo_system is None
        assert bridge.elo_adapter is None
        assert bridge.knowledge_mound is None
        assert bridge._last_sync is None
        assert bridge._sync_in_progress is False

    def test_initialization_with_dependencies(self):
        """Test initialization with all dependencies."""
        elo_system = MockEloSystem()
        elo_adapter = MockEloAdapter()
        km = MockKnowledgeMound()
        config = KMEloBridgeConfig(sync_interval_hours=12)

        bridge = KMEloBridge(
            elo_system=elo_system,
            elo_adapter=elo_adapter,
            knowledge_mound=km,
            config=config,
        )

        assert bridge.elo_system is elo_system
        assert bridge.elo_adapter is elo_adapter
        assert bridge.knowledge_mound is km
        assert bridge._config.sync_interval_hours == 12

    def test_factory_function(self):
        """Test factory function creates bridge."""
        bridge = create_km_elo_bridge()

        assert isinstance(bridge, KMEloBridge)
        assert bridge._config is not None


class TestKMEloBridgeSetters:
    """Tests for KMEloBridge setter methods."""

    def test_set_elo_system(self):
        """Test setting ELO system."""
        bridge = KMEloBridge()
        elo_system = MockEloSystem()
        adapter = MockEloAdapter()
        bridge._elo_adapter = adapter

        bridge.set_elo_system(elo_system)

        assert bridge.elo_system is elo_system

    def test_set_elo_adapter(self):
        """Test setting ELO adapter."""
        bridge = KMEloBridge()
        adapter = MockEloAdapter()

        bridge.set_elo_adapter(adapter)

        assert bridge.elo_adapter is adapter

    def test_set_knowledge_mound(self):
        """Test setting knowledge mound."""
        bridge = KMEloBridge()
        km = MockKnowledgeMound()

        bridge.set_knowledge_mound(km)

        assert bridge.knowledge_mound is km


class TestKMEloBridgeSync:
    """Tests for KMEloBridge sync operations."""

    @pytest.fixture
    def bridge(self):
        """Create bridge with mocks."""
        elo_system = MockEloSystem()
        elo_adapter = MockEloAdapter()
        km = MockKnowledgeMound()
        config = KMEloBridgeConfig(
            sync_interval_hours=0,  # No interval restriction for testing
        )

        return KMEloBridge(
            elo_system=elo_system,
            elo_adapter=elo_adapter,
            knowledge_mound=km,
            config=config,
        )

    @pytest.mark.asyncio
    async def test_sync_analyzes_agents(self, bridge):
        """Test sync analyzes all agents."""
        result = await bridge.sync_km_to_elo(force=True)

        assert result.agents_analyzed == 3  # claude, gpt4, gemini
        # Only claude has enough items (6) for pattern detection (min 5)
        # gpt4 has 2 items, gemini has 0 - both skip analyze call
        assert len(bridge._elo_adapter.analyze_calls) == 1  # Only claude analyzed

    @pytest.mark.asyncio
    async def test_sync_detects_patterns(self, bridge):
        """Test sync detects patterns from KM items."""
        result = await bridge.sync_km_to_elo(force=True)

        # Only claude has enough items (6) for pattern detection (min 5)
        assert result.patterns_detected >= 1

    @pytest.mark.asyncio
    async def test_sync_calls_adapter(self, bridge):
        """Test sync calls adapter's sync method."""
        result = await bridge.sync_km_to_elo(force=True)

        assert len(bridge._elo_adapter.sync_calls) == 1
        sync_call = bridge._elo_adapter.sync_calls[0]
        assert "claude" in sync_call["agent_patterns"]

    @pytest.mark.asyncio
    async def test_sync_updates_last_sync(self, bridge):
        """Test sync updates last sync timestamp."""
        assert bridge._last_sync is None

        await bridge.sync_km_to_elo(force=True)

        assert bridge._last_sync is not None

    @pytest.mark.asyncio
    async def test_sync_tracks_history(self, bridge):
        """Test sync tracks history."""
        assert len(bridge._sync_history) == 0

        await bridge.sync_km_to_elo(force=True)

        assert len(bridge._sync_history) == 1
        assert bridge._sync_history[0].agents_analyzed == 3

    @pytest.mark.asyncio
    async def test_sync_specific_agents(self, bridge):
        """Test syncing specific agents only."""
        result = await bridge.sync_km_to_elo(
            agent_names=["claude"],
            force=True,
        )

        assert result.agents_analyzed == 1
        assert len(bridge._elo_adapter.analyze_calls) == 1
        assert bridge._elo_adapter.analyze_calls[0]["agent_name"] == "claude"

    @pytest.mark.asyncio
    async def test_sync_respects_interval(self, bridge):
        """Test sync respects interval when not forced."""
        bridge._config.sync_interval_hours = 24
        bridge._last_sync = 0  # Some time in the past but recent

        import time

        bridge._last_sync = time.time()  # Just synced

        result = await bridge.sync_km_to_elo(force=False)

        # Should have interval error
        assert any("interval" in e.lower() for e in result.errors)

    @pytest.mark.asyncio
    async def test_sync_auto_apply(self, bridge):
        """Test sync with auto-apply enabled."""
        bridge._config.auto_apply = True

        result = await bridge.sync_km_to_elo(force=True)

        sync_call = bridge._elo_adapter.sync_calls[0]
        assert sync_call["auto_apply"] is True


class TestKMEloBridgePatterns:
    """Tests for pattern-related methods."""

    @pytest.fixture
    def bridge(self):
        """Create bridge with mocks."""
        elo_adapter = MockEloAdapter()
        elo_adapter.patterns = {
            "claude": [
                MockKMEloPattern("claude", "success_contributor", 0.85),
                MockKMEloPattern("claude", "domain_expert", 0.9),
            ]
        }
        return KMEloBridge(elo_adapter=elo_adapter)

    @pytest.mark.asyncio
    async def test_get_agent_patterns(self, bridge):
        """Test getting agent's KM patterns."""
        patterns = await bridge.get_agent_km_patterns("claude")

        assert len(patterns) == 2
        assert patterns[0].pattern_type == "success_contributor"

    @pytest.mark.asyncio
    async def test_get_patterns_no_patterns(self, bridge):
        """Test getting patterns for agent without patterns."""
        patterns = await bridge.get_agent_km_patterns("unknown")

        assert len(patterns) == 0


class TestKMEloBridgePendingAdjustments:
    """Tests for pending adjustment methods."""

    @pytest.fixture
    def bridge(self):
        """Create bridge with pending adjustments."""
        elo_adapter = MockEloAdapter()
        elo_adapter.pending_adjustments = [
            MagicMock(agent_name="claude", adjustment=15.0, confidence=0.85),
            MagicMock(agent_name="gpt4", adjustment=-5.0, confidence=0.6),
        ]
        return KMEloBridge(elo_adapter=elo_adapter)

    @pytest.mark.asyncio
    async def test_get_pending_adjustments(self, bridge):
        """Test getting pending adjustments."""
        pending = await bridge.get_pending_adjustments()

        assert len(pending) == 2

    @pytest.mark.asyncio
    async def test_get_pending_no_adapter(self):
        """Test getting pending adjustments without adapter."""
        bridge = KMEloBridge()
        pending = await bridge.get_pending_adjustments()

        assert len(pending) == 0


class TestKMEloBridgeStatus:
    """Tests for status and metrics methods."""

    @pytest.fixture
    def bridge(self):
        """Create bridge for status testing."""
        elo_system = MockEloSystem()
        elo_adapter = MockEloAdapter()
        km = MockKnowledgeMound()

        return KMEloBridge(
            elo_system=elo_system,
            elo_adapter=elo_adapter,
            knowledge_mound=km,
        )

    def test_get_status(self, bridge):
        """Test getting bridge status."""
        status = bridge.get_status()

        assert status["elo_system_available"] is True
        assert status["elo_adapter_available"] is True
        assert status["knowledge_mound_available"] is True
        assert status["sync_in_progress"] is False
        assert status["total_syncs"] == 0
        assert "config" in status

    def test_get_status_without_dependencies(self):
        """Test status when dependencies are missing."""
        bridge = KMEloBridge()
        status = bridge.get_status()

        assert status["elo_system_available"] is False
        assert status["elo_adapter_available"] is False
        assert status["knowledge_mound_available"] is False

    def test_get_sync_history(self, bridge):
        """Test getting sync history."""
        # Add some history
        bridge._sync_history = [
            KMEloBridgeSyncResult(agents_analyzed=3, timestamp="2024-01-01"),
            KMEloBridgeSyncResult(agents_analyzed=5, timestamp="2024-01-02"),
        ]

        history = bridge.get_sync_history(limit=1)

        assert len(history) == 1
        assert history[0].agents_analyzed == 5

    def test_reset_metrics(self, bridge):
        """Test resetting metrics."""
        bridge._total_syncs = 10
        bridge._total_adjustments = 25
        bridge._sync_history = [
            KMEloBridgeSyncResult(agents_analyzed=3),
        ]

        bridge.reset_metrics()

        assert bridge._total_syncs == 0
        assert bridge._total_adjustments == 0
        assert len(bridge._sync_history) == 0


class TestKMEloBridgeSyncResult:
    """Tests for KMEloBridgeSyncResult dataclass."""

    def test_default_values(self):
        """Test default values."""
        result = KMEloBridgeSyncResult()

        assert result.agents_analyzed == 0
        assert result.patterns_detected == 0
        assert result.adjustments_recommended == 0
        assert result.adjustments_applied == 0
        assert result.total_elo_change == 0.0
        assert result.agents_affected == []
        assert result.errors == []

    def test_custom_values(self):
        """Test custom values."""
        result = KMEloBridgeSyncResult(
            agents_analyzed=5,
            patterns_detected=3,
            adjustments_applied=2,
            total_elo_change=30.0,
            agents_affected=["claude", "gpt4"],
        )

        assert result.agents_analyzed == 5
        assert result.patterns_detected == 3
        assert result.adjustments_applied == 2
        assert result.total_elo_change == 30.0
        assert len(result.agents_affected) == 2


class TestKMEloBridgeErrorHandling:
    """Tests for error handling."""

    @pytest.mark.asyncio
    async def test_sync_without_agents(self):
        """Test sync with no agents available."""
        bridge = KMEloBridge(
            elo_system=MagicMock(get_all_ratings=lambda: []),
            elo_adapter=MockEloAdapter(),
        )

        result = await bridge.sync_km_to_elo(force=True)

        assert "No agents" in result.errors[0]

    @pytest.mark.asyncio
    async def test_sync_handles_pattern_error(self):
        """Test sync handles pattern analysis errors gracefully."""
        adapter = MockEloAdapter()

        async def failing_analyze(*args, **kwargs):
            raise ValueError("Analysis failed")

        adapter.analyze_km_patterns_for_agent = failing_analyze

        bridge = KMEloBridge(
            elo_system=MockEloSystem(),
            elo_adapter=adapter,
            knowledge_mound=MockKnowledgeMound(),
            config=KMEloBridgeConfig(sync_interval_hours=0),
        )

        result = await bridge.sync_km_to_elo(force=True)

        assert len(result.errors) > 0
        assert "Analysis failed" in result.errors[0]


class TestKMEloBridgeConcurrency:
    """Tests for concurrency handling."""

    @pytest.mark.asyncio
    async def test_concurrent_sync_blocked(self):
        """Test that concurrent syncs are blocked."""
        bridge = KMEloBridge(
            elo_system=MockEloSystem(),
            elo_adapter=MockEloAdapter(),
            knowledge_mound=MockKnowledgeMound(),
            config=KMEloBridgeConfig(sync_interval_hours=0),
        )

        # Manually set sync in progress
        bridge._sync_in_progress = True

        result = await bridge.sync_km_to_elo(force=True)

        assert "already in progress" in result.errors[0]
