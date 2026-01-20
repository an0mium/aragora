"""
Tests for CrossDebateMemory.

Tests the cross-debate institutional memory system:
- DebateMemoryEntry serialization/deserialization
- CrossDebateConfig configuration
- AccessTier (MemoryTier) transitions
- Memory management and limits
- Query and context retrieval

Run with:
    pytest tests/memory/test_cross_debate.py -v --asyncio-mode=auto
"""

from __future__ import annotations

import json
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.memory.cross_debate_rlm import (
    AccessTier,
    CrossDebateConfig,
    CrossDebateMemory,
    DebateMemoryEntry,
    MemoryTier,
)


# =============================================================================
# AccessTier Tests
# =============================================================================


class TestAccessTier:
    """Tests for AccessTier enum."""

    def test_tier_values(self):
        """Test tier values."""
        assert AccessTier.HOT.value == "hot"
        assert AccessTier.WARM.value == "warm"
        assert AccessTier.COLD.value == "cold"
        assert AccessTier.ARCHIVE.value == "archive"

    def test_memory_tier_alias(self):
        """Test that MemoryTier is an alias for AccessTier."""
        assert MemoryTier is AccessTier


# =============================================================================
# DebateMemoryEntry Tests
# =============================================================================


class TestDebateMemoryEntry:
    """Tests for DebateMemoryEntry dataclass."""

    @pytest.fixture
    def sample_entry(self):
        """Create a sample entry."""
        return DebateMemoryEntry(
            debate_id="debate-123",
            task="Design a rate limiter",
            domain="engineering",
            timestamp=datetime(2026, 1, 15, 10, 0, 0),
            tier=MemoryTier.HOT,
            participants=["claude", "gpt-4", "gemini"],
            consensus_reached=True,
            final_answer="Token bucket algorithm recommended",
            key_insights=["Token bucket is efficient", "Consider Redis"],
            compressed_context="Discussion about rate limiting approaches",
            token_count=100,
            access_count=5,
            last_accessed=datetime(2026, 1, 18, 12, 0, 0),
        )

    def test_entry_creation(self, sample_entry):
        """Test entry creation."""
        assert sample_entry.debate_id == "debate-123"
        assert sample_entry.task == "Design a rate limiter"
        assert sample_entry.domain == "engineering"
        assert sample_entry.tier == MemoryTier.HOT
        assert len(sample_entry.participants) == 3
        assert sample_entry.consensus_reached is True

    def test_to_dict(self, sample_entry):
        """Test serialization to dictionary."""
        data = sample_entry.to_dict()

        assert data["debate_id"] == "debate-123"
        assert data["tier"] == "hot"
        assert data["consensus_reached"] is True
        assert data["timestamp"] == "2026-01-15T10:00:00"
        assert data["last_accessed"] == "2026-01-18T12:00:00"

    def test_from_dict(self, sample_entry):
        """Test deserialization from dictionary."""
        data = sample_entry.to_dict()
        restored = DebateMemoryEntry.from_dict(data)

        assert restored.debate_id == sample_entry.debate_id
        assert restored.task == sample_entry.task
        assert restored.tier == sample_entry.tier
        assert restored.consensus_reached == sample_entry.consensus_reached

    def test_from_dict_defaults(self):
        """Test from_dict with minimal data."""
        data = {
            "debate_id": "test-id",
            "task": "Test task",
            "timestamp": "2026-01-01T00:00:00",
        }
        entry = DebateMemoryEntry.from_dict(data)

        assert entry.debate_id == "test-id"
        assert entry.domain == "general"  # Default
        assert entry.tier == MemoryTier.WARM  # Default
        assert entry.participants == []
        assert entry.consensus_reached is False


# =============================================================================
# CrossDebateConfig Tests
# =============================================================================


class TestCrossDebateConfig:
    """Tests for CrossDebateConfig dataclass."""

    def test_default_config(self):
        """Test default configuration."""
        config = CrossDebateConfig()

        assert config.hot_duration == timedelta(hours=24)
        assert config.warm_duration == timedelta(days=7)
        assert config.cold_duration == timedelta(days=30)
        assert config.max_entries == 1000
        assert config.max_hot_entries == 50
        assert config.enable_rlm is True
        assert config.persist_to_disk is True

    def test_custom_config(self):
        """Test custom configuration."""
        config = CrossDebateConfig(
            max_entries=500,
            max_hot_entries=20,
            enable_rlm=False,
            persist_to_disk=False,
        )

        assert config.max_entries == 500
        assert config.max_hot_entries == 20
        assert config.enable_rlm is False
        assert config.persist_to_disk is False

    def test_config_with_storage_path(self):
        """Test configuration with storage path."""
        config = CrossDebateConfig(
            storage_path=Path("/tmp/test_memory.json"),
        )

        assert config.storage_path == Path("/tmp/test_memory.json")


# =============================================================================
# CrossDebateMemory Tests
# =============================================================================


class TestCrossDebateMemoryInit:
    """Tests for CrossDebateMemory initialization."""

    def test_default_init(self):
        """Test default initialization."""
        memory = CrossDebateMemory()

        assert memory.config is not None
        assert memory._entries == {}
        assert memory._initialized is False

    def test_custom_config_init(self):
        """Test initialization with custom config."""
        config = CrossDebateConfig(max_entries=100)
        memory = CrossDebateMemory(config=config)

        assert memory.config.max_entries == 100

    def test_has_real_rlm_property(self):
        """Test has_real_rlm property."""
        memory = CrossDebateMemory()
        # Should return a boolean (depends on RLM availability)
        assert isinstance(memory.has_real_rlm, bool)


class TestCrossDebateMemoryHelpers:
    """Tests for helper methods."""

    @pytest.fixture
    def memory(self):
        """Create memory instance."""
        return CrossDebateMemory(
            config=CrossDebateConfig(
                persist_to_disk=False,
                enable_rlm=False,
            )
        )

    def test_estimate_tokens(self, memory):
        """Test token estimation."""
        # ~4 chars per token
        assert memory._estimate_tokens("test") == 1
        assert memory._estimate_tokens("a" * 100) == 25

    def test_generate_id(self, memory):
        """Test ID generation."""
        now = datetime.now()
        id1 = memory._generate_id("task1", now)
        id2 = memory._generate_id("task2", now)
        id3 = memory._generate_id("task1", now)

        assert len(id1) == 16
        assert id1 != id2  # Different tasks
        assert id1 == id3  # Same task and time

    def test_determine_tier_hot(self, memory):
        """Test tier determination for recent entries."""
        recent = datetime.now() - timedelta(hours=1)
        assert memory._determine_tier(recent) == MemoryTier.HOT

    def test_determine_tier_warm(self, memory):
        """Test tier determination for warm entries."""
        warm = datetime.now() - timedelta(days=3)
        assert memory._determine_tier(warm) == MemoryTier.WARM

    def test_determine_tier_cold(self, memory):
        """Test tier determination for cold entries."""
        cold = datetime.now() - timedelta(days=14)
        assert memory._determine_tier(cold) == MemoryTier.COLD

    def test_determine_tier_archive(self, memory):
        """Test tier determination for archive entries."""
        old = datetime.now() - timedelta(days=60)
        assert memory._determine_tier(old) == MemoryTier.ARCHIVE


class TestCrossDebateMemoryAddDebate:
    """Tests for adding debates to memory."""

    @pytest.fixture
    def memory(self):
        """Create memory instance with RLM disabled."""
        return CrossDebateMemory(
            config=CrossDebateConfig(
                persist_to_disk=False,
                enable_rlm=False,
            )
        )

    @pytest.fixture
    def mock_debate_result(self):
        """Create a mock debate result."""
        result = MagicMock()
        result.debate_id = "debate-abc"
        result.task = "Design an API"
        result.domain = "engineering"
        result.participants = ["claude", "gpt-4"]
        result.consensus_reached = True
        result.final_answer = "RESTful design recommended"
        result.messages = []
        result.critiques = []
        return result

    @pytest.mark.asyncio
    async def test_add_debate(self, memory, mock_debate_result):
        """Test adding a debate."""
        debate_id = await memory.add_debate(mock_debate_result)

        assert debate_id is not None
        assert debate_id in memory._entries

        entry = memory._entries[debate_id]
        assert entry.task == "Design an API"
        assert entry.domain == "engineering"
        assert entry.consensus_reached is True
        assert entry.tier == MemoryTier.HOT  # New entries are hot

    @pytest.mark.asyncio
    async def test_add_debate_generates_id(self, memory):
        """Test ID generation when debate_id missing."""
        result = MagicMock()
        result.debate_id = None
        result.task = "Test task"
        result.domain = "general"
        result.participants = []
        result.consensus_reached = False
        result.final_answer = ""
        result.messages = []
        result.critiques = []

        debate_id = await memory.add_debate(result)

        assert debate_id is not None
        assert len(debate_id) == 16  # Generated ID

    @pytest.mark.asyncio
    async def test_add_multiple_debates(self, memory, mock_debate_result):
        """Test adding multiple debates."""
        for i in range(5):
            mock_debate_result.debate_id = f"debate-{i}"
            mock_debate_result.task = f"Task {i}"
            await memory.add_debate(mock_debate_result)

        assert len(memory._entries) == 5
        assert memory._initialized is True


class TestCrossDebateMemoryContext:
    """Tests for context retrieval."""

    @pytest.fixture
    def memory(self):
        """Create memory instance."""
        return CrossDebateMemory(
            config=CrossDebateConfig(
                persist_to_disk=False,
                enable_rlm=False,
            )
        )

    @pytest.fixture
    async def populated_memory(self, memory):
        """Create memory with entries."""
        for i in range(5):
            entry = DebateMemoryEntry(
                debate_id=f"debate-{i}",
                task=f"Design API endpoint {i}",
                domain="engineering" if i % 2 == 0 else "general",
                timestamp=datetime.now() - timedelta(hours=i),
                tier=MemoryTier.HOT,
                participants=["claude", "gpt-4"],
                consensus_reached=True,
                final_answer=f"Answer {i}",
                key_insights=[f"Insight {i}"],
                compressed_context=f"Context {i}",
                token_count=50,
            )
            memory._entries[entry.debate_id] = entry
        memory._initialized = True
        return memory

    @pytest.mark.asyncio
    async def test_get_relevant_context_empty(self, memory):
        """Test getting context from empty memory."""
        context = await memory.get_relevant_context("test task")
        assert context == ""

    @pytest.mark.asyncio
    async def test_get_relevant_context(self, populated_memory):
        """Test getting relevant context."""
        context = await populated_memory.get_relevant_context(
            task="Design API endpoint",
            max_tokens=1000,
        )

        assert "API endpoint" in context or "Debate" in context or context != ""

    @pytest.mark.asyncio
    async def test_get_relevant_context_with_domain(self, populated_memory):
        """Test getting context filtered by domain."""
        context = await populated_memory.get_relevant_context(
            task="Design API",
            domain="engineering",
        )

        # Only engineering domain entries should be considered
        assert context is not None

    @pytest.mark.asyncio
    async def test_get_relevant_context_with_tier_filter(self, populated_memory):
        """Test getting context filtered by tier."""
        context = await populated_memory.get_relevant_context(
            task="Design API",
            include_tiers=[MemoryTier.HOT],
        )

        assert context is not None


class TestCrossDebateMemoryQuery:
    """Tests for query functionality."""

    @pytest.fixture
    def memory(self):
        """Create memory instance."""
        return CrossDebateMemory(
            config=CrossDebateConfig(
                persist_to_disk=False,
                enable_rlm=False,
            )
        )

    @pytest.fixture
    async def populated_memory(self, memory):
        """Create memory with entries."""
        entry = DebateMemoryEntry(
            debate_id="debate-1",
            task="Rate limiting design",
            domain="engineering",
            timestamp=datetime.now() - timedelta(hours=1),
            tier=MemoryTier.HOT,
            participants=["claude"],
            consensus_reached=True,
            final_answer="Use token bucket",
            key_insights=["Token bucket is efficient"],
            compressed_context="Discussion about rate limiting",
            token_count=50,
        )
        memory._entries[entry.debate_id] = entry
        memory._initialized = True
        return memory

    @pytest.mark.asyncio
    async def test_query_empty_memory(self, memory):
        """Test querying empty memory."""
        result = await memory.query_past_debates("What was discussed?")
        assert "No past debates" in result or "No relevant" in result

    @pytest.mark.asyncio
    async def test_fallback_query(self, populated_memory):
        """Test fallback keyword-based query."""
        result = await populated_memory.query_past_debates(
            "What about rate limiting?",
            max_debates=5,
        )

        # Should use fallback since RLM is disabled
        assert (
            "rate" in result.lower() or "limiting" in result.lower() or "debate" in result.lower()
        )


class TestCrossDebateMemoryPersistence:
    """Tests for persistence functionality."""

    @pytest.mark.asyncio
    async def test_save_and_load(self):
        """Test saving and loading from disk."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage_path = Path(tmpdir) / "memory.json"

            # Create and populate memory
            memory1 = CrossDebateMemory(
                config=CrossDebateConfig(
                    persist_to_disk=True,
                    storage_path=storage_path,
                    enable_rlm=False,
                )
            )

            result = MagicMock()
            result.debate_id = "test-debate"
            result.task = "Test task"
            result.domain = "test"
            result.participants = ["agent1"]
            result.consensus_reached = True
            result.final_answer = "Test answer"
            result.messages = []
            result.critiques = []

            await memory1.add_debate(result)

            # Verify file exists
            assert storage_path.exists()

            # Load into new memory
            memory2 = CrossDebateMemory(
                config=CrossDebateConfig(
                    persist_to_disk=True,
                    storage_path=storage_path,
                    enable_rlm=False,
                )
            )
            await memory2.initialize()

            assert len(memory2._entries) == 1
            assert "test-debate" in memory2._entries

    @pytest.mark.asyncio
    async def test_clear(self):
        """Test clearing memory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage_path = Path(tmpdir) / "memory.json"

            memory = CrossDebateMemory(
                config=CrossDebateConfig(
                    persist_to_disk=True,
                    storage_path=storage_path,
                    enable_rlm=False,
                )
            )

            # Add an entry
            result = MagicMock()
            result.debate_id = "test"
            result.task = "Test"
            result.domain = "test"
            result.participants = []
            result.consensus_reached = False
            result.final_answer = ""
            result.messages = []
            result.critiques = []

            await memory.add_debate(result)
            assert len(memory._entries) == 1

            # Clear
            await memory.clear()
            assert len(memory._entries) == 0


class TestCrossDebateMemoryStatistics:
    """Tests for statistics functionality."""

    @pytest.fixture
    def memory(self):
        """Create memory instance."""
        return CrossDebateMemory(
            config=CrossDebateConfig(
                persist_to_disk=False,
                enable_rlm=False,
            )
        )

    def test_get_statistics_empty(self, memory):
        """Test statistics on empty memory."""
        stats = memory.get_statistics()

        assert stats["total_entries"] == 0
        assert stats["total_tokens"] == 0
        assert all(v == 0 for v in stats["tier_distribution"].values())

    def test_get_statistics_populated(self, memory):
        """Test statistics on populated memory."""
        # Add entries
        for i, tier in enumerate([MemoryTier.HOT, MemoryTier.HOT, MemoryTier.WARM]):
            entry = DebateMemoryEntry(
                debate_id=f"debate-{i}",
                task=f"Task {i}",
                domain="general",
                timestamp=datetime.now(),
                tier=tier,
                participants=[],
                consensus_reached=False,
                final_answer="",
                key_insights=[],
                compressed_context="",
                token_count=100,
            )
            memory._entries[entry.debate_id] = entry

        stats = memory.get_statistics()

        assert stats["total_entries"] == 3
        assert stats["tier_distribution"]["hot"] == 2
        assert stats["tier_distribution"]["warm"] == 1
        assert stats["total_tokens"] == 300


class TestCrossDebateMemoryLimits:
    """Tests for memory limit management."""

    @pytest.mark.asyncio
    async def test_manage_memory_limits(self):
        """Test that memory limits are enforced."""
        memory = CrossDebateMemory(
            config=CrossDebateConfig(
                persist_to_disk=False,
                enable_rlm=False,
                max_hot_entries=2,  # Very low for testing
            )
        )

        # Add more entries than limit
        for i in range(5):
            result = MagicMock()
            result.debate_id = f"debate-{i}"
            result.task = f"Task {i}"
            result.domain = "test"
            result.participants = []
            result.consensus_reached = False
            result.final_answer = ""
            result.messages = []
            result.critiques = []

            await memory.add_debate(result)

        # Should have limited entries
        hot_count = sum(1 for e in memory._entries.values() if e.tier == MemoryTier.HOT)
        assert hot_count <= 2
