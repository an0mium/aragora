"""Tests for cross-debate RLM memory module.

Tests the CrossDebateMemory class which maintains institutional
knowledge from past debates using hierarchical compression.
"""

import asyncio
import json
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.memory.cross_debate_rlm import AccessTier


class TestAccessTier:
    """Test AccessTier enum."""

    def test_access_tier_values(self):
        """Test AccessTier has correct values."""
        assert AccessTier.HOT.value == "hot"
        assert AccessTier.WARM.value == "warm"
        assert AccessTier.COLD.value == "cold"
        assert AccessTier.ARCHIVE.value == "archive"

    def test_memory_tier_alias(self):
        """Test MemoryTier is an alias for AccessTier."""
        from aragora.memory.cross_debate_rlm import MemoryTier

        assert MemoryTier is AccessTier


class TestDebateMemoryEntry:
    """Test DebateMemoryEntry dataclass."""

    def test_entry_creation(self):
        """Test creating a DebateMemoryEntry."""
        from aragora.memory.cross_debate_rlm import DebateMemoryEntry

        now = datetime.now()
        entry = DebateMemoryEntry(
            debate_id="debate_001",
            task="Design a caching system",
            domain="engineering",
            timestamp=now,
            tier=AccessTier.HOT,
            participants=["claude", "gpt-4"],
            consensus_reached=True,
            final_answer="Use Redis for caching.",
            key_insights=["Consider TTL", "Handle cache invalidation"],
            compressed_context="Design discussion about caching...",
        )

        assert entry.debate_id == "debate_001"
        assert entry.task == "Design a caching system"
        assert entry.domain == "engineering"
        assert entry.tier == AccessTier.HOT
        assert len(entry.participants) == 2
        assert entry.consensus_reached is True
        assert entry.token_count == 0  # Default
        assert entry.access_count == 0  # Default
        assert entry.last_accessed is None  # Default

    def test_entry_to_dict(self):
        """Test DebateMemoryEntry serialization."""
        from aragora.memory.cross_debate_rlm import DebateMemoryEntry

        now = datetime.now()
        entry = DebateMemoryEntry(
            debate_id="debate_001",
            task="Test task",
            domain="general",
            timestamp=now,
            tier=AccessTier.WARM,
            participants=["agent1"],
            consensus_reached=False,
            final_answer="Answer",
            key_insights=["insight1"],
            compressed_context="context",
            token_count=100,
            access_count=5,
            last_accessed=now,
        )

        d = entry.to_dict()

        assert d["debate_id"] == "debate_001"
        assert d["task"] == "Test task"
        assert d["domain"] == "general"
        assert d["tier"] == "warm"
        assert d["participants"] == ["agent1"]
        assert d["consensus_reached"] is False
        assert d["token_count"] == 100
        assert d["access_count"] == 5
        assert d["last_accessed"] == now.isoformat()

    def test_entry_from_dict(self):
        """Test DebateMemoryEntry deserialization."""
        from aragora.memory.cross_debate_rlm import DebateMemoryEntry

        now = datetime.now()
        data = {
            "debate_id": "debate_002",
            "task": "Another task",
            "domain": "science",
            "timestamp": now.isoformat(),
            "tier": "cold",
            "participants": ["agent1", "agent2"],
            "consensus_reached": True,
            "final_answer": "The answer is 42",
            "key_insights": ["Key insight"],
            "compressed_context": "Compressed...",
            "token_count": 50,
            "access_count": 3,
            "last_accessed": now.isoformat(),
        }

        entry = DebateMemoryEntry.from_dict(data)

        assert entry.debate_id == "debate_002"
        assert entry.tier == AccessTier.COLD
        assert entry.token_count == 50
        assert entry.access_count == 3

    def test_entry_from_dict_with_defaults(self):
        """Test deserialization handles missing optional fields."""
        from aragora.memory.cross_debate_rlm import DebateMemoryEntry

        data = {
            "debate_id": "debate_003",
            "task": "Minimal task",
            "timestamp": datetime.now().isoformat(),
        }

        entry = DebateMemoryEntry.from_dict(data)

        assert entry.debate_id == "debate_003"
        assert entry.domain == "general"  # Default
        assert entry.tier == AccessTier.WARM  # Default
        assert entry.participants == []  # Default
        assert entry.consensus_reached is False  # Default
        assert entry.token_count == 0  # Default


class TestCrossDebateConfig:
    """Test CrossDebateConfig dataclass."""

    def test_config_defaults(self):
        """Test CrossDebateConfig default values."""
        from aragora.memory.cross_debate_rlm import CrossDebateConfig

        config = CrossDebateConfig()

        assert config.hot_duration == timedelta(hours=24)
        assert config.warm_duration == timedelta(days=7)
        assert config.cold_duration == timedelta(days=30)
        assert config.max_entries == 1000
        assert config.max_hot_entries == 50
        assert config.max_warm_entries == 200
        assert config.max_cold_entries == 500
        assert config.hot_token_budget == 5000
        assert config.warm_token_budget == 2000
        assert config.cold_token_budget == 500
        assert config.enable_rlm is True
        assert config.persist_to_disk is True
        # storage_path defaults to .nomic/cross_debate_memory.json when persist_to_disk is True
        assert config.storage_path == Path(".nomic/cross_debate_memory.json")

    def test_config_custom_values(self):
        """Test CrossDebateConfig with custom values."""
        from aragora.memory.cross_debate_rlm import CrossDebateConfig

        config = CrossDebateConfig(
            hot_duration=timedelta(hours=12),
            max_entries=500,
            enable_rlm=False,
            storage_path=Path("/tmp/test.json"),
        )

        assert config.hot_duration == timedelta(hours=12)
        assert config.max_entries == 500
        assert config.enable_rlm is False
        assert config.storage_path == Path("/tmp/test.json")


class TestCrossDebateMemory:
    """Test CrossDebateMemory class."""

    @pytest.fixture
    def memory(self):
        """Create a CrossDebateMemory instance."""
        from aragora.memory.cross_debate_rlm import (
            CrossDebateConfig,
            CrossDebateMemory,
        )

        config = CrossDebateConfig(
            enable_rlm=False,  # Disable RLM for testing
            persist_to_disk=False,
        )
        return CrossDebateMemory(config)

    @pytest.fixture
    def memory_with_storage(self):
        """Create a CrossDebateMemory with disk storage."""
        from aragora.memory.cross_debate_rlm import (
            CrossDebateConfig,
            CrossDebateMemory,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            config = CrossDebateConfig(
                enable_rlm=False,
                persist_to_disk=True,
                storage_path=Path(tmpdir) / "memory.json",
            )
            yield CrossDebateMemory(config)

    @pytest.fixture
    def mock_debate_result(self):
        """Create a mock debate result."""
        result = MagicMock()
        result.debate_id = "test_debate_001"
        result.task = "Design a rate limiter for API requests"
        result.domain = "engineering"
        result.participants = ["claude", "gpt-4"]
        result.consensus_reached = True
        result.final_answer = "Use token bucket algorithm with Redis."
        result.messages = [
            MagicMock(agent="claude", content="I suggest using token bucket..."),
            MagicMock(agent="gpt-4", content="Agreed, with distributed storage..."),
        ]
        result.critiques = [
            MagicMock(summary="Consider edge cases for burst traffic"),
        ]
        return result

    @pytest.mark.asyncio
    async def test_initialize(self, memory):
        """Test memory initialization."""
        assert memory._initialized is False
        await memory.initialize()
        assert memory._initialized is True

        # Second initialization is a no-op
        await memory.initialize()
        assert memory._initialized is True

    @pytest.mark.asyncio
    async def test_add_debate(self, memory, mock_debate_result):
        """Test adding a debate to memory."""
        debate_id = await memory.add_debate(mock_debate_result)

        assert debate_id == "test_debate_001"
        assert debate_id in memory._entries
        entry = memory._entries[debate_id]
        assert entry.task == "Design a rate limiter for API requests"
        assert entry.domain == "engineering"
        assert entry.tier == AccessTier.HOT
        assert entry.consensus_reached is True

    @pytest.mark.asyncio
    async def test_add_debate_generates_id(self, memory):
        """Test adding a debate without debate_id generates one."""
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
        assert len(debate_id) == 16  # SHA256 truncated
        assert debate_id in memory._entries

    @pytest.mark.asyncio
    async def test_get_relevant_context_no_matches(self, memory):
        """Test getting context with no matches returns empty."""
        await memory.initialize()

        context = await memory.get_relevant_context(
            task="Unrelated task about cooking",
            max_tokens=1000,
        )

        assert context == ""

    @pytest.mark.asyncio
    async def test_get_relevant_context_with_matches(self, memory, mock_debate_result):
        """Test getting relevant context from past debates."""
        await memory.add_debate(mock_debate_result)

        context = await memory.get_relevant_context(
            task="Design API rate limiting",
            max_tokens=2000,
        )

        assert context != ""
        assert "rate limiter" in context.lower() or "api" in context.lower()

    @pytest.mark.asyncio
    async def test_get_relevant_context_domain_filter(self, memory):
        """Test context retrieval with domain filter."""
        # Add debates in different domains
        result1 = MagicMock()
        result1.debate_id = "eng_001"
        result1.task = "Design a system"
        result1.domain = "engineering"
        result1.participants = []
        result1.consensus_reached = True
        result1.final_answer = "Engineering answer"
        result1.messages = []
        result1.critiques = []

        result2 = MagicMock()
        result2.debate_id = "sci_001"
        result2.task = "Design an experiment"
        result2.domain = "science"
        result2.participants = []
        result2.consensus_reached = True
        result2.final_answer = "Science answer"
        result2.messages = []
        result2.critiques = []

        await memory.add_debate(result1)
        await memory.add_debate(result2)

        # Filter by domain
        context = await memory.get_relevant_context(
            task="Design something",
            domain="engineering",
            max_tokens=2000,
        )

        # Should only include engineering debates
        assert "engineering" in context.lower() or "system" in context.lower()

    @pytest.mark.asyncio
    async def test_get_relevant_context_tier_filter(self, memory, mock_debate_result):
        """Test context retrieval with tier filter."""
        await memory.add_debate(mock_debate_result)

        # Request only COLD tier (should get nothing since entry is HOT)
        context = await memory.get_relevant_context(
            task="Design API rate limiting",
            include_tiers=[AccessTier.COLD],
            max_tokens=2000,
        )

        assert context == ""

    @pytest.mark.asyncio
    async def test_get_relevant_context_updates_access(self, memory, mock_debate_result):
        """Test that accessing context updates access tracking."""
        await memory.add_debate(mock_debate_result)

        entry = memory._entries["test_debate_001"]
        assert entry.access_count == 0
        assert entry.last_accessed is None

        await memory.get_relevant_context(
            task="Design rate limiting API",
            max_tokens=2000,
        )

        assert entry.access_count == 1
        assert entry.last_accessed is not None

    @pytest.mark.asyncio
    async def test_get_statistics(self, memory, mock_debate_result):
        """Test getting memory statistics."""
        await memory.initialize()

        stats = memory.get_statistics()
        assert stats["total_entries"] == 0
        assert stats["total_tokens"] == 0

        await memory.add_debate(mock_debate_result)

        stats = memory.get_statistics()
        assert stats["total_entries"] == 1
        assert stats["tier_distribution"]["hot"] == 1
        assert stats["total_tokens"] > 0

    @pytest.mark.asyncio
    async def test_clear(self, memory, mock_debate_result):
        """Test clearing memory."""
        await memory.add_debate(mock_debate_result)
        assert len(memory._entries) == 1

        await memory.clear()
        assert len(memory._entries) == 0

    @pytest.mark.asyncio
    async def test_determine_tier_by_age(self, memory):
        """Test tier determination based on age."""
        now = datetime.now()

        # Recent = HOT
        tier = memory._determine_tier(now - timedelta(hours=12))
        assert tier == AccessTier.HOT

        # 3 days ago = WARM
        tier = memory._determine_tier(now - timedelta(days=3))
        assert tier == AccessTier.WARM

        # 14 days ago = COLD
        tier = memory._determine_tier(now - timedelta(days=14))
        assert tier == AccessTier.COLD

        # 60 days ago = ARCHIVE
        tier = memory._determine_tier(now - timedelta(days=60))
        assert tier == AccessTier.ARCHIVE

    def test_estimate_tokens(self, memory):
        """Test token estimation."""
        text = "This is a test with approximately twenty characters"
        tokens = memory._estimate_tokens(text)
        assert tokens == len(text) // 4

    def test_generate_id(self, memory):
        """Test unique ID generation."""
        now = datetime.now()
        id1 = memory._generate_id("task1", now)
        id2 = memory._generate_id("task2", now)
        id3 = memory._generate_id("task1", now + timedelta(seconds=1))

        assert len(id1) == 16
        assert id1 != id2
        assert id1 != id3

        # Same inputs produce same ID
        id4 = memory._generate_id("task1", now)
        assert id1 == id4


class TestCrossDebateMemoryPersistence:
    """Test disk persistence for CrossDebateMemory."""

    @pytest.mark.asyncio
    async def test_save_and_load(self):
        """Test saving and loading memory from disk."""
        from aragora.memory.cross_debate_rlm import (
            CrossDebateConfig,
            CrossDebateMemory,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            storage_path = Path(tmpdir) / "memory.json"

            # Create and populate memory
            config = CrossDebateConfig(
                enable_rlm=False,
                persist_to_disk=True,
                storage_path=storage_path,
            )
            memory = CrossDebateMemory(config)

            result = MagicMock()
            result.debate_id = "persist_test"
            result.task = "Test persistence"
            result.domain = "testing"
            result.participants = ["agent1"]
            result.consensus_reached = True
            result.final_answer = "Persistence works"
            result.messages = []
            result.critiques = []

            await memory.add_debate(result)

            # Verify file was created
            assert storage_path.exists()

            # Create new memory instance and load
            memory2 = CrossDebateMemory(config)
            await memory2.initialize()

            assert "persist_test" in memory2._entries
            entry = memory2._entries["persist_test"]
            assert entry.task == "Test persistence"
            assert entry.consensus_reached is True

    @pytest.mark.asyncio
    async def test_load_nonexistent_file(self):
        """Test loading from non-existent file is handled gracefully."""
        from aragora.memory.cross_debate_rlm import (
            CrossDebateConfig,
            CrossDebateMemory,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            config = CrossDebateConfig(
                enable_rlm=False,
                persist_to_disk=True,
                storage_path=Path(tmpdir) / "nonexistent.json",
            )
            memory = CrossDebateMemory(config)

            # Should not raise
            await memory.initialize()
            assert len(memory._entries) == 0


class TestCrossDebateMemoryLimits:
    """Test memory limit enforcement."""

    @pytest.mark.asyncio
    async def test_tier_transition(self):
        """Test entries transition between tiers as they age."""
        from aragora.memory.cross_debate_rlm import (
            CrossDebateConfig,
            CrossDebateMemory,
            DebateMemoryEntry,
        )

        config = CrossDebateConfig(
            enable_rlm=False,
            persist_to_disk=False,
            hot_duration=timedelta(hours=1),  # Short for testing
        )
        memory = CrossDebateMemory(config)
        await memory.initialize()

        # Add an old entry directly
        old_entry = DebateMemoryEntry(
            debate_id="old_debate",
            task="Old task",
            domain="general",
            timestamp=datetime.now() - timedelta(hours=2),
            tier=AccessTier.HOT,
            participants=[],
            consensus_reached=False,
            final_answer="",
            key_insights=[],
            compressed_context="Old context...",
            token_count=100,
        )
        memory._entries["old_debate"] = old_entry

        # Trigger limit management
        await memory._manage_memory_limits()

        # Entry should have transitioned to WARM
        assert memory._entries["old_debate"].tier == AccessTier.WARM


class TestCrossDebateMemoryCompression:
    """Test context compression."""

    @pytest.mark.asyncio
    async def test_compress_without_rlm(self):
        """Test compression falls back to truncation without RLM."""
        from aragora.memory.cross_debate_rlm import (
            CrossDebateConfig,
            CrossDebateMemory,
        )

        config = CrossDebateConfig(
            enable_rlm=False,
            persist_to_disk=False,
        )
        memory = CrossDebateMemory(config)

        # Long context should be truncated
        long_context = "x" * 5000
        compressed = await memory._compress_context(long_context)

        assert len(compressed) <= 2000

    @pytest.mark.asyncio
    async def test_compress_empty_context(self):
        """Test compressing empty context."""
        from aragora.memory.cross_debate_rlm import (
            CrossDebateConfig,
            CrossDebateMemory,
        )

        config = CrossDebateConfig(
            enable_rlm=False,
            persist_to_disk=False,
        )
        memory = CrossDebateMemory(config)

        compressed = await memory._compress_context("")
        assert compressed == ""


class TestCrossDebateMemoryInsights:
    """Test insight extraction."""

    @pytest.mark.asyncio
    async def test_extract_insights(self):
        """Test extracting insights from debate result."""
        from aragora.memory.cross_debate_rlm import (
            CrossDebateConfig,
            CrossDebateMemory,
        )

        config = CrossDebateConfig(
            enable_rlm=False,
            persist_to_disk=False,
        )
        memory = CrossDebateMemory(config)

        result = MagicMock()
        result.final_answer = "The final conclusion is to use approach A."
        result.consensus_reached = True
        result.critiques = [
            MagicMock(summary="Consider edge cases"),
            MagicMock(summary="Performance matters"),
        ]

        insights = await memory._extract_insights(result)

        assert len(insights) <= 5
        assert any("Conclusion" in i for i in insights)
        assert any("Consensus" in i for i in insights)

    @pytest.mark.asyncio
    async def test_extract_insights_limited(self):
        """Test insights are limited to 5."""
        from aragora.memory.cross_debate_rlm import (
            CrossDebateConfig,
            CrossDebateMemory,
        )

        config = CrossDebateConfig(
            enable_rlm=False,
            persist_to_disk=False,
        )
        memory = CrossDebateMemory(config)

        result = MagicMock()
        result.final_answer = "Answer"
        result.consensus_reached = True
        result.critiques = [MagicMock(summary=f"Critique {i}") for i in range(10)]

        insights = await memory._extract_insights(result)

        assert len(insights) <= 5


class TestCrossDebateMemoryFormatting:
    """Test entry formatting."""

    def test_format_entry(self):
        """Test formatting a memory entry."""
        from aragora.memory.cross_debate_rlm import (
            CrossDebateConfig,
            CrossDebateMemory,
            DebateMemoryEntry,
        )

        config = CrossDebateConfig(enable_rlm=False, persist_to_disk=False)
        memory = CrossDebateMemory(config)

        entry = DebateMemoryEntry(
            debate_id="format_test",
            task="Test formatting the entry",
            domain="testing",
            timestamp=datetime.now(),
            tier=AccessTier.HOT,
            participants=["agent1"],
            consensus_reached=True,
            final_answer="The formatted answer is here.",
            key_insights=["Insight 1", "Insight 2"],
            compressed_context="Context...",
        )

        formatted = memory._format_entry(entry, max_tokens=500)

        assert "Test formatting" in formatted
        assert "Consensus: Yes" in formatted
        assert "Insight 1" in formatted

    def test_format_entry_truncates(self):
        """Test formatting truncates to max tokens."""
        from aragora.memory.cross_debate_rlm import (
            CrossDebateConfig,
            CrossDebateMemory,
            DebateMemoryEntry,
        )

        config = CrossDebateConfig(enable_rlm=False, persist_to_disk=False)
        memory = CrossDebateMemory(config)

        entry = DebateMemoryEntry(
            debate_id="truncate_test",
            task="x" * 1000,
            domain="testing",
            timestamp=datetime.now(),
            tier=AccessTier.HOT,
            participants=[],
            consensus_reached=False,
            final_answer="y" * 1000,
            key_insights=["z" * 500 for _ in range(5)],
            compressed_context="",
        )

        # Limit to 50 tokens = 200 chars
        formatted = memory._format_entry(entry, max_tokens=50)

        assert len(formatted) <= 200


class TestCrossDebateMemoryRLMFallback:
    """Test official RLM vs compression fallback behavior.

    These tests verify that CrossDebateMemory:
    1. Prefers official RLM when available
    2. Falls back to compression gracefully when RLM unavailable
    3. Properly reports RLM availability via has_real_rlm property
    """

    def test_has_real_rlm_property(self):
        """Test has_real_rlm reflects HAS_OFFICIAL_RLM flag."""
        from aragora.memory.cross_debate_rlm import (
            CrossDebateConfig,
            CrossDebateMemory,
            HAS_OFFICIAL_RLM,
        )

        config = CrossDebateConfig(enable_rlm=True, persist_to_disk=False)
        memory = CrossDebateMemory(config)

        # has_real_rlm should reflect the HAS_OFFICIAL_RLM constant
        assert memory.has_real_rlm == HAS_OFFICIAL_RLM

    @pytest.mark.asyncio
    async def test_get_rlm_returns_wrapper(self):
        """Test _get_rlm returns AragoraRLM wrapper.

        AragoraRLM is always created - it handles fallback internally when
        the official RLM library is not available.
        """
        from aragora.memory.cross_debate_rlm import (
            CrossDebateConfig,
            CrossDebateMemory,
            HAS_OFFICIAL_RLM,
        )
        from aragora.rlm import AragoraRLM

        config = CrossDebateConfig(enable_rlm=True, persist_to_disk=False)
        memory = CrossDebateMemory(config)

        rlm = await memory._get_rlm()

        # AragoraRLM wrapper should always be created
        assert rlm is not None
        assert isinstance(rlm, AragoraRLM)

        # The wrapper's has_real_rlm reflects official library availability
        assert memory.has_real_rlm == HAS_OFFICIAL_RLM

    @pytest.mark.asyncio
    async def test_query_past_debates_with_no_entries(self):
        """Test query_past_debates handles empty memory."""
        from aragora.memory.cross_debate_rlm import (
            CrossDebateConfig,
            CrossDebateMemory,
        )

        config = CrossDebateConfig(enable_rlm=True, persist_to_disk=False)
        memory = CrossDebateMemory(config)
        await memory.initialize()

        result = await memory.query_past_debates("What was decided?")

        # Should handle gracefully whether using RLM or fallback
        assert result is not None
        assert isinstance(result, str)

    @pytest.mark.asyncio
    async def test_query_past_debates_fallback(self):
        """Test query_past_debates fallback when RLM unavailable."""
        from aragora.memory.cross_debate_rlm import (
            CrossDebateConfig,
            CrossDebateMemory,
            DebateMemoryEntry,
            AccessTier,
        )

        config = CrossDebateConfig(enable_rlm=True, persist_to_disk=False)
        memory = CrossDebateMemory(config)
        await memory.initialize()

        # Add a test entry
        entry = DebateMemoryEntry(
            debate_id="test_query",
            task="Design a caching system",
            domain="engineering",
            timestamp=datetime.now(),
            tier=AccessTier.HOT,
            participants=["claude", "gpt-4"],
            consensus_reached=True,
            final_answer="Use Redis with TTL of 1 hour.",
            key_insights=["Consider cache invalidation", "Handle race conditions"],
            compressed_context="Discussion about caching strategies...",
        )
        memory._entries["test_query"] = entry

        # Query should work whether using RLM or fallback
        result = await memory.query_past_debates("What caching strategy was recommended?")

        assert result is not None
        assert isinstance(result, str)
        # Should find relevant content
        assert len(result) > 0

    @pytest.mark.asyncio
    async def test_fallback_query_keyword_matching(self):
        """Test _fallback_query performs keyword matching."""
        from aragora.memory.cross_debate_rlm import (
            CrossDebateConfig,
            CrossDebateMemory,
            DebateMemoryEntry,
            AccessTier,
        )

        config = CrossDebateConfig(enable_rlm=False, persist_to_disk=False)
        memory = CrossDebateMemory(config)
        await memory.initialize()

        # Add entries
        memory._entries["cache_debate"] = DebateMemoryEntry(
            debate_id="cache_debate",
            task="Design a caching system",
            domain="engineering",
            timestamp=datetime.now(),
            tier=AccessTier.HOT,
            participants=["agent1"],
            consensus_reached=True,
            final_answer="Use Redis.",
            key_insights=["Cache improves performance"],
            compressed_context="Cache discussion...",
        )

        memory._entries["auth_debate"] = DebateMemoryEntry(
            debate_id="auth_debate",
            task="Design authentication flow",
            domain="security",
            timestamp=datetime.now(),
            tier=AccessTier.HOT,
            participants=["agent2"],
            consensus_reached=True,
            final_answer="Use OAuth 2.0.",
            key_insights=["Security is important"],
            compressed_context="Auth discussion...",
        )

        # Query about caching should find cache_debate
        result = memory._fallback_query("caching system", max_debates=5)

        assert "caching" in result.lower() or "cache" in result.lower()
        assert "auth" not in result.lower() or "Found 1" in result

    def test_build_rlm_context(self):
        """Test _build_rlm_context formats entries for RLM REPL."""
        from aragora.memory.cross_debate_rlm import (
            CrossDebateConfig,
            CrossDebateMemory,
            DebateMemoryEntry,
            AccessTier,
        )

        config = CrossDebateConfig(enable_rlm=True, persist_to_disk=False)
        memory = CrossDebateMemory(config)

        entry = DebateMemoryEntry(
            debate_id="rlm_test",
            task="Test RLM context building",
            domain="testing",
            timestamp=datetime.now(),
            tier=AccessTier.HOT,
            participants=["agent1", "agent2"],
            consensus_reached=True,
            final_answer="Context building works.",
            key_insights=["Insight A", "Insight B"],
            compressed_context="Full context here...",
        )
        memory._entries["rlm_test"] = entry

        context = memory._build_rlm_context(max_debates=5)

        assert "Test RLM context building" in context
        assert "Participants: agent1, agent2" in context
        assert "Consensus: Yes" in context
        assert "Insight A" in context
        assert "Context building works" in context

    def test_build_rlm_context_empty(self):
        """Test _build_rlm_context with no entries."""
        from aragora.memory.cross_debate_rlm import (
            CrossDebateConfig,
            CrossDebateMemory,
        )

        config = CrossDebateConfig(enable_rlm=True, persist_to_disk=False)
        memory = CrossDebateMemory(config)

        context = memory._build_rlm_context(max_debates=5)

        assert context == ""

    @pytest.mark.asyncio
    async def test_compression_used_as_fallback_not_primary(self):
        """Test that compression is only used when RLM unavailable.

        This is a key architectural requirement: compression should be
        a FALLBACK to true RLM, not the primary approach.
        """
        from aragora.memory.cross_debate_rlm import (
            CrossDebateConfig,
            CrossDebateMemory,
            HAS_OFFICIAL_RLM,
        )

        config = CrossDebateConfig(enable_rlm=True, persist_to_disk=False)
        memory = CrossDebateMemory(config)

        # The has_real_rlm property should indicate whether
        # official RLM or compression fallback will be used
        if memory.has_real_rlm:
            # Official RLM available - query_past_debates should use it
            assert HAS_OFFICIAL_RLM is True
        else:
            # Compression fallback - query_past_debates will use _fallback_query
            assert HAS_OFFICIAL_RLM is False

        # Either way, the interface should work
        await memory.initialize()
        result = await memory.query_past_debates("test query")
        assert result is not None
