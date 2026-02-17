"""Tests for Extended Rounds module.

Tests the ExtendedDebateConfig, RLMContextManager, and context compression
functionality for debates with 50+ rounds.
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.debate.extended_rounds import (
    ContextStrategy,
    ExtendedContextState,
    ExtendedDebateConfig,
    RLMContextManager,
    RoundSummary,
    create_extended_config,
)


# =============================================================================
# Mock classes for testing
# =============================================================================


@dataclass
class MockEnvironment:
    """Mock Environment for testing."""

    task: str = "Test debate task"


@dataclass
class MockMessage:
    """Mock Message for testing."""

    role: str = "proposer"
    agent: str = "test-agent"
    content: str = "Test message content"
    timestamp: datetime = field(default_factory=datetime.now)
    round: int = 0


@dataclass
class MockDebateContext:
    """Mock DebateContext for testing."""

    env: MockEnvironment = field(default_factory=MockEnvironment)
    context_messages: list[MockMessage] = field(default_factory=list)
    historical_context_cache: str = ""


@dataclass
class MockRLMResult:
    """Mock RLMResult for testing."""

    answer: str = "Compressed summary of debate"
    used_true_rlm: bool = False
    used_compression_fallback: bool = True


# =============================================================================
# ExtendedDebateConfig Tests
# =============================================================================


class TestExtendedDebateConfig:
    """Tests for ExtendedDebateConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = ExtendedDebateConfig()

        assert config.max_rounds == 100
        assert config.soft_limit_rounds == 50
        assert config.compression_threshold == 10000
        assert config.context_window_rounds == 10
        assert config.min_context_ratio == 0.3
        assert config.enable_rlm is True
        assert config.rlm_max_levels == 4
        assert config.rlm_cache_enabled is True
        assert config.compression_timeout == 30.0
        assert config.parallel_compression is True
        assert config.context_strategy == ContextStrategy.ADAPTIVE

    def test_custom_values(self):
        """Test configuration with custom values."""
        config = ExtendedDebateConfig(
            max_rounds=200,
            soft_limit_rounds=100,
            compression_threshold=5000,
            context_window_rounds=5,
            min_context_ratio=0.2,
            enable_rlm=False,
            context_strategy=ContextStrategy.HIERARCHICAL,
        )

        assert config.max_rounds == 200
        assert config.soft_limit_rounds == 100
        assert config.compression_threshold == 5000
        assert config.context_window_rounds == 5
        assert config.min_context_ratio == 0.2
        assert config.enable_rlm is False
        assert config.context_strategy == ContextStrategy.HIERARCHICAL


class TestContextStrategy:
    """Tests for ContextStrategy enum."""

    def test_strategy_values(self):
        """Test strategy enum values."""
        assert ContextStrategy.FULL.value == "full"
        assert ContextStrategy.SLIDING_WINDOW.value == "sliding_window"
        assert ContextStrategy.HIERARCHICAL.value == "hierarchical"
        assert ContextStrategy.ADAPTIVE.value == "adaptive"


class TestRoundSummary:
    """Tests for RoundSummary dataclass."""

    def test_round_summary_creation(self):
        """Test RoundSummary creation."""
        summary = RoundSummary(
            round_num=5,
            proposals={"agent1": "proposal1", "agent2": "proposal2"},
            critiques=["critique1", "critique2"],
            key_points=["point1", "point2"],
            consensus_progress=0.75,
            token_count=500,
        )

        assert summary.round_num == 5
        assert len(summary.proposals) == 2
        assert len(summary.critiques) == 2
        assert summary.consensus_progress == 0.75
        assert summary.token_count == 500
        assert summary.compressed_at is None


class TestExtendedContextState:
    """Tests for ExtendedContextState dataclass."""

    def test_default_state(self):
        """Test default state initialization."""
        state = ExtendedContextState()

        assert state.current_round == 0
        assert state.total_tokens == 0
        assert state.compressed_tokens == 0
        assert state.round_summaries == {}
        assert state.compressed_history == ""
        assert state.rlm_context is None
        assert state.compressions_performed == 0
        assert state.tokens_saved == 0
        assert state.compression_time_total == 0.0


# =============================================================================
# RLMContextManager Initialization Tests
# =============================================================================


class TestRLMContextManagerInit:
    """Tests for RLMContextManager initialization."""

    def test_init_with_default_config(self):
        """Test initialization with default config."""
        manager = RLMContextManager()

        assert manager.config is not None
        assert manager.config.max_rounds == 100
        assert manager._state.current_round == 0
        assert manager._rlm is None

    def test_init_with_custom_config(self):
        """Test initialization with custom config."""
        config = ExtendedDebateConfig(max_rounds=50)
        manager = RLMContextManager(config)

        assert manager.config.max_rounds == 50

    def test_reset_clears_state(self):
        """Test reset method clears state."""
        manager = RLMContextManager()
        manager._state.current_round = 10
        manager._state.total_tokens = 5000
        manager._state.compressions_performed = 3

        manager.reset()

        assert manager._state.current_round == 0
        assert manager._state.total_tokens == 0
        assert manager._state.compressions_performed == 0


# =============================================================================
# Token Estimation Tests
# =============================================================================


class TestTokenEstimation:
    """Tests for token estimation."""

    def test_estimate_tokens(self):
        """Test token estimation (4 chars per token)."""
        manager = RLMContextManager()

        assert manager._estimate_tokens("") == 0
        assert manager._estimate_tokens("test") == 1
        assert manager._estimate_tokens("a" * 100) == 25

    def test_estimate_tokens_long_text(self):
        """Test token estimation for longer text."""
        manager = RLMContextManager()

        # 40000 chars = 10000 tokens
        long_text = "a" * 40000
        assert manager._estimate_tokens(long_text) == 10000


# =============================================================================
# Strategy Selection Tests
# =============================================================================


class TestStrategySelection:
    """Tests for context strategy selection."""

    def test_fixed_strategy_returns_configured(self):
        """Test fixed strategy returns configured value."""
        config = ExtendedDebateConfig(context_strategy=ContextStrategy.HIERARCHICAL)
        manager = RLMContextManager(config)

        result = manager._select_strategy(total_tokens=5000, round_num=10)
        assert result == ContextStrategy.HIERARCHICAL

    def test_adaptive_small_context_uses_full(self):
        """Test adaptive strategy uses FULL for small context."""
        config = ExtendedDebateConfig(
            context_strategy=ContextStrategy.ADAPTIVE,
            compression_threshold=10000,
        )
        manager = RLMContextManager(config)

        # Below threshold
        result = manager._select_strategy(total_tokens=5000, round_num=10)
        assert result == ContextStrategy.FULL

    def test_adaptive_medium_context_uses_sliding_window(self):
        """Test adaptive strategy uses SLIDING_WINDOW for medium context."""
        config = ExtendedDebateConfig(
            context_strategy=ContextStrategy.ADAPTIVE,
            compression_threshold=10000,
            soft_limit_rounds=50,
        )
        manager = RLMContextManager(config)

        # Above threshold, below soft limit
        result = manager._select_strategy(total_tokens=15000, round_num=30)
        assert result == ContextStrategy.SLIDING_WINDOW

    def test_adaptive_large_context_uses_hierarchical(self):
        """Test adaptive strategy uses HIERARCHICAL for large context."""
        config = ExtendedDebateConfig(
            context_strategy=ContextStrategy.ADAPTIVE,
            compression_threshold=10000,
            soft_limit_rounds=50,
        )
        manager = RLMContextManager(config)

        # Above threshold, above soft limit
        result = manager._select_strategy(total_tokens=20000, round_num=60)
        assert result == ContextStrategy.HIERARCHICAL


# =============================================================================
# Context Building Tests
# =============================================================================


class TestContextBuilding:
    """Tests for building debate context."""

    def test_build_full_context_empty(self):
        """Test building context with no messages."""
        manager = RLMContextManager()
        debate_context = MockDebateContext(
            env=MockEnvironment(task="Empty debate"),
            context_messages=[],
        )

        result = manager._build_full_context(debate_context)

        assert "## Task\nEmpty debate" in result

    def test_build_full_context_with_messages(self):
        """Test building context with messages."""
        manager = RLMContextManager()

        messages = [
            MockMessage(role="proposer", agent="agent1", content="Proposal 1", round=1),
            MockMessage(role="critic", agent="agent2", content="Critique 1", round=1),
        ]
        debate_context = MockDebateContext(
            env=MockEnvironment(task="Test task"),
            context_messages=messages,
        )

        result = manager._build_full_context(debate_context)

        assert "## Task\nTest task" in result
        assert "### Round 1 - agent1 (proposer)" in result
        assert "Proposal 1" in result
        assert "### Round 1 - agent2 (critic)" in result
        assert "Critique 1" in result

    def test_build_full_context_with_historical_context(self):
        """Test building context with historical context cache."""
        manager = RLMContextManager()
        debate_context = MockDebateContext(
            env=MockEnvironment(task="Test task"),
            context_messages=[],
            historical_context_cache="Historical context information",
        )

        result = manager._build_full_context(debate_context)

        assert "## Historical Context" in result
        assert "Historical context information" in result

    def test_build_old_rounds_content(self):
        """Test building old rounds content for compression."""
        manager = RLMContextManager()

        messages = [
            MockMessage(role="proposer", agent="agent1", content="Round 1 content", round=1),
            MockMessage(role="proposer", agent="agent2", content="Round 2 content", round=2),
            MockMessage(role="proposer", agent="agent1", content="Round 10 content", round=10),
            MockMessage(role="proposer", agent="agent2", content="Round 11 content", round=11),
        ]
        debate_context = MockDebateContext(context_messages=messages)

        # Window starts at round 10, so only rounds 1-9 should be included
        result = manager._build_old_rounds_content(debate_context, window_start=10)

        assert "Round 1 - agent1 (proposer): Round 1 content" in result
        assert "Round 2 - agent2 (proposer): Round 2 content" in result
        assert "Round 10" not in result
        assert "Round 11" not in result


# =============================================================================
# Sliding Window Tests
# =============================================================================


class TestSlidingWindow:
    """Tests for sliding window context management."""

    @pytest.mark.asyncio
    async def test_sliding_window_first_rounds(self):
        """Test sliding window with early rounds (no compression needed)."""
        config = ExtendedDebateConfig(context_window_rounds=10)
        manager = RLMContextManager(config)

        messages = [MockMessage(round=i, content=f"Content {i}") for i in range(5)]
        debate_context = MockDebateContext(
            env=MockEnvironment(task="Test"),
            context_messages=messages,
        )

        result = await manager._apply_sliding_window(debate_context, round_num=5)

        # All rounds should be included (window_start = max(0, 5-10) = 0)
        for i in range(5):
            assert f"Content {i}" in result

    @pytest.mark.asyncio
    async def test_sliding_window_with_older_rounds(self):
        """Test sliding window includes compressed history for older rounds."""
        config = ExtendedDebateConfig(context_window_rounds=5)
        manager = RLMContextManager(config)
        manager._state.compressed_history = "Summary of rounds 1-10"

        messages = [MockMessage(round=i, content=f"Content {i}") for i in range(15)]
        debate_context = MockDebateContext(
            env=MockEnvironment(task="Test"),
            context_messages=messages,
        )

        result = await manager._apply_sliding_window(debate_context, round_num=15)

        # Should include compressed history
        assert "Summary of rounds 1-10" in result
        # Should include recent rounds (10-14)
        assert "Content 10" in result
        assert "Content 14" in result
        # Should not include old rounds at full detail
        assert "### Round 5 -" not in result


# =============================================================================
# RLM Integration Tests
# =============================================================================


class TestRLMIntegration:
    """Tests for RLM-based context compression."""

    @pytest.mark.asyncio
    async def test_get_rlm_lazy_loading(self):
        """Test RLM is lazy-loaded."""
        config = ExtendedDebateConfig(enable_rlm=True)
        manager = RLMContextManager(config)

        assert manager._rlm is None

        with patch("aragora.debate.extended_rounds.logger"):
            with patch.dict(
                "sys.modules",
                {"aragora.rlm": MagicMock(get_rlm=lambda: MagicMock())},
            ):
                rlm = await manager._get_rlm()
                # Should now be cached
                assert manager._rlm is not None

    @pytest.mark.asyncio
    async def test_get_rlm_disabled(self):
        """Test RLM not loaded when disabled."""
        config = ExtendedDebateConfig(enable_rlm=False)
        manager = RLMContextManager(config)

        rlm = await manager._get_rlm()
        assert rlm is None
        assert manager._rlm is None

    @pytest.mark.asyncio
    async def test_hierarchical_compression_fallback_no_rlm(self):
        """Test hierarchical compression falls back to sliding window when no RLM."""
        config = ExtendedDebateConfig(enable_rlm=False, context_window_rounds=5)
        manager = RLMContextManager(config)

        messages = [MockMessage(round=i, content=f"Content {i}") for i in range(20)]
        debate_context = MockDebateContext(
            env=MockEnvironment(task="Test"),
            context_messages=messages,
        )

        result = await manager._apply_hierarchical_compression(debate_context, round_num=20)

        # Should fall back to sliding window
        assert "Content 15" in result
        assert "Content 19" in result

    @pytest.mark.asyncio
    async def test_hierarchical_compression_with_rlm(self):
        """Test hierarchical compression using RLM."""
        config = ExtendedDebateConfig(enable_rlm=True, context_window_rounds=5)
        manager = RLMContextManager(config)

        # Create mock RLM
        mock_rlm = MagicMock()
        mock_result = MockRLMResult(
            answer="Compressed: Key points from early rounds",
            used_true_rlm=False,
            used_compression_fallback=True,
        )
        mock_rlm.compress_and_query = AsyncMock(return_value=mock_result)
        manager._rlm = mock_rlm

        messages = [MockMessage(round=i, content=f"Content {i}") for i in range(20)]
        debate_context = MockDebateContext(
            env=MockEnvironment(task="Test"),
            context_messages=messages,
        )

        result = await manager._apply_hierarchical_compression(debate_context, round_num=20)

        # Should have called compress_and_query
        mock_rlm.compress_and_query.assert_called_once()

        # Should have updated compressed history
        assert manager._state.compressed_history == "Compressed: Key points from early rounds"
        assert manager._state.compressions_performed == 1

    @pytest.mark.asyncio
    async def test_hierarchical_compression_timeout(self):
        """Test hierarchical compression handles timeout."""
        config = ExtendedDebateConfig(
            enable_rlm=True,
            compression_timeout=0.1,
            context_window_rounds=5,
        )
        manager = RLMContextManager(config)

        # Create mock RLM that times out
        async def slow_compress(*args, **kwargs):
            await asyncio.sleep(1.0)  # Longer than timeout
            return MockRLMResult()

        mock_rlm = MagicMock()
        mock_rlm.compress_and_query = slow_compress
        manager._rlm = mock_rlm

        messages = [MockMessage(round=i, content=f"Content {i}") for i in range(20)]
        debate_context = MockDebateContext(
            env=MockEnvironment(task="Test"),
            context_messages=messages,
        )

        # Should not raise, should fall back to sliding window
        with patch("aragora.debate.extended_rounds.logger") as mock_logger:
            result = await manager._apply_hierarchical_compression(debate_context, round_num=20)

            # Should log warning about timeout
            mock_logger.warning.assert_called()


# =============================================================================
# Prepare Round Context Tests
# =============================================================================


class TestPrepareRoundContext:
    """Tests for the main prepare_round_context method."""

    @pytest.mark.asyncio
    async def test_prepare_context_small_debate(self):
        """Test context preparation for small debate uses full context."""
        config = ExtendedDebateConfig(compression_threshold=10000)
        manager = RLMContextManager(config)

        messages = [MockMessage(round=i, content=f"Short {i}") for i in range(3)]
        debate_context = MockDebateContext(
            env=MockEnvironment(task="Small debate"),
            context_messages=messages,
        )

        result = await manager.prepare_round_context(debate_context, round_num=3)

        # Should use full context
        assert "## Task\nSmall debate" in result
        assert "Short 0" in result
        assert "Short 2" in result
        assert manager._state.current_round == 3

    @pytest.mark.asyncio
    async def test_prepare_context_medium_debate(self):
        """Test context preparation for medium debate uses sliding window."""
        config = ExtendedDebateConfig(
            compression_threshold=500,  # Low threshold to trigger compression
            context_window_rounds=5,
            soft_limit_rounds=50,
        )
        manager = RLMContextManager(config)

        # Create enough messages to exceed threshold
        messages = [
            MockMessage(round=i, content=f"Long content that takes up space {i}" * 10)
            for i in range(20)
        ]
        debate_context = MockDebateContext(
            env=MockEnvironment(task="Medium debate"),
            context_messages=messages,
        )

        result = await manager.prepare_round_context(debate_context, round_num=20)

        # Should use sliding window - recent rounds should be present
        assert "Long content that takes up space 15" in result
        assert manager._state.current_round == 20

    @pytest.mark.asyncio
    async def test_prepare_context_first_round(self):
        """Test context preparation for first round."""
        manager = RLMContextManager()
        debate_context = MockDebateContext(
            env=MockEnvironment(task="First round debate"),
            context_messages=[],
        )

        result = await manager.prepare_round_context(debate_context, round_num=0)

        assert "## Task\nFirst round debate" in result
        assert manager._state.current_round == 0

    @pytest.mark.asyncio
    async def test_prepare_context_updates_state(self):
        """Test that prepare_round_context updates state correctly."""
        manager = RLMContextManager()

        messages = [MockMessage(round=i, content=f"Content {i}") for i in range(5)]
        debate_context = MockDebateContext(
            env=MockEnvironment(task="Test"),
            context_messages=messages,
        )

        await manager.prepare_round_context(debate_context, round_num=5)

        assert manager._state.current_round == 5
        assert manager._state.total_tokens > 0


# =============================================================================
# Multi-Round Scenario Tests
# =============================================================================


class TestMultiRoundScenarios:
    """Tests for multi-round debate scenarios."""

    @pytest.mark.asyncio
    async def test_10_round_debate(self):
        """Test 10-round debate context preparation."""
        config = ExtendedDebateConfig(
            context_window_rounds=5,
            compression_threshold=1000,
        )
        manager = RLMContextManager(config)

        messages = [
            MockMessage(
                role="proposer", agent=f"agent{i % 3}", content=f"Proposal round {i}", round=i
            )
            for i in range(10)
        ]
        debate_context = MockDebateContext(
            env=MockEnvironment(task="10 round debate"),
            context_messages=messages,
        )

        result = await manager.prepare_round_context(debate_context, round_num=10)

        assert manager._state.current_round == 10
        assert "10 round debate" in result

    @pytest.mark.asyncio
    async def test_25_round_debate(self):
        """Test 25-round debate context preparation."""
        config = ExtendedDebateConfig(
            context_window_rounds=10,
            compression_threshold=500,  # Low to trigger compression
            soft_limit_rounds=50,
        )
        manager = RLMContextManager(config)
        manager._state.compressed_history = "Summary of rounds 1-15"

        messages = [
            MockMessage(
                role="proposer",
                agent=f"agent{i % 4}",
                content=f"Detailed proposal content for round {i} with arguments" * 3,
                round=i,
            )
            for i in range(25)
        ]
        debate_context = MockDebateContext(
            env=MockEnvironment(task="25 round complex debate"),
            context_messages=messages,
        )

        result = await manager.prepare_round_context(debate_context, round_num=25)

        assert manager._state.current_round == 25
        # Recent rounds should be present
        assert "round 24" in result.lower() or "round 20" in result.lower()

    @pytest.mark.asyncio
    async def test_50_plus_round_debate(self):
        """Test 50+ round debate with hierarchical compression."""
        config = ExtendedDebateConfig(
            max_rounds=100,
            soft_limit_rounds=50,
            context_window_rounds=10,
            compression_threshold=1000,
            enable_rlm=False,  # Test without RLM
        )
        manager = RLMContextManager(config)
        manager._state.compressed_history = "Extensive summary of rounds 1-45"

        messages = [
            MockMessage(
                role="proposer",
                agent=f"agent{i % 5}",
                content=f"Round {i} with extensive content" * 5,
                round=i,
            )
            for i in range(55)
        ]
        debate_context = MockDebateContext(
            env=MockEnvironment(task="Long debate requiring compression"),
            context_messages=messages,
        )

        result = await manager.prepare_round_context(debate_context, round_num=55)

        assert manager._state.current_round == 55
        # Should use hierarchical strategy but fall back to sliding window
        assert "round 54" in result.lower() or "round 50" in result.lower()

    @pytest.mark.asyncio
    async def test_progressive_context_growth(self):
        """Test context management over progressively growing debate."""
        config = ExtendedDebateConfig(
            context_window_rounds=5,
            compression_threshold=200,
        )
        manager = RLMContextManager(config)

        rounds_processed = []
        for round_num in range(1, 15):
            messages = [
                MockMessage(round=i, content=f"Content for round {i}" * 2) for i in range(round_num)
            ]
            debate_context = MockDebateContext(
                env=MockEnvironment(task="Progressive debate"),
                context_messages=messages,
            )

            result = await manager.prepare_round_context(debate_context, round_num=round_num)
            rounds_processed.append(manager._state.current_round)

        # Should have processed all rounds
        assert rounds_processed[-1] == 14


# =============================================================================
# Drill Down Context Tests
# =============================================================================


class TestDrillDownContext:
    """Tests for drill-down context querying."""

    @pytest.mark.asyncio
    async def test_drill_down_empty_history(self):
        """Test drill-down with no compressed history."""
        manager = RLMContextManager()

        result = await manager.get_drill_down_context("query")
        assert result == ""

    @pytest.mark.asyncio
    async def test_drill_down_no_rlm(self):
        """Test drill-down falls back to compressed history without RLM."""
        config = ExtendedDebateConfig(enable_rlm=False)
        manager = RLMContextManager(config)
        manager._state.compressed_history = "Historical summary content"

        result = await manager.get_drill_down_context("query")
        assert result == "Historical summary content"

    @pytest.mark.asyncio
    async def test_drill_down_with_rlm(self):
        """Test drill-down using RLM."""
        config = ExtendedDebateConfig(enable_rlm=True)
        manager = RLMContextManager(config)
        manager._state.compressed_history = "Historical summary content"

        # Create mock RLM
        mock_rlm = MagicMock()
        mock_result = MockRLMResult(answer="Relevant details for query")
        mock_rlm.compress_and_query = AsyncMock(return_value=mock_result)
        manager._rlm = mock_rlm

        result = await manager.get_drill_down_context("specific query")

        mock_rlm.compress_and_query.assert_called_once()
        assert result == "Relevant details for query"

    @pytest.mark.asyncio
    async def test_drill_down_rlm_error_fallback(self):
        """Test drill-down falls back on RLM error."""
        config = ExtendedDebateConfig(enable_rlm=True)
        manager = RLMContextManager(config)
        manager._state.compressed_history = "Fallback content"

        # Create mock RLM that raises
        mock_rlm = MagicMock()
        mock_rlm.compress_and_query = AsyncMock(side_effect=RuntimeError("RLM error"))
        manager._rlm = mock_rlm

        result = await manager.get_drill_down_context("query")
        assert result == "Fallback content"


# =============================================================================
# Statistics Tests
# =============================================================================


class TestStatistics:
    """Tests for statistics retrieval."""

    def test_get_statistics_initial(self):
        """Test statistics for fresh manager."""
        manager = RLMContextManager()
        stats = manager.get_statistics()

        assert stats["current_round"] == 0
        assert stats["total_tokens"] == 0
        assert stats["compressed_tokens"] == 0
        assert stats["compressions_performed"] == 0
        assert stats["tokens_saved"] == 0
        assert stats["compression_time_total"] == 0.0
        assert stats["compression_ratio"] == 0

    def test_get_statistics_after_compression(self):
        """Test statistics after compression operations."""
        manager = RLMContextManager()
        manager._state.current_round = 20
        manager._state.total_tokens = 10000
        manager._state.compressed_tokens = 3000
        manager._state.compressions_performed = 3
        manager._state.tokens_saved = 7000
        manager._state.compression_time_total = 5.5

        stats = manager.get_statistics()

        assert stats["current_round"] == 20
        assert stats["total_tokens"] == 10000
        assert stats["compressed_tokens"] == 3000
        assert stats["compressions_performed"] == 3
        assert stats["tokens_saved"] == 7000
        assert stats["compression_time_total"] == 5.5
        assert stats["compression_ratio"] == 0.3


# =============================================================================
# Factory Function Tests
# =============================================================================


class TestCreateExtendedConfig:
    """Tests for create_extended_config factory function."""

    def test_create_default_config(self):
        """Test creating default extended config."""
        config = create_extended_config()

        assert config.max_rounds == 100
        assert config.soft_limit_rounds == 50
        assert config.compression_threshold == 10000
        assert config.context_window_rounds == 10
        assert config.context_strategy == ContextStrategy.ADAPTIVE

    def test_create_custom_max_rounds(self):
        """Test creating config with custom max rounds."""
        config = create_extended_config(max_rounds=200)

        assert config.max_rounds == 200
        assert config.soft_limit_rounds == 50  # Not affected

    def test_create_aggressive_config(self):
        """Test creating aggressive compression config."""
        config = create_extended_config(max_rounds=100, aggressive=True)

        assert config.max_rounds == 100
        assert config.soft_limit_rounds == 50  # max_rounds // 2
        assert config.compression_threshold == 5000
        assert config.context_window_rounds == 5
        assert config.min_context_ratio == 0.2
        assert config.rlm_max_levels == 5
        assert config.context_strategy == ContextStrategy.HIERARCHICAL

    def test_create_aggressive_large_debate(self):
        """Test aggressive config for large debate."""
        config = create_extended_config(max_rounds=200, aggressive=True)

        assert config.max_rounds == 200
        assert config.soft_limit_rounds == 100  # max_rounds // 2


# =============================================================================
# Edge Cases Tests
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_context_with_missing_env(self):
        """Test context building with missing environment."""
        manager = RLMContextManager()

        # Create context with None env
        @dataclass
        class ContextWithNoneEnv:
            env: Any = None
            context_messages: list = field(default_factory=list)
            historical_context_cache: str = ""

        debate_context = ContextWithNoneEnv()

        result = manager._build_full_context(debate_context)
        # Should not crash, just skip task
        assert isinstance(result, str)

    @pytest.mark.asyncio
    async def test_message_with_missing_attributes(self):
        """Test handling messages with missing attributes."""
        manager = RLMContextManager()

        # Create a mock message without some attributes
        class MinimalMessage:
            pass

        msg = MinimalMessage()
        debate_context = MockDebateContext(context_messages=[msg])

        result = manager._build_full_context(debate_context)
        # Should use defaults ("unknown") instead of crashing
        assert "unknown" in result

    @pytest.mark.asyncio
    async def test_concurrent_context_preparation(self):
        """Test concurrent access to context preparation."""
        manager = RLMContextManager()

        debate_context = MockDebateContext(
            env=MockEnvironment(task="Concurrent test"),
            context_messages=[MockMessage(round=i) for i in range(5)],
        )

        # Run multiple preparations concurrently
        results = await asyncio.gather(
            manager.prepare_round_context(debate_context, round_num=5),
            manager.prepare_round_context(debate_context, round_num=5),
            manager.prepare_round_context(debate_context, round_num=5),
        )

        # All should complete successfully
        assert len(results) == 3
        for result in results:
            assert "Concurrent test" in result

    @pytest.mark.asyncio
    async def test_zero_window_rounds(self):
        """Test with zero context window rounds."""
        config = ExtendedDebateConfig(context_window_rounds=0)
        manager = RLMContextManager(config)

        messages = [MockMessage(round=i, content=f"Content {i}") for i in range(10)]
        debate_context = MockDebateContext(
            env=MockEnvironment(task="Zero window"),
            context_messages=messages,
        )

        # Should not crash with zero window
        result = await manager._apply_sliding_window(debate_context, round_num=10)
        assert "Zero window" in result

    @pytest.mark.asyncio
    async def test_round_zero_handling(self):
        """Test handling of round 0."""
        manager = RLMContextManager()
        debate_context = MockDebateContext(
            env=MockEnvironment(task="Round zero test"),
            context_messages=[],
        )

        result = await manager.prepare_round_context(debate_context, round_num=0)

        assert manager._state.current_round == 0
        assert "Round zero test" in result

    def test_compression_ratio_zero_tokens(self):
        """Test compression ratio calculation with zero tokens."""
        manager = RLMContextManager()
        manager._state.total_tokens = 0
        manager._state.compressed_tokens = 0

        stats = manager.get_statistics()
        assert stats["compression_ratio"] == 0  # Should not divide by zero

    @pytest.mark.asyncio
    async def test_rlm_import_error_handling(self):
        """Test graceful handling of RLM import error."""
        config = ExtendedDebateConfig(enable_rlm=True)
        manager = RLMContextManager(config)

        # Mock the import to raise ImportError
        with patch.dict("sys.modules", {"aragora.rlm": None}):
            with patch("aragora.debate.extended_rounds.logger"):
                rlm = await manager._get_rlm()
                # Should handle import error gracefully
                # Note: The actual behavior depends on how the import is structured


# =============================================================================
# Context Overflow Tests
# =============================================================================


class TestContextOverflow:
    """Tests for context overflow handling."""

    @pytest.mark.asyncio
    async def test_large_context_triggers_compression(self):
        """Test that large context triggers compression."""
        config = ExtendedDebateConfig(
            compression_threshold=1000,
            context_window_rounds=5,
        )
        manager = RLMContextManager(config)

        # Create messages that exceed threshold
        messages = [
            MockMessage(round=i, content="Very long content " * 100)  # ~1800 chars each
            for i in range(20)
        ]
        debate_context = MockDebateContext(
            env=MockEnvironment(task="Large context test"),
            context_messages=messages,
        )

        await manager.prepare_round_context(debate_context, round_num=20)

        # Strategy should have been SLIDING_WINDOW or HIERARCHICAL
        assert manager._state.total_tokens > config.compression_threshold

    @pytest.mark.asyncio
    async def test_historical_context_truncation(self):
        """Test that historical context is truncated."""
        manager = RLMContextManager()

        long_history = "Historical context " * 1000  # > 2000 chars
        debate_context = MockDebateContext(
            env=MockEnvironment(task="History test"),
            historical_context_cache=long_history,
            context_messages=[],
        )

        result = manager._build_full_context(debate_context)

        # Should be truncated to 2000 chars
        assert len(result.split("## Historical Context\n")[1].split("\n\n")[0]) <= 2000


# =============================================================================
# Token Tracking Tests
# =============================================================================


class TestTokenTracking:
    """Tests for token tracking and savings calculation."""

    @pytest.mark.asyncio
    async def test_tokens_saved_calculation(self):
        """Test tokens saved calculation after compression."""
        config = ExtendedDebateConfig(enable_rlm=True, context_window_rounds=5)
        manager = RLMContextManager(config)

        # Create mock RLM that returns a short compression
        mock_rlm = MagicMock()
        # Original content will be ~1000+ chars, compressed to ~50
        mock_result = MockRLMResult(answer="Short compressed summary")
        mock_rlm.compress_and_query = AsyncMock(return_value=mock_result)
        manager._rlm = mock_rlm

        messages = [
            MockMessage(round=i, content="Content that needs compression " * 10) for i in range(20)
        ]
        debate_context = MockDebateContext(
            env=MockEnvironment(task="Token tracking test"),
            context_messages=messages,
        )

        await manager._apply_hierarchical_compression(debate_context, round_num=20)

        # Should have saved tokens
        assert manager._state.tokens_saved > 0
        assert manager._state.compressions_performed == 1


# =============================================================================
# Additional Comprehensive Tests
# =============================================================================


class TestRoundExtensionLogic:
    """Additional tests for round extension logic."""

    def test_max_rounds_boundary(self):
        """Test configuration at max rounds boundary."""
        config = ExtendedDebateConfig(max_rounds=50)
        manager = RLMContextManager(config)

        # Manager should accept max_rounds config
        assert manager.config.max_rounds == 50

    def test_soft_limit_trigger(self):
        """Test soft limit triggers hierarchical strategy."""
        config = ExtendedDebateConfig(
            context_strategy=ContextStrategy.ADAPTIVE,
            compression_threshold=100,  # Very low to ensure threshold is exceeded
            soft_limit_rounds=10,
        )
        manager = RLMContextManager(config)

        # Above soft limit should use HIERARCHICAL
        result = manager._select_strategy(total_tokens=500, round_num=15)
        assert result == ContextStrategy.HIERARCHICAL

        # Below soft limit should use SLIDING_WINDOW
        result = manager._select_strategy(total_tokens=500, round_num=8)
        assert result == ContextStrategy.SLIDING_WINDOW

    def test_all_strategies_configured(self):
        """Test all context strategies can be configured."""
        for strategy in ContextStrategy:
            config = ExtendedDebateConfig(context_strategy=strategy)
            manager = RLMContextManager(config)
            assert manager.config.context_strategy == strategy


class TestContextCompressionSummarization:
    """Additional tests for context compression and summarization."""

    @pytest.mark.asyncio
    async def test_compression_preserves_task(self):
        """Test that compression always preserves the task."""
        config = ExtendedDebateConfig(
            context_window_rounds=2,
            compression_threshold=100,
        )
        manager = RLMContextManager(config)
        manager._state.compressed_history = "Compressed history"

        messages = [MockMessage(round=i, content=f"Content {i}" * 50) for i in range(20)]
        debate_context = MockDebateContext(
            env=MockEnvironment(task="Critical task to preserve"),
            context_messages=messages,
        )

        result = await manager._apply_sliding_window(debate_context, round_num=20)

        # Task should always be present
        assert "Critical task to preserve" in result

    @pytest.mark.asyncio
    async def test_compression_statistics_accuracy(self):
        """Test compression statistics are accurately tracked."""
        config = ExtendedDebateConfig(enable_rlm=True, context_window_rounds=3)
        manager = RLMContextManager(config)

        mock_rlm = MagicMock()
        mock_result = MockRLMResult(answer="Tiny")  # Very short compression
        mock_rlm.compress_and_query = AsyncMock(return_value=mock_result)
        manager._rlm = mock_rlm

        messages = [
            MockMessage(round=i, content="Verbose content " * 50)  # Long messages
            for i in range(15)
        ]
        debate_context = MockDebateContext(
            env=MockEnvironment(task="Stats test"),
            context_messages=messages,
        )

        initial_compressions = manager._state.compressions_performed
        await manager._apply_hierarchical_compression(debate_context, round_num=14)

        # Verify compression was counted
        assert manager._state.compressions_performed == initial_compressions + 1
        # Verify time was tracked
        assert manager._state.compression_time_total > 0

    @pytest.mark.asyncio
    async def test_rlm_result_empty_answer_handling(self):
        """Test handling of RLM result with empty answer."""
        config = ExtendedDebateConfig(enable_rlm=True, context_window_rounds=3)
        manager = RLMContextManager(config)

        mock_rlm = MagicMock()
        # Return result with empty answer
        mock_result = MockRLMResult(answer="")
        mock_rlm.compress_and_query = AsyncMock(return_value=mock_result)
        manager._rlm = mock_rlm

        messages = [MockMessage(round=i, content=f"Content {i}" * 20) for i in range(15)]
        debate_context = MockDebateContext(
            env=MockEnvironment(task="Empty answer test"),
            context_messages=messages,
        )

        # Should not crash with empty answer
        result = await manager._apply_hierarchical_compression(debate_context, round_num=14)
        assert result is not None

    @pytest.mark.asyncio
    async def test_rlm_result_none_handling(self):
        """Test handling of None RLM result."""
        config = ExtendedDebateConfig(enable_rlm=True, context_window_rounds=3)
        manager = RLMContextManager(config)

        mock_rlm = MagicMock()
        mock_rlm.compress_and_query = AsyncMock(return_value=None)
        manager._rlm = mock_rlm

        messages = [MockMessage(round=i, content=f"Content {i}" * 20) for i in range(15)]
        debate_context = MockDebateContext(
            env=MockEnvironment(task="None result test"),
            context_messages=messages,
        )

        # Should not crash with None result
        result = await manager._apply_hierarchical_compression(debate_context, round_num=14)
        assert result is not None


class TestTokenLimitManagement:
    """Additional tests for token limit management."""

    def test_min_context_ratio_config(self):
        """Test min_context_ratio configuration."""
        config = ExtendedDebateConfig(min_context_ratio=0.5)
        assert config.min_context_ratio == 0.5

        config2 = ExtendedDebateConfig(min_context_ratio=0.1)
        assert config2.min_context_ratio == 0.1

    def test_token_estimation_unicode(self):
        """Test token estimation with unicode characters."""
        manager = RLMContextManager()

        # Unicode characters (typically more bytes but still 1 char)
        unicode_text = "\u4e2d\u6587\u6587\u672c" * 100  # Chinese characters
        estimated = manager._estimate_tokens(unicode_text)
        assert estimated == 100  # 400 chars / 4

    def test_token_estimation_whitespace(self):
        """Test token estimation with various whitespace."""
        manager = RLMContextManager()

        whitespace_text = "word   word\n\nword\tword"
        estimated = manager._estimate_tokens(whitespace_text)
        assert estimated > 0

    @pytest.mark.asyncio
    async def test_very_high_token_count(self):
        """Test handling of very high token counts."""
        config = ExtendedDebateConfig(
            compression_threshold=100,
            context_window_rounds=2,
            soft_limit_rounds=5,
        )
        manager = RLMContextManager(config)

        # Create massive context
        messages = [
            MockMessage(round=i, content="x" * 10000)  # 2500 tokens each
            for i in range(100)
        ]
        debate_context = MockDebateContext(
            env=MockEnvironment(task="High token test"),
            context_messages=messages,
        )

        # Should handle without crashing
        result = await manager.prepare_round_context(debate_context, round_num=99)
        assert result is not None
        assert manager._state.total_tokens > 100000  # Very high token count


class TestMultiRoundExtendedScenarios:
    """Additional multi-round scenario tests."""

    @pytest.mark.asyncio
    async def test_exactly_50_rounds(self):
        """Test exactly 50 rounds - at soft limit boundary."""
        config = ExtendedDebateConfig(
            soft_limit_rounds=50,
            context_window_rounds=10,
            compression_threshold=500,
        )
        manager = RLMContextManager(config)
        manager._state.compressed_history = "Rounds 1-40 summary"

        messages = [MockMessage(round=i, content=f"Round {i} content" * 5) for i in range(50)]
        debate_context = MockDebateContext(
            env=MockEnvironment(task="50 rounds"),
            context_messages=messages,
        )

        result = await manager.prepare_round_context(debate_context, round_num=50)

        assert manager._state.current_round == 50
        assert result is not None

    @pytest.mark.asyncio
    async def test_100_round_debate_stress(self):
        """Stress test with 100 rounds."""
        config = ExtendedDebateConfig(
            max_rounds=100,
            soft_limit_rounds=50,
            context_window_rounds=10,
            compression_threshold=1000,
            enable_rlm=False,
        )
        manager = RLMContextManager(config)
        manager._state.compressed_history = "Extensive summary of rounds 1-90"

        messages = [
            MockMessage(
                round=i,
                agent=f"agent{i % 5}",
                content=f"Round {i}: Extended discussion point " * 3,
            )
            for i in range(100)
        ]
        debate_context = MockDebateContext(
            env=MockEnvironment(task="100 round stress test"),
            context_messages=messages,
        )

        result = await manager.prepare_round_context(debate_context, round_num=100)

        assert manager._state.current_round == 100
        assert result is not None
        # Should include recent rounds
        assert "Round 99" in result or "Round 95" in result

    @pytest.mark.asyncio
    async def test_sparse_round_numbers(self):
        """Test with non-contiguous round numbers."""
        config = ExtendedDebateConfig(context_window_rounds=5)
        manager = RLMContextManager(config)

        # Create messages with gaps in round numbers
        messages = [
            MockMessage(round=1, content="Round 1"),
            MockMessage(round=5, content="Round 5"),
            MockMessage(round=10, content="Round 10"),
            MockMessage(round=20, content="Round 20"),
        ]
        debate_context = MockDebateContext(
            env=MockEnvironment(task="Sparse rounds"),
            context_messages=messages,
        )

        result = await manager.prepare_round_context(debate_context, round_num=20)

        assert manager._state.current_round == 20


class TestEdgeCasesExtended:
    """Extended edge case tests."""

    @pytest.mark.asyncio
    async def test_empty_message_content(self):
        """Test handling of messages with empty content."""
        manager = RLMContextManager()

        messages = [
            MockMessage(round=1, content=""),
            MockMessage(round=2, content="Non-empty"),
            MockMessage(round=3, content=""),
        ]
        debate_context = MockDebateContext(
            env=MockEnvironment(task="Empty content test"),
            context_messages=messages,
        )

        result = manager._build_full_context(debate_context)
        assert "Non-empty" in result

    @pytest.mark.asyncio
    async def test_special_characters_in_content(self):
        """Test handling of special characters in content."""
        manager = RLMContextManager()

        messages = [
            MockMessage(round=1, content="Content with <html> tags"),
            MockMessage(round=2, content="Content with 'quotes' and \"double quotes\""),
            MockMessage(round=3, content="Content with $pecial ch@racters!"),
        ]
        debate_context = MockDebateContext(
            env=MockEnvironment(task="Special chars test"),
            context_messages=messages,
        )

        result = manager._build_full_context(debate_context)
        assert "<html>" in result
        assert "quotes" in result
        assert "$pecial" in result

    @pytest.mark.asyncio
    async def test_very_long_agent_names(self):
        """Test handling of very long agent names."""
        manager = RLMContextManager()

        long_name = "agent_" + "x" * 500
        messages = [MockMessage(round=1, agent=long_name, content="Content")]
        debate_context = MockDebateContext(
            env=MockEnvironment(task="Long agent name test"),
            context_messages=messages,
        )

        result = manager._build_full_context(debate_context)
        assert long_name in result

    @pytest.mark.asyncio
    async def test_first_round_edge_case(self):
        """Test first round with specific edge conditions."""
        manager = RLMContextManager()

        # First round with one message
        debate_context = MockDebateContext(
            env=MockEnvironment(task="First round"),
            context_messages=[MockMessage(round=0, content="Initial proposal")],
        )

        result = await manager.prepare_round_context(debate_context, round_num=0)

        assert manager._state.current_round == 0
        assert "First round" in result
        assert "Initial proposal" in result

    @pytest.mark.asyncio
    async def test_window_larger_than_messages(self):
        """Test when context window is larger than message count."""
        config = ExtendedDebateConfig(context_window_rounds=100)  # Very large window
        manager = RLMContextManager(config)

        messages = [MockMessage(round=i, content=f"Content {i}") for i in range(5)]
        debate_context = MockDebateContext(
            env=MockEnvironment(task="Small debate"),
            context_messages=messages,
        )

        result = await manager._apply_sliding_window(debate_context, round_num=5)

        # All messages should be included
        for i in range(5):
            assert f"Content {i}" in result

    def test_reset_preserves_config(self):
        """Test that reset preserves configuration."""
        config = ExtendedDebateConfig(max_rounds=200, compression_threshold=5000)
        manager = RLMContextManager(config)

        manager._state.current_round = 50
        manager._state.compressions_performed = 10

        manager.reset()

        # State should be reset
        assert manager._state.current_round == 0
        assert manager._state.compressions_performed == 0

        # Config should be preserved
        assert manager.config.max_rounds == 200
        assert manager.config.compression_threshold == 5000

    @pytest.mark.asyncio
    async def test_multiple_sequential_compressions(self):
        """Test multiple sequential compression operations."""
        config = ExtendedDebateConfig(
            enable_rlm=True,
            context_window_rounds=3,
        )
        manager = RLMContextManager(config)

        mock_rlm = MagicMock()
        call_count = 0

        async def track_calls(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return MockRLMResult(answer=f"Compression {call_count}")

        mock_rlm.compress_and_query = track_calls
        manager._rlm = mock_rlm

        for round_num in range(10, 30, 5):
            messages = [MockMessage(round=i, content=f"Content {i}" * 20) for i in range(round_num)]
            debate_context = MockDebateContext(
                env=MockEnvironment(task="Sequential compressions"),
                context_messages=messages,
            )
            await manager._apply_hierarchical_compression(debate_context, round_num=round_num)

        # Should have performed multiple compressions
        assert manager._state.compressions_performed > 1


class TestRLMContextManagerLevelHints:
    """Test level hints for drill-down queries."""

    @pytest.mark.asyncio
    async def test_drill_down_abstract_level(self):
        """Test drill-down with ABSTRACT level hint."""
        manager = RLMContextManager()
        manager._state.compressed_history = "Summary content"

        mock_rlm = MagicMock()
        mock_rlm.compress_and_query = AsyncMock(
            return_value=MockRLMResult(answer="Abstract answer")
        )
        manager._rlm = mock_rlm

        result = await manager.get_drill_down_context("query", level="ABSTRACT")
        assert result == "Abstract answer"

    @pytest.mark.asyncio
    async def test_drill_down_full_level(self):
        """Test drill-down with FULL level hint."""
        manager = RLMContextManager()
        manager._state.compressed_history = "Detailed content"

        mock_rlm = MagicMock()
        mock_rlm.compress_and_query = AsyncMock(return_value=MockRLMResult(answer="Full answer"))
        manager._rlm = mock_rlm

        result = await manager.get_drill_down_context("query", level="FULL")
        assert result == "Full answer"


class TestRLMTrueRLMFlag:
    """Test handling of used_true_rlm flag."""

    @pytest.mark.asyncio
    async def test_true_rlm_used_logging(self):
        """Test logging when TRUE RLM is used."""
        config = ExtendedDebateConfig(enable_rlm=True, context_window_rounds=3)
        manager = RLMContextManager(config)

        mock_rlm = MagicMock()
        mock_result = MockRLMResult(
            answer="Compressed content",
            used_true_rlm=True,
            used_compression_fallback=False,
        )
        mock_rlm.compress_and_query = AsyncMock(return_value=mock_result)
        manager._rlm = mock_rlm

        messages = [MockMessage(round=i, content=f"Content {i}" * 20) for i in range(15)]
        debate_context = MockDebateContext(
            env=MockEnvironment(task="TRUE RLM test"),
            context_messages=messages,
        )

        with patch("aragora.debate.extended_rounds.logger") as mock_logger:
            await manager._apply_hierarchical_compression(debate_context, round_num=14)
            # Should log that TRUE RLM was used
            assert any("TRUE RLM" in str(call) for call in mock_logger.debug.call_args_list)

    @pytest.mark.asyncio
    async def test_compression_fallback_logging(self):
        """Test logging when compression fallback is used."""
        config = ExtendedDebateConfig(enable_rlm=True, context_window_rounds=3)
        manager = RLMContextManager(config)

        mock_rlm = MagicMock()
        mock_result = MockRLMResult(
            answer="Compressed content",
            used_true_rlm=False,
            used_compression_fallback=True,
        )
        mock_rlm.compress_and_query = AsyncMock(return_value=mock_result)
        manager._rlm = mock_rlm

        messages = [MockMessage(round=i, content=f"Content {i}" * 20) for i in range(15)]
        debate_context = MockDebateContext(
            env=MockEnvironment(task="Fallback test"),
            context_messages=messages,
        )

        with patch("aragora.debate.extended_rounds.logger") as mock_logger:
            await manager._apply_hierarchical_compression(debate_context, round_num=14)
            # Should log that compression fallback was used
            assert any(
                "compression fallback" in str(call) for call in mock_logger.debug.call_args_list
            )


class TestRoundSummaryExtended:
    """Extended tests for RoundSummary dataclass."""

    def test_round_summary_with_compressed_at(self):
        """Test RoundSummary with compressed_at timestamp."""
        import time

        current_time = time.time()
        summary = RoundSummary(
            round_num=10,
            proposals={"agent1": "proposal"},
            critiques=["critique1"],
            key_points=["key1", "key2"],
            consensus_progress=0.9,
            token_count=750,
            compressed_at=current_time,
        )

        assert summary.compressed_at == current_time
        assert summary.round_num == 10
        assert summary.consensus_progress == 0.9

    def test_round_summary_empty_collections(self):
        """Test RoundSummary with empty collections."""
        summary = RoundSummary(
            round_num=1,
            proposals={},
            critiques=[],
            key_points=[],
            consensus_progress=0.0,
            token_count=0,
        )

        assert len(summary.proposals) == 0
        assert len(summary.critiques) == 0
        assert len(summary.key_points) == 0


class TestExtendedContextStateMutability:
    """Test ExtendedContextState mutability and independence."""

    def test_state_round_summaries_independence(self):
        """Test that round_summaries are independent between instances."""
        state1 = ExtendedContextState()
        state2 = ExtendedContextState()

        state1.round_summaries[1] = RoundSummary(
            round_num=1,
            proposals={"a": "b"},
            critiques=[],
            key_points=[],
            consensus_progress=0.5,
            token_count=100,
        )

        assert 1 not in state2.round_summaries
        assert len(state2.round_summaries) == 0

    def test_state_rlm_context_none_default(self):
        """Test that rlm_context defaults to None."""
        state = ExtendedContextState()
        assert state.rlm_context is None

    def test_state_all_counters_zero(self):
        """Test all statistical counters start at zero."""
        state = ExtendedContextState()

        assert state.current_round == 0
        assert state.total_tokens == 0
        assert state.compressed_tokens == 0
        assert state.compressions_performed == 0
        assert state.tokens_saved == 0
        assert state.compression_time_total == 0.0
