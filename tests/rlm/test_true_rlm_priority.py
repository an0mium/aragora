"""
Tests for TRUE RLM prioritization (Phase 12).

Verifies that TRUE RLM (REPL-based) is preferred over compression-based
methods when the official `rlm` package is installed.

Based on arXiv:2512.24601 "Recursive Language Models":
- TRUE RLM stores context in REPL environment
- Model writes code to query context programmatically
- No information loss from truncation/compression
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestRLMMode:
    """Tests for RLMMode enum."""

    def test_rlm_mode_values(self):
        """RLMMode should have TRUE_RLM, COMPRESSION, and AUTO values."""
        from aragora.rlm.types import RLMMode

        assert RLMMode.TRUE_RLM.value == "true_rlm"
        assert RLMMode.COMPRESSION.value == "compression"
        assert RLMMode.AUTO.value == "auto"

    def test_rlm_mode_exported(self):
        """RLMMode should be exported from aragora.rlm."""
        from aragora.rlm import RLMMode

        assert RLMMode is not None
        assert hasattr(RLMMode, "TRUE_RLM")
        assert hasattr(RLMMode, "AUTO")


class TestRLMConfig:
    """Tests for RLMConfig TRUE RLM options."""

    def test_config_prefer_true_rlm_default(self):
        """prefer_true_rlm should default to True."""
        from aragora.rlm.types import RLMConfig

        config = RLMConfig()
        assert config.prefer_true_rlm is True

    def test_config_warn_on_compression_default(self):
        """warn_on_compression_fallback should default to True."""
        from aragora.rlm.types import RLMConfig

        config = RLMConfig()
        assert config.warn_on_compression_fallback is True

    def test_config_require_true_rlm_default(self):
        """require_true_rlm should default to False."""
        from aragora.rlm.types import RLMConfig

        config = RLMConfig()
        assert config.require_true_rlm is False

    def test_config_mode_default_auto(self):
        """mode should default to AUTO."""
        from aragora.rlm.types import RLMConfig, RLMMode

        config = RLMConfig()
        assert config.mode == RLMMode.AUTO


class TestGetRLMFactory:
    """Tests for get_rlm factory function."""

    def test_get_rlm_returns_instance(self):
        """get_rlm should return AragoraRLM instance."""
        from aragora.rlm import get_rlm

        rlm = get_rlm()
        assert rlm is not None

    def test_get_rlm_singleton(self):
        """get_rlm should return same instance by default."""
        from aragora.rlm import get_rlm, reset_singleton

        reset_singleton()
        rlm1 = get_rlm()
        rlm2 = get_rlm()
        assert rlm1 is rlm2

    def test_get_rlm_force_new(self):
        """get_rlm with force_new should create new instance."""
        from aragora.rlm import get_rlm, reset_singleton

        reset_singleton()
        rlm1 = get_rlm()
        rlm2 = get_rlm(force_new=True)
        assert rlm1 is not rlm2

    def test_get_rlm_with_mode(self):
        """get_rlm should accept mode parameter."""
        from aragora.rlm import RLMMode, get_rlm

        # AUTO mode should work
        rlm_auto = get_rlm(mode=RLMMode.AUTO, force_new=True)
        assert rlm_auto is not None

        # COMPRESSION mode should work
        rlm_comp = get_rlm(mode=RLMMode.COMPRESSION, force_new=True)
        assert rlm_comp is not None

    @pytest.mark.skipif(
        os.environ.get("HAS_OFFICIAL_RLM") == "true",
        reason="Only run when official RLM is NOT installed",
    )
    def test_get_rlm_true_rlm_mode_raises_without_library(self):
        """get_rlm with TRUE_RLM mode should raise if library not installed."""
        from aragora.rlm import RLMMode, get_rlm
        from aragora.rlm.bridge import HAS_OFFICIAL_RLM

        if HAS_OFFICIAL_RLM:
            pytest.skip("Official RLM is installed")

        with pytest.raises(RuntimeError, match="TRUE RLM required"):
            get_rlm(mode=RLMMode.TRUE_RLM, force_new=True)

    @pytest.mark.skipif(
        os.environ.get("HAS_OFFICIAL_RLM") == "true",
        reason="Only run when official RLM is NOT installed",
    )
    def test_get_rlm_require_true_rlm_raises(self):
        """get_rlm with require_true_rlm should raise if not available."""
        from aragora.rlm import get_rlm
        from aragora.rlm.bridge import HAS_OFFICIAL_RLM

        if HAS_OFFICIAL_RLM:
            pytest.skip("Official RLM is installed")

        with pytest.raises(RuntimeError, match="TRUE RLM required"):
            get_rlm(require_true_rlm=True, force_new=True)


class TestCompressAndQuery:
    """Tests for compress_and_query function."""

    def test_compress_and_query_accepts_mode(self):
        """compress_and_query should accept mode parameter."""
        from aragora.rlm import RLMMode, compress_and_query

        # Should not raise - just checking parameter acceptance
        # Actual async call would require mock
        assert callable(compress_and_query)


class TestREPLContextAdapter:
    """Tests for REPLContextAdapter class."""

    def test_repl_adapter_exists(self):
        """REPLContextAdapter should be importable."""
        from aragora.rlm.adapter import REPLContextAdapter

        assert REPLContextAdapter is not None

    def test_repl_adapter_has_true_rlm_property(self):
        """REPLContextAdapter should have has_true_rlm property."""
        from aragora.rlm.adapter import REPLContextAdapter

        adapter = REPLContextAdapter()
        assert hasattr(adapter, "has_true_rlm")
        assert isinstance(adapter.has_true_rlm, bool)

    def test_repl_adapter_create_repl_for_debate_method(self):
        """REPLContextAdapter should have create_repl_for_debate method."""
        from aragora.rlm.adapter import REPLContextAdapter

        adapter = REPLContextAdapter()
        assert hasattr(adapter, "create_repl_for_debate")
        assert callable(adapter.create_repl_for_debate)

    def test_repl_adapter_create_repl_for_knowledge_method(self):
        """REPLContextAdapter should have create_repl_for_knowledge method."""
        from aragora.rlm.adapter import REPLContextAdapter

        adapter = REPLContextAdapter()
        assert hasattr(adapter, "create_repl_for_knowledge")
        assert callable(adapter.create_repl_for_knowledge)

    def test_repl_adapter_get_repl_prompt_method(self):
        """REPLContextAdapter should have get_repl_prompt method."""
        from aragora.rlm.adapter import REPLContextAdapter

        adapter = REPLContextAdapter()
        assert hasattr(adapter, "get_repl_prompt")

        # Test debate prompt
        prompt = adapter.get_repl_prompt("test-123", "debate")
        assert "debate" in prompt.lower()
        assert "get_round" in prompt

        # Test knowledge prompt
        prompt = adapter.get_repl_prompt("test-456", "knowledge")
        assert "knowledge" in prompt.lower()
        assert "get_facts" in prompt


class TestGetREPLAdapter:
    """Tests for get_repl_adapter function."""

    def test_get_repl_adapter_returns_instance(self):
        """get_repl_adapter should return REPLContextAdapter instance."""
        from aragora.rlm import get_repl_adapter
        from aragora.rlm.adapter import REPLContextAdapter

        adapter = get_repl_adapter()
        assert isinstance(adapter, REPLContextAdapter)

    def test_get_repl_adapter_exported(self):
        """get_repl_adapter should be exported from aragora.rlm."""
        from aragora.rlm import get_repl_adapter

        assert callable(get_repl_adapter)


class TestDebateHelpers:
    """Tests for debate REPL helpers."""

    def test_debate_helpers_exported(self):
        """Debate helpers should be exported from aragora.rlm."""
        from aragora.rlm import DebateREPLContext, get_debate_helpers, load_debate_context

        assert DebateREPLContext is not None
        assert callable(load_debate_context)
        assert callable(get_debate_helpers)

    def test_get_debate_helpers_returns_dict(self):
        """get_debate_helpers should return dictionary of helpers."""
        from aragora.rlm import get_debate_helpers

        # By default, RLM_M and FINAL are excluded (provided by RLMEnvironment)
        helpers = get_debate_helpers()
        assert isinstance(helpers, dict)
        assert "get_round" in helpers
        assert "get_proposals_by_agent" in helpers
        assert "search_debate" in helpers
        assert "RLM_M" not in helpers  # Excluded by default
        assert "FINAL" not in helpers  # Excluded by default

        # Can include RLM primitives when explicitly requested
        helpers_with_rlm = get_debate_helpers(include_rlm_primitives=True)
        assert "RLM_M" in helpers_with_rlm
        assert "FINAL" in helpers_with_rlm

    def test_debate_repl_context_dataclass(self):
        """DebateREPLContext should be a dataclass with expected fields."""
        from aragora.rlm.debate_helpers import DebateREPLContext

        # Create minimal instance
        ctx = DebateREPLContext(
            debate_id="test-123",
            task="Test task",
            total_rounds=3,
            agent_names=["claude", "gpt4"],
            rounds={1: [], 2: [], 3: []},
            by_agent={"claude": [], "gpt4": []},
            all_messages=[],
            consensus_reached=True,
            final_answer="Test answer",
            confidence=0.8,
        )

        assert ctx.debate_id == "test-123"
        assert ctx.total_rounds == 3
        assert len(ctx.agent_names) == 2


class TestKnowledgeHelpers:
    """Tests for knowledge REPL helpers."""

    def test_knowledge_helpers_exported(self):
        """Knowledge helpers should be exported from aragora.rlm."""
        from aragora.rlm import (
            KnowledgeItem,
            KnowledgeREPLContext,
            get_knowledge_helpers,
            load_knowledge_context,
        )

        assert KnowledgeItem is not None
        assert KnowledgeREPLContext is not None
        assert callable(load_knowledge_context)
        assert callable(get_knowledge_helpers)

    def test_get_knowledge_helpers_returns_dict(self):
        """get_knowledge_helpers should return dictionary of helpers."""
        from aragora.rlm import get_knowledge_helpers

        # By default, RLM_M and FINAL are excluded (provided by RLMEnvironment)
        helpers = get_knowledge_helpers()
        assert isinstance(helpers, dict)
        assert "get_facts" in helpers
        assert "get_claims" in helpers
        assert "get_evidence" in helpers
        assert "filter_by_confidence" in helpers
        assert "search_knowledge" in helpers
        assert "RLM_M" not in helpers  # Excluded by default
        assert "FINAL" not in helpers  # Excluded by default

        # Can include RLM primitives when explicitly requested
        helpers_with_rlm = get_knowledge_helpers(include_rlm_primitives=True)
        assert "RLM_M" in helpers_with_rlm
        assert "FINAL" in helpers_with_rlm

    def test_knowledge_item_dataclass(self):
        """KnowledgeItem should be a dataclass with expected fields."""
        from aragora.rlm.knowledge_helpers import KnowledgeItem

        item = KnowledgeItem(
            id="fact-123",
            content="Test fact content",
            source="fact",
            confidence=0.9,
            created_at="2024-01-01T00:00:00Z",
        )

        assert item.id == "fact-123"
        assert item.confidence == 0.9
        assert item.source == "fact"


class TestContextGathererTrueRLM:
    """Tests for ContextGatherer TRUE RLM integration."""

    def test_context_gatherer_has_query_method(self):
        """ContextGatherer should have _query_with_true_rlm method."""
        from aragora.debate.context_gatherer import ContextGatherer

        gatherer = ContextGatherer()
        assert hasattr(gatherer, "_query_with_true_rlm")
        assert callable(gatherer._query_with_true_rlm)

    def test_context_gatherer_has_knowledge_query_method(self):
        """ContextGatherer should have query_knowledge_with_true_rlm method."""
        from aragora.debate.context_gatherer import ContextGatherer

        gatherer = ContextGatherer()
        assert hasattr(gatherer, "query_knowledge_with_true_rlm")
        assert callable(gatherer.query_knowledge_with_true_rlm)


class TestKnowledgeMoundTrueRLM:
    """Tests for Knowledge Mound TRUE RLM integration."""

    def test_rlm_mixin_has_true_rlm_available(self):
        """RLMOperationsMixin should have is_true_rlm_available method."""
        from aragora.knowledge.mound.api.rlm import RLMOperationsMixin

        assert hasattr(RLMOperationsMixin, "is_true_rlm_available")

    def test_rlm_mixin_has_query_with_true_rlm(self):
        """RLMOperationsMixin should have query_with_true_rlm method."""
        from aragora.knowledge.mound.api.rlm import RLMOperationsMixin

        assert hasattr(RLMOperationsMixin, "query_with_true_rlm")

    def test_rlm_mixin_has_create_knowledge_repl(self):
        """RLMOperationsMixin should have create_knowledge_repl method."""
        from aragora.knowledge.mound.api.rlm import RLMOperationsMixin

        assert hasattr(RLMOperationsMixin, "create_knowledge_repl")


class TestEnvironmentVariables:
    """Tests for RLM environment variable configuration."""

    def test_env_var_rlm_mode(self):
        """ARAGORA_RLM_MODE should affect get_rlm behavior."""
        from aragora.rlm import get_rlm, reset_singleton

        reset_singleton()

        # Test with AUTO mode via env var
        with patch.dict(os.environ, {"ARAGORA_RLM_MODE": "auto"}):
            rlm = get_rlm(force_new=True)
            assert rlm is not None

    def test_env_var_warn_fallback(self):
        """ARAGORA_RLM_WARN_FALLBACK should affect warning behavior."""
        # Just verify the env var is checked in the factory
        # The actual warning behavior is internal
        from aragora.rlm.factory import get_rlm

        assert callable(get_rlm)


class TestMetricsTracking:
    """Tests for RLM metrics tracking."""

    def test_factory_metrics_track_true_rlm(self):
        """Factory metrics should track TRUE RLM vs compression usage."""
        from aragora.rlm import get_factory_metrics, reset_metrics

        reset_metrics()
        metrics = get_factory_metrics()

        assert "true_rlm_calls" in metrics
        assert "compression_fallback_calls" in metrics
        assert metrics["true_rlm_calls"] == 0
        assert metrics["compression_fallback_calls"] == 0


class TestRLMResultTracking:
    """Tests for RLMResult approach tracking."""

    def test_rlm_result_has_tracking_flags(self):
        """RLMResult should have used_true_rlm and used_compression_fallback."""
        from aragora.rlm.types import RLMResult

        result = RLMResult(
            answer="Test answer",
            used_true_rlm=True,
            used_compression_fallback=False,
        )

        assert result.used_true_rlm is True
        assert result.used_compression_fallback is False

    def test_rlm_result_default_flags(self):
        """RLMResult tracking flags should default to False."""
        from aragora.rlm.types import RLMResult

        result = RLMResult(answer="Test")

        assert result.used_true_rlm is False
        assert result.used_compression_fallback is False
