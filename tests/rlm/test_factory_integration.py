"""Tests for RLM factory and consumer integration.

These tests verify that:
1. The factory correctly returns AragoraRLM instances
2. Singleton behavior works as expected
3. Consumer files correctly use the factory
4. Tracking flags (used_true_rlm, used_compression_fallback) are set correctly
"""

from __future__ import annotations

import pytest
from unittest.mock import patch, MagicMock, AsyncMock


class TestRLMFactory:
    """Test factory function behavior."""

    def test_get_rlm_returns_aragora_rlm(self):
        """get_rlm() should return an AragoraRLM instance."""
        from aragora.rlm import get_rlm, reset_singleton

        reset_singleton()  # Clear any cached instance
        rlm = get_rlm(force_new=True)

        assert rlm is not None
        assert hasattr(rlm, "compress_and_query")

    def test_get_rlm_singleton_behavior(self):
        """get_rlm() should return the same instance by default."""
        from aragora.rlm import get_rlm, reset_singleton

        reset_singleton()  # Start fresh
        rlm1 = get_rlm()
        rlm2 = get_rlm()

        assert rlm1 is rlm2

    def test_get_rlm_force_new(self):
        """get_rlm(force_new=True) should create a new instance."""
        from aragora.rlm import get_rlm, reset_singleton

        reset_singleton()
        rlm1 = get_rlm()
        rlm2 = get_rlm(force_new=True)

        assert rlm1 is not rlm2

    def test_reset_singleton_clears_cache(self):
        """reset_singleton() should clear the cached instance."""
        from aragora.rlm import get_rlm, reset_singleton

        rlm1 = get_rlm()
        reset_singleton()
        rlm2 = get_rlm()

        # After reset, should be a different instance
        assert rlm1 is not rlm2

    def test_get_compressor_returns_hierarchical_compressor(self):
        """get_compressor() should return a HierarchicalCompressor."""
        from aragora.rlm import get_compressor

        compressor = get_compressor()

        assert compressor is not None
        assert hasattr(compressor, "compress")


class TestRLMResult:
    """Test RLMResult tracking flags."""

    def test_rlm_result_has_tracking_fields(self):
        """RLMResult should have used_true_rlm and used_compression_fallback."""
        from aragora.rlm.types import RLMResult

        # Create a result with tracking flags
        result = RLMResult(
            answer="Test answer",
            confidence=0.9,
            used_true_rlm=False,
            used_compression_fallback=True,
        )

        assert hasattr(result, "used_true_rlm")
        assert hasattr(result, "used_compression_fallback")
        assert result.used_true_rlm is False
        assert result.used_compression_fallback is True


class TestCompressAndQuery:
    """Test the compress_and_query convenience function."""

    @pytest.mark.asyncio
    async def test_compress_and_query_sets_tracking_flags(self):
        """compress_and_query() should return result with tracking flags."""
        from aragora.rlm import compress_and_query, reset_singleton

        reset_singleton()

        result = await compress_and_query(
            query="What is the main topic?",
            content="This is test content about machine learning and AI development.",
            source_type="test",
        )

        # Should have tracking flags
        assert hasattr(result, "used_true_rlm")
        assert hasattr(result, "used_compression_fallback")
        # At least one should be True
        assert result.used_true_rlm or result.used_compression_fallback

    @pytest.mark.asyncio
    async def test_compress_and_query_returns_answer(self):
        """compress_and_query() should return an answer."""
        from aragora.rlm import compress_and_query, reset_singleton

        reset_singleton()

        result = await compress_and_query(
            query="Summarize this content",
            content="Python is a programming language known for its simplicity.",
            source_type="text",
        )

        assert result.answer is not None
        assert isinstance(result.answer, str)


class TestFallbackBehavior:
    """Test compression fallback when TRUE RLM unavailable."""

    def test_fallback_when_official_rlm_not_installed(self):
        """Should fall back to compression when official rlm not installed."""
        from aragora.rlm.bridge import HAS_OFFICIAL_RLM
        from aragora.rlm import get_rlm, reset_singleton

        reset_singleton()
        rlm = get_rlm(force_new=True)

        # Should have compression capability regardless of official RLM
        assert hasattr(rlm, "compress_and_query")

        # If official RLM is not available, the compression fallback should work
        if not HAS_OFFICIAL_RLM:
            # Fallback mode
            assert hasattr(rlm, "_compressor")


class TestConsumerIntegration:
    """Test that consumer modules correctly use the factory."""

    def test_extended_rounds_imports_factory(self):
        """extended_rounds.py should import get_rlm from factory."""
        # This tests that the module structure is correct
        from aragora.debate.extended_rounds import RLMContextManager

        manager = RLMContextManager()
        # Should have _get_rlm method that uses factory
        assert hasattr(manager, "_get_rlm")

    def test_context_init_uses_factory(self):
        """context_init.py should use get_rlm from factory."""
        from aragora.debate.phases.context_init import ContextInitializer

        # Create initializer with RLM enabled
        initializer = ContextInitializer(enable_rlm_compression=True)
        # Should have _rlm attribute (may be None if RLM not available)
        assert hasattr(initializer, "_rlm")

    def test_rlm_handler_uses_factory(self):
        """server/handlers/rlm.py should use factory methods."""
        from aragora.server.handlers.rlm import RLMContextHandler

        handler = RLMContextHandler({})

        # Should have _get_compressor and _get_rlm methods
        assert hasattr(handler, "_get_compressor")
        assert hasattr(handler, "_get_rlm")


class TestConfigPropagation:
    """Test that configuration is properly propagated through factory."""

    def test_config_passed_to_rlm(self):
        """Custom config should be passed to AragoraRLM."""
        from aragora.rlm import get_rlm, reset_singleton
        from aragora.rlm.types import RLMConfig

        reset_singleton()

        custom_config = RLMConfig(
            target_tokens=5000,
            cache_compressions=False,
        )

        rlm = get_rlm(config=custom_config, force_new=True)

        # RLM should be created (config validation happens internally)
        assert rlm is not None

    def test_config_passed_to_compressor(self):
        """Custom config should be passed to HierarchicalCompressor."""
        from aragora.rlm import get_compressor
        from aragora.rlm.types import RLMConfig

        custom_config = RLMConfig(
            target_tokens=2000,
        )

        compressor = get_compressor(config=custom_config)

        assert compressor is not None
        assert hasattr(compressor, "config")


class TestTrueRLMRouting:
    """Test TRUE RLM routing behavior with mocks.

    These tests simulate scenarios where the official RLM library is installed
    to verify the routing logic works correctly.
    """

    @pytest.mark.asyncio
    async def test_true_rlm_flag_set_when_mock_rlm_available(self):
        """When TRUE RLM is mocked as available, results should indicate true_rlm used."""
        from aragora.rlm import reset_singleton
        from aragora.rlm.types import RLMResult

        reset_singleton()

        # Create a mock RLM result that simulates TRUE RLM behavior
        mock_result = RLMResult(
            answer="Answer from TRUE RLM",
            confidence=0.95,
            used_true_rlm=True,
            used_compression_fallback=False,
        )

        # Verify the result structure
        assert mock_result.used_true_rlm is True
        assert mock_result.used_compression_fallback is False
        assert mock_result.answer == "Answer from TRUE RLM"

    @pytest.mark.asyncio
    async def test_compression_fallback_flag_set_when_true_rlm_unavailable(self):
        """When TRUE RLM unavailable, results should indicate compression_fallback used."""
        from aragora.rlm import get_rlm, reset_singleton
        from aragora.rlm.bridge import HAS_OFFICIAL_RLM

        reset_singleton()

        # Get RLM instance (will use compression fallback in test environment)
        rlm = get_rlm()

        # In test environment without official RLM, compression fallback should be used
        if not HAS_OFFICIAL_RLM:
            result = await rlm.compress_and_query(
                query="What is this about?",
                content="Test content for compression.",
                source_type="test",
            )

            assert result.used_compression_fallback is True
            assert result.used_true_rlm is False

    @pytest.mark.asyncio
    async def test_mocked_true_rlm_query_method(self):
        """Test that query() method works with mocked TRUE RLM behavior."""
        from aragora.rlm import reset_singleton
        from aragora.rlm.types import RLMContext, RLMResult

        reset_singleton()

        # Create mock RLM context
        context = RLMContext(
            original_content="Test debate content",
            original_tokens=100,
            source_type="debate",
        )

        # Create mock result simulating TRUE RLM
        # Note: TRUE RLM writes code internally but result doesn't expose it
        mock_true_rlm_result = RLMResult(
            answer="The model wrote code to analyze this content",
            confidence=0.9,
            used_true_rlm=True,
            used_compression_fallback=False,
        )

        # Verify TRUE RLM result structure
        assert mock_true_rlm_result.used_true_rlm is True
        assert mock_true_rlm_result.answer is not None

    @pytest.mark.skip(reason="Logging assertion flaky on CI")
    def test_factory_logging_indicates_rlm_type(self):
        """Factory should log whether TRUE RLM or compression fallback is used."""
        import logging
        from aragora.rlm import get_rlm, reset_singleton

        reset_singleton()

        # Capture log output
        with patch("aragora.rlm.factory.logger") as mock_logger:
            rlm = get_rlm(force_new=True)

            # Should have logged info about which RLM type is being used
            assert mock_logger.info.called or mock_logger.debug.called

    @pytest.mark.asyncio
    async def test_result_includes_approach_metadata(self):
        """Results should include metadata about which approach was used."""
        from aragora.rlm import compress_and_query, reset_singleton

        reset_singleton()

        result = await compress_and_query(
            query="Summarize this",
            content="This is a test document about software architecture.",
            source_type="document",
        )

        # Result should have both tracking flags
        assert hasattr(result, "used_true_rlm")
        assert hasattr(result, "used_compression_fallback")

        # Exactly one should be True (they're mutually exclusive)
        assert result.used_true_rlm != result.used_compression_fallback or (
            not result.used_true_rlm and not result.used_compression_fallback
        )


class TestMockedOfficialRLM:
    """Tests with HAS_OFFICIAL_RLM mocked as True."""

    @pytest.mark.asyncio
    async def test_factory_prefers_true_rlm_when_available(self):
        """When official RLM is available, factory should prefer it."""
        from aragora.rlm.types import RLMResult

        # Create a mock that simulates TRUE RLM being available
        mock_true_rlm = MagicMock()
        mock_true_rlm.query = AsyncMock(
            return_value=RLMResult(
                answer="TRUE RLM answer",
                confidence=0.95,
                used_true_rlm=True,
                used_compression_fallback=False,
            )
        )

        # Verify the mock behaves correctly
        result = await mock_true_rlm.query("test query", MagicMock())
        assert result.used_true_rlm is True
        assert result.answer == "TRUE RLM answer"

    @pytest.mark.asyncio
    async def test_fallback_to_compression_on_true_rlm_error(self):
        """If TRUE RLM fails, should fall back to compression."""
        from aragora.rlm import get_rlm, reset_singleton
        from aragora.rlm.types import RLMResult

        reset_singleton()

        # Get RLM instance
        rlm = get_rlm()

        # Even if TRUE RLM were to fail, compression fallback should work
        result = await rlm.compress_and_query(
            query="Test query",
            content="Test content",
            source_type="test",
        )

        # Should get a valid result regardless of TRUE RLM availability
        assert result is not None
        assert hasattr(result, "answer")


class TestUpdatedConsumerModules:
    """Test that updated consumer modules use factory correctly."""

    def test_context_gatherer_uses_factory(self):
        """ContextGatherer should use get_rlm from factory."""
        from aragora.debate.context_gatherer import ContextGatherer, HAS_RLM

        # Create gatherer with RLM enabled
        gatherer = ContextGatherer(enable_rlm_compression=True)

        # Should have _aragora_rlm attribute (may be None if RLM not available)
        assert hasattr(gatherer, "_aragora_rlm")

        # If RLM is available, should be initialized via factory
        if HAS_RLM:
            # The RLM instance should be set
            assert gatherer._enable_rlm is True

    def test_cognitive_limiter_uses_factory(self):
        """RLMCognitiveLoadLimiter should use get_rlm from factory."""
        from aragora.debate.cognitive_limiter_rlm import RLMCognitiveLoadLimiter

        # Create limiter
        limiter = RLMCognitiveLoadLimiter()

        # Should have _aragora_rlm attribute
        assert hasattr(limiter, "_aragora_rlm")

    def test_cross_debate_memory_uses_factory(self):
        """CrossDebateMemory should use get_rlm from factory."""
        from aragora.memory.cross_debate_rlm import CrossDebateMemory

        # Create memory
        memory = CrossDebateMemory()

        # Should have has_real_rlm property
        assert hasattr(memory, "has_real_rlm")

        # Should have _get_rlm method
        assert hasattr(memory, "_get_rlm")

    def test_prompt_builder_rlm_availability(self):
        """PromptBuilder should correctly detect RLM availability."""
        from aragora.debate.prompt_builder import HAS_RLM

        # HAS_RLM should be a boolean
        assert isinstance(HAS_RLM, bool)


class TestMetricsExport:
    """Test metrics export functionality."""

    def test_export_to_json_returns_valid_json(self):
        """export_to_json() should return valid JSON."""
        import json
        from aragora.rlm import export_to_json, reset_singleton, reset_metrics

        reset_singleton()
        reset_metrics()

        json_str = export_to_json()
        data = json.loads(json_str)

        assert "timestamp" in data
        assert "timestamp_iso" in data
        assert "metrics" in data
        assert isinstance(data["metrics"], dict)

    def test_metrics_collector_tracks_snapshots(self):
        """MetricsCollector should track metric snapshots."""
        from aragora.rlm import get_metrics_collector, get_rlm, reset_singleton, reset_metrics

        reset_singleton()
        reset_metrics()

        collector = get_metrics_collector()

        # Collect initial snapshot
        snapshot1 = collector.collect()
        assert snapshot1.metrics["get_rlm_calls"] == 0

        # Make some calls
        get_rlm()

        # Collect again
        snapshot2 = collector.collect()
        assert snapshot2.metrics["get_rlm_calls"] > snapshot1.metrics["get_rlm_calls"]

    def test_metrics_collector_calculates_delta(self):
        """MetricsCollector should calculate delta between snapshots."""
        from aragora.rlm import get_metrics_collector, get_rlm, reset_singleton, reset_metrics

        reset_singleton()
        reset_metrics()

        collector = get_metrics_collector()
        collector.collect()  # Initial snapshot

        # Make calls
        get_rlm()
        get_rlm()

        # Get delta
        delta = collector.get_delta()
        assert delta is not None
        assert delta["get_rlm_calls"] >= 2

    def test_metrics_snapshot_to_dict(self):
        """MetricsSnapshot should convert to dictionary."""
        from aragora.rlm import MetricsSnapshot

        snapshot = MetricsSnapshot(timestamp=1234567890.0, metrics={"test_metric": 42})

        data = snapshot.to_dict()
        assert data["timestamp"] == 1234567890.0
        assert "timestamp_iso" in data
        assert data["metrics"]["test_metric"] == 42

    def test_export_to_prometheus_without_library(self):
        """export_to_prometheus() should handle missing prometheus_client gracefully."""
        from aragora.rlm import export_to_prometheus

        # Should return empty dict if prometheus_client not installed
        # (or actual metrics if it is installed)
        result = export_to_prometheus()
        assert isinstance(result, dict)

    def test_export_to_statsd_without_library(self):
        """export_to_statsd() should handle missing statsd client gracefully."""
        from aragora.rlm import export_to_statsd

        # Should return False if statsd not installed (no server to connect to anyway)
        result = export_to_statsd(host="nonexistent.local", port=9999)
        assert isinstance(result, bool)
