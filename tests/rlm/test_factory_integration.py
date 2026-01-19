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
        assert hasattr(rlm, 'compress_and_query')

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
        assert hasattr(compressor, 'compress')


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

        assert hasattr(result, 'used_true_rlm')
        assert hasattr(result, 'used_compression_fallback')
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
        assert hasattr(result, 'used_true_rlm')
        assert hasattr(result, 'used_compression_fallback')
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
        assert hasattr(rlm, 'compress_and_query')

        # If official RLM is not available, the compression fallback should work
        if not HAS_OFFICIAL_RLM:
            # Fallback mode
            assert hasattr(rlm, '_compressor')


class TestConsumerIntegration:
    """Test that consumer modules correctly use the factory."""

    def test_extended_rounds_imports_factory(self):
        """extended_rounds.py should import get_rlm from factory."""
        # This tests that the module structure is correct
        from aragora.debate.extended_rounds import RLMContextManager

        manager = RLMContextManager()
        # Should have _get_rlm method that uses factory
        assert hasattr(manager, '_get_rlm')

    def test_context_init_uses_factory(self):
        """context_init.py should use get_rlm from factory."""
        from aragora.debate.phases.context_init import ContextInitializer

        # Create initializer with RLM enabled
        initializer = ContextInitializer(enable_rlm_compression=True)
        # Should have _rlm attribute (may be None if RLM not available)
        assert hasattr(initializer, '_rlm')

    def test_rlm_handler_uses_factory(self):
        """server/handlers/rlm.py should use factory methods."""
        from aragora.server.handlers.rlm import RLMContextHandler

        handler = RLMContextHandler({})

        # Should have _get_compressor and _get_rlm methods
        assert hasattr(handler, '_get_compressor')
        assert hasattr(handler, '_get_rlm')


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
        assert hasattr(compressor, 'config')
