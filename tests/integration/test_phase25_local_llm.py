"""Integration tests for Phase 25: Local LLM Factory Integration.

Tests the local LLM detection and fallback chain functionality.
"""

import pytest
from unittest.mock import patch, AsyncMock, MagicMock


class TestLocalLLMDetection:
    """Test AgentRegistry local LLM detection."""

    def test_detect_local_agents_returns_list(self):
        """detect_local_agents() should return a list."""
        from aragora.agents.registry import AgentRegistry

        # Mock the detector to avoid actual network calls
        with patch("aragora.agents.registry.LocalLLMDetector") as mock_detector:
            mock_instance = MagicMock()
            mock_detector.return_value = mock_instance

            # Create a mock status with empty servers
            mock_status = MagicMock()
            mock_status.servers = []
            mock_instance.detect_all = AsyncMock(return_value=mock_status)

            result = AgentRegistry.detect_local_agents()
            assert isinstance(result, list)

    def test_get_local_status_returns_dict(self):
        """get_local_status() should return a dict with expected keys."""
        from aragora.agents.registry import AgentRegistry

        with patch("aragora.agents.registry.LocalLLMDetector") as mock_detector:
            mock_instance = MagicMock()
            mock_detector.return_value = mock_instance

            mock_status = MagicMock()
            mock_status.servers = []
            mock_status.any_available = False
            mock_status.total_models = 0
            mock_status.recommended_server = None
            mock_status.recommended_model = None
            mock_status.get_available_agents.return_value = []
            mock_instance.detect_all = AsyncMock(return_value=mock_status)

            result = AgentRegistry.get_local_status()

            assert isinstance(result, dict)
            assert "any_available" in result
            assert "servers" in result
            assert "recommended_server" in result


class TestLocalFallbackChain:
    """Test fallback chain with local LLM support."""

    def test_get_local_fallback_providers_no_servers(self):
        """get_local_fallback_providers() returns empty when no servers available."""
        from aragora.agents.fallback import get_local_fallback_providers

        with patch("aragora.agents.fallback.AgentRegistry") as mock_registry:
            mock_registry.detect_local_agents.return_value = []

            result = get_local_fallback_providers()
            assert result == []

    def test_get_local_fallback_providers_with_servers(self):
        """get_local_fallback_providers() returns available server names."""
        from aragora.agents.fallback import get_local_fallback_providers

        with patch("aragora.agents.fallback.AgentRegistry") as mock_registry:
            mock_registry.detect_local_agents.return_value = [
                {"name": "ollama", "available": True},
                {"name": "lm-studio", "available": False},
            ]

            result = get_local_fallback_providers()
            assert result == ["ollama"]

    def test_build_fallback_chain_without_local(self):
        """build_fallback_chain_with_local() without local returns original chain."""
        from aragora.agents.fallback import build_fallback_chain_with_local

        chain = ["openai", "anthropic"]
        result = build_fallback_chain_with_local(chain, include_local=False)
        assert result == chain

    def test_build_fallback_chain_with_local_priority(self):
        """build_fallback_chain_with_local() with priority inserts before OpenRouter."""
        from aragora.agents.fallback import build_fallback_chain_with_local

        with patch("aragora.agents.fallback.get_local_fallback_providers") as mock_providers:
            mock_providers.return_value = ["ollama"]

            chain = ["openai", "openrouter", "anthropic"]
            result = build_fallback_chain_with_local(
                chain,
                include_local=True,
                local_priority=True,
            )

            # Local should come before openrouter
            assert result.index("ollama") < result.index("openrouter")

    def test_is_local_llm_available_false(self):
        """is_local_llm_available() returns False when no servers."""
        from aragora.agents.fallback import is_local_llm_available

        with patch("aragora.agents.fallback.AgentRegistry") as mock_registry:
            mock_registry.get_local_status.return_value = {"any_available": False}

            result = is_local_llm_available()
            assert result is False


class TestAgentSettingsLocal:
    """Test AgentSettings local fallback configuration."""

    def test_local_fallback_defaults(self):
        """AgentSettings should have local fallback disabled by default."""
        from aragora.config.settings import AgentSettings

        settings = AgentSettings()
        assert settings.local_fallback_enabled is False
        assert settings.local_fallback_priority is False

    def test_local_fallback_can_be_enabled(self):
        """AgentSettings local fallback can be enabled via env."""
        import os
        from aragora.config.settings import AgentSettings

        # Test with environment variable
        with patch.dict(os.environ, {"ARAGORA_LOCAL_FALLBACK_ENABLED": "true"}):
            settings = AgentSettings()
            assert settings.local_fallback_enabled is True
