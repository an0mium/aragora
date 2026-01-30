"""
Tests for Specialist Agent Factory.

Tests the specialist agent factory functionality including:
- AgentConfig dataclass
- SpecialistAgentInfo dataclass
- SpecialistAgentFactory initialization and methods
- Vertical-specific agent creation
- Fallback to base models
- Task-based vertical inference
- Global factory instance management
"""

from __future__ import annotations

import inspect
from dataclasses import fields
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# =============================================================================
# AgentConfig Tests
# =============================================================================


class TestAgentConfigDataclass:
    """Test AgentConfig dataclass."""

    def test_init_minimal(self):
        """Test minimal initialization with only required vertical."""
        from aragora.agents.specialist_factory import AgentConfig
        from aragora.training.specialist_models import Vertical

        config = AgentConfig(vertical=Vertical.LEGAL)

        assert config.vertical == Vertical.LEGAL
        assert config.org_id is None
        assert config.fallback_to_base is True
        assert config.prefer_speed is False
        assert config.require_specialist is False
        assert config.extra_context == ""
        assert config.temperature == 0.7
        assert config.max_tokens == 4096
        assert config.metadata == {}

    def test_init_with_all_fields(self):
        """Test initialization with all fields."""
        from aragora.agents.specialist_factory import AgentConfig
        from aragora.training.specialist_models import Vertical

        config = AgentConfig(
            vertical=Vertical.SECURITY,
            org_id="org-123",
            fallback_to_base=False,
            prefer_speed=True,
            require_specialist=True,
            extra_context="Additional security context",
            temperature=0.5,
            max_tokens=2048,
            metadata={"key": "value"},
        )

        assert config.vertical == Vertical.SECURITY
        assert config.org_id == "org-123"
        assert config.fallback_to_base is False
        assert config.prefer_speed is True
        assert config.require_specialist is True
        assert config.extra_context == "Additional security context"
        assert config.temperature == 0.5
        assert config.max_tokens == 2048
        assert config.metadata == {"key": "value"}

    def test_has_expected_fields(self):
        """Test AgentConfig has all expected fields."""
        from aragora.agents.specialist_factory import AgentConfig

        field_names = {f.name for f in fields(AgentConfig)}

        expected = {
            "vertical",
            "org_id",
            "fallback_to_base",
            "prefer_speed",
            "require_specialist",
            "extra_context",
            "temperature",
            "max_tokens",
            "metadata",
        }

        assert field_names == expected


class TestAgentConfigVerticals:
    """Test AgentConfig with different verticals."""

    def test_legal_vertical(self):
        """Test AgentConfig with legal vertical."""
        from aragora.agents.specialist_factory import AgentConfig
        from aragora.training.specialist_models import Vertical

        config = AgentConfig(vertical=Vertical.LEGAL)

        assert config.vertical == Vertical.LEGAL
        assert config.vertical.value == "legal"

    def test_healthcare_vertical(self):
        """Test AgentConfig with healthcare vertical."""
        from aragora.agents.specialist_factory import AgentConfig
        from aragora.training.specialist_models import Vertical

        config = AgentConfig(vertical=Vertical.HEALTHCARE)

        assert config.vertical == Vertical.HEALTHCARE

    def test_all_verticals(self):
        """Test AgentConfig works with all verticals."""
        from aragora.agents.specialist_factory import AgentConfig
        from aragora.training.specialist_models import Vertical

        for vertical in Vertical:
            config = AgentConfig(vertical=vertical)
            assert config.vertical == vertical


# =============================================================================
# SpecialistAgentInfo Tests
# =============================================================================


class TestSpecialistAgentInfoDataclass:
    """Test SpecialistAgentInfo dataclass."""

    def test_has_expected_fields(self):
        """Test SpecialistAgentInfo has all expected fields."""
        from aragora.agents.specialist_factory import SpecialistAgentInfo

        field_names = {f.name for f in fields(SpecialistAgentInfo)}

        expected = {
            "agent",
            "specialist_model",
            "is_specialist",
            "vertical",
            "model_id",
            "base_model",
            "adapter_name",
        }

        assert field_names == expected

    def test_specialist_info_with_specialist(self):
        """Test SpecialistAgentInfo for specialist agent."""
        from aragora.agents.specialist_factory import SpecialistAgentInfo
        from aragora.training.specialist_models import Vertical, SpecialistModel, TrainingStatus

        mock_agent = MagicMock()
        specialist_model = SpecialistModel(
            id="sm_legal_abc123",
            base_model="llama-3.3-70b",
            adapter_name="legal-expert",
            vertical=Vertical.LEGAL,
            org_id=None,
            status=TrainingStatus.READY,
        )

        info = SpecialistAgentInfo(
            agent=mock_agent,
            specialist_model=specialist_model,
            is_specialist=True,
            vertical=Vertical.LEGAL,
            model_id="sm_legal_abc123",
            base_model="llama-3.3-70b",
            adapter_name="legal-expert",
        )

        assert info.is_specialist is True
        assert info.specialist_model is specialist_model
        assert info.adapter_name == "legal-expert"

    def test_specialist_info_with_base_model(self):
        """Test SpecialistAgentInfo for base model fallback."""
        from aragora.agents.specialist_factory import SpecialistAgentInfo
        from aragora.training.specialist_models import Vertical

        mock_agent = MagicMock()

        info = SpecialistAgentInfo(
            agent=mock_agent,
            specialist_model=None,
            is_specialist=False,
            vertical=Vertical.GENERAL,
            model_id="meta-llama/llama-3.3-70b-instruct",
            base_model="llama-3.3-70b",
            adapter_name=None,
        )

        assert info.is_specialist is False
        assert info.specialist_model is None
        assert info.adapter_name is None


# =============================================================================
# SpecialistAgentFactory Initialization Tests
# =============================================================================


class TestSpecialistAgentFactoryInit:
    """Test SpecialistAgentFactory initialization."""

    def test_init_default(self):
        """Test default initialization."""
        from aragora.agents.specialist_factory import SpecialistAgentFactory

        with patch("aragora.agents.specialist_factory.get_specialist_registry") as mock_get:
            mock_registry = MagicMock()
            mock_get.return_value = mock_registry

            factory = SpecialistAgentFactory()

            assert factory._registry is mock_registry
            mock_get.assert_called_once()

    def test_init_with_registry(self):
        """Test initialization with provided registry."""
        from aragora.agents.specialist_factory import SpecialistAgentFactory
        from aragora.training.specialist_models import SpecialistModelRegistry

        registry = SpecialistModelRegistry()
        factory = SpecialistAgentFactory(registry=registry)

        assert factory._registry is registry

    def test_has_base_models_mapping(self):
        """Test factory has base models mapping."""
        from aragora.agents.specialist_factory import SpecialistAgentFactory
        from aragora.training.specialist_models import SpecialistModelRegistry

        factory = SpecialistAgentFactory(registry=SpecialistModelRegistry())

        assert "llama-3.3-70b" in factory._base_models
        assert "qwen-2.5-72b" in factory._base_models
        assert "deepseek-v3" in factory._base_models

    def test_has_vertical_prompts(self):
        """Test factory has vertical-specific prompts."""
        from aragora.agents.specialist_factory import SpecialistAgentFactory
        from aragora.training.specialist_models import SpecialistModelRegistry, Vertical

        factory = SpecialistAgentFactory(registry=SpecialistModelRegistry())

        # Check prompts exist for all verticals
        for vertical in Vertical:
            assert vertical in factory._vertical_prompts


# =============================================================================
# Vertical Prompt Tests
# =============================================================================


class TestVerticalPrompts:
    """Test vertical-specific system prompts."""

    @pytest.fixture
    def factory(self):
        """Create factory for testing."""
        from aragora.agents.specialist_factory import SpecialistAgentFactory
        from aragora.training.specialist_models import SpecialistModelRegistry

        return SpecialistAgentFactory(registry=SpecialistModelRegistry())

    def test_legal_prompt_contains_expertise(self, factory):
        """Test legal prompt contains relevant expertise."""
        from aragora.training.specialist_models import Vertical

        prompt = factory._vertical_prompts[Vertical.LEGAL]

        assert "legal" in prompt.lower()
        assert "contract" in prompt.lower()
        assert "compliance" in prompt.lower()

    def test_healthcare_prompt_contains_expertise(self, factory):
        """Test healthcare prompt contains relevant expertise."""
        from aragora.training.specialist_models import Vertical

        prompt = factory._vertical_prompts[Vertical.HEALTHCARE]

        assert "healthcare" in prompt.lower() or "medical" in prompt.lower()
        assert "hipaa" in prompt.lower()
        assert "patient" in prompt.lower()

    def test_security_prompt_contains_expertise(self, factory):
        """Test security prompt contains relevant expertise."""
        from aragora.training.specialist_models import Vertical

        prompt = factory._vertical_prompts[Vertical.SECURITY]

        assert "security" in prompt.lower()
        assert "vulnerability" in prompt.lower()
        assert "owasp" in prompt.lower()

    def test_software_prompt_contains_expertise(self, factory):
        """Test software prompt contains relevant expertise."""
        from aragora.training.specialist_models import Vertical

        prompt = factory._vertical_prompts[Vertical.SOFTWARE]

        assert "software" in prompt.lower()
        assert "code" in prompt.lower() or "architecture" in prompt.lower()


# =============================================================================
# System Prompt Building Tests
# =============================================================================


class TestGetSystemPrompt:
    """Test _get_system_prompt method."""

    @pytest.fixture
    def factory(self):
        """Create factory for testing."""
        from aragora.agents.specialist_factory import SpecialistAgentFactory
        from aragora.training.specialist_models import SpecialistModelRegistry

        return SpecialistAgentFactory(registry=SpecialistModelRegistry())

    def test_get_prompt_for_vertical(self, factory):
        """Test getting prompt for a specific vertical."""
        from aragora.agents.specialist_factory import AgentConfig
        from aragora.training.specialist_models import Vertical

        config = AgentConfig(vertical=Vertical.LEGAL)
        prompt = factory._get_system_prompt(config)

        assert "legal" in prompt.lower()

    def test_get_prompt_with_extra_context(self, factory):
        """Test prompt includes extra context."""
        from aragora.agents.specialist_factory import AgentConfig
        from aragora.training.specialist_models import Vertical

        extra = "Focus on GDPR compliance specifically."
        config = AgentConfig(vertical=Vertical.LEGAL, extra_context=extra)
        prompt = factory._get_system_prompt(config)

        assert extra in prompt
        assert "legal" in prompt.lower()

    def test_get_prompt_falls_back_to_general(self, factory):
        """Test unknown vertical falls back to general prompt."""
        from aragora.agents.specialist_factory import AgentConfig
        from aragora.training.specialist_models import Vertical

        config = AgentConfig(vertical=Vertical.GENERAL)
        prompt = factory._get_system_prompt(config)

        assert "assistant" in prompt.lower() or "multi-domain" in prompt.lower()


# =============================================================================
# Agent Creation Tests
# =============================================================================


class TestCreateAgent:
    """Test agent creation methods."""

    @pytest.fixture
    def factory(self):
        """Create factory for testing."""
        from aragora.agents.specialist_factory import SpecialistAgentFactory
        from aragora.training.specialist_models import SpecialistModelRegistry

        return SpecialistAgentFactory(registry=SpecialistModelRegistry())

    @pytest.mark.asyncio
    async def test_create_falls_back_to_base_agent(self, factory):
        """Test create falls back to base agent when no specialist."""
        from aragora.agents.specialist_factory import AgentConfig
        from aragora.training.specialist_models import Vertical

        config = AgentConfig(vertical=Vertical.LEGAL)

        with patch(
            "aragora.agents.api_agents.openrouter.get_api_key",
            return_value="test-key",
        ):
            result = await factory.create(config)

            assert result.is_specialist is False
            assert result.specialist_model is None
            assert result.vertical == Vertical.LEGAL

    @pytest.mark.asyncio
    async def test_create_with_specialist_available(self, factory):
        """Test create uses specialist when available."""
        from aragora.agents.specialist_factory import AgentConfig
        from aragora.training.specialist_models import (
            Vertical,
            SpecialistModel,
            TrainingStatus,
        )

        # Register a specialist model
        specialist = SpecialistModel(
            id="sm_legal_test",
            base_model="llama-3.3-70b",
            adapter_name="legal-expert",
            vertical=Vertical.LEGAL,
            org_id=None,
            status=TrainingStatus.READY,
        )
        factory._registry.register(specialist)

        config = AgentConfig(vertical=Vertical.LEGAL)

        # Mock TinkerAgent at the import point in specialist_factory
        mock_tinker_agent = MagicMock()
        with patch(
            "aragora.agents.specialist_factory.TinkerAgent",
            return_value=mock_tinker_agent,
        ):
            result = await factory.create(config)

            assert result.is_specialist is True
            assert result.specialist_model is specialist
            assert result.adapter_name == "legal-expert"

    @pytest.mark.asyncio
    async def test_create_require_specialist_raises(self, factory):
        """Test create raises when specialist required but not available."""
        from aragora.agents.specialist_factory import AgentConfig
        from aragora.training.specialist_models import Vertical

        config = AgentConfig(vertical=Vertical.LEGAL, require_specialist=True)

        with pytest.raises(ValueError, match="No specialist model available"):
            await factory.create(config)

    @pytest.mark.asyncio
    async def test_create_with_org_specific_specialist(self, factory):
        """Test create prioritizes org-specific specialist."""
        from aragora.agents.specialist_factory import AgentConfig
        from aragora.training.specialist_models import (
            Vertical,
            SpecialistModel,
            TrainingStatus,
        )

        # Register global specialist
        global_specialist = SpecialistModel(
            id="sm_legal_global",
            base_model="llama-3.3-70b",
            adapter_name="legal-global",
            vertical=Vertical.LEGAL,
            org_id=None,
            status=TrainingStatus.READY,
        )
        factory._registry.register(global_specialist)

        # Register org-specific specialist
        org_specialist = SpecialistModel(
            id="sm_legal_org",
            base_model="llama-3.3-70b",
            adapter_name="legal-org-specific",
            vertical=Vertical.LEGAL,
            org_id="org-123",
            status=TrainingStatus.READY,
        )
        factory._registry.register(org_specialist)

        config = AgentConfig(vertical=Vertical.LEGAL, org_id="org-123")

        # Mock TinkerAgent at the import point in specialist_factory
        mock_tinker_agent = MagicMock()
        with patch(
            "aragora.agents.specialist_factory.TinkerAgent",
            return_value=mock_tinker_agent,
        ):
            result = await factory.create(config)

            # Should get org-specific specialist
            assert result.specialist_model is org_specialist
            assert result.adapter_name == "legal-org-specific"


# =============================================================================
# Base Agent Creation Tests
# =============================================================================


class TestCreateBaseAgent:
    """Test _create_base_agent method."""

    @pytest.fixture
    def factory(self):
        """Create factory for testing."""
        from aragora.agents.specialist_factory import SpecialistAgentFactory
        from aragora.training.specialist_models import SpecialistModelRegistry

        return SpecialistAgentFactory(registry=SpecialistModelRegistry())

    @pytest.mark.asyncio
    async def test_creates_openrouter_agent(self, factory):
        """Test creates OpenRouterAgent for base model."""
        from aragora.agents.specialist_factory import AgentConfig
        from aragora.training.specialist_models import Vertical

        config = AgentConfig(vertical=Vertical.LEGAL)

        with patch(
            "aragora.agents.api_agents.openrouter.get_api_key",
            return_value="test-key",
        ):
            result = await factory._create_base_agent(config)

            assert result.is_specialist is False
            assert "llama" in result.model_id.lower() or "openrouter" in result.model_id.lower()

    @pytest.mark.asyncio
    async def test_prefer_speed_uses_smaller_model(self, factory):
        """Test prefer_speed flag uses smaller model."""
        from aragora.agents.specialist_factory import AgentConfig
        from aragora.training.specialist_models import Vertical

        config = AgentConfig(vertical=Vertical.LEGAL, prefer_speed=True)

        with patch(
            "aragora.agents.api_agents.openrouter.get_api_key",
            return_value="test-key",
        ):
            result = await factory._create_base_agent(config)

            # Should use 8b model instead of 70b
            assert "8b" in result.base_model or "7b" in result.base_model


# =============================================================================
# Convenience Method Tests
# =============================================================================


class TestConvenienceMethods:
    """Test convenience methods for creating vertical-specific agents."""

    @pytest.fixture
    def factory(self):
        """Create factory for testing."""
        from aragora.agents.specialist_factory import SpecialistAgentFactory
        from aragora.training.specialist_models import SpecialistModelRegistry

        return SpecialistAgentFactory(registry=SpecialistModelRegistry())

    @pytest.mark.asyncio
    async def test_create_legal_agent(self, factory):
        """Test create_legal_agent method."""
        from aragora.training.specialist_models import Vertical

        with patch(
            "aragora.agents.api_agents.openrouter.get_api_key",
            return_value="test-key",
        ):
            result = await factory.create_legal_agent()

            assert result.vertical == Vertical.LEGAL

    @pytest.mark.asyncio
    async def test_create_healthcare_agent(self, factory):
        """Test create_healthcare_agent method."""
        from aragora.training.specialist_models import Vertical

        with patch(
            "aragora.agents.api_agents.openrouter.get_api_key",
            return_value="test-key",
        ):
            result = await factory.create_healthcare_agent()

            assert result.vertical == Vertical.HEALTHCARE

    @pytest.mark.asyncio
    async def test_create_security_agent(self, factory):
        """Test create_security_agent method."""
        from aragora.training.specialist_models import Vertical

        with patch(
            "aragora.agents.api_agents.openrouter.get_api_key",
            return_value="test-key",
        ):
            result = await factory.create_security_agent()

            assert result.vertical == Vertical.SECURITY

    @pytest.mark.asyncio
    async def test_create_accounting_agent(self, factory):
        """Test create_accounting_agent method."""
        from aragora.training.specialist_models import Vertical

        with patch(
            "aragora.agents.api_agents.openrouter.get_api_key",
            return_value="test-key",
        ):
            result = await factory.create_accounting_agent()

            assert result.vertical == Vertical.ACCOUNTING

    @pytest.mark.asyncio
    async def test_create_regulatory_agent(self, factory):
        """Test create_regulatory_agent method."""
        from aragora.training.specialist_models import Vertical

        with patch(
            "aragora.agents.api_agents.openrouter.get_api_key",
            return_value="test-key",
        ):
            result = await factory.create_regulatory_agent()

            assert result.vertical == Vertical.REGULATORY

    @pytest.mark.asyncio
    async def test_create_academic_agent(self, factory):
        """Test create_academic_agent method."""
        from aragora.training.specialist_models import Vertical

        with patch(
            "aragora.agents.api_agents.openrouter.get_api_key",
            return_value="test-key",
        ):
            result = await factory.create_academic_agent()

            assert result.vertical == Vertical.ACADEMIC

    @pytest.mark.asyncio
    async def test_create_software_agent(self, factory):
        """Test create_software_agent method."""
        from aragora.training.specialist_models import Vertical

        with patch(
            "aragora.agents.api_agents.openrouter.get_api_key",
            return_value="test-key",
        ):
            result = await factory.create_software_agent()

            assert result.vertical == Vertical.SOFTWARE

    @pytest.mark.asyncio
    async def test_convenience_methods_pass_kwargs(self, factory):
        """Test convenience methods pass additional kwargs."""
        with patch(
            "aragora.agents.api_agents.openrouter.get_api_key",
            return_value="test-key",
        ):
            result = await factory.create_legal_agent(
                org_id="org-test",
                temperature=0.5,
                extra_context="Extra context",
            )

            # Verify it used the org_id by checking the config was processed
            assert result.vertical.value == "legal"


# =============================================================================
# Vertical Inference Tests
# =============================================================================


class TestInferVertical:
    """Test _infer_vertical method for task-based selection."""

    @pytest.fixture
    def factory(self):
        """Create factory for testing."""
        from aragora.agents.specialist_factory import SpecialistAgentFactory
        from aragora.training.specialist_models import SpecialistModelRegistry

        return SpecialistAgentFactory(registry=SpecialistModelRegistry())

    def test_infer_legal_vertical(self, factory):
        """Test infers legal vertical from task."""
        from aragora.training.specialist_models import Vertical

        task = "Review this contract for potential legal issues"
        result = factory._infer_vertical(task)

        assert result == Vertical.LEGAL

    def test_infer_healthcare_vertical(self, factory):
        """Test infers healthcare vertical from task."""
        from aragora.training.specialist_models import Vertical

        task = "Analyze patient medical records for diagnosis"
        result = factory._infer_vertical(task)

        assert result == Vertical.HEALTHCARE

    def test_infer_security_vertical(self, factory):
        """Test infers security vertical from task."""
        from aragora.training.specialist_models import Vertical

        task = "Check this code for security vulnerabilities"
        result = factory._infer_vertical(task)

        assert result == Vertical.SECURITY

    def test_infer_software_vertical(self, factory):
        """Test infers software vertical from task."""
        from aragora.training.specialist_models import Vertical

        task = "Review this API architecture design"
        result = factory._infer_vertical(task)

        assert result == Vertical.SOFTWARE

    def test_infer_defaults_to_general(self, factory):
        """Test defaults to general for unrecognized task."""
        from aragora.training.specialist_models import Vertical

        task = "Help me with this random thing"
        result = factory._infer_vertical(task)

        assert result == Vertical.GENERAL


# =============================================================================
# Select For Task Tests
# =============================================================================


class TestSelectForTask:
    """Test select_for_task method."""

    @pytest.fixture
    def factory(self):
        """Create factory for testing."""
        from aragora.agents.specialist_factory import SpecialistAgentFactory
        from aragora.training.specialist_models import SpecialistModelRegistry

        return SpecialistAgentFactory(registry=SpecialistModelRegistry())

    @pytest.mark.asyncio
    async def test_select_for_legal_task(self, factory):
        """Test selects legal agent for legal task."""
        from aragora.training.specialist_models import Vertical

        task = "Draft a contract clause for liability limitation"

        with patch(
            "aragora.agents.api_agents.openrouter.get_api_key",
            return_value="test-key",
        ):
            result = await factory.select_for_task(task)

            assert result.vertical == Vertical.LEGAL

    @pytest.mark.asyncio
    async def test_select_for_task_with_org(self, factory):
        """Test select_for_task uses org_id."""
        task = "Review security authentication flow"

        with patch(
            "aragora.agents.api_agents.openrouter.get_api_key",
            return_value="test-key",
        ):
            result = await factory.select_for_task(task, org_id="org-123")

            # Task should be processed with the org context
            assert result.vertical.value in ["security", "software"]


# =============================================================================
# List Available Specialists Tests
# =============================================================================


class TestListAvailableSpecialists:
    """Test list_available_specialists method."""

    @pytest.fixture
    def factory(self):
        """Create factory for testing."""
        from aragora.agents.specialist_factory import SpecialistAgentFactory
        from aragora.training.specialist_models import SpecialistModelRegistry

        return SpecialistAgentFactory(registry=SpecialistModelRegistry())

    def test_empty_registry_returns_empty(self, factory):
        """Test returns empty dict when no specialists registered."""
        result = factory.list_available_specialists()

        assert result == {}

    def test_lists_ready_specialists_by_vertical(self, factory):
        """Test lists ready specialists grouped by vertical."""
        from aragora.training.specialist_models import (
            Vertical,
            SpecialistModel,
            TrainingStatus,
        )

        # Register ready specialist
        legal_specialist = SpecialistModel(
            id="sm_legal_1",
            base_model="llama-3.3-70b",
            adapter_name="legal-v1",
            vertical=Vertical.LEGAL,
            org_id=None,
            status=TrainingStatus.READY,
        )
        factory._registry.register(legal_specialist)

        # Register pending specialist (should not appear)
        pending_specialist = SpecialistModel(
            id="sm_security_1",
            base_model="llama-3.3-70b",
            adapter_name="security-v1",
            vertical=Vertical.SECURITY,
            org_id=None,
            status=TrainingStatus.PENDING,
        )
        factory._registry.register(pending_specialist)

        result = factory.list_available_specialists()

        assert Vertical.LEGAL in result
        assert len(result[Vertical.LEGAL]) == 1
        assert Vertical.SECURITY not in result  # Pending not included

    def test_filters_by_org_id(self, factory):
        """Test filters specialists by org_id."""
        from aragora.training.specialist_models import (
            Vertical,
            SpecialistModel,
            TrainingStatus,
        )

        # Global specialist
        global_specialist = SpecialistModel(
            id="sm_legal_global",
            base_model="llama-3.3-70b",
            adapter_name="legal-global",
            vertical=Vertical.LEGAL,
            org_id=None,
            status=TrainingStatus.READY,
        )
        factory._registry.register(global_specialist)

        # Org-specific specialist
        org_specialist = SpecialistModel(
            id="sm_legal_org",
            base_model="llama-3.3-70b",
            adapter_name="legal-org",
            vertical=Vertical.LEGAL,
            org_id="org-123",
            status=TrainingStatus.READY,
        )
        factory._registry.register(org_specialist)

        # Different org specialist
        other_org_specialist = SpecialistModel(
            id="sm_legal_other",
            base_model="llama-3.3-70b",
            adapter_name="legal-other",
            vertical=Vertical.LEGAL,
            org_id="org-456",
            status=TrainingStatus.READY,
        )
        factory._registry.register(other_org_specialist)

        result = factory.list_available_specialists(org_id="org-123")

        # Should include global and org-123 specific, not org-456
        assert Vertical.LEGAL in result
        model_ids = [m.id for m in result[Vertical.LEGAL]]
        assert "sm_legal_global" in model_ids
        assert "sm_legal_org" in model_ids
        assert "sm_legal_other" not in model_ids


# =============================================================================
# Global Factory Instance Tests
# =============================================================================


class TestGetSpecialistFactory:
    """Test get_specialist_factory function."""

    def test_creates_singleton(self):
        """Test returns same instance on repeated calls."""
        from aragora.agents import specialist_factory

        # Reset the global instance
        specialist_factory._specialist_factory = None

        factory1 = specialist_factory.get_specialist_factory()
        factory2 = specialist_factory.get_specialist_factory()

        assert factory1 is factory2

    def test_accepts_custom_registry(self):
        """Test accepts custom registry on first call."""
        from aragora.agents import specialist_factory
        from aragora.training.specialist_models import SpecialistModelRegistry

        # Reset the global instance
        specialist_factory._specialist_factory = None

        custom_registry = SpecialistModelRegistry()
        factory = specialist_factory.get_specialist_factory(registry=custom_registry)

        assert factory._registry is custom_registry


# =============================================================================
# Module Exports Tests
# =============================================================================


class TestModuleExports:
    """Test module exports."""

    def test_agent_config_exportable(self):
        """Test AgentConfig can be imported."""
        from aragora.agents.specialist_factory import AgentConfig

        assert AgentConfig is not None

    def test_specialist_agent_info_exportable(self):
        """Test SpecialistAgentInfo can be imported."""
        from aragora.agents.specialist_factory import SpecialistAgentInfo

        assert SpecialistAgentInfo is not None

    def test_specialist_agent_factory_exportable(self):
        """Test SpecialistAgentFactory can be imported."""
        from aragora.agents.specialist_factory import SpecialistAgentFactory

        assert SpecialistAgentFactory is not None

    def test_get_specialist_factory_exportable(self):
        """Test get_specialist_factory can be imported."""
        from aragora.agents.specialist_factory import get_specialist_factory

        assert get_specialist_factory is not None

    def test_all_exports_in_dunder_all(self):
        """Test __all__ contains expected exports."""
        from aragora.agents import specialist_factory

        expected = [
            "AgentConfig",
            "SpecialistAgentInfo",
            "SpecialistAgentFactory",
            "get_specialist_factory",
        ]

        for name in expected:
            assert name in specialist_factory.__all__


# =============================================================================
# Method Signature Tests
# =============================================================================


class TestMethodSignatures:
    """Test method signatures."""

    def test_create_signature(self):
        """Test create method signature."""
        from aragora.agents.specialist_factory import SpecialistAgentFactory

        sig = inspect.signature(SpecialistAgentFactory.create)
        params = list(sig.parameters.keys())

        assert "self" in params
        assert "config" in params

    def test_select_for_task_signature(self):
        """Test select_for_task method signature."""
        from aragora.agents.specialist_factory import SpecialistAgentFactory

        sig = inspect.signature(SpecialistAgentFactory.select_for_task)
        params = list(sig.parameters.keys())

        assert "self" in params
        assert "task" in params
        assert "org_id" in params
        assert "kwargs" in params

    def test_list_available_specialists_signature(self):
        """Test list_available_specialists method signature."""
        from aragora.agents.specialist_factory import SpecialistAgentFactory

        sig = inspect.signature(SpecialistAgentFactory.list_available_specialists)
        params = list(sig.parameters.keys())

        assert "self" in params
        assert "org_id" in params
