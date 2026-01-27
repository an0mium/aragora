"""
Tests for aragora.skills.registry module.

Covers:
- SkillExecutionMetrics
- RateLimitState
- SkillRegistry registration, invocation, rate limiting, metrics
- Function schema generation
- Hooks system
"""

from __future__ import annotations

import asyncio
import time
from datetime import datetime, timezone
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.skills.base import (
    Skill,
    SkillCapability,
    SkillContext,
    SkillManifest,
    SkillResult,
    SkillStatus,
)
from aragora.skills.registry import (
    RateLimitState,
    SkillExecutionMetrics,
    SkillRegistry,
    get_skill_registry,
    reset_skill_registry,
)


# =============================================================================
# Test Fixtures
# =============================================================================


class MockSkill(Skill):
    """Mock skill for testing."""

    def __init__(
        self,
        name: str = "mock_skill",
        capabilities: List[SkillCapability] = None,
        required_permissions: List[str] = None,
        rate_limit: int = None,
        max_execution_time: float = 60.0,
        debate_compatible: bool = True,
        requires_debate_context: bool = False,
    ):
        self._name = name
        self._capabilities = capabilities or [SkillCapability.WEB_SEARCH]
        self._required_permissions = required_permissions or []
        self._rate_limit = rate_limit
        self._max_execution_time = max_execution_time
        self._debate_compatible = debate_compatible
        self._requires_debate_context = requires_debate_context
        self._execute_mock = AsyncMock(return_value=SkillResult.create_success({"mock": True}))

    @property
    def manifest(self) -> SkillManifest:
        return SkillManifest(
            name=self._name,
            version="1.0.0",
            capabilities=self._capabilities,
            input_schema={"query": {"type": "string", "required": True}},
            required_permissions=self._required_permissions,
            rate_limit_per_minute=self._rate_limit,
            max_execution_time_seconds=self._max_execution_time,
            debate_compatible=self._debate_compatible,
            requires_debate_context=self._requires_debate_context,
            tags=["test", "mock"],
        )

    async def execute(
        self,
        input_data: Dict[str, Any],
        context: SkillContext,
    ) -> SkillResult:
        return await self._execute_mock(input_data, context)


class SlowSkill(Skill):
    """Skill that takes time to execute."""

    def __init__(self, delay: float = 0.5):
        self._delay = delay

    @property
    def manifest(self) -> SkillManifest:
        return SkillManifest(
            name="slow_skill",
            version="1.0.0",
            capabilities=[SkillCapability.EXTERNAL_API],
            input_schema={},
            max_execution_time_seconds=0.1,  # Very short timeout
        )

    async def execute(
        self,
        input_data: Dict[str, Any],
        context: SkillContext,
    ) -> SkillResult:
        await asyncio.sleep(self._delay)
        return SkillResult.create_success({"slow": True})


class FailingSkill(Skill):
    """Skill that always fails."""

    @property
    def manifest(self) -> SkillManifest:
        return SkillManifest(
            name="failing_skill",
            version="1.0.0",
            capabilities=[],
            input_schema={},
        )

    async def execute(
        self,
        input_data: Dict[str, Any],
        context: SkillContext,
    ) -> SkillResult:
        raise ValueError("Skill execution failed!")


@pytest.fixture
def registry() -> SkillRegistry:
    """Create a fresh registry for testing."""
    return SkillRegistry()


@pytest.fixture
def context() -> SkillContext:
    """Create a basic context for testing."""
    return SkillContext(
        user_id="user123",
        tenant_id="tenant456",
        permissions=["skill:execute"],
    )


# =============================================================================
# SkillExecutionMetrics Tests
# =============================================================================


class TestSkillExecutionMetrics:
    """Tests for SkillExecutionMetrics dataclass."""

    def test_initial_values(self):
        """Test initial metric values."""
        metrics = SkillExecutionMetrics()

        assert metrics.total_invocations == 0
        assert metrics.successful_invocations == 0
        assert metrics.failed_invocations == 0
        assert metrics.total_execution_time_ms == 0
        assert metrics.average_execution_time_ms == 0
        assert metrics.last_invocation is None
        assert metrics.last_error is None

    def test_record_successful_execution(self):
        """Test recording a successful execution."""
        metrics = SkillExecutionMetrics()
        metrics.record_execution(success=True, duration_ms=100.0)

        assert metrics.total_invocations == 1
        assert metrics.successful_invocations == 1
        assert metrics.failed_invocations == 0
        assert metrics.total_execution_time_ms == 100.0
        assert metrics.average_execution_time_ms == 100.0
        assert metrics.last_invocation is not None

    def test_record_failed_execution(self):
        """Test recording a failed execution."""
        metrics = SkillExecutionMetrics()
        metrics.record_execution(success=False, duration_ms=50.0, error="Test error")

        assert metrics.total_invocations == 1
        assert metrics.successful_invocations == 0
        assert metrics.failed_invocations == 1
        assert metrics.last_error == "Test error"

    def test_average_calculation(self):
        """Test average execution time calculation."""
        metrics = SkillExecutionMetrics()
        metrics.record_execution(success=True, duration_ms=100.0)
        metrics.record_execution(success=True, duration_ms=200.0)
        metrics.record_execution(success=True, duration_ms=300.0)

        assert metrics.total_invocations == 3
        assert metrics.total_execution_time_ms == 600.0
        assert metrics.average_execution_time_ms == 200.0

    def test_success_rate(self):
        """Test success rate calculation."""
        metrics = SkillExecutionMetrics()
        metrics.record_execution(success=True, duration_ms=100.0)
        metrics.record_execution(success=True, duration_ms=100.0)
        metrics.record_execution(success=False, duration_ms=100.0)
        metrics.record_execution(success=True, duration_ms=100.0)

        assert metrics.success_rate == 75.0

    def test_success_rate_zero_invocations(self):
        """Test success rate with no invocations."""
        metrics = SkillExecutionMetrics()
        assert metrics.success_rate == 0.0


# =============================================================================
# RateLimitState Tests
# =============================================================================


class TestRateLimitState:
    """Tests for RateLimitState dataclass."""

    def test_initial_values(self):
        """Test initial state values."""
        state = RateLimitState()

        assert state.window_start == 0.0
        assert state.request_count == 0


# =============================================================================
# SkillRegistry Registration Tests
# =============================================================================


class TestSkillRegistryRegistration:
    """Tests for SkillRegistry registration functionality."""

    def test_register_skill(self, registry: SkillRegistry):
        """Test registering a skill."""
        skill = MockSkill(name="test_skill")
        registry.register(skill)

        assert registry.has_skill("test_skill")
        assert registry.skill_count == 1

    def test_register_duplicate_raises(self, registry: SkillRegistry):
        """Test registering duplicate skill raises error."""
        skill = MockSkill(name="test_skill")
        registry.register(skill)

        with pytest.raises(ValueError, match="already registered"):
            registry.register(skill)

    def test_register_with_replace(self, registry: SkillRegistry):
        """Test registering with replace=True."""
        skill1 = MockSkill(name="test_skill")
        skill2 = MockSkill(name="test_skill")

        registry.register(skill1)
        registry.register(skill2, replace=True)

        assert registry.skill_count == 1

    def test_unregister_skill(self, registry: SkillRegistry):
        """Test unregistering a skill."""
        skill = MockSkill(name="test_skill")
        registry.register(skill)

        result = registry.unregister("test_skill")

        assert result is True
        assert not registry.has_skill("test_skill")
        assert registry.skill_count == 0

    def test_unregister_nonexistent(self, registry: SkillRegistry):
        """Test unregistering nonexistent skill."""
        result = registry.unregister("nonexistent")
        assert result is False

    def test_get_skill(self, registry: SkillRegistry):
        """Test getting a skill by name."""
        skill = MockSkill(name="test_skill")
        registry.register(skill)

        retrieved = registry.get("test_skill")
        assert retrieved is skill

    def test_get_nonexistent_skill(self, registry: SkillRegistry):
        """Test getting nonexistent skill returns None."""
        retrieved = registry.get("nonexistent")
        assert retrieved is None

    def test_has_skill(self, registry: SkillRegistry):
        """Test has_skill check."""
        skill = MockSkill(name="test_skill")
        registry.register(skill)

        assert registry.has_skill("test_skill") is True
        assert registry.has_skill("nonexistent") is False


# =============================================================================
# SkillRegistry List and Filter Tests
# =============================================================================


class TestSkillRegistryListAndFilter:
    """Tests for SkillRegistry listing and filtering."""

    @pytest.fixture
    def populated_registry(self, registry: SkillRegistry) -> SkillRegistry:
        """Create a registry with multiple skills."""
        registry.register(
            MockSkill(
                name="web_search",
                capabilities=[SkillCapability.WEB_SEARCH],
                debate_compatible=True,
            )
        )
        registry.register(
            MockSkill(
                name="code_exec",
                capabilities=[SkillCapability.CODE_EXECUTION],
                debate_compatible=False,
            )
        )
        registry.register(
            MockSkill(
                name="llm_skill",
                capabilities=[SkillCapability.LLM_INFERENCE, SkillCapability.EXTERNAL_API],
                debate_compatible=True,
            )
        )
        return registry

    def test_list_all_skills(self, populated_registry: SkillRegistry):
        """Test listing all skills."""
        manifests = populated_registry.list_skills()
        assert len(manifests) == 3

    def test_list_by_capability(self, populated_registry: SkillRegistry):
        """Test filtering by capability."""
        manifests = populated_registry.list_skills(capability=SkillCapability.WEB_SEARCH)
        assert len(manifests) == 1
        assert manifests[0].name == "web_search"

    def test_list_by_tag(self, populated_registry: SkillRegistry):
        """Test filtering by tag."""
        manifests = populated_registry.list_skills(tag="test")
        assert len(manifests) == 3  # All mock skills have "test" tag

    def test_list_debate_compatible_only(self, populated_registry: SkillRegistry):
        """Test filtering debate-compatible skills."""
        manifests = populated_registry.list_skills(debate_compatible_only=True)
        assert len(manifests) == 2
        names = [m.name for m in manifests]
        assert "code_exec" not in names

    def test_get_capabilities(self, populated_registry: SkillRegistry):
        """Test getting all capabilities."""
        caps = populated_registry.get_capabilities()

        assert SkillCapability.WEB_SEARCH in caps
        assert SkillCapability.CODE_EXECUTION in caps
        assert SkillCapability.LLM_INFERENCE in caps
        assert SkillCapability.EXTERNAL_API in caps


# =============================================================================
# SkillRegistry Invocation Tests
# =============================================================================


class TestSkillRegistryInvocation:
    """Tests for SkillRegistry invocation."""

    @pytest.mark.asyncio
    async def test_invoke_skill(self, registry: SkillRegistry, context: SkillContext):
        """Test invoking a skill."""
        skill = MockSkill(name="test_skill")
        registry.register(skill)

        result = await registry.invoke("test_skill", {"query": "test"}, context)

        assert result.success is True
        assert result.data["mock"] is True
        skill._execute_mock.assert_called_once()

    @pytest.mark.asyncio
    async def test_invoke_nonexistent_skill(self, registry: SkillRegistry, context: SkillContext):
        """Test invoking nonexistent skill."""
        result = await registry.invoke("nonexistent", {}, context)

        assert result.success is False
        assert result.status == SkillStatus.FAILURE
        assert "not found" in result.error_message.lower()

    @pytest.mark.asyncio
    async def test_invoke_validates_input(self, registry: SkillRegistry, context: SkillContext):
        """Test that invocation validates input."""
        skill = MockSkill(name="test_skill")
        registry.register(skill)

        result = await registry.invoke("test_skill", {}, context)  # Missing required "query"

        assert result.success is False
        assert result.status == SkillStatus.INVALID_INPUT

    @pytest.mark.asyncio
    async def test_invoke_checks_permissions(self, registry: SkillRegistry):
        """Test that invocation checks permissions."""
        skill = MockSkill(
            name="test_skill",
            required_permissions=["special:permission"],
        )
        registry.register(skill)
        context = SkillContext(permissions=["basic:permission"])

        result = await registry.invoke("test_skill", {"query": "test"}, context)

        assert result.success is False
        assert result.status == SkillStatus.PERMISSION_DENIED

    @pytest.mark.asyncio
    async def test_invoke_requires_debate_context(
        self, registry: SkillRegistry, context: SkillContext
    ):
        """Test skill requiring debate context."""
        skill = MockSkill(
            name="debate_skill",
            requires_debate_context=True,
        )
        registry.register(skill)

        result = await registry.invoke("debate_skill", {"query": "test"}, context)

        assert result.success is False
        assert "debate context" in result.error_message.lower()

    @pytest.mark.asyncio
    async def test_invoke_with_debate_context(self, registry: SkillRegistry):
        """Test skill with debate context provided."""
        skill = MockSkill(
            name="debate_skill",
            requires_debate_context=True,
        )
        registry.register(skill)
        context = SkillContext(
            debate_id="debate123",
            permissions=["skill:execute"],
        )

        result = await registry.invoke("debate_skill", {"query": "test"}, context)

        assert result.success is True

    @pytest.mark.asyncio
    async def test_invoke_timeout(self, registry: SkillRegistry, context: SkillContext):
        """Test skill timeout handling."""
        skill = SlowSkill(delay=1.0)  # Takes 1 second, timeout is 0.1s
        registry.register(skill)

        result = await registry.invoke("slow_skill", {}, context)

        assert result.success is False
        assert result.status == SkillStatus.TIMEOUT

    @pytest.mark.asyncio
    async def test_invoke_exception_handling(self, registry: SkillRegistry, context: SkillContext):
        """Test exception handling during invocation."""
        skill = FailingSkill()
        registry.register(skill)

        result = await registry.invoke("failing_skill", {}, context)

        assert result.success is False
        assert result.status == SkillStatus.FAILURE
        assert "failed" in result.error_message.lower()

    @pytest.mark.asyncio
    async def test_invoke_sets_timing(self, registry: SkillRegistry):
        """Test that invocation sets timing fields."""
        skill = MockSkill(name="test_skill")
        registry.register(skill)
        context = SkillContext(permissions=["skill:execute"])

        result = await registry.invoke("test_skill", {"query": "test"}, context)

        assert result.success is True
        assert result.started_at is not None
        assert result.completed_at is not None
        # Note: started_at is set by registry after skill execution,
        # so we just verify both timestamps exist


# =============================================================================
# SkillRegistry Rate Limiting Tests
# =============================================================================


class TestSkillRegistryRateLimiting:
    """Tests for SkillRegistry rate limiting."""

    @pytest.mark.asyncio
    async def test_rate_limit_enforced(self, context: SkillContext):
        """Test rate limiting is enforced."""
        registry = SkillRegistry(enable_rate_limiting=True)
        skill = MockSkill(name="limited_skill", rate_limit=2)  # 2 per minute
        registry.register(skill)

        # First two should succeed
        result1 = await registry.invoke("limited_skill", {"query": "test"}, context)
        result2 = await registry.invoke("limited_skill", {"query": "test"}, context)

        # Third should be rate limited
        result3 = await registry.invoke("limited_skill", {"query": "test"}, context)

        assert result1.success is True
        assert result2.success is True
        assert result3.success is False
        assert result3.status == SkillStatus.RATE_LIMITED

    @pytest.mark.asyncio
    async def test_rate_limit_disabled(self, context: SkillContext):
        """Test rate limiting can be disabled."""
        registry = SkillRegistry(enable_rate_limiting=False)
        skill = MockSkill(name="limited_skill", rate_limit=1)
        registry.register(skill)

        # All should succeed even with rate_limit=1
        for _ in range(5):
            result = await registry.invoke("limited_skill", {"query": "test"}, context)
            assert result.success is True

    @pytest.mark.asyncio
    async def test_no_rate_limit_no_enforcement(self, context: SkillContext):
        """Test skills without rate limit are not limited."""
        registry = SkillRegistry(enable_rate_limiting=True)
        skill = MockSkill(name="unlimited_skill", rate_limit=None)
        registry.register(skill)

        # All should succeed
        for _ in range(10):
            result = await registry.invoke("unlimited_skill", {"query": "test"}, context)
            assert result.success is True


# =============================================================================
# SkillRegistry Metrics Tests
# =============================================================================


class TestSkillRegistryMetrics:
    """Tests for SkillRegistry metrics collection."""

    @pytest.mark.asyncio
    async def test_metrics_recorded(self, registry: SkillRegistry, context: SkillContext):
        """Test metrics are recorded for invocations."""
        skill = MockSkill(name="test_skill")
        registry.register(skill)

        await registry.invoke("test_skill", {"query": "test"}, context)

        metrics = registry.get_metrics("test_skill")
        assert metrics["total_invocations"] == 1
        assert metrics["successful"] == 1
        assert metrics["failed"] == 0

    @pytest.mark.asyncio
    async def test_metrics_count_failures(self, registry: SkillRegistry, context: SkillContext):
        """Test metrics count failures."""
        skill = FailingSkill()
        registry.register(skill)

        await registry.invoke("failing_skill", {}, context)

        metrics = registry.get_metrics("failing_skill")
        assert metrics["failed"] == 1
        assert metrics["successful"] == 0

    @pytest.mark.asyncio
    async def test_metrics_disabled(self, context: SkillContext):
        """Test metrics can be disabled."""
        registry = SkillRegistry(enable_metrics=False)
        skill = MockSkill(name="test_skill")
        registry.register(skill)

        await registry.invoke("test_skill", {"query": "test"}, context)

        metrics = registry.get_metrics("test_skill")
        assert metrics == {}

    def test_get_all_metrics(self, registry: SkillRegistry):
        """Test getting all metrics."""
        skill1 = MockSkill(name="skill1")
        skill2 = MockSkill(name="skill2")
        registry.register(skill1)
        registry.register(skill2)

        all_metrics = registry.get_metrics()
        assert isinstance(all_metrics, dict)


# =============================================================================
# SkillRegistry Function Schema Tests
# =============================================================================


class TestSkillRegistryFunctionSchemas:
    """Tests for function schema generation."""

    def test_get_all_function_schemas(self, registry: SkillRegistry):
        """Test getting all function schemas."""
        registry.register(MockSkill(name="skill1"))
        registry.register(MockSkill(name="skill2"))

        schemas = registry.get_function_schemas()

        assert len(schemas) == 2
        names = [s["name"] for s in schemas]
        assert "skill1" in names
        assert "skill2" in names

    def test_get_filtered_function_schemas(self, registry: SkillRegistry):
        """Test getting filtered function schemas."""
        registry.register(MockSkill(name="skill1"))
        registry.register(MockSkill(name="skill2"))
        registry.register(MockSkill(name="skill3"))

        schemas = registry.get_function_schemas(skills=["skill1", "skill3"])

        assert len(schemas) == 2
        names = [s["name"] for s in schemas]
        assert "skill1" in names
        assert "skill3" in names
        assert "skill2" not in names

    def test_get_debate_compatible_schemas(self, registry: SkillRegistry):
        """Test getting only debate-compatible schemas."""
        registry.register(MockSkill(name="debate_skill", debate_compatible=True))
        registry.register(MockSkill(name="non_debate_skill", debate_compatible=False))

        schemas = registry.get_function_schemas(debate_compatible_only=True)

        assert len(schemas) == 1
        assert schemas[0]["name"] == "debate_skill"


# =============================================================================
# SkillRegistry Hooks Tests
# =============================================================================


class TestSkillRegistryHooks:
    """Tests for SkillRegistry hooks system."""

    @pytest.mark.asyncio
    async def test_pre_invoke_hook(self, registry: SkillRegistry, context: SkillContext):
        """Test pre-invoke hook is called."""
        hook = MagicMock()
        skill = MockSkill(name="test_skill")
        registry.register(skill)
        registry.add_hook("pre_invoke", hook)

        await registry.invoke("test_skill", {"query": "test"}, context)

        hook.assert_called_once_with("test_skill", {"query": "test"}, context)

    @pytest.mark.asyncio
    async def test_post_invoke_hook(self, registry: SkillRegistry, context: SkillContext):
        """Test post-invoke hook is called."""
        hook = MagicMock()
        skill = MockSkill(name="test_skill")
        registry.register(skill)
        registry.add_hook("post_invoke", hook)

        await registry.invoke("test_skill", {"query": "test"}, context)

        hook.assert_called_once()
        args = hook.call_args[0]
        assert args[0] == "test_skill"
        assert isinstance(args[1], SkillResult)
        assert args[2] == context

    @pytest.mark.asyncio
    async def test_on_error_hook(self, registry: SkillRegistry, context: SkillContext):
        """Test on-error hook is called on exception."""
        hook = MagicMock()
        skill = FailingSkill()
        registry.register(skill)
        registry.add_hook("on_error", hook)

        await registry.invoke("failing_skill", {}, context)

        hook.assert_called_once()
        args = hook.call_args[0]
        assert args[0] == "failing_skill"
        assert isinstance(args[1], ValueError)
        assert args[2] == context

    def test_add_invalid_hook_type(self, registry: SkillRegistry):
        """Test adding invalid hook type raises error."""
        with pytest.raises(ValueError, match="Unknown hook type"):
            registry.add_hook("invalid_hook", MagicMock())

    @pytest.mark.asyncio
    async def test_hook_unregister(self, registry: SkillRegistry, context: SkillContext):
        """Test unregistering a hook."""
        hook = MagicMock()
        skill = MockSkill(name="test_skill")
        registry.register(skill)

        unregister = registry.add_hook("pre_invoke", hook)
        unregister()

        await registry.invoke("test_skill", {"query": "test"}, context)

        hook.assert_not_called()

    @pytest.mark.asyncio
    async def test_hook_exception_doesnt_break_invocation(
        self, registry: SkillRegistry, context: SkillContext
    ):
        """Test hook exception doesn't break skill invocation."""

        def failing_hook(*args):
            raise RuntimeError("Hook failed!")

        skill = MockSkill(name="test_skill")
        registry.register(skill)
        registry.add_hook("pre_invoke", failing_hook)

        # Should still succeed despite hook failure
        result = await registry.invoke("test_skill", {"query": "test"}, context)
        assert result.success is True


# =============================================================================
# SkillRegistry RBAC Integration Tests
# =============================================================================


class TestSkillRegistryRBAC:
    """Tests for RBAC integration."""

    @pytest.mark.asyncio
    async def test_rbac_checker_used(self):
        """Test RBAC checker is used when provided."""
        rbac_checker = MagicMock()
        rbac_checker.check_permission = MagicMock(return_value=True)

        registry = SkillRegistry(rbac_checker=rbac_checker)
        skill = MockSkill(name="test_skill", required_permissions=["test:perm"])
        registry.register(skill)
        # Context has the permission so skill's own check passes, then RBAC is used
        context = SkillContext(permissions=["test:perm"])

        result = await registry.invoke("test_skill", {"query": "test"}, context)

        assert result.success is True
        rbac_checker.check_permission.assert_called()

    @pytest.mark.asyncio
    async def test_rbac_checker_denies_access(self):
        """Test RBAC checker can deny access."""
        rbac_checker = MagicMock()
        rbac_checker.check_permission = MagicMock(return_value=False)

        registry = SkillRegistry(rbac_checker=rbac_checker)
        skill = MockSkill(name="test_skill", required_permissions=["test:perm"])
        registry.register(skill)
        # Context has the permission so skill's own check passes, then RBAC denies
        context = SkillContext(permissions=["test:perm"])

        result = await registry.invoke("test_skill", {"query": "test"}, context)

        assert result.success is False
        assert result.status == SkillStatus.PERMISSION_DENIED

    @pytest.mark.asyncio
    async def test_async_rbac_checker(self):
        """Test async RBAC checker is awaited."""
        rbac_checker = MagicMock()
        rbac_checker.check_permission = AsyncMock(return_value=True)

        registry = SkillRegistry(rbac_checker=rbac_checker)
        skill = MockSkill(name="test_skill", required_permissions=["test:perm"])
        registry.register(skill)
        # Context has the permission so skill's own check passes
        context = SkillContext(permissions=["test:perm"])

        result = await registry.invoke("test_skill", {"query": "test"}, context)

        assert result.success is True


# =============================================================================
# Global Registry Tests
# =============================================================================


class TestGlobalRegistry:
    """Tests for global registry singleton."""

    def test_get_skill_registry_returns_singleton(self):
        """Test get_skill_registry returns same instance."""
        reset_skill_registry()  # Start fresh

        registry1 = get_skill_registry()
        registry2 = get_skill_registry()

        assert registry1 is registry2

    def test_reset_skill_registry(self):
        """Test reset_skill_registry creates new instance."""
        registry1 = get_skill_registry()
        reset_skill_registry()
        registry2 = get_skill_registry()

        assert registry1 is not registry2
