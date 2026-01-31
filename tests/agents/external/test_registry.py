"""Tests for external agent registry."""

import pytest

from aragora.agents.external.base import ExternalAgentAdapter
from aragora.agents.external.config import ExternalAgentConfig
from aragora.agents.external.models import (
    HealthStatus,
    TaskRequest,
    TaskResult,
    TaskStatus,
)
from aragora.agents.external.registry import (
    ExternalAdapterSpec,
    ExternalAgentRegistry,
)


class MockConfig(ExternalAgentConfig):
    """Mock config for testing."""

    adapter_name: str = "mock"


class MockAdapter(ExternalAgentAdapter):
    """Mock adapter for testing."""

    adapter_name = "mock"

    async def submit_task(self, request: TaskRequest) -> str:
        return "mock-task-123"

    async def get_task_status(self, task_id: str) -> TaskStatus:
        return TaskStatus.COMPLETED

    async def get_task_result(self, task_id: str) -> TaskResult:
        return TaskResult(task_id=task_id, status=TaskStatus.COMPLETED)

    async def cancel_task(self, task_id: str) -> bool:
        return True

    async def health_check(self) -> HealthStatus:
        from datetime import datetime, timezone

        return HealthStatus(
            adapter_name=self.adapter_name,
            healthy=True,
            last_check=datetime.now(timezone.utc),
            response_time_ms=10.0,
        )


class TestExternalAdapterSpec:
    """Tests for ExternalAdapterSpec."""

    def test_spec_creation(self) -> None:
        """Test creating an adapter spec."""
        spec = ExternalAdapterSpec(
            name="test",
            adapter_class=MockAdapter,
            config_class=MockConfig,
            description="Test adapter",
            requires="Nothing",
            env_vars="TEST_VAR",
        )
        assert spec.name == "test"
        assert spec.adapter_class == MockAdapter
        assert spec.config_class == MockConfig
        assert spec.description == "Test adapter"

    def test_spec_immutability(self) -> None:
        """Test that spec is immutable (frozen)."""
        spec = ExternalAdapterSpec(
            name="test",
            adapter_class=MockAdapter,
            config_class=MockConfig,
        )
        with pytest.raises(AttributeError):
            spec.name = "changed"  # type: ignore


class TestExternalAgentRegistry:
    """Tests for ExternalAgentRegistry."""

    def setup_method(self) -> None:
        """Clear registry before each test."""
        ExternalAgentRegistry.clear()

    def teardown_method(self) -> None:
        """Clear registry after each test."""
        ExternalAgentRegistry.clear()

    def test_register_decorator(self) -> None:
        """Test registering an adapter via decorator."""

        @ExternalAgentRegistry.register(
            "test_adapter",
            config_class=MockConfig,
            description="Test adapter",
        )
        class TestAdapter(MockAdapter):
            adapter_name = "test_adapter"

        assert ExternalAgentRegistry.is_registered("test_adapter")
        spec = ExternalAgentRegistry.get_spec("test_adapter")
        assert spec is not None
        assert spec.adapter_class == TestAdapter

    def test_create_adapter(self) -> None:
        """Test creating an adapter from registry."""

        @ExternalAgentRegistry.register(
            "createable",
            config_class=MockConfig,
        )
        class CreateableAdapter(MockAdapter):
            adapter_name = "createable"

        adapter = ExternalAgentRegistry.create("createable")
        assert adapter is not None
        assert adapter.adapter_name == "createable"
        assert isinstance(adapter.config, MockConfig)

    def test_create_with_custom_config(self) -> None:
        """Test creating adapter with custom config."""

        @ExternalAgentRegistry.register(
            "configurable",
            config_class=MockConfig,
        )
        class ConfigurableAdapter(MockAdapter):
            adapter_name = "configurable"

        config = MockConfig(timeout_seconds=500.0)
        adapter = ExternalAgentRegistry.create("configurable", config=config)
        assert adapter.config.timeout_seconds == 500.0

    def test_create_unknown_adapter_raises(self) -> None:
        """Test that creating unknown adapter raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            ExternalAgentRegistry.create("nonexistent")
        assert "Unknown adapter" in str(exc_info.value)

    def test_list_all(self) -> None:
        """Test listing all registered adapters."""

        @ExternalAgentRegistry.register(
            "listed1",
            config_class=MockConfig,
            description="First adapter",
        )
        class Listed1(MockAdapter):
            pass

        @ExternalAgentRegistry.register(
            "listed2",
            config_class=MockConfig,
            description="Second adapter",
        )
        class Listed2(MockAdapter):
            pass

        all_adapters = ExternalAgentRegistry.list_all()
        assert "listed1" in all_adapters
        assert "listed2" in all_adapters
        assert all_adapters["listed1"]["description"] == "First adapter"

    def test_get_registered_names(self) -> None:
        """Test getting list of registered names."""

        @ExternalAgentRegistry.register("alpha", config_class=MockConfig)
        class Alpha(MockAdapter):
            pass

        @ExternalAgentRegistry.register("beta", config_class=MockConfig)
        class Beta(MockAdapter):
            pass

        names = ExternalAgentRegistry.get_registered_names()
        assert "alpha" in names
        assert "beta" in names
        assert names == sorted(names)  # Should be sorted

    def test_is_registered(self) -> None:
        """Test checking if adapter is registered."""

        @ExternalAgentRegistry.register("exists", config_class=MockConfig)
        class Exists(MockAdapter):
            pass

        assert ExternalAgentRegistry.is_registered("exists")
        assert not ExternalAgentRegistry.is_registered("not_exists")

    def test_clear(self) -> None:
        """Test clearing the registry."""

        @ExternalAgentRegistry.register("clearable", config_class=MockConfig)
        class Clearable(MockAdapter):
            pass

        assert ExternalAgentRegistry.is_registered("clearable")
        ExternalAgentRegistry.clear()
        assert not ExternalAgentRegistry.is_registered("clearable")
