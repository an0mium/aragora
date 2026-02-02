"""
Tests for device connector registry.

Tests cover:
- DeviceConnectorRegistry initialization
- Connector loading and lazy initialization
- Get, register, and unregister operations
- Platform discovery
- Device type lookup
- Health aggregation
- Global registry functions
"""

import os
import pytest
import threading
from unittest.mock import AsyncMock, MagicMock, patch

from aragora.connectors.devices.registry import (
    DeviceConnectorRegistry,
    get_registry,
    get_connector,
    register_connector,
    list_available_platforms,
    get_configured_platforms,
)
from aragora.connectors.devices.base import DeviceConnector, DeviceConnectorConfig
from aragora.connectors.devices.models import DeviceType


class MockDeviceConnector(DeviceConnector):
    """Mock connector for testing."""

    def __init__(self, config=None, platform="mock"):
        super().__init__(config)
        self._platform = platform

    @property
    def platform_name(self) -> str:
        return self._platform

    @property
    def platform_display_name(self) -> str:
        return f"Mock {self._platform.upper()}"

    @property
    def supported_device_types(self):
        return [DeviceType.ANDROID]

    async def send_notification(self, device, message, **kwargs):
        pass


class TestDeviceConnectorRegistry:
    """Tests for DeviceConnectorRegistry class."""

    def test_initialization(self):
        """Should initialize with empty state."""
        registry = DeviceConnectorRegistry()

        assert registry._connectors == {}
        assert registry._initialized is False
        assert isinstance(registry._lock, type(threading.Lock()))

    def test_load_connector_classes_idempotent(self):
        """Should only load classes once."""
        registry = DeviceConnectorRegistry()

        registry._load_connector_classes()
        first_state = dict(registry.CONNECTOR_CLASSES)

        registry._load_connector_classes()
        second_state = dict(registry.CONNECTOR_CLASSES)

        # Should be the same after second call
        assert first_state == second_state

    def test_register_connector(self):
        """Should register connector instance."""
        registry = DeviceConnectorRegistry()
        connector = MockDeviceConnector()

        registry.register("mock", connector)

        assert "mock" in registry._connectors
        assert registry._connectors["mock"] is connector

    def test_unregister_connector(self):
        """Should unregister connector and return it."""
        registry = DeviceConnectorRegistry()
        connector = MockDeviceConnector()
        registry.register("mock", connector)

        result = registry.unregister("mock")

        assert result is connector
        assert "mock" not in registry._connectors

    def test_unregister_nonexistent(self):
        """Should return None for nonexistent connector."""
        registry = DeviceConnectorRegistry()

        result = registry.unregister("nonexistent")

        assert result is None

    def test_all_connectors(self):
        """Should return copy of all connectors."""
        registry = DeviceConnectorRegistry()
        connector1 = MockDeviceConnector(platform="mock1")
        connector2 = MockDeviceConnector(platform="mock2")

        registry.register("mock1", connector1)
        registry.register("mock2", connector2)

        all_connectors = registry.all()

        assert len(all_connectors) == 2
        assert all_connectors["mock1"] is connector1
        assert all_connectors["mock2"] is connector2

    def test_list_platforms(self):
        """Should list registered platform names."""
        registry = DeviceConnectorRegistry()
        registry.register("platform_a", MockDeviceConnector(platform="a"))
        registry.register("platform_b", MockDeviceConnector(platform="b"))

        platforms = registry.list_platforms()

        assert set(platforms) == {"platform_a", "platform_b"}

    def test_list_available_platforms(self):
        """Should list loadable platform names."""
        registry = DeviceConnectorRegistry()

        # Add a test class
        registry.CONNECTOR_CLASSES["test_platform"] = MockDeviceConnector
        registry._initialized = True

        available = registry.list_available_platforms()

        assert "test_platform" in available

    def test_get_for_device_type(self):
        """Should find connectors supporting device type."""
        registry = DeviceConnectorRegistry()
        android_connector = MockDeviceConnector(platform="android_connector")
        registry.register("android", android_connector)

        matching = registry.get_for_device_type(DeviceType.ANDROID)

        assert len(matching) == 1
        assert matching[0] is android_connector

    def test_get_for_device_type_no_match(self):
        """Should return empty list when no match."""
        registry = DeviceConnectorRegistry()
        connector = MockDeviceConnector()
        registry.register("mock", connector)

        matching = registry.get_for_device_type(DeviceType.ALEXA)

        assert matching == []


class TestRegistryGetMethod:
    """Tests for registry.get() method."""

    def test_get_cached_connector(self):
        """Should return cached connector."""
        registry = DeviceConnectorRegistry()
        connector = MockDeviceConnector()
        registry.register("mock", connector)

        result = registry.get("mock", auto_initialize=False)

        assert result is connector

    def test_get_unknown_platform(self):
        """Should raise KeyError for unknown platform."""
        registry = DeviceConnectorRegistry()
        registry._initialized = True  # Skip loading

        with pytest.raises(KeyError, match="Unknown device connector"):
            registry.get("unknown_platform", auto_initialize=False)

    def test_get_creates_new_connector(self):
        """Should create connector from registered class."""
        registry = DeviceConnectorRegistry()
        registry.CONNECTOR_CLASSES["mock"] = MockDeviceConnector
        registry._initialized = True

        connector = registry.get("mock", auto_initialize=False)

        assert isinstance(connector, MockDeviceConnector)
        assert "mock" in registry._connectors


class TestRegistryConfiguredPlatforms:
    """Tests for get_configured_platforms method."""

    def test_no_platforms_configured(self):
        """Should return empty list when nothing configured."""
        registry = DeviceConnectorRegistry()

        with patch.dict(os.environ, {}, clear=True):
            configured = registry.get_configured_platforms()

        assert configured == []

    def test_fcm_configured(self):
        """Should detect FCM configuration."""
        registry = DeviceConnectorRegistry()

        env = {
            "FCM_PROJECT_ID": "test-project",
            "FCM_PRIVATE_KEY": "test-key",
        }

        with patch.dict(os.environ, env, clear=True):
            configured = registry.get_configured_platforms()

        assert "fcm" in configured

    def test_fcm_with_google_credentials(self):
        """Should detect FCM with GOOGLE_APPLICATION_CREDENTIALS."""
        registry = DeviceConnectorRegistry()

        env = {
            "FCM_PROJECT_ID": "test-project",
            "GOOGLE_APPLICATION_CREDENTIALS": "/path/to/creds.json",
        }

        with patch.dict(os.environ, env, clear=True):
            configured = registry.get_configured_platforms()

        assert "fcm" in configured

    def test_apns_configured(self):
        """Should detect APNs configuration."""
        registry = DeviceConnectorRegistry()

        env = {
            "APNS_KEY_ID": "ABC123",
            "APNS_TEAM_ID": "TEAM123",
            "APNS_BUNDLE_ID": "com.test.app",
            "APNS_PRIVATE_KEY": "-----BEGIN PRIVATE KEY-----",
        }

        with patch.dict(os.environ, env, clear=True):
            configured = registry.get_configured_platforms()

        assert "apns" in configured

    def test_apns_partial_config(self):
        """Should not detect APNs with partial config."""
        registry = DeviceConnectorRegistry()

        env = {
            "APNS_KEY_ID": "ABC123",
            # Missing other required keys
        }

        with patch.dict(os.environ, env, clear=True):
            configured = registry.get_configured_platforms()

        assert "apns" not in configured

    def test_web_push_configured(self):
        """Should detect Web Push configuration."""
        registry = DeviceConnectorRegistry()

        env = {
            "VAPID_PUBLIC_KEY": "BTestPublicKey",
            "VAPID_PRIVATE_KEY": "TestPrivateKey",
            "VAPID_SUBJECT": "mailto:test@example.com",
        }

        with patch.dict(os.environ, env, clear=True):
            configured = registry.get_configured_platforms()

        assert "web_push" in configured

    def test_alexa_configured(self):
        """Should detect Alexa configuration."""
        registry = DeviceConnectorRegistry()

        env = {
            "ALEXA_CLIENT_ID": "client123",
            "ALEXA_CLIENT_SECRET": "secret456",
            "ALEXA_SKILL_ID": "skill789",
        }

        with patch.dict(os.environ, env, clear=True):
            configured = registry.get_configured_platforms()

        assert "alexa" in configured

    def test_google_home_configured(self):
        """Should detect Google Home configuration."""
        registry = DeviceConnectorRegistry()

        env = {
            "GOOGLE_HOME_PROJECT_ID": "project123",
            "GOOGLE_HOME_CREDENTIALS": "/path/to/creds.json",
        }

        with patch.dict(os.environ, env, clear=True):
            configured = registry.get_configured_platforms()

        assert "google_home" in configured


class TestRegistryInitializeAll:
    """Tests for initialize_all method."""

    @pytest.mark.asyncio
    async def test_initialize_all_success(self):
        """Should initialize all connector classes."""
        registry = DeviceConnectorRegistry()
        registry.CONNECTOR_CLASSES["mock"] = MockDeviceConnector
        registry._initialized = True

        results = await registry.initialize_all()

        assert "mock" in results
        # Result depends on initialize() implementation
        assert isinstance(results["mock"], bool)

    @pytest.mark.asyncio
    async def test_initialize_all_handles_errors(self):
        """Should handle initialization errors gracefully."""
        registry = DeviceConnectorRegistry()

        class FailingConnector(MockDeviceConnector):
            def __init__(self, config=None):
                raise ValueError("Init failed")

        registry.CONNECTOR_CLASSES["failing"] = FailingConnector
        registry._initialized = True

        results = await registry.initialize_all()

        assert results["failing"] is False


class TestRegistryShutdownAll:
    """Tests for shutdown_all method."""

    @pytest.mark.asyncio
    async def test_shutdown_all(self):
        """Should shutdown all connectors."""
        registry = DeviceConnectorRegistry()
        connector1 = MockDeviceConnector(platform="c1")
        connector2 = MockDeviceConnector(platform="c2")
        connector1.shutdown = AsyncMock()
        connector2.shutdown = AsyncMock()

        registry.register("c1", connector1)
        registry.register("c2", connector2)

        await registry.shutdown_all()

        connector1.shutdown.assert_called_once()
        connector2.shutdown.assert_called_once()
        assert registry._connectors == {}

    @pytest.mark.asyncio
    async def test_shutdown_handles_errors(self):
        """Should handle shutdown errors gracefully."""
        registry = DeviceConnectorRegistry()
        connector = MockDeviceConnector()
        connector.shutdown = AsyncMock(side_effect=RuntimeError("Shutdown failed"))

        registry.register("failing", connector)

        # Should not raise
        await registry.shutdown_all()


class TestRegistryHealth:
    """Tests for get_health method."""

    @pytest.mark.asyncio
    async def test_health_aggregation(self):
        """Should aggregate health from all connectors."""
        registry = DeviceConnectorRegistry()

        connector1 = MockDeviceConnector(platform="healthy1")
        connector1.get_health = AsyncMock(
            return_value={"status": "healthy", "platform": "healthy1"}
        )

        connector2 = MockDeviceConnector(platform="healthy2")
        connector2.get_health = AsyncMock(
            return_value={"status": "healthy", "platform": "healthy2"}
        )

        registry.register("h1", connector1)
        registry.register("h2", connector2)

        health = await registry.get_health()

        assert health["status"] == "healthy"
        assert len(health["connectors"]) == 2

    @pytest.mark.asyncio
    async def test_health_degraded_on_partial_failure(self):
        """Should report degraded when some connectors unhealthy."""
        registry = DeviceConnectorRegistry()

        healthy = MockDeviceConnector(platform="healthy")
        healthy.get_health = AsyncMock(return_value={"status": "healthy"})

        unhealthy = MockDeviceConnector(platform="unhealthy")
        unhealthy.get_health = AsyncMock(return_value={"status": "unhealthy"})

        registry.register("healthy", healthy)
        registry.register("unhealthy", unhealthy)

        health = await registry.get_health()

        assert health["status"] == "degraded"

    @pytest.mark.asyncio
    async def test_health_unhealthy_on_all_failure(self):
        """Should report unhealthy when all connectors fail."""
        registry = DeviceConnectorRegistry()

        connector = MockDeviceConnector()
        connector.get_health = AsyncMock(return_value={"status": "unhealthy"})

        registry.register("unhealthy", connector)

        health = await registry.get_health()

        assert health["status"] == "unhealthy"

    @pytest.mark.asyncio
    async def test_health_handles_exceptions(self):
        """Should handle exceptions during health check."""
        registry = DeviceConnectorRegistry()

        connector = MockDeviceConnector()
        connector.get_health = AsyncMock(side_effect=RuntimeError("Health check failed"))

        registry.register("error", connector)

        health = await registry.get_health()

        assert health["connectors"]["error"]["status"] == "error"
        assert "Health check failed" in health["connectors"]["error"]["error"]

    @pytest.mark.asyncio
    async def test_health_includes_platform_info(self):
        """Should include platform discovery info."""
        registry = DeviceConnectorRegistry()

        health = await registry.get_health()

        assert "configured_platforms" in health
        assert "available_platforms" in health


class TestGlobalRegistryFunctions:
    """Tests for module-level registry functions."""

    def test_get_registry_singleton(self):
        """Should return same registry instance."""
        # Reset global registry for test isolation
        import aragora.connectors.devices.registry as registry_module

        registry_module._registry = None

        registry1 = get_registry()
        registry2 = get_registry()

        assert registry1 is registry2

    def test_get_connector_delegates(self):
        """Should delegate to registry.get()."""
        registry = get_registry()

        # Register a mock connector
        mock_connector = MockDeviceConnector()
        registry.register("test_mock", mock_connector)

        try:
            result = get_connector("test_mock")
            assert result is mock_connector
        finally:
            registry.unregister("test_mock")

    def test_register_connector_delegates(self):
        """Should delegate to registry.register()."""
        registry = get_registry()
        connector = MockDeviceConnector()

        try:
            register_connector("test_reg", connector)
            assert registry._connectors.get("test_reg") is connector
        finally:
            registry.unregister("test_reg")

    def test_list_available_platforms_delegates(self):
        """Should delegate to registry."""
        result = list_available_platforms()
        assert isinstance(result, list)

    def test_get_configured_platforms_delegates(self):
        """Should delegate to registry."""
        result = get_configured_platforms()
        assert isinstance(result, list)


class TestThreadSafety:
    """Tests for thread safety."""

    def test_concurrent_registration(self):
        """Should handle concurrent registration safely."""
        registry = DeviceConnectorRegistry()
        results = []

        def register_connector(name):
            connector = MockDeviceConnector(platform=name)
            registry.register(name, connector)
            results.append(name)

        threads = [
            threading.Thread(target=register_connector, args=(f"conn_{i}",)) for i in range(10)
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(results) == 10
        assert len(registry._connectors) == 10

    def test_concurrent_access(self):
        """Should handle concurrent access safely."""
        registry = DeviceConnectorRegistry()

        for i in range(5):
            registry.register(f"conn_{i}", MockDeviceConnector(platform=f"c{i}"))

        access_count = 0

        def access_registry():
            nonlocal access_count
            _ = registry.all()
            _ = registry.list_platforms()
            access_count += 1

        threads = [threading.Thread(target=access_registry) for _ in range(20)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert access_count == 20
