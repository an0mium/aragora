"""Tests for the unified connector runtime registry and management handler.

Covers:
- ConnectorStatus enum values
- ConnectorInfo dataclass and serialization
- ConnectorRegistry singleton, discovery, CRUD, health-checks, and summary
- ConnectorManagementHandler routing and response formatting
"""

from __future__ import annotations

import time
from typing import Any
from unittest.mock import patch

import pytest

from aragora.connectors.runtime_registry import (
    ConnectorInfo,
    ConnectorRegistry,
    ConnectorStatus,
    get_connector_registry,
)
from aragora.server.handlers.connectors.management import ConnectorManagementHandler


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_singleton():
    """Ensure every test starts with a fresh registry singleton."""
    ConnectorRegistry.reset_instance()
    yield
    ConnectorRegistry.reset_instance()


@pytest.fixture()
def empty_registry() -> ConnectorRegistry:
    """Return a registry that skipped automatic discovery."""
    with patch.object(ConnectorRegistry, "_discover_connectors"):
        registry = ConnectorRegistry()
    return registry


@pytest.fixture()
def sample_info() -> ConnectorInfo:
    return ConnectorInfo(
        name="test_connector",
        connector_type="chat",
        module_path="aragora.connectors.chat.test_connector",
        status=ConnectorStatus.HEALTHY,
        last_health_check=time.time(),
        capabilities=["messaging", "webhooks"],
        metadata={"importable": True},
    )


@pytest.fixture()
def populated_registry(
    empty_registry: ConnectorRegistry, sample_info: ConnectorInfo
) -> ConnectorRegistry:
    """Registry with a handful of connectors for filtering tests."""
    empty_registry.register(sample_info)
    empty_registry.register(
        ConnectorInfo(
            name="stripe",
            connector_type="payment",
            module_path="aragora.connectors.payments.stripe",
            status=ConnectorStatus.HEALTHY,
            capabilities=["payments", "subscriptions"],
        )
    )
    empty_registry.register(
        ConnectorInfo(
            name="kafka",
            connector_type="enterprise",
            module_path="aragora.connectors.enterprise.streaming.kafka",
            status=ConnectorStatus.UNHEALTHY,
            capabilities=["streaming", "events"],
            metadata={"importable": False, "import_error": "no module"},
        )
    )
    return empty_registry


@pytest.fixture()
def handler(populated_registry: ConnectorRegistry) -> ConnectorManagementHandler:
    """Management handler backed by the populated_registry."""
    h = ConnectorManagementHandler(server_context={})
    h._registry = populated_registry
    return h


# ===================================================================
# ConnectorStatus
# ===================================================================


class TestConnectorStatus:
    def test_enum_values(self):
        assert ConnectorStatus.HEALTHY.value == "healthy"
        assert ConnectorStatus.DEGRADED.value == "degraded"
        assert ConnectorStatus.UNHEALTHY.value == "unhealthy"
        assert ConnectorStatus.UNKNOWN.value == "unknown"

    def test_enum_from_value(self):
        assert ConnectorStatus("healthy") is ConnectorStatus.HEALTHY

    def test_enum_members_count(self):
        assert len(ConnectorStatus) == 4


# ===================================================================
# ConnectorInfo
# ===================================================================


class TestConnectorInfo:
    def test_defaults(self):
        info = ConnectorInfo(name="x", connector_type="chat", module_path="a.b.c")
        assert info.status is ConnectorStatus.UNKNOWN
        assert info.last_health_check is None
        assert info.capabilities == []
        assert info.metadata == {}

    def test_to_dict(self, sample_info: ConnectorInfo):
        d = sample_info.to_dict()
        assert d["name"] == "test_connector"
        assert d["connector_type"] == "chat"
        assert d["status"] == "healthy"
        assert "messaging" in d["capabilities"]
        assert isinstance(d["metadata"], dict)

    def test_to_dict_returns_copies(self, sample_info: ConnectorInfo):
        """Mutating the returned dict must not affect the original."""
        d = sample_info.to_dict()
        d["capabilities"].append("evil")
        assert "evil" not in sample_info.capabilities

    def test_to_dict_status_serialization(self):
        for status in ConnectorStatus:
            info = ConnectorInfo(name="s", connector_type="ai", module_path="m", status=status)
            assert info.to_dict()["status"] == status.value


# ===================================================================
# ConnectorRegistry — basics
# ===================================================================


class TestRegistryBasics:
    def test_singleton(self):
        a = ConnectorRegistry.get_instance()
        b = ConnectorRegistry.get_instance()
        assert a is b

    def test_reset_clears_singleton(self):
        a = ConnectorRegistry.get_instance()
        ConnectorRegistry.reset_instance()
        b = ConnectorRegistry.get_instance()
        assert a is not b

    def test_register_and_get(self, empty_registry: ConnectorRegistry, sample_info: ConnectorInfo):
        empty_registry.register(sample_info)
        assert empty_registry.get("test_connector") is sample_info

    def test_get_missing_returns_none(self, empty_registry: ConnectorRegistry):
        assert empty_registry.get("nonexistent") is None

    def test_unregister(self, empty_registry: ConnectorRegistry, sample_info: ConnectorInfo):
        empty_registry.register(sample_info)
        assert empty_registry.unregister("test_connector") is True
        assert empty_registry.get("test_connector") is None

    def test_unregister_missing(self, empty_registry: ConnectorRegistry):
        assert empty_registry.unregister("nope") is False


# ===================================================================
# ConnectorRegistry — listing & filtering
# ===================================================================


class TestRegistryListing:
    def test_list_all(self, populated_registry: ConnectorRegistry):
        names = [c.name for c in populated_registry.list_all()]
        assert "kafka" in names
        assert "stripe" in names
        assert "test_connector" in names

    def test_list_all_sorted(self, populated_registry: ConnectorRegistry):
        names = [c.name for c in populated_registry.list_all()]
        assert names == sorted(names)

    def test_list_by_type(self, populated_registry: ConnectorRegistry):
        payments = populated_registry.list_by_type("payment")
        assert len(payments) == 1
        assert payments[0].name == "stripe"

    def test_list_by_type_empty(self, populated_registry: ConnectorRegistry):
        assert populated_registry.list_by_type("nonexistent_type") == []

    def test_list_by_status(self, populated_registry: ConnectorRegistry):
        unhealthy = populated_registry.list_by_status(ConnectorStatus.UNHEALTHY)
        assert len(unhealthy) == 1
        assert unhealthy[0].name == "kafka"


# ===================================================================
# ConnectorRegistry — health checks
# ===================================================================


class TestRegistryHealthCheck:
    def test_health_check_unknown_connector(self, empty_registry: ConnectorRegistry):
        assert empty_registry.health_check("ghost") is ConnectorStatus.UNKNOWN

    def test_health_check_importable(self, empty_registry: ConnectorRegistry):
        """A connector whose module is importable should become HEALTHY."""
        info = ConnectorInfo(
            name="json_mod",
            connector_type="ai",
            module_path="json",  # stdlib, always importable
        )
        empty_registry.register(info)
        status = empty_registry.health_check("json_mod")
        assert status is ConnectorStatus.HEALTHY
        assert info.last_health_check is not None

    def test_health_check_not_importable(self, empty_registry: ConnectorRegistry):
        info = ConnectorInfo(
            name="fake",
            connector_type="ai",
            module_path="aragora.connectors.does_not_exist_xyz",
        )
        empty_registry.register(info)
        status = empty_registry.health_check("fake")
        assert status is ConnectorStatus.UNHEALTHY
        assert info.metadata.get("importable") is False

    def test_health_check_all(self, populated_registry: ConnectorRegistry):
        results = populated_registry.health_check_all()
        assert isinstance(results, dict)
        assert set(results.keys()) == {"test_connector", "stripe", "kafka"}
        for v in results.values():
            assert isinstance(v, ConnectorStatus)


# ===================================================================
# ConnectorRegistry — summary
# ===================================================================


class TestRegistrySummary:
    def test_summary_structure(self, populated_registry: ConnectorRegistry):
        s = populated_registry.get_summary()
        assert s["total"] == 3
        assert "by_type" in s
        assert "by_status" in s
        assert "connectors" in s

    def test_summary_by_type(self, populated_registry: ConnectorRegistry):
        s = populated_registry.get_summary()
        assert s["by_type"]["chat"] == 1
        assert s["by_type"]["payment"] == 1
        assert s["by_type"]["enterprise"] == 1

    def test_summary_by_status(self, populated_registry: ConnectorRegistry):
        s = populated_registry.get_summary()
        assert s["by_status"]["healthy"] == 2
        assert s["by_status"]["unhealthy"] == 1


# ===================================================================
# ConnectorRegistry — discovery integration
# ===================================================================


class TestDiscovery:
    def test_full_discovery_finds_connectors(self):
        """When the real discovery runs it should find at least some connectors."""
        registry = ConnectorRegistry()
        assert len(registry.list_all()) > 0

    def test_discovered_connectors_have_status(self):
        registry = ConnectorRegistry()
        for c in registry.list_all():
            assert isinstance(c.status, ConnectorStatus)


# ===================================================================
# get_connector_registry convenience function
# ===================================================================


class TestGetConnectorRegistry:
    def test_returns_singleton(self):
        a = get_connector_registry()
        b = get_connector_registry()
        assert a is b


# ===================================================================
# ConnectorManagementHandler — GET routes
# ===================================================================


class TestManagementHandlerGET:
    def test_list_all(self, handler: ConnectorManagementHandler):
        result = handler.handle("/api/v1/connectors", {}, None)
        assert result is not None
        data, status, _ = result
        assert status == 200
        assert data["total"] == 3

    def test_list_with_trailing_slash(self, handler: ConnectorManagementHandler):
        result = handler.handle("/api/v1/connectors/", {}, None)
        assert result is not None
        _, status, _ = result
        assert status == 200

    def test_list_by_type_filter(self, handler: ConnectorManagementHandler):
        result = handler.handle("/api/v1/connectors", {"type": "payment"}, None)
        assert result is not None
        data, status, _ = result
        assert status == 200
        assert data["total"] == 1
        assert data["connectors"][0]["name"] == "stripe"

    def test_list_by_status_filter(self, handler: ConnectorManagementHandler):
        result = handler.handle("/api/v1/connectors", {"status": "unhealthy"}, None)
        assert result is not None
        data, status, _ = result
        assert status == 200
        assert data["total"] == 1
        assert data["connectors"][0]["name"] == "kafka"

    def test_list_invalid_status_filter(self, handler: ConnectorManagementHandler):
        result = handler.handle("/api/v1/connectors", {"status": "bogus"}, None)
        assert result is not None
        _, status, _ = result
        assert status == 400

    def test_summary(self, handler: ConnectorManagementHandler):
        result = handler.handle("/api/v1/connectors/summary", {}, None)
        assert result is not None
        data, status, _ = result
        assert status == 200
        assert data["total"] == 3
        assert "by_type" in data

    def test_detail(self, handler: ConnectorManagementHandler):
        result = handler.handle("/api/v1/connectors/stripe", {}, None)
        assert result is not None
        data, status, _ = result
        assert status == 200
        assert data["name"] == "stripe"
        assert data["connector_type"] == "payment"

    def test_detail_not_found(self, handler: ConnectorManagementHandler):
        result = handler.handle("/api/v1/connectors/nonexistent", {}, None)
        assert result is not None
        _, status, _ = result
        assert status == 404

    def test_detail_invalid_name(self, handler: ConnectorManagementHandler):
        result = handler.handle("/api/v1/connectors/bad-name!", {}, None)
        assert result is not None
        _, status, _ = result
        assert status == 400

    def test_health_check(self, handler: ConnectorManagementHandler):
        result = handler.handle("/api/v1/connectors/stripe/health", {}, None)
        assert result is not None
        data, status, _ = result
        assert status == 200
        assert data["name"] == "stripe"
        assert "status" in data

    def test_health_check_not_found(self, handler: ConnectorManagementHandler):
        result = handler.handle("/api/v1/connectors/ghost/health", {}, None)
        assert result is not None
        _, status, _ = result
        assert status == 404

    def test_unrelated_path_returns_none(self, handler: ConnectorManagementHandler):
        result = handler.handle("/api/v1/debates", {}, None)
        assert result is None


# ===================================================================
# ConnectorManagementHandler — POST routes
# ===================================================================


class TestManagementHandlerPOST:
    def test_test_connector(self, handler: ConnectorManagementHandler):
        result = handler.handle_post("/api/v1/connectors/stripe/test", {}, None)
        assert result is not None
        data, status, _ = result
        assert status == 200
        assert data["name"] == "stripe"
        assert "status" in data
        assert "capabilities" in data

    def test_test_not_found(self, handler: ConnectorManagementHandler):
        result = handler.handle_post("/api/v1/connectors/nope/test", {}, None)
        assert result is not None
        _, status, _ = result
        assert status == 404

    def test_test_unhealthy_includes_error(self, handler: ConnectorManagementHandler):
        """A connector whose module cannot be imported should surface an error."""
        # Register a genuinely non-importable connector so the /test endpoint
        # reports an error or warning in its response.
        registry = handler._get_registry()
        registry.register(
            ConnectorInfo(
                name="broken",
                connector_type="ai",
                module_path="aragora.connectors.does_not_exist_xyz",
                status=ConnectorStatus.UNHEALTHY,
                capabilities=[],
                metadata={"importable": False, "import_error": "no module"},
            )
        )
        result = handler.handle_post("/api/v1/connectors/broken/test", {}, None)
        assert result is not None
        data, status, _ = result
        assert status == 200
        assert "error" in data or "warning" in data

    def test_post_unrelated_path_returns_none(self, handler: ConnectorManagementHandler):
        result = handler.handle_post("/api/v1/debates", {}, None)
        assert result is None
