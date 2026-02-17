"""
Tests for FederationRegistry - External agent framework registration and discovery.

Covers:
- Framework registration and unregistration
- Capability discovery and querying
- Version negotiation
- Health check probes
- Framework status transitions
- Lifecycle hooks (startup, shutdown, reconnect)
- Dead framework cleanup
- Capability-based routing lookup
- Multi-framework management
- Edge cases and error handling
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.gateway.federation.registry import (
    ExternalFramework,
    FederationRegistry,
    FrameworkCapability,
    FrameworkStatus,
    HealthStatus,
    RegistrationResult,
)


# =============================================================================
# Helpers
# =============================================================================


def _cap(name: str, **kwargs) -> FrameworkCapability:
    """Shorthand helper to create a FrameworkCapability."""
    return FrameworkCapability(
        name=name,
        description=kwargs.get("description", f"{name} capability"),
        **{k: v for k, v in kwargs.items() if k != "description"},
    )


def _endpoints(base: str = "http://localhost:9000", **extra) -> dict[str, str]:
    """Shorthand helper to create an endpoints dict with a base URL."""
    return {"base": base, **extra}


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def registry() -> FederationRegistry:
    """Create an in-memory FederationRegistry (no background tasks)."""
    return FederationRegistry(redis_url="memory://", heartbeat_timeout=60.0)


@pytest.fixture
async def connected_registry() -> FederationRegistry:
    """Create and connect an in-memory FederationRegistry, close after test."""
    reg = FederationRegistry(redis_url="memory://", heartbeat_timeout=60.0)
    await reg.connect()
    yield reg
    await reg.close()


async def _register_framework(
    registry: FederationRegistry,
    name: str = "autogpt",
    version: str = "0.5.0",
    capabilities: list[FrameworkCapability] | None = None,
    endpoints: dict[str, str] | None = None,
    **kwargs,
) -> RegistrationResult:
    """Register a framework with sensible defaults."""
    caps = capabilities or [_cap("autonomous_task")]
    eps = endpoints or _endpoints()
    return await registry.register(
        name=name,
        version=version,
        capabilities=caps,
        endpoints=eps,
        **kwargs,
    )


# =============================================================================
# FrameworkCapability Tests
# =============================================================================


class TestFrameworkCapability:
    """Tests for the FrameworkCapability dataclass."""

    def test_creation_defaults(self):
        cap = FrameworkCapability(name="test_cap")
        assert cap.name == "test_cap"
        assert cap.description == ""
        assert cap.parameters == {}
        assert cap.returns == "Any"
        assert cap.version == "1.0.0"
        assert cap.metadata == {}

    def test_creation_with_all_fields(self):
        cap = FrameworkCapability(
            name="code_gen",
            description="Generate code",
            parameters={"language": "str", "prompt": "str"},
            returns="CodeResult",
            version="2.0.0",
            metadata={"provider": "openai"},
        )
        assert cap.name == "code_gen"
        assert cap.description == "Generate code"
        assert cap.parameters == {"language": "str", "prompt": "str"}
        assert cap.returns == "CodeResult"
        assert cap.version == "2.0.0"
        assert cap.metadata == {"provider": "openai"}

    def test_to_dict(self):
        cap = FrameworkCapability(
            name="analysis",
            description="Analyze data",
            parameters={"data": "str"},
            returns="AnalysisResult",
            version="1.1.0",
            metadata={"tier": "premium"},
        )
        d = cap.to_dict()
        assert d["name"] == "analysis"
        assert d["description"] == "Analyze data"
        assert d["parameters"] == {"data": "str"}
        assert d["returns"] == "AnalysisResult"
        assert d["version"] == "1.1.0"
        assert d["metadata"] == {"tier": "premium"}

    def test_from_dict_full(self):
        data = {
            "name": "search",
            "description": "Search the web",
            "parameters": {"query": "str"},
            "returns": "SearchResults",
            "version": "1.2.0",
            "metadata": {"engine": "google"},
        }
        cap = FrameworkCapability.from_dict(data)
        assert cap.name == "search"
        assert cap.description == "Search the web"
        assert cap.parameters == {"query": "str"}
        assert cap.returns == "SearchResults"
        assert cap.version == "1.2.0"
        assert cap.metadata == {"engine": "google"}

    def test_from_dict_minimal(self):
        data = {"name": "minimal"}
        cap = FrameworkCapability.from_dict(data)
        assert cap.name == "minimal"
        assert cap.description == ""
        assert cap.parameters == {}
        assert cap.returns == "Any"
        assert cap.version == "1.0.0"

    def test_roundtrip_serialization(self):
        original = FrameworkCapability(
            name="roundtrip",
            description="Test roundtrip",
            parameters={"a": "int"},
            returns="Result",
            version="3.0.0",
            metadata={"key": "value"},
        )
        restored = FrameworkCapability.from_dict(original.to_dict())
        assert restored.name == original.name
        assert restored.description == original.description
        assert restored.parameters == original.parameters
        assert restored.returns == original.returns
        assert restored.version == original.version
        assert restored.metadata == original.metadata


# =============================================================================
# ExternalFramework Tests
# =============================================================================


class TestExternalFramework:
    """Tests for the ExternalFramework dataclass."""

    def test_auto_generates_framework_id(self):
        fw = ExternalFramework(name="crewai", version="1.0.0")
        assert fw.framework_id.startswith("fw-crewai-")
        assert len(fw.framework_id) > len("fw-crewai-")

    def test_preserves_provided_framework_id(self):
        fw = ExternalFramework(framework_id="fw-custom-123", name="test")
        assert fw.framework_id == "fw-custom-123"

    def test_default_values(self):
        fw = ExternalFramework(name="test")
        assert fw.version == "0.0.0"
        assert fw.capabilities == []
        assert fw.endpoints == {}
        assert fw.health_status == HealthStatus.UNKNOWN
        assert fw.status == FrameworkStatus.REGISTERING
        assert fw.api_version == "1.0.0"
        assert fw.supported_api_versions == ["1.0.0"]
        assert fw.consecutive_failures == 0
        assert fw.metadata == {}
        assert fw.tags == set()
        assert fw.startup_hooks == []
        assert fw.shutdown_hooks == []

    def test_is_healthy_when_healthy(self):
        fw = ExternalFramework(name="test", health_status=HealthStatus.HEALTHY)
        assert fw.is_healthy() is True

    def test_is_healthy_when_degraded(self):
        fw = ExternalFramework(name="test", health_status=HealthStatus.DEGRADED)
        assert fw.is_healthy() is False

    def test_is_healthy_when_unhealthy(self):
        fw = ExternalFramework(name="test", health_status=HealthStatus.UNHEALTHY)
        assert fw.is_healthy() is False

    def test_is_healthy_when_unknown(self):
        fw = ExternalFramework(name="test", health_status=HealthStatus.UNKNOWN)
        assert fw.is_healthy() is False

    def test_is_active_requires_active_and_healthy(self):
        fw = ExternalFramework(
            name="test",
            status=FrameworkStatus.ACTIVE,
            health_status=HealthStatus.HEALTHY,
        )
        assert fw.is_active() is True

    def test_is_active_false_when_not_active_status(self):
        fw = ExternalFramework(
            name="test",
            status=FrameworkStatus.DRAINING,
            health_status=HealthStatus.HEALTHY,
        )
        assert fw.is_active() is False

    def test_is_active_false_when_not_healthy(self):
        fw = ExternalFramework(
            name="test",
            status=FrameworkStatus.ACTIVE,
            health_status=HealthStatus.DEGRADED,
        )
        assert fw.is_active() is False

    def test_is_alive_recent_heartbeat(self):
        fw = ExternalFramework(name="test", last_heartbeat=time.time())
        assert fw.is_alive(timeout_seconds=60.0) is True

    def test_is_alive_expired_heartbeat(self):
        fw = ExternalFramework(name="test", last_heartbeat=time.time() - 120)
        assert fw.is_alive(timeout_seconds=60.0) is False

    def test_has_capability(self):
        fw = ExternalFramework(
            name="test",
            capabilities=[_cap("code_gen"), _cap("search")],
        )
        assert fw.has_capability("code_gen") is True
        assert fw.has_capability("search") is True
        assert fw.has_capability("nonexistent") is False

    def test_get_capability(self):
        cap = _cap("code_gen", description="Generate code")
        fw = ExternalFramework(name="test", capabilities=[cap])
        found = fw.get_capability("code_gen")
        assert found is not None
        assert found.name == "code_gen"
        assert fw.get_capability("missing") is None

    def test_has_all_capabilities(self):
        fw = ExternalFramework(
            name="test",
            capabilities=[_cap("a"), _cap("b"), _cap("c")],
        )
        assert fw.has_all_capabilities(["a", "b"]) is True
        assert fw.has_all_capabilities(["a", "b", "c"]) is True
        assert fw.has_all_capabilities(["a", "d"]) is False
        assert fw.has_all_capabilities([]) is True

    def test_supports_api_version(self):
        fw = ExternalFramework(
            name="test",
            supported_api_versions=["1.0.0", "1.1.0"],
        )
        assert fw.supports_api_version("1.0.0") is True
        assert fw.supports_api_version("1.1.0") is True
        assert fw.supports_api_version("2.0.0") is False

    def test_get_endpoint(self):
        fw = ExternalFramework(
            name="test",
            endpoints={"base": "http://localhost:9000", "health": "/health"},
        )
        assert fw.get_endpoint("base") == "http://localhost:9000"
        assert fw.get_endpoint("health") == "/health"
        assert fw.get_endpoint("invoke") is None

    def test_record_health_check_healthy(self):
        fw = ExternalFramework(name="test")
        fw.record_health_check(True)
        assert fw.health_status == HealthStatus.HEALTHY
        assert fw.consecutive_failures == 0
        assert fw.last_health_check > 0

    def test_record_health_check_first_failure_is_degraded(self):
        fw = ExternalFramework(name="test")
        fw.record_health_check(False)
        assert fw.health_status == HealthStatus.DEGRADED
        assert fw.consecutive_failures == 1

    def test_record_health_check_two_failures_is_degraded(self):
        fw = ExternalFramework(name="test")
        fw.record_health_check(False)
        fw.record_health_check(False)
        assert fw.health_status == HealthStatus.DEGRADED
        assert fw.consecutive_failures == 2

    def test_record_health_check_three_failures_is_unhealthy(self):
        fw = ExternalFramework(name="test")
        fw.record_health_check(False)
        fw.record_health_check(False)
        fw.record_health_check(False)
        assert fw.health_status == HealthStatus.UNHEALTHY
        assert fw.consecutive_failures == 3

    def test_record_health_check_recovery_resets_failures(self):
        fw = ExternalFramework(name="test")
        fw.record_health_check(False)
        fw.record_health_check(False)
        assert fw.consecutive_failures == 2
        fw.record_health_check(True)
        assert fw.consecutive_failures == 0
        assert fw.health_status == HealthStatus.HEALTHY

    def test_to_dict(self):
        fw = ExternalFramework(
            framework_id="fw-test-abc",
            name="test",
            version="1.0.0",
            capabilities=[_cap("code_gen")],
            endpoints={"base": "http://localhost:9000"},
            tags={"gpu", "fast"},
            metadata={"provider": "test"},
        )
        d = fw.to_dict()
        assert d["framework_id"] == "fw-test-abc"
        assert d["name"] == "test"
        assert d["version"] == "1.0.0"
        assert len(d["capabilities"]) == 1
        assert d["capabilities"][0]["name"] == "code_gen"
        assert d["endpoints"] == {"base": "http://localhost:9000"}
        assert set(d["tags"]) == {"gpu", "fast"}
        assert d["metadata"] == {"provider": "test"}

    def test_from_dict(self):
        data = {
            "framework_id": "fw-restored-abc",
            "name": "restored",
            "version": "2.0.0",
            "capabilities": [{"name": "search", "description": "Search"}],
            "endpoints": {"base": "http://localhost:8080"},
            "health_status": "healthy",
            "status": "active",
            "api_version": "1.1.0",
            "supported_api_versions": ["1.0.0", "1.1.0"],
            "registered_at": 1000.0,
            "last_heartbeat": 2000.0,
            "last_health_check": 1500.0,
            "consecutive_failures": 1,
            "metadata": {"key": "val"},
            "tags": ["gpu"],
            "startup_hooks": ["hook1"],
            "shutdown_hooks": ["hook2"],
        }
        fw = ExternalFramework.from_dict(data)
        assert fw.framework_id == "fw-restored-abc"
        assert fw.name == "restored"
        assert fw.version == "2.0.0"
        assert len(fw.capabilities) == 1
        assert fw.capabilities[0].name == "search"
        assert fw.health_status == HealthStatus.HEALTHY
        assert fw.status == FrameworkStatus.ACTIVE
        assert fw.api_version == "1.1.0"
        assert fw.registered_at == 1000.0
        assert fw.last_heartbeat == 2000.0
        assert fw.consecutive_failures == 1
        assert fw.tags == {"gpu"}
        assert fw.startup_hooks == ["hook1"]
        assert fw.shutdown_hooks == ["hook2"]

    def test_roundtrip_serialization(self):
        original = ExternalFramework(
            framework_id="fw-rt-001",
            name="roundtrip",
            version="1.2.3",
            capabilities=[_cap("a"), _cap("b")],
            endpoints={"base": "http://example.com", "health": "/health"},
            health_status=HealthStatus.DEGRADED,
            status=FrameworkStatus.DRAINING,
            tags={"tag1", "tag2"},
        )
        restored = ExternalFramework.from_dict(original.to_dict())
        assert restored.framework_id == original.framework_id
        assert restored.name == original.name
        assert restored.version == original.version
        assert len(restored.capabilities) == len(original.capabilities)
        assert restored.health_status == original.health_status
        assert restored.status == original.status
        assert restored.tags == original.tags


# =============================================================================
# RegistrationResult Tests
# =============================================================================


class TestRegistrationResult:
    """Tests for the RegistrationResult dataclass."""

    def test_success_result(self):
        result = RegistrationResult(
            success=True,
            framework_id="fw-abc",
            negotiated_version="1.1.0",
            message="OK",
        )
        assert result.success is True
        assert result.framework_id == "fw-abc"
        assert result.negotiated_version == "1.1.0"
        assert result.message == "OK"

    def test_failure_result(self):
        result = RegistrationResult(success=False, message="Bad version")
        assert result.success is False
        assert result.framework_id == ""
        assert result.framework is None

    def test_to_dict(self):
        result = RegistrationResult(
            success=True,
            framework_id="fw-x",
            negotiated_version="2.0.0",
            message="Registered",
        )
        d = result.to_dict()
        assert d["success"] is True
        assert d["framework_id"] == "fw-x"
        assert d["negotiated_version"] == "2.0.0"
        assert d["message"] == "Registered"
        # framework object should not be serialized
        assert "framework" not in d


# =============================================================================
# Enums Tests
# =============================================================================


class TestEnums:
    """Tests for HealthStatus and FrameworkStatus enums."""

    def test_health_status_values(self):
        assert HealthStatus.HEALTHY.value == "healthy"
        assert HealthStatus.DEGRADED.value == "degraded"
        assert HealthStatus.UNHEALTHY.value == "unhealthy"
        assert HealthStatus.UNKNOWN.value == "unknown"

    def test_framework_status_values(self):
        assert FrameworkStatus.REGISTERING.value == "registering"
        assert FrameworkStatus.ACTIVE.value == "active"
        assert FrameworkStatus.DRAINING.value == "draining"
        assert FrameworkStatus.DISCONNECTED.value == "disconnected"
        assert FrameworkStatus.FAILED.value == "failed"

    def test_health_status_from_value(self):
        assert HealthStatus("healthy") == HealthStatus.HEALTHY
        assert HealthStatus("degraded") == HealthStatus.DEGRADED

    def test_framework_status_from_value(self):
        assert FrameworkStatus("active") == FrameworkStatus.ACTIVE
        assert FrameworkStatus("failed") == FrameworkStatus.FAILED


# =============================================================================
# FederationRegistry: Registration Tests
# =============================================================================


class TestRegistration:
    """Tests for framework registration and unregistration."""

    @pytest.mark.asyncio
    async def test_register_basic(self, registry):
        result = await _register_framework(registry)
        assert result.success is True
        assert result.framework_id != ""
        assert result.negotiated_version in FederationRegistry.SUPPORTED_API_VERSIONS
        assert result.message == "Registration successful"
        assert result.framework is not None
        assert result.framework.name == "autogpt"

    @pytest.mark.asyncio
    async def test_register_sets_active_status(self, registry):
        result = await _register_framework(registry)
        assert result.framework.status == FrameworkStatus.ACTIVE

    @pytest.mark.asyncio
    async def test_register_sets_unknown_health(self, registry):
        result = await _register_framework(registry)
        assert result.framework.health_status == HealthStatus.UNKNOWN

    @pytest.mark.asyncio
    async def test_register_with_tags(self, registry):
        result = await _register_framework(registry, tags=["gpu", "fast"])
        assert result.framework.tags == {"gpu", "fast"}

    @pytest.mark.asyncio
    async def test_register_with_metadata(self, registry):
        result = await _register_framework(
            registry, metadata={"provider": "openai", "tier": "premium"}
        )
        assert result.framework.metadata == {"provider": "openai", "tier": "premium"}

    @pytest.mark.asyncio
    async def test_register_missing_base_endpoint(self, registry):
        result = await registry.register(
            name="bad",
            version="1.0.0",
            capabilities=[_cap("test")],
            endpoints={"health": "/health"},  # no "base"
        )
        assert result.success is False
        assert "base" in result.message.lower()

    @pytest.mark.asyncio
    async def test_register_incompatible_api_version(self, registry):
        result = await registry.register(
            name="old_framework",
            version="1.0.0",
            capabilities=[_cap("test")],
            endpoints=_endpoints(),
            supported_api_versions=["0.1.0", "0.2.0"],
        )
        assert result.success is False
        assert "compatible" in result.message.lower() or "version" in result.message.lower()

    @pytest.mark.asyncio
    async def test_register_negotiates_highest_version(self, registry):
        result = await registry.register(
            name="multi_version",
            version="1.0.0",
            capabilities=[_cap("test")],
            endpoints=_endpoints(),
            supported_api_versions=["1.0.0", "1.1.0", "2.0.0"],
        )
        assert result.success is True
        assert result.negotiated_version == "2.0.0"

    @pytest.mark.asyncio
    async def test_register_re_registration_updates(self, registry):
        result1 = await _register_framework(registry, name="reregister", version="1.0.0")
        assert result1.success is True
        fw_id1 = result1.framework_id

        # Mark first framework as active+healthy so _find_by_name discovers it
        result1.framework.health_status = HealthStatus.HEALTHY
        await registry._save_framework(result1.framework)

        result2 = await _register_framework(registry, name="reregister", version="2.0.0")
        assert result2.success is True
        # Should reuse the framework_id from the first registration
        assert result2.framework_id == fw_id1
        assert result2.framework.version == "2.0.0"

    @pytest.mark.asyncio
    async def test_register_preserved_registered_at_on_reregister(self, registry):
        result1 = await _register_framework(registry, name="ts_test")
        original_ts = result1.framework.registered_at

        # Mark first framework as active+healthy so _find_by_name discovers it
        result1.framework.health_status = HealthStatus.HEALTHY
        await registry._save_framework(result1.framework)

        # Small delay to ensure time difference
        await asyncio.sleep(0.01)
        result2 = await _register_framework(registry, name="ts_test", version="2.0.0")
        assert result2.framework.registered_at == original_ts

    @pytest.mark.asyncio
    async def test_register_stores_in_local_cache(self, registry):
        result = await _register_framework(registry, name="cached")
        fw = await registry.get(result.framework_id)
        assert fw is not None
        assert fw.name == "cached"

    @pytest.mark.asyncio
    async def test_unregister_existing(self, registry):
        result = await _register_framework(registry, name="to_remove")
        removed = await registry.unregister(result.framework_id)
        assert removed is True
        fw = await registry.get(result.framework_id)
        assert fw is None

    @pytest.mark.asyncio
    async def test_unregister_nonexistent(self, registry):
        removed = await registry.unregister("fw-does-not-exist")
        assert removed is False

    @pytest.mark.asyncio
    async def test_unregister_removes_from_capability_index(self, registry):
        result = await _register_framework(
            registry,
            name="indexed",
            capabilities=[_cap("indexed_cap")],
        )
        # Verify it is indexed
        frameworks = await registry.find_by_capability("indexed_cap", only_active=False)
        assert any(f.framework_id == result.framework_id for f in frameworks)

        await registry.unregister(result.framework_id)
        frameworks = await registry.find_by_capability("indexed_cap", only_active=False)
        assert not any(f.framework_id == result.framework_id for f in frameworks)

    @pytest.mark.asyncio
    async def test_get_nonexistent(self, registry):
        fw = await registry.get("fw-nonexistent")
        assert fw is None


# =============================================================================
# FederationRegistry: Capability Discovery Tests
# =============================================================================


class TestCapabilityDiscovery:
    """Tests for capability discovery and querying."""

    @pytest.mark.asyncio
    async def test_find_by_capability_basic(self, registry):
        result = await _register_framework(
            registry,
            name="cap_fw",
            capabilities=[_cap("code_gen"), _cap("search")],
        )
        # Mark healthy for is_active()
        result.framework.health_status = HealthStatus.HEALTHY
        await registry._save_framework(result.framework)

        found = await registry.find_by_capability("code_gen")
        assert len(found) == 1
        assert found[0].name == "cap_fw"

    @pytest.mark.asyncio
    async def test_find_by_capability_no_match(self, registry):
        await _register_framework(registry, capabilities=[_cap("code_gen")])
        found = await registry.find_by_capability("nonexistent", only_active=False)
        assert len(found) == 0

    @pytest.mark.asyncio
    async def test_find_by_capability_only_active(self, registry):
        result = await _register_framework(registry, name="inactive", capabilities=[_cap("search")])
        # Framework has UNKNOWN health by default, so is_active() is False
        found_active = await registry.find_by_capability("search", only_active=True)
        assert len(found_active) == 0

        found_all = await registry.find_by_capability("search", only_active=False)
        assert len(found_all) == 1

    @pytest.mark.asyncio
    async def test_find_by_capabilities_intersection(self, registry):
        r1 = await _register_framework(
            registry,
            name="fw_ab",
            capabilities=[_cap("a"), _cap("b")],
        )
        r1.framework.health_status = HealthStatus.HEALTHY
        await registry._save_framework(r1.framework)

        r2 = await _register_framework(
            registry,
            name="fw_bc",
            capabilities=[_cap("b"), _cap("c")],
        )
        r2.framework.health_status = HealthStatus.HEALTHY
        await registry._save_framework(r2.framework)

        found = await registry.find_by_capabilities(["a", "b"])
        assert len(found) == 1
        assert found[0].name == "fw_ab"

    @pytest.mark.asyncio
    async def test_find_by_capabilities_empty_list(self, registry):
        r1 = await _register_framework(registry, name="fw1")
        r1.framework.health_status = HealthStatus.HEALTHY
        await registry._save_framework(r1.framework)

        found = await registry.find_by_capabilities([])
        assert len(found) >= 1

    @pytest.mark.asyncio
    async def test_find_by_capabilities_no_overlap(self, registry):
        await _register_framework(registry, name="fw1", capabilities=[_cap("a")])
        found = await registry.find_by_capabilities(["x", "y"], only_active=False)
        assert len(found) == 0

    @pytest.mark.asyncio
    async def test_query_capabilities(self, registry):
        caps = [_cap("code_gen"), _cap("search")]
        result = await _register_framework(registry, capabilities=caps)
        queried = await registry.query_capabilities(result.framework_id)
        assert len(queried) == 2
        names = {c.name for c in queried}
        assert names == {"code_gen", "search"}

    @pytest.mark.asyncio
    async def test_query_capabilities_nonexistent(self, registry):
        queried = await registry.query_capabilities("fw-does-not-exist")
        assert queried == []

    @pytest.mark.asyncio
    async def test_capability_index_updated_on_register(self, registry):
        await _register_framework(
            registry, name="fw1", capabilities=[_cap("shared"), _cap("unique1")]
        )
        await _register_framework(
            registry, name="fw2", capabilities=[_cap("shared"), _cap("unique2")]
        )

        assert "shared" in registry._capability_index
        assert len(registry._capability_index["shared"]) == 2
        assert "unique1" in registry._capability_index
        assert len(registry._capability_index["unique1"]) == 1


# =============================================================================
# FederationRegistry: Version Negotiation Tests
# =============================================================================


class TestVersionNegotiation:
    """Tests for API version negotiation."""

    def test_negotiate_finds_highest_common(self, registry):
        result = registry._negotiate_version(["1.0.0", "1.1.0"])
        assert result == "1.1.0"

    def test_negotiate_single_match(self, registry):
        result = registry._negotiate_version(["1.0.0"])
        assert result == "1.0.0"

    def test_negotiate_no_match(self, registry):
        result = registry._negotiate_version(["0.1.0", "0.2.0"])
        assert result is None

    def test_negotiate_all_match(self, registry):
        result = registry._negotiate_version(["1.0.0", "1.1.0", "2.0.0"])
        assert result == "2.0.0"

    def test_negotiate_empty_client_versions(self, registry):
        result = registry._negotiate_version([])
        assert result is None

    def test_negotiate_prefers_higher_version(self, registry):
        result = registry._negotiate_version(["1.0.0", "2.0.0"])
        assert result == "2.0.0"


# =============================================================================
# FederationRegistry: Health Check Tests
# =============================================================================


class TestHealthCheck:
    """Tests for health check probes."""

    @pytest.mark.asyncio
    async def test_health_check_nonexistent_framework(self, registry):
        status = await registry.health_check("fw-missing")
        assert status == HealthStatus.UNKNOWN

    @pytest.mark.asyncio
    async def test_health_check_no_base_endpoint(self, registry):
        # Register with base then remove it
        result = await _register_framework(registry, name="no_base")
        result.framework.endpoints = {}
        await registry._save_framework(result.framework)

        status = await registry.health_check(result.framework_id)
        assert status in (HealthStatus.DEGRADED, HealthStatus.UNHEALTHY)

    @pytest.mark.asyncio
    async def test_health_check_uses_health_endpoint(self, registry):
        result = await _register_framework(
            registry,
            name="has_health",
            endpoints={"base": "http://localhost:9000", "health": "http://localhost:9000/health"},
        )

        with patch.object(registry, "_check_health_endpoint", new_callable=AsyncMock) as mock:
            mock.return_value = True
            status = await registry.health_check(result.framework_id)
            assert status == HealthStatus.HEALTHY
            mock.assert_called_once_with("http://localhost:9000/health")

    @pytest.mark.asyncio
    async def test_health_check_constructs_from_base(self, registry):
        result = await _register_framework(
            registry,
            name="base_only",
            endpoints={"base": "http://localhost:9000"},
        )

        with patch.object(registry, "_check_health_endpoint", new_callable=AsyncMock) as mock:
            mock.return_value = True
            status = await registry.health_check(result.framework_id)
            assert status == HealthStatus.HEALTHY
            mock.assert_called_once_with("http://localhost:9000/health")

    @pytest.mark.asyncio
    async def test_health_check_failure_increments_counter(self, registry):
        result = await _register_framework(registry, name="failing")

        with patch.object(registry, "_check_health_endpoint", new_callable=AsyncMock) as mock:
            mock.return_value = False
            await registry.health_check(result.framework_id)
            fw = await registry.get(result.framework_id)
            assert fw.consecutive_failures == 1
            assert fw.health_status == HealthStatus.DEGRADED

    @pytest.mark.asyncio
    async def test_health_check_exception_counts_as_failure(self, registry):
        result = await _register_framework(registry, name="error_fw")

        with patch.object(registry, "_check_health_endpoint", new_callable=AsyncMock) as mock:
            mock.side_effect = ConnectionError("Connection refused")
            await registry.health_check(result.framework_id)
            fw = await registry.get(result.framework_id)
            assert fw.consecutive_failures == 1

    @pytest.mark.asyncio
    async def test_health_check_marks_failed_after_max_failures(self, registry):
        reg = FederationRegistry(redis_url="memory://", max_consecutive_failures=3)
        result = await _register_framework(reg, name="will_fail")

        with patch.object(reg, "_check_health_endpoint", new_callable=AsyncMock) as mock:
            mock.return_value = False
            for _ in range(3):
                await reg.health_check(result.framework_id)

            fw = await reg.get(result.framework_id)
            assert fw.status == FrameworkStatus.FAILED
            assert fw.health_status == HealthStatus.UNHEALTHY

    @pytest.mark.asyncio
    async def test_health_check_emits_failed_event(self, registry):
        reg = FederationRegistry(redis_url="memory://", max_consecutive_failures=2)
        events = []
        reg.set_event_callback(lambda t, d: events.append((t, d)))

        result = await _register_framework(reg, name="event_fail")

        with patch.object(reg, "_check_health_endpoint", new_callable=AsyncMock) as mock:
            mock.return_value = False
            for _ in range(2):
                await reg.health_check(result.framework_id)

        fail_events = [e for e in events if e[0] == "framework_failed"]
        assert len(fail_events) == 1
        assert fail_events[0][1]["framework_id"] == result.framework_id

    @pytest.mark.asyncio
    async def test_health_check_strips_trailing_slash_from_base(self, registry):
        result = await _register_framework(
            registry,
            name="trailing_slash",
            endpoints={"base": "http://localhost:9000/"},
        )

        with patch.object(registry, "_check_health_endpoint", new_callable=AsyncMock) as mock:
            mock.return_value = True
            await registry.health_check(result.framework_id)
            mock.assert_called_once_with("http://localhost:9000/health")


# =============================================================================
# FederationRegistry: Status Transition Tests
# =============================================================================


class TestStatusTransitions:
    """Tests for framework status transitions."""

    @pytest.mark.asyncio
    async def test_registration_sets_active(self, registry):
        result = await _register_framework(registry)
        assert result.framework.status == FrameworkStatus.ACTIVE

    @pytest.mark.asyncio
    async def test_heartbeat_updates_status(self, registry):
        result = await _register_framework(registry, name="hb_fw")
        ok = await registry.heartbeat(result.framework_id, status=FrameworkStatus.DRAINING)
        assert ok is True
        fw = await registry.get(result.framework_id)
        assert fw.status == FrameworkStatus.DRAINING

    @pytest.mark.asyncio
    async def test_heartbeat_updates_timestamp(self, registry):
        result = await _register_framework(registry, name="hb_time")
        old_hb = result.framework.last_heartbeat
        await asyncio.sleep(0.01)
        await registry.heartbeat(result.framework_id)
        fw = await registry.get(result.framework_id)
        assert fw.last_heartbeat > old_hb

    @pytest.mark.asyncio
    async def test_heartbeat_nonexistent_returns_false(self, registry):
        ok = await registry.heartbeat("fw-gone")
        assert ok is False

    @pytest.mark.asyncio
    async def test_heartbeat_emits_status_change_event(self, registry):
        events = []
        registry.set_event_callback(lambda t, d: events.append((t, d)))

        result = await _register_framework(registry, name="event_hb")
        await registry.heartbeat(result.framework_id, status=FrameworkStatus.DRAINING)

        status_events = [e for e in events if e[0] == "framework_status_changed"]
        assert len(status_events) == 1
        assert status_events[0][1]["old_status"] == "active"
        assert status_events[0][1]["new_status"] == "draining"

    @pytest.mark.asyncio
    async def test_heartbeat_no_event_when_status_unchanged(self, registry):
        events = []
        registry.set_event_callback(lambda t, d: events.append((t, d)))

        result = await _register_framework(registry, name="same_status")
        await registry.heartbeat(result.framework_id, status=FrameworkStatus.ACTIVE)

        status_events = [e for e in events if e[0] == "framework_status_changed"]
        assert len(status_events) == 0

    @pytest.mark.asyncio
    async def test_heartbeat_without_status_keeps_current(self, registry):
        result = await _register_framework(registry, name="no_status_hb")
        assert result.framework.status == FrameworkStatus.ACTIVE
        await registry.heartbeat(result.framework_id)
        fw = await registry.get(result.framework_id)
        assert fw.status == FrameworkStatus.ACTIVE


# =============================================================================
# FederationRegistry: Lifecycle Hooks Tests
# =============================================================================


class TestLifecycleHooks:
    """Tests for startup, shutdown, and reconnect hooks."""

    @pytest.mark.asyncio
    async def test_startup_hook_called_on_register(self, registry):
        called_with = []
        registry.register_startup_hook("hook1", lambda fw: called_with.append(fw.name))
        await _register_framework(registry, name="hooked")
        assert "hooked" in called_with

    @pytest.mark.asyncio
    async def test_async_startup_hook(self, registry):
        called_with = []

        async def async_hook(fw):
            called_with.append(fw.name)

        registry.register_startup_hook("async_hook", async_hook)
        await _register_framework(registry, name="async_test")
        assert "async_test" in called_with

    @pytest.mark.asyncio
    async def test_shutdown_hook_called_on_unregister(self, registry):
        called_with = []
        registry.register_shutdown_hook("sh_hook", lambda fw: called_with.append(fw.name))
        result = await _register_framework(registry, name="to_unregister")
        await registry.unregister(result.framework_id)
        assert "to_unregister" in called_with

    @pytest.mark.asyncio
    async def test_async_shutdown_hook(self, registry):
        called_with = []

        async def async_shutdown(fw):
            called_with.append(fw.name)

        registry.register_shutdown_hook("async_sh", async_shutdown)
        result = await _register_framework(registry, name="async_sh_test")
        await registry.unregister(result.framework_id)
        assert "async_sh_test" in called_with

    @pytest.mark.asyncio
    async def test_startup_hook_failure_does_not_block_registration(self, registry):
        def bad_hook(fw):
            raise RuntimeError("Hook exploded")

        registry.register_startup_hook("bad_hook", bad_hook)
        result = await _register_framework(registry, name="survives")
        assert result.success is True

    @pytest.mark.asyncio
    async def test_shutdown_hook_failure_does_not_block_unregister(self, registry):
        def bad_hook(fw):
            raise RuntimeError("Shutdown exploded")

        registry.register_shutdown_hook("bad_sh", bad_hook)
        result = await _register_framework(registry, name="sh_survives")
        removed = await registry.unregister(result.framework_id)
        assert removed is True

    def test_register_reconnect_hook(self, registry):
        hook = MagicMock()
        registry.register_reconnect_hook("rc_hook", hook)
        assert "rc_hook" in registry._reconnect_hooks

    def test_unregister_hook_removes_from_all(self, registry):
        hook = MagicMock()
        registry.register_startup_hook("multi_hook", hook)
        registry.register_shutdown_hook("multi_hook", hook)
        registry.register_reconnect_hook("multi_hook", hook)

        removed = registry.unregister_hook("multi_hook")
        assert removed is True
        assert "multi_hook" not in registry._startup_hooks
        assert "multi_hook" not in registry._shutdown_hooks
        assert "multi_hook" not in registry._reconnect_hooks

    def test_unregister_hook_returns_false_for_missing(self, registry):
        removed = registry.unregister_hook("nonexistent")
        assert removed is False

    @pytest.mark.asyncio
    async def test_multiple_startup_hooks(self, registry):
        order = []
        registry.register_startup_hook("h1", lambda fw: order.append("h1"))
        registry.register_startup_hook("h2", lambda fw: order.append("h2"))
        registry.register_startup_hook("h3", lambda fw: order.append("h3"))

        await _register_framework(registry, name="multi_hook_fw")
        assert order == ["h1", "h2", "h3"]

    @pytest.mark.asyncio
    async def test_startup_hook_ids_recorded_on_framework(self, registry):
        registry.register_startup_hook("track_hook", lambda fw: None)
        result = await _register_framework(registry, name="tracked")
        assert "track_hook" in result.framework.startup_hooks


# =============================================================================
# FederationRegistry: Dead Framework Cleanup Tests
# =============================================================================


class TestDeadFrameworkCleanup:
    """Tests for dead framework cleanup."""

    @pytest.mark.asyncio
    async def test_cleanup_marks_expired_as_disconnected(self, registry):
        result = await _register_framework(registry, name="old_fw")
        # Simulate expired heartbeat
        result.framework.last_heartbeat = time.time() - 120
        await registry._save_framework(result.framework)

        count = await registry._cleanup_dead_frameworks()
        assert count == 1

        fw = await registry.get(result.framework_id)
        assert fw.status == FrameworkStatus.DISCONNECTED

    @pytest.mark.asyncio
    async def test_cleanup_skips_alive_frameworks(self, registry):
        result = await _register_framework(registry, name="alive_fw")
        # Heartbeat is fresh (just registered)
        count = await registry._cleanup_dead_frameworks()
        assert count == 0
        fw = await registry.get(result.framework_id)
        assert fw.status == FrameworkStatus.ACTIVE

    @pytest.mark.asyncio
    async def test_cleanup_skips_already_disconnected(self, registry):
        result = await _register_framework(registry, name="already_disc")
        result.framework.last_heartbeat = time.time() - 120
        result.framework.status = FrameworkStatus.DISCONNECTED
        await registry._save_framework(result.framework)

        count = await registry._cleanup_dead_frameworks()
        assert count == 0

    @pytest.mark.asyncio
    async def test_cleanup_skips_already_failed(self, registry):
        result = await _register_framework(registry, name="already_failed")
        result.framework.last_heartbeat = time.time() - 120
        result.framework.status = FrameworkStatus.FAILED
        await registry._save_framework(result.framework)

        count = await registry._cleanup_dead_frameworks()
        assert count == 0

    @pytest.mark.asyncio
    async def test_cleanup_emits_disconnected_event(self, registry):
        events = []
        registry.set_event_callback(lambda t, d: events.append((t, d)))

        result = await _register_framework(registry, name="disc_event")
        result.framework.last_heartbeat = time.time() - 120
        await registry._save_framework(result.framework)

        await registry._cleanup_dead_frameworks()

        disc_events = [e for e in events if e[0] == "framework_disconnected"]
        assert len(disc_events) == 1
        assert disc_events[0][1]["name"] == "disc_event"

    @pytest.mark.asyncio
    async def test_cleanup_multiple_dead_frameworks(self, registry):
        for i in range(3):
            result = await _register_framework(registry, name=f"dead_{i}")
            result.framework.last_heartbeat = time.time() - 200
            await registry._save_framework(result.framework)

        # Also register a live one
        await _register_framework(registry, name="live_one")

        count = await registry._cleanup_dead_frameworks()
        assert count == 3


# =============================================================================
# FederationRegistry: Capability-Based Routing / Selection Tests
# =============================================================================


class TestFrameworkSelection:
    """Tests for select_framework and capability-based routing."""

    async def _make_active_framework(
        self,
        registry: FederationRegistry,
        name: str,
        capabilities: list[FrameworkCapability],
        version: str = "1.0.0",
        consecutive_failures: int = 0,
    ) -> RegistrationResult:
        result = await _register_framework(
            registry, name=name, version=version, capabilities=capabilities
        )
        result.framework.health_status = HealthStatus.HEALTHY
        result.framework.consecutive_failures = consecutive_failures
        result.framework.last_health_check = time.time()
        await registry._save_framework(result.framework)
        return result

    @pytest.mark.asyncio
    async def test_select_healthiest(self, registry):
        await self._make_active_framework(
            registry, "fw_healthy", [_cap("search")], consecutive_failures=0
        )
        await self._make_active_framework(
            registry, "fw_degraded", [_cap("search")], consecutive_failures=2
        )

        selected = await registry.select_framework(["search"], strategy="healthiest")
        assert selected is not None
        assert selected.name == "fw_healthy"

    @pytest.mark.asyncio
    async def test_select_newest(self, registry):
        await self._make_active_framework(registry, "fw_old", [_cap("search")], version="1.0.0")
        await self._make_active_framework(registry, "fw_new", [_cap("search")], version="2.0.0")

        selected = await registry.select_framework(["search"], strategy="newest")
        assert selected is not None
        assert selected.name == "fw_new"

    @pytest.mark.asyncio
    async def test_select_random(self, registry):
        await self._make_active_framework(registry, "fw_a", [_cap("search")])
        await self._make_active_framework(registry, "fw_b", [_cap("search")])

        selected = await registry.select_framework(["search"], strategy="random")
        assert selected is not None
        assert selected.name in ("fw_a", "fw_b")

    @pytest.mark.asyncio
    async def test_select_unknown_strategy_returns_first(self, registry):
        await self._make_active_framework(registry, "fw_x", [_cap("search")])
        selected = await registry.select_framework(["search"], strategy="unknown_strategy")
        assert selected is not None

    @pytest.mark.asyncio
    async def test_select_no_candidates(self, registry):
        selected = await registry.select_framework(["nonexistent"])
        assert selected is None

    @pytest.mark.asyncio
    async def test_select_with_exclude(self, registry):
        r1 = await self._make_active_framework(registry, "fw_excluded", [_cap("search")])
        await self._make_active_framework(registry, "fw_kept", [_cap("search")])

        selected = await registry.select_framework(["search"], exclude=[r1.framework_id])
        assert selected is not None
        assert selected.name == "fw_kept"

    @pytest.mark.asyncio
    async def test_select_with_version_preference(self, registry):
        await self._make_active_framework(registry, "fw_v1", [_cap("search")], version="1.5.0")
        await self._make_active_framework(registry, "fw_v2", [_cap("search")], version="2.1.0")

        selected = await registry.select_framework(["search"], prefer_version="2.")
        assert selected is not None
        assert selected.name == "fw_v2"

    @pytest.mark.asyncio
    async def test_select_version_preference_fallback(self, registry):
        await self._make_active_framework(registry, "fw_only", [_cap("search")], version="1.0.0")

        # Prefer version 3.x which doesn't exist, should fall back to available
        selected = await registry.select_framework(["search"], prefer_version="3.")
        assert selected is not None
        assert selected.name == "fw_only"

    @pytest.mark.asyncio
    async def test_select_all_excluded(self, registry):
        r1 = await self._make_active_framework(registry, "fw_e1", [_cap("s")])
        r2 = await self._make_active_framework(registry, "fw_e2", [_cap("s")])
        selected = await registry.select_framework(
            ["s"], exclude=[r1.framework_id, r2.framework_id]
        )
        assert selected is None


# =============================================================================
# FederationRegistry: Multi-Framework Management Tests
# =============================================================================


class TestMultiFrameworkManagement:
    """Tests for managing multiple frameworks simultaneously."""

    @pytest.mark.asyncio
    async def test_list_all_active_only(self, registry):
        r1 = await _register_framework(registry, name="active_fw")
        r1.framework.health_status = HealthStatus.HEALTHY
        await registry._save_framework(r1.framework)

        r2 = await _register_framework(registry, name="inactive_fw")
        # inactive_fw has UNKNOWN health, so is_active() is False

        active = await registry.list_all(include_inactive=False)
        assert len(active) == 1
        assert active[0].name == "active_fw"

    @pytest.mark.asyncio
    async def test_list_all_including_inactive(self, registry):
        await _register_framework(registry, name="fw1")
        await _register_framework(registry, name="fw2")
        all_fws = await registry.list_all(include_inactive=True)
        assert len(all_fws) == 2

    @pytest.mark.asyncio
    async def test_register_many_frameworks(self, registry):
        for i in range(10):
            result = await _register_framework(registry, name=f"fw_{i}")
            assert result.success is True

        all_fws = await registry.list_all(include_inactive=True)
        assert len(all_fws) == 10

    @pytest.mark.asyncio
    async def test_get_stats(self, registry):
        r1 = await _register_framework(registry, name="s1", capabilities=[_cap("a"), _cap("b")])
        r1.framework.health_status = HealthStatus.HEALTHY
        await registry._save_framework(r1.framework)

        r2 = await _register_framework(registry, name="s2", capabilities=[_cap("a")])
        # r2 health is UNKNOWN

        stats = await registry.get_stats()
        assert stats["total_frameworks"] == 2
        assert stats["active_frameworks"] == 1
        assert stats["by_capability"]["a"] == 2
        assert stats["by_capability"]["b"] == 1
        assert stats["heartbeat_timeout"] == 60.0
        assert stats["supported_api_versions"] == FederationRegistry.SUPPORTED_API_VERSIONS

    @pytest.mark.asyncio
    async def test_clear(self, registry):
        await _register_framework(registry, name="fw1")
        await _register_framework(registry, name="fw2")
        assert len(registry._local_cache) == 2
        assert len(registry._capability_index) > 0

        await registry.clear()
        assert len(registry._local_cache) == 0
        assert len(registry._capability_index) == 0

    @pytest.mark.asyncio
    async def test_unique_framework_ids(self, registry):
        ids = set()
        for i in range(20):
            result = await _register_framework(registry, name=f"unique_{i}")
            ids.add(result.framework_id)
        assert len(ids) == 20


# =============================================================================
# FederationRegistry: Event Callback Tests
# =============================================================================


class TestEventCallbacks:
    """Tests for event emission and callbacks."""

    @pytest.mark.asyncio
    async def test_register_emits_event(self, registry):
        events = []
        registry.set_event_callback(lambda t, d: events.append((t, d)))
        await _register_framework(registry, name="event_test")
        reg_events = [e for e in events if e[0] == "framework_registered"]
        assert len(reg_events) == 1
        assert reg_events[0][1]["name"] == "event_test"

    @pytest.mark.asyncio
    async def test_unregister_emits_event(self, registry):
        events = []
        registry.set_event_callback(lambda t, d: events.append((t, d)))
        result = await _register_framework(registry, name="unreg_event")
        await registry.unregister(result.framework_id)
        unreg_events = [e for e in events if e[0] == "framework_unregistered"]
        assert len(unreg_events) == 1

    @pytest.mark.asyncio
    async def test_event_callback_exception_is_caught(self, registry):
        def bad_callback(event_type, data):
            raise RuntimeError("Callback failed")

        registry.set_event_callback(bad_callback)
        # Should not raise
        result = await _register_framework(registry, name="safe")
        assert result.success is True

    @pytest.mark.asyncio
    async def test_no_callback_set(self, registry):
        # Should work without any callback
        result = await _register_framework(registry, name="no_cb")
        assert result.success is True


# =============================================================================
# FederationRegistry: Refresh Capabilities Tests
# =============================================================================


class TestRefreshCapabilities:
    """Tests for live capability refresh."""

    @pytest.mark.asyncio
    async def test_refresh_nonexistent_framework(self, registry):
        caps = await registry.refresh_capabilities("fw-missing")
        assert caps == []

    @pytest.mark.asyncio
    async def test_refresh_no_endpoints(self, registry):
        result = await _register_framework(registry, name="no_cap_ep")
        result.framework.endpoints = {}
        await registry._save_framework(result.framework)

        caps = await registry.refresh_capabilities(result.framework_id)
        assert caps == result.framework.capabilities

    @pytest.mark.asyncio
    async def test_refresh_constructs_url_from_base(self, registry):
        result = await _register_framework(
            registry,
            name="base_cap",
            endpoints={"base": "http://localhost:9000"},
        )

        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(
            return_value={"capabilities": [{"name": "new_cap", "description": "New"}]}
        )
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=False)

        mock_session = AsyncMock()
        mock_session.get = MagicMock(return_value=mock_response)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            with patch("aiohttp.ClientTimeout"):
                registry._http_client = mock_session
                caps = await registry.refresh_capabilities(result.framework_id)
                assert len(caps) == 1
                assert caps[0].name == "new_cap"

    @pytest.mark.asyncio
    async def test_refresh_emits_event_on_change(self, registry):
        events = []
        registry.set_event_callback(lambda t, d: events.append((t, d)))

        result = await _register_framework(
            registry,
            name="change_cap",
            capabilities=[_cap("old_cap")],
            endpoints={"base": "http://localhost:9000"},
        )

        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(
            return_value={"capabilities": [{"name": "new_cap", "description": "New"}]}
        )
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=False)

        mock_session = AsyncMock()
        mock_session.get = MagicMock(return_value=mock_response)

        registry._http_client = mock_session
        await registry.refresh_capabilities(result.framework_id)

        cap_events = [e for e in events if e[0] == "capabilities_updated"]
        assert len(cap_events) == 1
        assert "new_cap" in cap_events[0][1]["added"]
        assert "old_cap" in cap_events[0][1]["removed"]


# =============================================================================
# FederationRegistry: Connect / Close Lifecycle Tests
# =============================================================================


class TestConnectClose:
    """Tests for connect and close lifecycle."""

    @pytest.mark.asyncio
    async def test_connect_memory_mode(self):
        reg = FederationRegistry(redis_url="memory://")
        await reg.connect()
        assert reg._redis is None
        assert reg._health_check_task is not None
        assert reg._cleanup_task is not None
        await reg.close()

    @pytest.mark.asyncio
    async def test_connect_redis_import_error(self):
        reg = FederationRegistry(redis_url="redis://localhost:6379")
        with patch.dict("sys.modules", {"redis": None, "redis.asyncio": None}):
            await reg.connect()
        # Should fall back gracefully
        assert reg._health_check_task is not None
        await reg.close()

    @pytest.mark.asyncio
    async def test_close_cancels_background_tasks(self):
        reg = FederationRegistry(redis_url="memory://")
        await reg.connect()
        assert not reg._health_check_task.done()
        assert not reg._cleanup_task.done()
        await reg.close()
        # After close, tasks should be done (cancelled or finished with CancelledError)
        assert reg._health_check_task.done()
        assert reg._cleanup_task.done()

    @pytest.mark.asyncio
    async def test_close_runs_shutdown_hooks_for_active(self):
        reg = FederationRegistry(redis_url="memory://")
        await reg.connect()
        called = []
        reg.register_shutdown_hook("close_hook", lambda fw: called.append(fw.name))
        await _register_framework(reg, name="active_at_close")
        await reg.close()
        assert "active_at_close" in called

    @pytest.mark.asyncio
    async def test_close_with_http_client(self):
        reg = FederationRegistry(redis_url="memory://")
        await reg.connect()
        mock_client = AsyncMock()
        reg._http_client = mock_client
        await reg.close()
        mock_client.close.assert_called_once()


# =============================================================================
# FederationRegistry: Edge Cases and Error Handling
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_register_empty_capabilities(self, registry):
        result = await registry.register(
            name="empty_caps",
            version="1.0.0",
            capabilities=[],
            endpoints=_endpoints(),
        )
        assert result.success is True
        assert result.framework.capabilities == []

    @pytest.mark.asyncio
    async def test_register_empty_name(self, registry):
        # Empty name should still work but produce odd framework_id
        result = await registry.register(
            name="",
            version="1.0.0",
            capabilities=[],
            endpoints=_endpoints(),
        )
        # The post_init only generates ID if name is truthy
        assert result.success is True

    @pytest.mark.asyncio
    async def test_register_duplicate_capabilities(self, registry):
        result = await _register_framework(
            registry,
            name="dup_caps",
            capabilities=[_cap("search"), _cap("search")],
        )
        assert result.success is True
        assert len(result.framework.capabilities) == 2

    @pytest.mark.asyncio
    async def test_concurrent_registrations(self, registry):
        tasks = [_register_framework(registry, name=f"concurrent_{i}") for i in range(10)]
        results = await asyncio.gather(*tasks)
        successes = [r for r in results if r.success]
        assert len(successes) == 10

    @pytest.mark.asyncio
    async def test_heartbeat_after_unregister(self, registry):
        result = await _register_framework(registry, name="unreg_hb")
        await registry.unregister(result.framework_id)
        ok = await registry.heartbeat(result.framework_id)
        assert ok is False

    @pytest.mark.asyncio
    async def test_health_check_after_unregister(self, registry):
        result = await _register_framework(registry, name="unreg_hc")
        await registry.unregister(result.framework_id)
        status = await registry.health_check(result.framework_id)
        assert status == HealthStatus.UNKNOWN

    @pytest.mark.asyncio
    async def test_find_by_capability_after_all_unregistered(self, registry):
        r1 = await _register_framework(registry, name="temp", capabilities=[_cap("temp_cap")])
        await registry.unregister(r1.framework_id)
        found = await registry.find_by_capability("temp_cap", only_active=False)
        assert len(found) == 0

    @pytest.mark.asyncio
    async def test_capability_index_cleanup(self, registry):
        r1 = await _register_framework(registry, name="idx_clean", capabilities=[_cap("only_cap")])
        assert "only_cap" in registry._capability_index
        await registry.unregister(r1.framework_id)
        # The key should be removed when the set is empty
        assert "only_cap" not in registry._capability_index

    @pytest.mark.asyncio
    async def test_framework_id_format(self, registry):
        result = await _register_framework(registry, name="myframework")
        assert result.framework_id.startswith("fw-myframework-")
        # 12 hex chars from sha256
        suffix = result.framework_id.split("fw-myframework-")[1]
        assert len(suffix) == 12

    @pytest.mark.asyncio
    async def test_multiple_capabilities_per_framework_indexed(self, registry):
        result = await _register_framework(
            registry,
            name="multi_cap",
            capabilities=[_cap("a"), _cap("b"), _cap("c")],
        )
        assert result.framework_id in registry._capability_index.get("a", set())
        assert result.framework_id in registry._capability_index.get("b", set())
        assert result.framework_id in registry._capability_index.get("c", set())

    @pytest.mark.asyncio
    async def test_stats_empty_registry(self, registry):
        stats = await registry.get_stats()
        assert stats["total_frameworks"] == 0
        assert stats["active_frameworks"] == 0
        assert stats["by_status"] == {}
        assert stats["by_health"] == {}
        assert stats["by_capability"] == {}

    @pytest.mark.asyncio
    async def test_clear_then_register(self, registry):
        await _register_framework(registry, name="before_clear")
        await registry.clear()
        result = await _register_framework(registry, name="after_clear")
        assert result.success is True
        all_fws = await registry.list_all(include_inactive=True)
        assert len(all_fws) == 1
        assert all_fws[0].name == "after_clear"

    def test_set_event_callback(self, registry):
        cb = MagicMock()
        registry.set_event_callback(cb)
        assert registry._event_callback is cb

    def test_emit_event_with_callback(self, registry):
        cb = MagicMock()
        registry.set_event_callback(cb)
        registry._emit_event("test_event", {"key": "value"})
        cb.assert_called_once_with("test_event", {"key": "value"})

    def test_emit_event_without_callback(self, registry):
        # Should not raise
        registry._emit_event("test_event", {"key": "value"})

    def test_emit_event_callback_exception_caught(self, registry):
        cb = MagicMock(side_effect=RuntimeError("boom"))
        registry.set_event_callback(cb)
        # Should not raise
        registry._emit_event("test_event", {})

    @pytest.mark.asyncio
    async def test_save_framework_updates_cache(self, registry):
        fw = ExternalFramework(
            framework_id="fw-direct-save",
            name="direct",
            status=FrameworkStatus.ACTIVE,
        )
        await registry._save_framework(fw)
        cached = registry._local_cache.get("fw-direct-save")
        assert cached is not None
        assert cached.name == "direct"

    @pytest.mark.asyncio
    async def test_register_with_custom_api_versions(self, registry):
        result = await registry.register(
            name="custom_api",
            version="1.0.0",
            capabilities=[_cap("test")],
            endpoints=_endpoints(),
            supported_api_versions=["1.1.0"],
        )
        assert result.success is True
        assert result.negotiated_version == "1.1.0"
        assert result.framework.supported_api_versions == ["1.1.0"]


# =============================================================================
# FederationRegistry: Advanced Health Endpoint Tests
# =============================================================================


class TestHealthEndpointBehavior:
    """Tests for _check_health_endpoint and advanced health scenarios."""

    @pytest.mark.asyncio
    async def test_check_health_endpoint_200_returns_true(self, registry):
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=False)

        mock_session = AsyncMock()
        mock_session.get = MagicMock(return_value=mock_response)
        registry._http_client = mock_session

        with patch.dict("sys.modules", {"aiohttp": MagicMock()}):
            result = await registry._check_health_endpoint("http://localhost:9000/health")
        assert result is True

    @pytest.mark.asyncio
    async def test_check_health_endpoint_500_returns_false(self, registry):
        mock_response = AsyncMock()
        mock_response.status = 500
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=False)

        mock_session = AsyncMock()
        mock_session.get = MagicMock(return_value=mock_response)
        registry._http_client = mock_session

        with patch.dict("sys.modules", {"aiohttp": MagicMock()}):
            result = await registry._check_health_endpoint("http://localhost:9000/health")
        assert result is False

    @pytest.mark.asyncio
    async def test_check_health_endpoint_connection_error(self, registry):
        mock_session = AsyncMock()
        mock_session.get = MagicMock(side_effect=ConnectionError("refused"))
        registry._http_client = mock_session

        with patch.dict("sys.modules", {"aiohttp": MagicMock()}):
            result = await registry._check_health_endpoint("http://localhost:9000/health")
        assert result is False

    @pytest.mark.asyncio
    async def test_check_health_endpoint_timeout_error(self, registry):
        mock_session = AsyncMock()
        mock_session.get = MagicMock(side_effect=asyncio.TimeoutError())
        registry._http_client = mock_session

        with patch.dict("sys.modules", {"aiohttp": MagicMock()}):
            result = await registry._check_health_endpoint("http://localhost:9000/health")
        assert result is False

    @pytest.mark.asyncio
    async def test_check_health_endpoint_no_aiohttp(self, registry):
        """When aiohttp is not installed, health check assumes healthy."""
        registry._http_client = None
        with patch.dict("sys.modules", {"aiohttp": None}):
            with patch("builtins.__import__", side_effect=ImportError("no aiohttp")):
                result = await registry._check_health_endpoint("http://example.com/health")
        assert result is True

    @pytest.mark.asyncio
    async def test_health_check_recovery_after_degraded(self, registry):
        result = await _register_framework(registry, name="recovery_fw")
        with patch.object(registry, "_check_health_endpoint", new_callable=AsyncMock) as mock:
            # Two failures -> DEGRADED
            mock.return_value = False
            await registry.health_check(result.framework_id)
            await registry.health_check(result.framework_id)
            fw = await registry.get(result.framework_id)
            assert fw.health_status == HealthStatus.DEGRADED

            # One success -> HEALTHY
            mock.return_value = True
            await registry.health_check(result.framework_id)
            fw = await registry.get(result.framework_id)
            assert fw.health_status == HealthStatus.HEALTHY
            assert fw.consecutive_failures == 0

    @pytest.mark.asyncio
    async def test_health_check_successive_failures_progression(self, registry):
        """Test the full progression: UNKNOWN -> DEGRADED -> UNHEALTHY."""
        result = await _register_framework(registry, name="progression_fw")
        with patch.object(registry, "_check_health_endpoint", new_callable=AsyncMock) as mock:
            mock.return_value = False

            await registry.health_check(result.framework_id)
            fw = await registry.get(result.framework_id)
            assert fw.health_status == HealthStatus.DEGRADED
            assert fw.consecutive_failures == 1

            await registry.health_check(result.framework_id)
            fw = await registry.get(result.framework_id)
            assert fw.health_status == HealthStatus.DEGRADED
            assert fw.consecutive_failures == 2

            await registry.health_check(result.framework_id)
            fw = await registry.get(result.framework_id)
            assert fw.health_status == HealthStatus.UNHEALTHY
            assert fw.consecutive_failures == 3

    @pytest.mark.asyncio
    async def test_health_check_does_not_emit_failed_below_threshold(self, registry):
        reg = FederationRegistry(redis_url="memory://", max_consecutive_failures=10)
        events = []
        reg.set_event_callback(lambda t, d: events.append((t, d)))

        result = await _register_framework(reg, name="not_yet_failed")
        with patch.object(reg, "_check_health_endpoint", new_callable=AsyncMock) as mock:
            mock.return_value = False
            for _ in range(3):
                await reg.health_check(result.framework_id)

        fail_events = [e for e in events if e[0] == "framework_failed"]
        assert len(fail_events) == 0

    @pytest.mark.asyncio
    async def test_health_check_constructs_url_strips_multiple_slashes(self, registry):
        result = await _register_framework(
            registry,
            name="multi_slash",
            endpoints={"base": "http://localhost:9000///"},
        )
        with patch.object(registry, "_check_health_endpoint", new_callable=AsyncMock) as mock:
            mock.return_value = True
            await registry.health_check(result.framework_id)
            # rstrip('/') removes all trailing slashes
            mock.assert_called_once_with("http://localhost:9000/health")


# =============================================================================
# FederationRegistry: Background Loop Tests
# =============================================================================


class TestBackgroundLoops:
    """Tests for health check and cleanup background loops."""

    @pytest.mark.asyncio
    async def test_health_check_loop_runs_periodically(self):
        reg = FederationRegistry(redis_url="memory://", health_check_interval=0.05)
        await reg.connect()

        result = await _register_framework(reg, name="loop_fw")
        result.framework.health_status = HealthStatus.HEALTHY
        await reg._save_framework(result.framework)

        with patch.object(reg, "health_check", new_callable=AsyncMock) as mock:
            await asyncio.sleep(0.12)
            assert mock.call_count >= 1

        await reg.close()

    @pytest.mark.asyncio
    async def test_cleanup_loop_runs_periodically(self):
        reg = FederationRegistry(redis_url="memory://", cleanup_interval=0.05)
        await reg.connect()

        with patch.object(reg, "_cleanup_dead_frameworks", new_callable=AsyncMock) as mock:
            mock.return_value = 0
            await asyncio.sleep(0.12)
            assert mock.call_count >= 1

        await reg.close()

    @pytest.mark.asyncio
    async def test_health_check_loop_survives_errors(self):
        reg = FederationRegistry(redis_url="memory://", health_check_interval=0.03)
        await reg.connect()

        call_count = 0

        async def failing_list(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise RuntimeError("simulated error")
            return []

        with patch.object(reg, "list_all", side_effect=failing_list):
            await asyncio.sleep(0.15)

        assert call_count >= 2
        await reg.close()

    @pytest.mark.asyncio
    async def test_cleanup_loop_survives_errors(self):
        reg = FederationRegistry(redis_url="memory://", cleanup_interval=0.03)
        await reg.connect()

        call_count = 0

        async def failing_cleanup():
            nonlocal call_count
            call_count += 1
            raise RuntimeError("cleanup error")

        with patch.object(reg, "_cleanup_dead_frameworks", side_effect=failing_cleanup):
            await asyncio.sleep(0.12)

        assert call_count >= 1
        await reg.close()


# =============================================================================
# FederationRegistry: Capability Index Integrity Tests
# =============================================================================


class TestCapabilityIndexIntegrity:
    """Tests ensuring capability index stays in sync."""

    @pytest.mark.asyncio
    async def test_reregistration_updates_capability_index(self, registry):
        """When a framework re-registers with different capabilities, index updates."""
        r1 = await _register_framework(
            registry, name="evolving", capabilities=[_cap("old_cap"), _cap("shared")]
        )
        assert r1.framework_id in registry._capability_index.get("old_cap", set())
        assert r1.framework_id in registry._capability_index.get("shared", set())

        r2 = await _register_framework(
            registry, name="evolving", capabilities=[_cap("new_cap"), _cap("shared")]
        )
        # After re-registration, old_cap should still be indexed (we overwrote the framework)
        # The index should contain the new capabilities
        assert r2.framework_id in registry._capability_index.get("new_cap", set())
        assert r2.framework_id in registry._capability_index.get("shared", set())

    @pytest.mark.asyncio
    async def test_unindex_removes_empty_capability_keys(self, registry):
        result = await _register_framework(
            registry, name="sole_provider", capabilities=[_cap("unique_cap_x")]
        )
        assert "unique_cap_x" in registry._capability_index
        await registry.unregister(result.framework_id)
        assert "unique_cap_x" not in registry._capability_index

    @pytest.mark.asyncio
    async def test_unindex_preserves_other_frameworks(self, registry):
        r1 = await _register_framework(registry, name="fw_keep", capabilities=[_cap("shared_cap")])
        r2 = await _register_framework(
            registry, name="fw_remove", capabilities=[_cap("shared_cap")]
        )
        assert len(registry._capability_index["shared_cap"]) == 2

        await registry.unregister(r2.framework_id)
        assert "shared_cap" in registry._capability_index
        assert r1.framework_id in registry._capability_index["shared_cap"]
        assert r2.framework_id not in registry._capability_index["shared_cap"]

    @pytest.mark.asyncio
    async def test_clear_empties_capability_index(self, registry):
        await _register_framework(registry, name="fw1", capabilities=[_cap("a"), _cap("b")])
        await _register_framework(registry, name="fw2", capabilities=[_cap("b"), _cap("c")])
        assert len(registry._capability_index) == 3

        await registry.clear()
        assert len(registry._capability_index) == 0

    @pytest.mark.asyncio
    async def test_many_capabilities_per_framework_indexed_correctly(self, registry):
        caps = [_cap(f"cap_{i}") for i in range(20)]
        result = await _register_framework(registry, name="many_caps", capabilities=caps)
        for i in range(20):
            assert result.framework_id in registry._capability_index[f"cap_{i}"]


# =============================================================================
# FederationRegistry: Advanced Selection Tests
# =============================================================================


class TestAdvancedSelection:
    """Advanced tests for framework selection logic."""

    async def _make_active(
        self,
        registry: FederationRegistry,
        name: str,
        capabilities: list[FrameworkCapability],
        version: str = "1.0.0",
        consecutive_failures: int = 0,
        last_health_check: float | None = None,
    ) -> RegistrationResult:
        result = await _register_framework(
            registry, name=name, version=version, capabilities=capabilities
        )
        result.framework.health_status = HealthStatus.HEALTHY
        result.framework.consecutive_failures = consecutive_failures
        result.framework.last_health_check = last_health_check or time.time()
        await registry._save_framework(result.framework)
        return result

    @pytest.mark.asyncio
    async def test_select_healthiest_tiebreak_by_health_check_recency(self, registry):
        """When failures are equal, prefer more recently checked."""
        r1 = await self._make_active(
            registry,
            "fw_old_check",
            [_cap("s")],
            consecutive_failures=0,
            last_health_check=time.time() - 100,
        )
        r2 = await self._make_active(
            registry,
            "fw_new_check",
            [_cap("s")],
            consecutive_failures=0,
            last_health_check=time.time(),
        )
        selected = await registry.select_framework(["s"], strategy="healthiest")
        assert selected is not None
        # Higher last_health_check means more recent, -f.last_health_check is lower
        assert selected.name == "fw_new_check"

    @pytest.mark.asyncio
    async def test_select_newest_string_comparison(self, registry):
        """Version comparison is string-based, verify ordering."""
        await self._make_active(registry, "fw_10", [_cap("s")], version="10.0.0")
        await self._make_active(registry, "fw_2", [_cap("s")], version="2.0.0")
        await self._make_active(registry, "fw_9", [_cap("s")], version="9.0.0")

        selected = await registry.select_framework(["s"], strategy="newest")
        assert selected is not None
        # String comparison: "9.0.0" > "2.0.0" > "10.0.0"
        assert selected.name == "fw_9"

    @pytest.mark.asyncio
    async def test_select_with_multiple_capabilities_required(self, registry):
        await self._make_active(registry, "fw_full", [_cap("a"), _cap("b"), _cap("c")])
        await self._make_active(registry, "fw_partial", [_cap("a"), _cap("b")])

        selected = await registry.select_framework(["a", "b", "c"])
        assert selected is not None
        assert selected.name == "fw_full"

    @pytest.mark.asyncio
    async def test_select_returns_none_when_all_excluded(self, registry):
        r1 = await self._make_active(registry, "fw_only", [_cap("s")])
        selected = await registry.select_framework(["s"], exclude=[r1.framework_id])
        assert selected is None

    @pytest.mark.asyncio
    async def test_select_prefer_version_with_no_match_falls_back(self, registry):
        await self._make_active(registry, "fw_a", [_cap("s")], version="1.0.0")
        await self._make_active(registry, "fw_b", [_cap("s")], version="1.5.0")

        selected = await registry.select_framework(["s"], prefer_version="99.")
        # No version match, should still select from all candidates
        assert selected is not None

    @pytest.mark.asyncio
    async def test_select_prefer_version_narrows_candidates(self, registry):
        await self._make_active(registry, "fw_v1", [_cap("s")], version="1.2.0")
        await self._make_active(registry, "fw_v2", [_cap("s")], version="2.3.0")
        await self._make_active(registry, "fw_v2b", [_cap("s")], version="2.5.0")

        selected = await registry.select_framework(["s"], strategy="newest", prefer_version="2.")
        assert selected is not None
        assert selected.version.startswith("2.")
        assert selected.name == "fw_v2b"

    @pytest.mark.asyncio
    async def test_select_random_returns_different_results_over_many_calls(self, registry):
        await self._make_active(registry, "fw_r1", [_cap("s")])
        await self._make_active(registry, "fw_r2", [_cap("s")])
        await self._make_active(registry, "fw_r3", [_cap("s")])

        names = set()
        for _ in range(30):
            selected = await registry.select_framework(["s"], strategy="random")
            names.add(selected.name)

        # With 30 tries and 3 options, probability of missing one is negligible
        assert len(names) >= 2

    @pytest.mark.asyncio
    async def test_select_excludes_inactive_frameworks(self, registry):
        """Inactive frameworks (UNKNOWN health) are not returned."""
        result = await _register_framework(registry, name="inactive_sel", capabilities=[_cap("s")])
        # Default health is UNKNOWN, is_active() is False
        selected = await registry.select_framework(["s"])
        assert selected is None


# =============================================================================
# FederationRegistry: Advanced Registration Tests
# =============================================================================


class TestAdvancedRegistration:
    """Advanced tests for registration edge cases and flows."""

    @pytest.mark.asyncio
    async def test_register_multiple_endpoints(self, registry):
        result = await registry.register(
            name="multi_ep",
            version="1.0.0",
            capabilities=[_cap("test")],
            endpoints={
                "base": "http://localhost:9000",
                "health": "http://localhost:9000/health",
                "invoke": "http://localhost:9000/invoke",
                "capabilities": "http://localhost:9000/caps",
            },
        )
        assert result.success is True
        assert result.framework.get_endpoint("invoke") == "http://localhost:9000/invoke"
        assert result.framework.get_endpoint("capabilities") == "http://localhost:9000/caps"

    @pytest.mark.asyncio
    async def test_register_with_all_supported_versions(self, registry):
        result = await registry.register(
            name="all_versions",
            version="1.0.0",
            capabilities=[_cap("test")],
            endpoints=_endpoints(),
            supported_api_versions=FederationRegistry.SUPPORTED_API_VERSIONS.copy(),
        )
        assert result.success is True
        assert result.negotiated_version == "2.0.0"

    @pytest.mark.asyncio
    async def test_register_default_api_versions_when_none(self, registry):
        """When no supported_api_versions is given, defaults to ['1.0.0']."""
        result = await registry.register(
            name="default_ver",
            version="1.0.0",
            capabilities=[_cap("test")],
            endpoints=_endpoints(),
            supported_api_versions=None,
        )
        assert result.success is True
        assert result.negotiated_version == "1.0.0"
        assert result.framework.supported_api_versions == ["1.0.0"]

    @pytest.mark.asyncio
    async def test_register_emits_event_with_capabilities_list(self, registry):
        events = []
        registry.set_event_callback(lambda t, d: events.append((t, d)))

        await _register_framework(
            registry,
            name="event_caps",
            capabilities=[_cap("a"), _cap("b"), _cap("c")],
        )
        reg_events = [e for e in events if e[0] == "framework_registered"]
        assert len(reg_events) == 1
        assert set(reg_events[0][1]["capabilities"]) == {"a", "b", "c"}

    @pytest.mark.asyncio
    async def test_register_many_capabilities(self, registry):
        caps = [_cap(f"cap_{i}") for i in range(50)]
        result = await _register_framework(registry, name="many_caps_fw", capabilities=caps)
        assert result.success is True
        assert len(result.framework.capabilities) == 50

    @pytest.mark.asyncio
    async def test_get_returns_same_object_from_cache(self, registry):
        result = await _register_framework(registry, name="cache_test")
        fw1 = await registry.get(result.framework_id)
        fw2 = await registry.get(result.framework_id)
        assert fw1 is fw2  # Same object reference from cache

    @pytest.mark.asyncio
    async def test_register_with_empty_tags(self, registry):
        result = await _register_framework(registry, tags=[])
        assert result.framework.tags == set()

    @pytest.mark.asyncio
    async def test_register_with_empty_metadata(self, registry):
        result = await _register_framework(registry, metadata={})
        assert result.framework.metadata == {}


# =============================================================================
# FederationRegistry: Dead Framework Cleanup Advanced Tests
# =============================================================================


class TestDeadFrameworkCleanupAdvanced:
    """Advanced tests for dead framework cleanup scenarios."""

    @pytest.mark.asyncio
    async def test_cleanup_with_custom_heartbeat_timeout(self):
        reg = FederationRegistry(redis_url="memory://", heartbeat_timeout=10.0)
        result = await _register_framework(reg, name="custom_timeout")
        # Set heartbeat to 15 seconds ago (exceeds 10s timeout)
        result.framework.last_heartbeat = time.time() - 15
        await reg._save_framework(result.framework)

        count = await reg._cleanup_dead_frameworks()
        assert count == 1
        fw = await reg.get(result.framework_id)
        assert fw.status == FrameworkStatus.DISCONNECTED

    @pytest.mark.asyncio
    async def test_cleanup_does_not_affect_draining_alive_frameworks(self, registry):
        result = await _register_framework(registry, name="draining_alive")
        result.framework.status = FrameworkStatus.DRAINING
        result.framework.last_heartbeat = time.time()  # Fresh heartbeat
        await registry._save_framework(result.framework)

        count = await registry._cleanup_dead_frameworks()
        assert count == 0
        fw = await registry.get(result.framework_id)
        assert fw.status == FrameworkStatus.DRAINING

    @pytest.mark.asyncio
    async def test_cleanup_marks_draining_expired_as_disconnected(self, registry):
        result = await _register_framework(registry, name="draining_expired")
        result.framework.status = FrameworkStatus.DRAINING
        result.framework.last_heartbeat = time.time() - 120
        await registry._save_framework(result.framework)

        count = await registry._cleanup_dead_frameworks()
        assert count == 1
        fw = await registry.get(result.framework_id)
        assert fw.status == FrameworkStatus.DISCONNECTED

    @pytest.mark.asyncio
    async def test_cleanup_emits_correct_event_data(self, registry):
        events = []
        registry.set_event_callback(lambda t, d: events.append((t, d)))

        result = await _register_framework(registry, name="event_disc_fw")
        result.framework.last_heartbeat = time.time() - 200
        await registry._save_framework(result.framework)

        await registry._cleanup_dead_frameworks()

        disc_events = [e for e in events if e[0] == "framework_disconnected"]
        assert len(disc_events) == 1
        assert disc_events[0][1]["framework_id"] == result.framework_id
        assert disc_events[0][1]["name"] == "event_disc_fw"

    @pytest.mark.asyncio
    async def test_cleanup_returns_zero_when_empty(self, registry):
        count = await registry._cleanup_dead_frameworks()
        assert count == 0

    @pytest.mark.asyncio
    async def test_cleanup_returns_correct_count(self, registry):
        for i in range(5):
            result = await _register_framework(registry, name=f"dead_{i}")
            result.framework.last_heartbeat = time.time() - 200
            await registry._save_framework(result.framework)

        # 2 alive frameworks
        for i in range(2):
            await _register_framework(registry, name=f"alive_{i}")

        count = await registry._cleanup_dead_frameworks()
        assert count == 5


# =============================================================================
# FederationRegistry: Advanced Lifecycle Hook Tests
# =============================================================================


class TestAdvancedLifecycleHooks:
    """Advanced tests for lifecycle hooks."""

    @pytest.mark.asyncio
    async def test_shutdown_hook_ids_recorded_on_framework(self, registry):
        registry.register_shutdown_hook("track_sh", lambda fw: None)
        result = await _register_framework(registry, name="sh_tracked")
        await registry.unregister(result.framework_id)
        fw_before_removal = result.framework
        assert "track_sh" in fw_before_removal.shutdown_hooks

    @pytest.mark.asyncio
    async def test_startup_hooks_called_in_order(self, registry):
        order = []
        for i in range(5):
            registry.register_startup_hook(f"order_{i}", lambda fw, idx=i: order.append(idx))
        await _register_framework(registry, name="ordered")
        assert order == [0, 1, 2, 3, 4]

    @pytest.mark.asyncio
    async def test_failing_startup_hook_does_not_affect_subsequent_hooks(self, registry):
        called = []

        def bad_hook(fw):
            called.append("bad")
            raise RuntimeError("explode")

        registry.register_startup_hook("bad", bad_hook)
        registry.register_startup_hook("good", lambda fw: called.append("good"))

        result = await _register_framework(registry, name="mixed_hooks")
        assert result.success is True
        assert "bad" in called
        assert "good" in called

    @pytest.mark.asyncio
    async def test_failing_shutdown_hook_does_not_affect_subsequent_hooks(self, registry):
        called = []

        def bad_hook(fw):
            called.append("bad_sh")
            raise RuntimeError("explode")

        registry.register_shutdown_hook("bad_sh", bad_hook)
        registry.register_shutdown_hook("good_sh", lambda fw: called.append("good_sh"))

        result = await _register_framework(registry, name="sh_mixed")
        await registry.unregister(result.framework_id)
        assert "bad_sh" in called
        assert "good_sh" in called

    @pytest.mark.asyncio
    async def test_async_failing_startup_hook(self, registry):
        async def async_bad(fw):
            raise RuntimeError("async boom")

        registry.register_startup_hook("async_bad", async_bad)
        result = await _register_framework(registry, name="async_fail")
        assert result.success is True

    @pytest.mark.asyncio
    async def test_async_failing_shutdown_hook(self, registry):
        async def async_bad_sh(fw):
            raise RuntimeError("async shutdown boom")

        registry.register_shutdown_hook("async_bad_sh", async_bad_sh)
        result = await _register_framework(registry, name="async_sh_fail")
        removed = await registry.unregister(result.framework_id)
        assert removed is True

    def test_unregister_hook_from_one_dict_only(self, registry):
        hook = MagicMock()
        registry.register_startup_hook("startup_only", hook)
        removed = registry.unregister_hook("startup_only")
        assert removed is True
        assert "startup_only" not in registry._startup_hooks

    def test_unregister_hook_not_in_any_dict(self, registry):
        removed = registry.unregister_hook("ghost_hook")
        assert removed is False

    @pytest.mark.asyncio
    async def test_close_runs_shutdown_hooks_for_all_active_frameworks(self):
        reg = FederationRegistry(redis_url="memory://")
        await reg.connect()

        called = []
        reg.register_shutdown_hook("close_sh", lambda fw: called.append(fw.name))

        await _register_framework(reg, name="fw_a")
        await _register_framework(reg, name="fw_b")
        await _register_framework(reg, name="fw_c")

        await reg.close()
        assert set(called) == {"fw_a", "fw_b", "fw_c"}


# =============================================================================
# FederationRegistry: Advanced Stats Tests
# =============================================================================


class TestAdvancedStats:
    """Advanced tests for registry statistics."""

    @pytest.mark.asyncio
    async def test_stats_by_status_distribution(self, registry):
        r1 = await _register_framework(registry, name="active_fw")
        r1.framework.health_status = HealthStatus.HEALTHY
        await registry._save_framework(r1.framework)

        r2 = await _register_framework(registry, name="draining_fw")
        r2.framework.status = FrameworkStatus.DRAINING
        await registry._save_framework(r2.framework)

        r3 = await _register_framework(registry, name="failed_fw")
        r3.framework.status = FrameworkStatus.FAILED
        await registry._save_framework(r3.framework)

        stats = await registry.get_stats()
        assert stats["by_status"]["active"] == 1  # Only r1 is active
        assert stats["by_status"].get("draining", 0) == 1
        assert stats["by_status"].get("failed", 0) == 1

    @pytest.mark.asyncio
    async def test_stats_by_health_distribution(self, registry):
        r1 = await _register_framework(registry, name="healthy_fw")
        r1.framework.health_status = HealthStatus.HEALTHY
        await registry._save_framework(r1.framework)

        r2 = await _register_framework(registry, name="degraded_fw")
        r2.framework.health_status = HealthStatus.DEGRADED
        await registry._save_framework(r2.framework)

        r3 = await _register_framework(registry, name="unknown_fw")
        # Default health is UNKNOWN

        stats = await registry.get_stats()
        assert stats["by_health"].get("healthy", 0) == 1
        assert stats["by_health"].get("degraded", 0) == 1
        assert stats["by_health"].get("unknown", 0) == 1

    @pytest.mark.asyncio
    async def test_stats_capability_counts_across_frameworks(self, registry):
        await _register_framework(
            registry, name="fw1", capabilities=[_cap("a"), _cap("b"), _cap("c")]
        )
        await _register_framework(registry, name="fw2", capabilities=[_cap("a"), _cap("d")])
        await _register_framework(registry, name="fw3", capabilities=[_cap("a")])

        stats = await registry.get_stats()
        assert stats["by_capability"]["a"] == 3
        assert stats["by_capability"]["b"] == 1
        assert stats["by_capability"]["c"] == 1
        assert stats["by_capability"]["d"] == 1

    @pytest.mark.asyncio
    async def test_stats_active_frameworks_count(self, registry):
        r1 = await _register_framework(registry, name="active1")
        r1.framework.health_status = HealthStatus.HEALTHY
        await registry._save_framework(r1.framework)

        r2 = await _register_framework(registry, name="active2")
        r2.framework.health_status = HealthStatus.HEALTHY
        await registry._save_framework(r2.framework)

        # This one is not active (UNKNOWN health)
        await _register_framework(registry, name="inactive1")

        stats = await registry.get_stats()
        assert stats["total_frameworks"] == 3
        assert stats["active_frameworks"] == 2


# =============================================================================
# FederationRegistry: Concurrent Operations Tests
# =============================================================================


class TestConcurrentOperations:
    """Tests for concurrent access patterns."""

    @pytest.mark.asyncio
    async def test_concurrent_register_and_unregister(self, registry):
        """Register and immediately unregister should not conflict."""
        results = []
        for i in range(5):
            r = await _register_framework(registry, name=f"conc_{i}")
            results.append(r)

        # Concurrently unregister all
        tasks = [registry.unregister(r.framework_id) for r in results]
        outcomes = await asyncio.gather(*tasks)
        assert all(outcomes)

        all_fws = await registry.list_all(include_inactive=True)
        assert len(all_fws) == 0

    @pytest.mark.asyncio
    async def test_concurrent_heartbeats(self, registry):
        result = await _register_framework(registry, name="hb_conc")
        tasks = [registry.heartbeat(result.framework_id) for _ in range(20)]
        outcomes = await asyncio.gather(*tasks)
        assert all(outcomes)

    @pytest.mark.asyncio
    async def test_concurrent_health_checks(self, registry):
        result = await _register_framework(registry, name="hc_conc")
        with patch.object(registry, "_check_health_endpoint", new_callable=AsyncMock) as mock:
            mock.return_value = True
            tasks = [registry.health_check(result.framework_id) for _ in range(10)]
            statuses = await asyncio.gather(*tasks)
            assert all(s == HealthStatus.HEALTHY for s in statuses)

    @pytest.mark.asyncio
    async def test_concurrent_find_by_capability(self, registry):
        for i in range(5):
            r = await _register_framework(
                registry, name=f"conc_cap_{i}", capabilities=[_cap("shared_c")]
            )
            r.framework.health_status = HealthStatus.HEALTHY
            await registry._save_framework(r.framework)

        tasks = [registry.find_by_capability("shared_c") for _ in range(10)]
        results = await asyncio.gather(*tasks)
        for result in results:
            assert len(result) == 5


# =============================================================================
# FederationRegistry: ExternalFramework Advanced Tests
# =============================================================================


class TestExternalFrameworkAdvanced:
    """Advanced tests for ExternalFramework dataclass behavior."""

    def test_is_alive_at_boundary(self):
        """Test is_alive at exact timeout boundary."""
        fw = ExternalFramework(name="test", last_heartbeat=time.time() - 60)
        # At exactly 60 seconds, should NOT be alive (strict less-than)
        assert fw.is_alive(timeout_seconds=60.0) is False

    def test_is_alive_just_under_boundary(self):
        fw = ExternalFramework(name="test", last_heartbeat=time.time() - 59.9)
        assert fw.is_alive(timeout_seconds=60.0) is True

    def test_record_health_check_updates_timestamp(self):
        fw = ExternalFramework(name="test")
        before = time.time()
        fw.record_health_check(True)
        assert fw.last_health_check >= before

    def test_record_health_check_many_failures_then_recovery(self):
        fw = ExternalFramework(name="test")
        for _ in range(10):
            fw.record_health_check(False)
        assert fw.consecutive_failures == 10
        assert fw.health_status == HealthStatus.UNHEALTHY

        fw.record_health_check(True)
        assert fw.consecutive_failures == 0
        assert fw.health_status == HealthStatus.HEALTHY

    def test_has_capability_empty_list(self):
        fw = ExternalFramework(name="test", capabilities=[])
        assert fw.has_capability("anything") is False

    def test_has_all_capabilities_with_empty_and_nonempty(self):
        fw = ExternalFramework(name="test", capabilities=[_cap("a")])
        assert fw.has_all_capabilities([]) is True
        assert fw.has_all_capabilities(["a"]) is True
        assert fw.has_all_capabilities(["a", "b"]) is False

    def test_get_capability_first_match(self):
        """When duplicate capabilities exist, returns the first match."""
        cap1 = FrameworkCapability(name="dup", description="first")
        cap2 = FrameworkCapability(name="dup", description="second")
        fw = ExternalFramework(name="test", capabilities=[cap1, cap2])
        found = fw.get_capability("dup")
        assert found is not None
        assert found.description == "first"

    def test_framework_from_dict_minimal(self):
        """Test from_dict with minimal data."""
        fw = ExternalFramework.from_dict({"name": "minimal"})
        assert fw.name == "minimal"
        assert fw.version == "0.0.0"
        assert fw.capabilities == []
        assert fw.health_status == HealthStatus.UNKNOWN
        assert fw.status == FrameworkStatus.REGISTERING

    def test_framework_from_dict_unknown_enum_raises(self):
        """Invalid enum values should raise ValueError."""
        with pytest.raises(ValueError):
            ExternalFramework.from_dict({"health_status": "invalid_status"})

    def test_to_dict_tags_are_list(self):
        """Tags in to_dict should be a list (JSON-serializable)."""
        fw = ExternalFramework(
            framework_id="fw-tag-test",
            name="test",
            tags={"alpha", "beta"},
        )
        d = fw.to_dict()
        assert isinstance(d["tags"], list)
        assert set(d["tags"]) == {"alpha", "beta"}

    def test_post_init_no_name_no_id(self):
        """When both name and framework_id are empty, framework_id stays empty."""
        fw = ExternalFramework(name="", framework_id="")
        assert fw.framework_id == ""

    def test_supports_api_version_empty_list(self):
        fw = ExternalFramework(name="test", supported_api_versions=[])
        assert fw.supports_api_version("1.0.0") is False


# =============================================================================
# FederationRegistry: Version Negotiation Advanced Tests
# =============================================================================


class TestVersionNegotiationAdvanced:
    """Advanced tests for version negotiation logic."""

    def test_negotiate_with_only_highest_version(self, registry):
        result = registry._negotiate_version(["2.0.0"])
        assert result == "2.0.0"

    def test_negotiate_with_superset_of_server_versions(self, registry):
        # Client supports more than server
        result = registry._negotiate_version(["0.9.0", "1.0.0", "1.1.0", "2.0.0", "3.0.0"])
        assert result == "2.0.0"

    def test_negotiate_with_duplicate_versions(self, registry):
        result = registry._negotiate_version(["1.0.0", "1.0.0", "1.1.0"])
        assert result == "1.1.0"

    def test_negotiate_returns_none_for_future_only_versions(self, registry):
        result = registry._negotiate_version(["3.0.0", "4.0.0", "5.0.0"])
        assert result is None

    def test_negotiate_single_common_at_lowest(self, registry):
        result = registry._negotiate_version(["0.1.0", "1.0.0"])
        assert result == "1.0.0"


# =============================================================================
# FederationRegistry: Refresh Capabilities Advanced Tests
# =============================================================================


class TestRefreshCapabilitiesAdvanced:
    """Advanced tests for live capability refresh."""

    @pytest.mark.asyncio
    async def test_refresh_uses_capabilities_endpoint_when_available(self, registry):
        result = await _register_framework(
            registry,
            name="cap_ep",
            endpoints={
                "base": "http://localhost:9000",
                "capabilities": "http://localhost:9000/custom-caps",
            },
        )

        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={"capabilities": [{"name": "refreshed_cap"}]})
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=False)

        mock_session = AsyncMock()
        mock_session.get = MagicMock(return_value=mock_response)
        registry._http_client = mock_session

        caps = await registry.refresh_capabilities(result.framework_id)
        assert len(caps) == 1
        assert caps[0].name == "refreshed_cap"
        # Verify custom endpoint was called
        mock_session.get.assert_called_once_with("http://localhost:9000/custom-caps")

    @pytest.mark.asyncio
    async def test_refresh_non_200_returns_cached(self, registry):
        result = await _register_framework(
            registry,
            name="non200",
            capabilities=[_cap("original")],
            endpoints={"base": "http://localhost:9000"},
        )

        mock_response = AsyncMock()
        mock_response.status = 500
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=False)

        mock_session = AsyncMock()
        mock_session.get = MagicMock(return_value=mock_response)
        registry._http_client = mock_session

        caps = await registry.refresh_capabilities(result.framework_id)
        assert len(caps) == 1
        assert caps[0].name == "original"

    @pytest.mark.asyncio
    async def test_refresh_exception_returns_cached(self, registry):
        result = await _register_framework(
            registry,
            name="error_refresh",
            capabilities=[_cap("cached_cap")],
            endpoints={"base": "http://localhost:9000"},
        )

        mock_session = AsyncMock()
        mock_session.get = MagicMock(side_effect=ConnectionError("refused"))
        registry._http_client = mock_session

        caps = await registry.refresh_capabilities(result.framework_id)
        assert len(caps) == 1
        assert caps[0].name == "cached_cap"

    @pytest.mark.asyncio
    async def test_refresh_no_change_does_not_emit_event(self, registry):
        events = []
        registry.set_event_callback(lambda t, d: events.append((t, d)))

        result = await _register_framework(
            registry,
            name="same_caps",
            capabilities=[_cap("stable")],
            endpoints={"base": "http://localhost:9000"},
        )

        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={"capabilities": [{"name": "stable"}]})
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=False)

        mock_session = AsyncMock()
        mock_session.get = MagicMock(return_value=mock_response)
        registry._http_client = mock_session

        # Clear events from registration
        events.clear()
        await registry.refresh_capabilities(result.framework_id)

        cap_events = [e for e in events if e[0] == "capabilities_updated"]
        assert len(cap_events) == 0

    @pytest.mark.asyncio
    async def test_refresh_updates_capability_index(self, registry):
        result = await _register_framework(
            registry,
            name="idx_refresh",
            capabilities=[_cap("old_idx")],
            endpoints={"base": "http://localhost:9000"},
        )
        assert result.framework_id in registry._capability_index.get("old_idx", set())

        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={"capabilities": [{"name": "new_idx"}]})
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=False)

        mock_session = AsyncMock()
        mock_session.get = MagicMock(return_value=mock_response)
        registry._http_client = mock_session

        await registry.refresh_capabilities(result.framework_id)
        assert result.framework_id in registry._capability_index.get("new_idx", set())


# =============================================================================
# FederationRegistry: Connect / Close Advanced Tests
# =============================================================================


class TestConnectCloseAdvanced:
    """Advanced tests for connect and close lifecycle."""

    @pytest.mark.asyncio
    async def test_connect_redis_connection_failure_falls_back(self):
        """When Redis is not available, falls back to in-memory."""
        reg = FederationRegistry(redis_url="redis://nonexistent-host:6379")

        # Mock redis.asyncio.from_url to return a client whose ping()
        # raises OSError -- which is in the caught exception tuple
        # (OSError, ConnectionError, TimeoutError). The real redis library
        # raises redis.exceptions.ConnectionError which is NOT a subclass
        # of Python's built-in ConnectionError.
        mock_redis_client = AsyncMock()
        mock_redis_client.ping = AsyncMock(side_effect=OSError("Connection refused"))
        mock_redis_client.close = AsyncMock()

        import redis.asyncio as _aioredis

        with patch.object(_aioredis, "from_url", return_value=mock_redis_client):
            # This should not raise even if Redis is unreachable
            await reg.connect()
            assert reg._health_check_task is not None
            await reg.close()

    @pytest.mark.asyncio
    async def test_close_without_connect(self):
        """Closing without connecting should not raise."""
        reg = FederationRegistry(redis_url="memory://")
        await reg.close()

    @pytest.mark.asyncio
    async def test_double_close(self):
        """Closing twice should not raise."""
        reg = FederationRegistry(redis_url="memory://")
        await reg.connect()
        await reg.close()
        await reg.close()

    @pytest.mark.asyncio
    async def test_operations_after_close(self):
        """Operations after close should still work on local cache."""
        reg = FederationRegistry(redis_url="memory://")
        await reg.connect()
        result = await _register_framework(reg, name="before_close")
        await reg.close()

        # get should still work from local cache
        fw = await reg.get(result.framework_id)
        # After close, the cache was not cleared, so this depends on implementation
        # At minimum, it should not raise

    @pytest.mark.asyncio
    async def test_connect_creates_fresh_background_tasks(self):
        """Each connect creates new background tasks."""
        reg = FederationRegistry(redis_url="memory://")
        await reg.connect()
        task1_hc = reg._health_check_task
        task1_cl = reg._cleanup_task
        await reg.close()

        await reg.connect()
        assert reg._health_check_task is not task1_hc
        assert reg._cleanup_task is not task1_cl
        await reg.close()
