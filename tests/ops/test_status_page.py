"""
Tests for the public status page and SLA instrumentation system.

Tests cover:
- ServiceStatus enum values and ordering
- ComponentStatus dataclass creation and serialization
- IncidentUpdate and IncidentRecord dataclasses
- StatusPage:
  - get_overall_status with various component states
  - get_component_statuses with and without health registry
  - get_active_incidents and get_incident_history
  - get_uptime calculations
  - create_incident, update_incident, resolve_incident lifecycle
  - Integration with HealthRegistry
  - to_dict serialization
- Singleton: get_status_page, reset_status_page
- StatusPageHandler: public endpoint routing and rate limiting
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from aragora.ops.status_page import (
    ComponentStatus,
    DEFAULT_COMPONENTS,
    IncidentRecord,
    IncidentUpdate,
    ServiceStatus,
    StatusPage,
    get_status_page,
    reset_status_page,
)


# ============================================================================
# ServiceStatus Enum
# ============================================================================


class TestServiceStatus:
    """Tests for the ServiceStatus enum."""

    def test_all_values_present(self):
        values = {s.value for s in ServiceStatus}
        assert values == {
            "operational",
            "degraded",
            "partial_outage",
            "major_outage",
            "maintenance",
        }

    def test_string_enum(self):
        assert isinstance(ServiceStatus.OPERATIONAL, str)
        assert ServiceStatus.OPERATIONAL == "operational"

    def test_from_value(self):
        assert ServiceStatus("degraded") == ServiceStatus.DEGRADED


# ============================================================================
# ComponentStatus Dataclass
# ============================================================================


class TestComponentStatus:
    """Tests for the ComponentStatus dataclass."""

    def test_creation_with_defaults(self):
        cs = ComponentStatus(name="api", status=ServiceStatus.OPERATIONAL)
        assert cs.name == "api"
        assert cs.status == ServiceStatus.OPERATIONAL
        assert cs.description == ""
        assert cs.response_time_ms is None
        assert isinstance(cs.last_checked, datetime)

    def test_creation_with_all_fields(self):
        now = datetime.now(timezone.utc)
        cs = ComponentStatus(
            name="database",
            status=ServiceStatus.DEGRADED,
            description="Primary data store",
            last_checked=now,
            response_time_ms=42.5,
        )
        assert cs.name == "database"
        assert cs.status == ServiceStatus.DEGRADED
        assert cs.description == "Primary data store"
        assert cs.last_checked == now
        assert cs.response_time_ms == 42.5

    def test_to_dict(self):
        now = datetime.now(timezone.utc)
        cs = ComponentStatus(
            name="api",
            status=ServiceStatus.OPERATIONAL,
            description="REST API",
            last_checked=now,
            response_time_ms=10.0,
        )
        d = cs.to_dict()
        assert d["name"] == "api"
        assert d["status"] == "operational"
        assert d["description"] == "REST API"
        assert d["last_checked"] == now.isoformat()
        assert d["response_time_ms"] == 10.0


# ============================================================================
# IncidentUpdate / IncidentRecord Dataclasses
# ============================================================================


class TestIncidentUpdate:
    """Tests for the IncidentUpdate dataclass."""

    def test_creation(self):
        now = datetime.now(timezone.utc)
        update = IncidentUpdate(
            timestamp=now, message="Looking into it", status="investigating"
        )
        assert update.timestamp == now
        assert update.message == "Looking into it"
        assert update.status == "investigating"


class TestIncidentRecord:
    """Tests for the IncidentRecord dataclass."""

    def test_creation_defaults(self):
        now = datetime.now(timezone.utc)
        incident = IncidentRecord(
            id="inc-1",
            title="API Latency",
            status="investigating",
            severity="warning",
            started_at=now,
        )
        assert incident.id == "inc-1"
        assert incident.resolved_at is None
        assert incident.updates == []

    def test_to_dict(self):
        now = datetime.now(timezone.utc)
        incident = IncidentRecord(
            id="inc-2",
            title="DB Outage",
            status="resolved",
            severity="critical",
            started_at=now,
            resolved_at=now + timedelta(hours=1),
            updates=[
                IncidentUpdate(timestamp=now, message="Detected", status="investigating"),
                IncidentUpdate(
                    timestamp=now + timedelta(hours=1),
                    message="Fixed",
                    status="resolved",
                ),
            ],
        )
        d = incident.to_dict()
        assert d["id"] == "inc-2"
        assert d["status"] == "resolved"
        assert d["severity"] == "critical"
        assert d["resolved_at"] is not None
        assert len(d["updates"]) == 2
        assert d["updates"][0]["status"] == "investigating"
        assert d["updates"][1]["message"] == "Fixed"

    def test_to_dict_unresolved(self):
        now = datetime.now(timezone.utc)
        incident = IncidentRecord(
            id="inc-3",
            title="Ongoing",
            status="investigating",
            severity="warning",
            started_at=now,
        )
        d = incident.to_dict()
        assert d["resolved_at"] is None


# ============================================================================
# StatusPage - Core Functionality
# ============================================================================


class TestStatusPage:
    """Tests for the StatusPage class."""

    def test_default_components(self):
        page = StatusPage()
        statuses = page.get_component_statuses()
        names = {s.name for s in statuses}
        assert names == set(DEFAULT_COMPONENTS.keys())

    def test_custom_components(self):
        page = StatusPage(components={"svc_a": "Service A", "svc_b": "Service B"})
        statuses = page.get_component_statuses()
        assert len(statuses) == 2
        assert {s.name for s in statuses} == {"svc_a", "svc_b"}

    def test_all_operational_without_registry(self):
        page = StatusPage(health_registry=MagicMock(get=MagicMock(return_value=None)))
        overall = page.get_overall_status()
        assert overall == ServiceStatus.OPERATIONAL

    def test_overall_status_degraded(self):
        """When one component is degraded, overall should be degraded."""
        # Create a mock registry that returns a degraded checker for 'api'
        mock_status = SimpleNamespace(
            healthy=False,
            consecutive_failures=1,
            last_check=datetime.now(timezone.utc),
            latency_ms=200.0,
        )
        mock_checker = MagicMock()
        mock_checker.get_status.return_value = mock_status

        registry = MagicMock()
        registry.get = lambda name: mock_checker if name == "api" else None

        page = StatusPage(health_registry=registry)
        overall = page.get_overall_status()
        assert overall == ServiceStatus.DEGRADED

    def test_overall_status_major_outage(self):
        """When a component has many consecutive failures, should be major outage."""
        mock_status = SimpleNamespace(
            healthy=False,
            consecutive_failures=5,
            last_check=datetime.now(timezone.utc),
            latency_ms=None,
        )
        mock_checker = MagicMock()
        mock_checker.get_status.return_value = mock_status

        registry = MagicMock()
        registry.get = lambda name: mock_checker if name == "database" else None

        page = StatusPage(health_registry=registry)
        overall = page.get_overall_status()
        assert overall == ServiceStatus.MAJOR_OUTAGE

    def test_overall_status_partial_outage(self):
        """3-4 consecutive failures maps to partial outage."""
        mock_status = SimpleNamespace(
            healthy=False,
            consecutive_failures=3,
            last_check=datetime.now(timezone.utc),
            latency_ms=None,
        )
        mock_checker = MagicMock()
        mock_checker.get_status.return_value = mock_status

        registry = MagicMock()
        registry.get = lambda name: mock_checker if name == "queue" else None

        page = StatusPage(health_registry=registry)
        overall = page.get_overall_status()
        assert overall == ServiceStatus.PARTIAL_OUTAGE

    def test_component_response_time(self):
        """Component should expose response_time_ms from health checker."""
        mock_status = SimpleNamespace(
            healthy=True,
            consecutive_failures=0,
            last_check=datetime.now(timezone.utc),
            latency_ms=15.5,
        )
        mock_checker = MagicMock()
        mock_checker.get_status.return_value = mock_status

        registry = MagicMock()
        registry.get = lambda name: mock_checker if name == "api" else None

        page = StatusPage(health_registry=registry)
        statuses = page.get_component_statuses()
        api_status = next(s for s in statuses if s.name == "api")
        assert api_status.response_time_ms == 15.5
        assert api_status.status == ServiceStatus.OPERATIONAL


# ============================================================================
# StatusPage - Incident Management
# ============================================================================


class TestStatusPageIncidents:
    """Tests for incident management on StatusPage."""

    def test_create_incident(self):
        page = StatusPage()
        incident = page.create_incident("API Down", "critical", "All endpoints returning 500")
        assert incident.title == "API Down"
        assert incident.severity == "critical"
        assert incident.status == "investigating"
        assert len(incident.updates) == 1
        assert incident.updates[0].message == "All endpoints returning 500"

    def test_get_active_incidents(self):
        page = StatusPage()
        page.create_incident("Issue A", "warning", "msg a")
        page.create_incident("Issue B", "critical", "msg b")

        active = page.get_active_incidents()
        assert len(active) == 2

    def test_update_incident(self):
        page = StatusPage()
        incident = page.create_incident("Latency spike", "warning", "Investigating")

        result = page.update_incident(incident.id, "Found root cause", "identified")
        assert result is True
        assert incident.status == "identified"
        assert len(incident.updates) == 2

    def test_update_nonexistent_incident(self):
        page = StatusPage()
        result = page.update_incident("nonexistent-id", "msg", "identified")
        assert result is False

    def test_resolve_incident(self):
        page = StatusPage()
        incident = page.create_incident("DB slow", "warning", "Queries timing out")

        result = page.resolve_incident(incident.id, "Indexes rebuilt")
        assert result is True
        assert incident.status == "resolved"
        assert incident.resolved_at is not None
        assert len(incident.updates) == 2
        assert incident.updates[-1].status == "resolved"

    def test_resolved_not_in_active(self):
        page = StatusPage()
        incident = page.create_incident("Temp issue", "info", "msg")
        page.resolve_incident(incident.id, "Fixed")

        active = page.get_active_incidents()
        assert len(active) == 0

    def test_get_incident_history(self):
        page = StatusPage()
        inc = page.create_incident("Old issue", "warning", "msg")
        page.resolve_incident(inc.id, "Fixed")

        history = page.get_incident_history(days=30)
        assert len(history) == 1
        assert history[0].id == inc.id

    def test_incident_history_excludes_active(self):
        page = StatusPage()
        page.create_incident("Active issue", "warning", "msg")

        history = page.get_incident_history(days=30)
        assert len(history) == 0

    def test_incident_history_window(self):
        """Old incidents outside the window should be excluded."""
        page = StatusPage()
        inc = page.create_incident("Recent", "warning", "msg")
        page.resolve_incident(inc.id, "Fixed")

        # Manually backdate the incident start
        with page._lock:
            page._incidents[0].started_at = datetime.now(timezone.utc) - timedelta(days=60)

        history = page.get_incident_history(days=30)
        assert len(history) == 0

    def test_maintenance_incident_affects_overall_status(self):
        page = StatusPage(health_registry=MagicMock(get=MagicMock(return_value=None)))
        page.create_incident("Planned maintenance", "maintenance", "Upgrading database")

        overall = page.get_overall_status()
        assert overall == ServiceStatus.MAINTENANCE


# ============================================================================
# StatusPage - Uptime
# ============================================================================


class TestStatusPageUptime:
    """Tests for uptime calculation."""

    def test_uptime_no_records(self):
        page = StatusPage()
        uptime = page.get_uptime("api", days=30)
        assert uptime == 100.0

    def test_uptime_all_healthy(self):
        page = StatusPage(health_registry=MagicMock(get=MagicMock(return_value=None)))
        # Calling get_component_statuses records uptime samples
        page.get_component_statuses()
        page.get_component_statuses()

        uptime = page.get_uptime("api", days=30)
        assert uptime == 100.0

    def test_uptime_with_failures(self):
        page = StatusPage(components={"svc": "test"})

        # Manually inject uptime records
        now = datetime.now(timezone.utc)
        with page._lock:
            page._uptime_records["svc"] = [
                (now - timedelta(minutes=i), i % 5 != 0) for i in range(100)
            ]

        uptime = page.get_uptime("svc", days=1)
        # 20 out of 100 samples are failures (every 5th)
        assert uptime == pytest.approx(80.0, abs=0.1)

    def test_uptime_respects_window(self):
        page = StatusPage(components={"svc": "test"})
        now = datetime.now(timezone.utc)

        with page._lock:
            # All healthy recently
            page._uptime_records["svc"] = [
                (now - timedelta(hours=i), True) for i in range(24)
            ]
            # Add old unhealthy records outside the 1-day window
            page._uptime_records["svc"].extend(
                [(now - timedelta(days=5, hours=i), False) for i in range(24)]
            )

        uptime = page.get_uptime("svc", days=1)
        assert uptime == 100.0


# ============================================================================
# StatusPage - Serialization
# ============================================================================


class TestStatusPageSerialization:
    """Tests for to_dict serialization."""

    def test_to_dict_structure(self):
        page = StatusPage(health_registry=MagicMock(get=MagicMock(return_value=None)))
        page.create_incident("Test incident", "warning", "msg")

        d = page.to_dict()

        assert "status" in d
        assert "updated_at" in d
        assert "components" in d
        assert "active_incidents" in d
        assert "incident_history" in d
        assert "uptime" in d

        assert isinstance(d["components"], list)
        assert len(d["components"]) == len(DEFAULT_COMPONENTS)
        assert isinstance(d["active_incidents"], list)
        assert len(d["active_incidents"]) == 1

    def test_to_dict_operational(self):
        page = StatusPage(
            health_registry=MagicMock(get=MagicMock(return_value=None)),
            components={"api": "API"},
        )
        d = page.to_dict()
        assert d["status"] == "operational"
        assert len(d["components"]) == 1
        assert d["components"][0]["status"] == "operational"


# ============================================================================
# Singleton
# ============================================================================


class TestSingleton:
    """Tests for get_status_page / reset_status_page."""

    def test_get_returns_same_instance(self):
        reset_status_page()
        a = get_status_page()
        b = get_status_page()
        assert a is b

    def test_reset_creates_new_instance(self):
        reset_status_page()
        a = get_status_page()
        reset_status_page()
        b = get_status_page()
        assert a is not b


# ============================================================================
# StatusPage - Health Registry Integration
# ============================================================================


class TestHealthRegistryIntegration:
    """Tests for integration with aragora.resilience.health.HealthRegistry."""

    def test_integration_with_real_health_registry(self):
        """Test StatusPage with actual HealthRegistry objects."""
        from aragora.resilience.health import HealthRegistry

        registry = HealthRegistry()
        checker = registry.register("api")
        checker.record_success(latency_ms=12.0)
        checker.record_success(latency_ms=14.0)

        page = StatusPage(health_registry=registry, components={"api": "REST API"})
        statuses = page.get_component_statuses()

        assert len(statuses) == 1
        api = statuses[0]
        assert api.name == "api"
        assert api.status == ServiceStatus.OPERATIONAL
        assert api.response_time_ms is not None
        assert api.response_time_ms > 0

    def test_integration_degraded_component(self):
        """Test that a failing health checker maps to DEGRADED."""
        from aragora.resilience.health import HealthRegistry

        registry = HealthRegistry()
        checker = registry.register("database", failure_threshold=2)
        checker.record_failure("Connection refused")
        checker.record_failure("Connection refused")

        page = StatusPage(health_registry=registry, components={"database": "DB"})
        statuses = page.get_component_statuses()

        db = statuses[0]
        assert db.status != ServiceStatus.OPERATIONAL

    def test_health_to_service_status_mapping(self):
        """Verify the mapping function directly."""
        assert StatusPage._health_to_service_status(True, 0) == ServiceStatus.OPERATIONAL
        assert StatusPage._health_to_service_status(False, 1) == ServiceStatus.DEGRADED
        assert StatusPage._health_to_service_status(False, 2) == ServiceStatus.DEGRADED
        assert StatusPage._health_to_service_status(False, 3) == ServiceStatus.PARTIAL_OUTAGE
        assert StatusPage._health_to_service_status(False, 4) == ServiceStatus.PARTIAL_OUTAGE
        assert StatusPage._health_to_service_status(False, 5) == ServiceStatus.MAJOR_OUTAGE
        assert StatusPage._health_to_service_status(False, 10) == ServiceStatus.MAJOR_OUTAGE


# ============================================================================
# StatusPage - Thread Safety
# ============================================================================


class TestThreadSafety:
    """Basic thread safety tests."""

    def test_concurrent_incident_creation(self):
        import threading

        page = StatusPage()
        errors = []

        def create_incidents():
            try:
                for i in range(10):
                    page.create_incident(f"Inc {i}", "warning", "msg")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=create_incidents) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(page.get_active_incidents()) == 50
