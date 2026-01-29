"""
Tests for Incident Persistence Store.

Validates CRUD operations for status page incidents.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from aragora.observability.incident_store import (
    IncidentStore,
    reset_incident_store,
)


@pytest.fixture
def store(tmp_path: Path) -> IncidentStore:
    return IncidentStore(db_path=str(tmp_path / "test_incidents.db"))


@pytest.fixture(autouse=True)
def reset_global():
    reset_incident_store()
    yield
    reset_incident_store()


class TestIncidentCRUD:
    """Test basic incident create/read/update operations."""

    def test_create_incident(self, store: IncidentStore):
        incident_id = store.create_incident(
            title="API degraded",
            severity="major",
            components=["api"],
        )

        assert incident_id
        incident = store.get_incident(incident_id)
        assert incident is not None
        assert incident.title == "API degraded"
        assert incident.severity == "major"
        assert incident.status == "investigating"
        assert incident.components == ["api"]
        assert incident.resolved_at is None

    def test_create_with_initial_message(self, store: IncidentStore):
        incident_id = store.create_incident(
            title="DB outage",
            severity="critical",
            components=["database"],
            initial_message="Database is unreachable",
        )

        incident = store.get_incident(incident_id)
        assert len(incident.updates) == 1
        assert incident.updates[0].message == "Database is unreachable"
        assert incident.updates[0].status == "investigating"

    def test_add_update(self, store: IncidentStore):
        incident_id = store.create_incident(
            title="Latency spike",
            severity="minor",
            components=["api"],
        )

        store.add_update(incident_id, "identified", "Root cause: cache miss storm")
        store.add_update(incident_id, "monitoring", "Cache primed, monitoring recovery")

        incident = store.get_incident(incident_id)
        assert incident.status == "monitoring"
        assert len(incident.updates) == 2
        assert incident.updates[0].status == "identified"
        assert incident.updates[1].status == "monitoring"
        assert incident.resolved_at is None

    def test_resolve_incident(self, store: IncidentStore):
        incident_id = store.create_incident(
            title="Auth outage",
            severity="critical",
            components=["auth"],
        )

        store.add_update(incident_id, "identified", "Token validation failing")
        store.resolve_incident(incident_id, "Certificate renewed")

        incident = store.get_incident(incident_id)
        assert incident.status == "resolved"
        assert incident.resolved_at is not None
        assert len(incident.updates) == 2
        assert incident.updates[-1].message == "Certificate renewed"

    def test_get_nonexistent_incident(self, store: IncidentStore):
        assert store.get_incident("nonexistent") is None


class TestIncidentQueries:
    """Test incident query methods."""

    def test_get_active_incidents(self, store: IncidentStore):
        store.create_incident(title="Active 1", severity="minor", components=["api"])
        store.create_incident(title="Active 2", severity="major", components=["db"])

        resolved_id = store.create_incident(title="Resolved", severity="minor", components=["api"])
        store.resolve_incident(resolved_id)

        active = store.get_active_incidents()
        assert len(active) == 2
        titles = {i.title for i in active}
        assert "Active 1" in titles
        assert "Active 2" in titles
        assert "Resolved" not in titles

    def test_get_recent_incidents(self, store: IncidentStore):
        # Create and resolve an incident
        iid = store.create_incident(
            title="Resolved recently",
            severity="minor",
            components=["api"],
        )
        store.resolve_incident(iid, "Fixed")

        recent = store.get_recent_incidents(days=7)
        assert len(recent) == 1
        assert recent[0].title == "Resolved recently"

    def test_active_incidents_excludes_resolved(self, store: IncidentStore):
        iid = store.create_incident(title="Test", severity="minor", components=[])
        store.resolve_incident(iid)

        active = store.get_active_incidents()
        assert len(active) == 0


class TestSLOViolationIntegration:
    """Test auto-creation of incidents from SLO violations."""

    def test_create_from_slo_violation(self, store: IncidentStore):
        iid = store.create_from_slo_violation(
            slo_name="availability",
            severity="critical",
            message="Availability dropped below 99.9%",
        )

        incident = store.get_incident(iid)
        assert incident is not None
        assert "availability" in incident.title.lower()
        assert incident.severity == "critical"
        assert incident.source == "slo_violation"
        assert "api" in incident.components

    def test_slo_component_mapping(self, store: IncidentStore):
        mappings = {
            "availability": ["api"],
            "debate_success": ["debates"],
            "knowledge_retrieval": ["knowledge"],
            "websocket_latency": ["websocket"],
            "auth_latency": ["auth"],
        }
        for slo_name, expected_components in mappings.items():
            iid = store.create_from_slo_violation(
                slo_name=slo_name,
                severity="warning",
                message=f"SLO {slo_name} violation",
            )
            incident = store.get_incident(iid)
            assert incident.components == expected_components, (
                f"SLO '{slo_name}' should map to {expected_components}"
            )

    def test_slo_with_custom_components(self, store: IncidentStore):
        iid = store.create_from_slo_violation(
            slo_name="custom_slo",
            severity="major",
            message="Custom SLO violated",
            components=["database", "redis"],
        )
        incident = store.get_incident(iid)
        assert incident.components == ["database", "redis"]


class TestIncidentSerialization:
    """Test incident to_dict serialization."""

    def test_to_dict(self, store: IncidentStore):
        iid = store.create_incident(
            title="Test incident",
            severity="major",
            components=["api", "database"],
            source="manual",
            initial_message="Initial investigation",
        )
        store.add_update(iid, "identified", "Found root cause")

        incident = store.get_incident(iid)
        d = incident.to_dict()

        assert d["id"] == iid
        assert d["title"] == "Test incident"
        assert d["severity"] == "major"
        assert d["components"] == ["api", "database"]
        assert d["source"] == "manual"
        assert len(d["updates"]) == 2
        assert d["updates"][0]["message"] == "Initial investigation"
        assert d["updates"][1]["message"] == "Found root cause"
