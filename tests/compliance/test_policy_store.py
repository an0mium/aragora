"""
Tests for the PolicyStore.

Covers:
- Policy CRUD operations
- Violation CRUD operations
- Filtering and pagination
- Audit logging
- Status updates
"""

from __future__ import annotations

import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pytest

from aragora.compliance.policy_store import (
    Policy,
    PolicyRule,
    PolicyStore,
    Violation,
)


@pytest.fixture
def temp_db():
    """Create a temporary database file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir) / "test_policy_store.db"


@pytest.fixture
def store(temp_db):
    """Create a PolicyStore instance."""
    return PolicyStore(temp_db)


@pytest.fixture
def sample_policy():
    """Create a sample policy."""
    return Policy(
        id="pol_test123",
        name="Test GDPR Policy",
        description="Test policy for GDPR compliance",
        framework_id="gdpr",
        workspace_id="ws_default",
        vertical_id="legal",
        level="mandatory",
        enabled=True,
        rules=[
            PolicyRule(
                rule_id="gdpr_consent",
                name="Consent Required",
                description="Must obtain user consent",
                severity="high",
                enabled=True,
            ),
            PolicyRule(
                rule_id="gdpr_retention",
                name="Data Retention",
                description="Limit data retention period",
                severity="medium",
                enabled=True,
            ),
        ],
        created_by="test_user",
    )


@pytest.fixture
def sample_violation():
    """Create a sample violation."""
    return Violation(
        id="viol_test123",
        policy_id="pol_test123",
        rule_id="gdpr_consent",
        rule_name="Consent Required",
        framework_id="gdpr",
        vertical_id="legal",
        workspace_id="ws_default",
        severity="high",
        status="open",
        description="Missing consent banner on signup page",
        source="forms/signup.tsx:42",
    )


# =============================================================================
# Policy CRUD Tests
# =============================================================================


class TestPolicyCRUD:
    """Tests for policy CRUD operations."""

    def test_create_policy(self, store, sample_policy):
        """Test creating a policy."""
        created = store.create_policy(sample_policy)

        assert created.id == sample_policy.id
        assert created.name == sample_policy.name
        assert created.framework_id == "gdpr"
        assert len(created.rules) == 2

    def test_get_policy(self, store, sample_policy):
        """Test retrieving a policy."""
        store.create_policy(sample_policy)

        retrieved = store.get_policy(sample_policy.id)

        assert retrieved is not None
        assert retrieved.id == sample_policy.id
        assert retrieved.name == sample_policy.name
        assert retrieved.framework_id == "gdpr"
        assert len(retrieved.rules) == 2
        assert retrieved.rules[0].rule_id == "gdpr_consent"

    def test_get_policy_not_found(self, store):
        """Test retrieving a non-existent policy."""
        result = store.get_policy("nonexistent")
        assert result is None

    def test_list_policies(self, store, sample_policy):
        """Test listing policies."""
        store.create_policy(sample_policy)

        # Create another policy
        another = Policy(
            id="pol_test456",
            name="HIPAA Policy",
            description="Healthcare compliance",
            framework_id="hipaa",
            workspace_id="ws_default",
            vertical_id="healthcare",
            level="mandatory",
            enabled=True,
        )
        store.create_policy(another)

        policies = store.list_policies()

        assert len(policies) == 2

    def test_list_policies_filter_by_workspace(self, store, sample_policy):
        """Test filtering policies by workspace."""
        store.create_policy(sample_policy)

        # Create policy in different workspace
        another = Policy(
            id="pol_test456",
            name="Other Workspace Policy",
            description="Different workspace",
            framework_id="sox",
            workspace_id="ws_other",
            vertical_id="accounting",
        )
        store.create_policy(another)

        policies = store.list_policies(workspace_id="ws_default")

        assert len(policies) == 1
        assert policies[0].id == sample_policy.id

    def test_list_policies_filter_by_vertical(self, store, sample_policy):
        """Test filtering policies by vertical."""
        store.create_policy(sample_policy)

        another = Policy(
            id="pol_test456",
            name="Healthcare Policy",
            description="Healthcare",
            framework_id="hipaa",
            workspace_id="ws_default",
            vertical_id="healthcare",
        )
        store.create_policy(another)

        policies = store.list_policies(vertical_id="legal")

        assert len(policies) == 1
        assert policies[0].vertical_id == "legal"

    def test_list_policies_enabled_only(self, store, sample_policy):
        """Test filtering to only enabled policies."""
        store.create_policy(sample_policy)

        disabled = Policy(
            id="pol_disabled",
            name="Disabled Policy",
            description="This is disabled",
            framework_id="owasp",
            workspace_id="ws_default",
            vertical_id="software",
            enabled=False,
        )
        store.create_policy(disabled)

        policies = store.list_policies(enabled_only=True)

        assert len(policies) == 1
        assert policies[0].enabled is True

    def test_update_policy(self, store, sample_policy):
        """Test updating a policy."""
        store.create_policy(sample_policy)

        updated = store.update_policy(
            sample_policy.id,
            {"name": "Updated GDPR Policy", "description": "Updated description"},
            changed_by="admin_user",
        )

        assert updated is not None
        assert updated.name == "Updated GDPR Policy"
        assert updated.description == "Updated description"

    def test_update_policy_not_found(self, store):
        """Test updating a non-existent policy."""
        result = store.update_policy("nonexistent", {"name": "New Name"})
        assert result is None

    def test_delete_policy(self, store, sample_policy):
        """Test deleting a policy."""
        store.create_policy(sample_policy)

        success = store.delete_policy(sample_policy.id)
        assert success is True

        # Verify deleted
        retrieved = store.get_policy(sample_policy.id)
        assert retrieved is None

    def test_delete_policy_not_found(self, store):
        """Test deleting a non-existent policy."""
        success = store.delete_policy("nonexistent")
        assert success is False

    def test_toggle_policy(self, store, sample_policy):
        """Test toggling policy enabled status."""
        store.create_policy(sample_policy)

        # Disable
        success = store.toggle_policy(sample_policy.id, False)
        assert success is True

        policy = store.get_policy(sample_policy.id)
        assert policy.enabled is False

        # Re-enable
        success = store.toggle_policy(sample_policy.id, True)
        assert success is True

        policy = store.get_policy(sample_policy.id)
        assert policy.enabled is True


# =============================================================================
# Violation CRUD Tests
# =============================================================================


class TestViolationCRUD:
    """Tests for violation CRUD operations."""

    def test_create_violation(self, store, sample_violation):
        """Test creating a violation."""
        created = store.create_violation(sample_violation)

        assert created.id == sample_violation.id
        assert created.rule_id == "gdpr_consent"
        assert created.status == "open"

    def test_get_violation(self, store, sample_violation):
        """Test retrieving a violation."""
        store.create_violation(sample_violation)

        retrieved = store.get_violation(sample_violation.id)

        assert retrieved is not None
        assert retrieved.id == sample_violation.id
        assert retrieved.severity == "high"
        assert retrieved.source == "forms/signup.tsx:42"

    def test_get_violation_not_found(self, store):
        """Test retrieving a non-existent violation."""
        result = store.get_violation("nonexistent")
        assert result is None

    def test_list_violations(self, store, sample_violation):
        """Test listing violations."""
        store.create_violation(sample_violation)

        another = Violation(
            id="viol_test456",
            policy_id="pol_test123",
            rule_id="gdpr_retention",
            rule_name="Data Retention",
            framework_id="gdpr",
            vertical_id="legal",
            workspace_id="ws_default",
            severity="medium",
            status="investigating",
            description="Data retained too long",
            source="db/cleanup.py:100",
        )
        store.create_violation(another)

        violations = store.list_violations()

        assert len(violations) == 2

    def test_list_violations_filter_by_status(self, store, sample_violation):
        """Test filtering violations by status."""
        store.create_violation(sample_violation)

        resolved = Violation(
            id="viol_resolved",
            policy_id="pol_test123",
            rule_id="gdpr_retention",
            rule_name="Data Retention",
            framework_id="gdpr",
            vertical_id="legal",
            workspace_id="ws_default",
            severity="low",
            status="resolved",
            description="Fixed",
            source="db/cleanup.py",
        )
        store.create_violation(resolved)

        violations = store.list_violations(status="open")

        assert len(violations) == 1
        assert violations[0].status == "open"

    def test_list_violations_filter_by_severity(self, store, sample_violation):
        """Test filtering violations by severity."""
        store.create_violation(sample_violation)

        low = Violation(
            id="viol_low",
            policy_id="pol_test123",
            rule_id="gdpr_minor",
            rule_name="Minor Issue",
            framework_id="gdpr",
            vertical_id="legal",
            workspace_id="ws_default",
            severity="low",
            status="open",
            description="Minor issue",
            source="app.py:10",
        )
        store.create_violation(low)

        violations = store.list_violations(severity="high")

        assert len(violations) == 1
        assert violations[0].severity == "high"

    def test_update_violation_status(self, store, sample_violation):
        """Test updating violation status."""
        store.create_violation(sample_violation)

        updated = store.update_violation_status(
            sample_violation.id,
            status="resolved",
            resolved_by="admin_user",
            resolution_notes="Fixed in PR #123",
        )

        assert updated is not None
        assert updated.status == "resolved"
        assert updated.resolved_by == "admin_user"
        assert updated.resolution_notes == "Fixed in PR #123"
        assert updated.resolved_at is not None

    def test_update_violation_status_investigating(self, store, sample_violation):
        """Test updating violation to investigating status."""
        store.create_violation(sample_violation)

        updated = store.update_violation_status(
            sample_violation.id,
            status="investigating",
        )

        assert updated is not None
        assert updated.status == "investigating"
        assert updated.resolved_at is None  # Not resolved yet

    def test_update_violation_status_false_positive(self, store, sample_violation):
        """Test marking violation as false positive."""
        store.create_violation(sample_violation)

        updated = store.update_violation_status(
            sample_violation.id,
            status="false_positive",
            resolution_notes="Not actually a violation",
        )

        assert updated is not None
        assert updated.status == "false_positive"
        assert updated.resolved_at is not None

    def test_delete_violation(self, store, sample_violation):
        """Test deleting a violation."""
        store.create_violation(sample_violation)

        success = store.delete_violation(sample_violation.id)
        assert success is True

        # Verify deleted
        retrieved = store.get_violation(sample_violation.id)
        assert retrieved is None

    def test_count_violations(self, store, sample_violation):
        """Test counting violations."""
        store.create_violation(sample_violation)

        # Add more violations
        critical = Violation(
            id="viol_critical",
            policy_id="",
            rule_id="sec_vuln",
            rule_name="Security Vulnerability",
            framework_id="owasp",
            vertical_id="software",
            workspace_id="ws_default",
            severity="critical",
            status="open",
            description="Critical security issue",
            source="api/auth.py:15",
        )
        store.create_violation(critical)

        counts = store.count_violations()

        assert counts["total"] == 2
        assert counts["high"] == 1
        assert counts["critical"] == 1

    def test_count_violations_by_workspace(self, store, sample_violation):
        """Test counting violations filtered by workspace."""
        store.create_violation(sample_violation)

        other_ws = Violation(
            id="viol_other",
            policy_id="",
            rule_id="test",
            rule_name="Test",
            framework_id="test",
            vertical_id="test",
            workspace_id="ws_other",
            severity="low",
            status="open",
            description="Other workspace",
            source="test.py",
        )
        store.create_violation(other_ws)

        counts = store.count_violations(workspace_id="ws_default")

        assert counts["total"] == 1


# =============================================================================
# Serialization Tests
# =============================================================================


class TestSerialization:
    """Tests for data serialization."""

    def test_policy_to_dict(self, sample_policy):
        """Test policy serialization to dict."""
        data = sample_policy.to_dict()

        assert data["id"] == "pol_test123"
        assert data["name"] == "Test GDPR Policy"
        assert data["framework_id"] == "gdpr"
        assert data["rules_count"] == 2
        assert len(data["rules"]) == 2
        assert data["rules"][0]["rule_id"] == "gdpr_consent"

    def test_policy_from_dict(self):
        """Test policy deserialization from dict."""
        data = {
            "id": "pol_test",
            "name": "Test Policy",
            "description": "Test",
            "framework_id": "owasp",
            "workspace_id": "default",
            "vertical_id": "software",
            "level": "recommended",
            "enabled": True,
            "rules": [
                {
                    "rule_id": "xss",
                    "name": "XSS Prevention",
                    "description": "Prevent XSS",
                    "severity": "high",
                }
            ],
            "created_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }

        policy = Policy.from_dict(data)

        assert policy.id == "pol_test"
        assert policy.framework_id == "owasp"
        assert len(policy.rules) == 1
        assert policy.rules[0].severity == "high"

    def test_violation_to_dict(self, sample_violation):
        """Test violation serialization to dict."""
        data = sample_violation.to_dict()

        assert data["id"] == "viol_test123"
        assert data["rule_id"] == "gdpr_consent"
        assert data["severity"] == "high"
        assert data["status"] == "open"
        assert data["resolved_at"] is None

    def test_violation_from_dict(self):
        """Test violation deserialization from dict."""
        data = {
            "id": "viol_test",
            "policy_id": "pol_test",
            "rule_id": "test_rule",
            "rule_name": "Test Rule",
            "framework_id": "test",
            "vertical_id": "test",
            "workspace_id": "default",
            "severity": "medium",
            "status": "investigating",
            "description": "Test violation",
            "source": "test.py:1",
            "detected_at": datetime.now(timezone.utc).isoformat(),
        }

        violation = Violation.from_dict(data)

        assert violation.id == "viol_test"
        assert violation.severity == "medium"
        assert violation.status == "investigating"


# =============================================================================
# Audit Log Tests
# =============================================================================


class TestAuditLog:
    """Tests for policy audit logging."""

    def test_audit_log_on_create(self, store, sample_policy):
        """Test that policy creation is logged."""
        store.create_policy(sample_policy)

        # Check audit log
        with store.connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM policy_audit WHERE policy_id = ?",
                (sample_policy.id,),
            )
            rows = cursor.fetchall()

        assert len(rows) == 1
        assert rows[0][2] == "create"  # action column

    def test_audit_log_on_update(self, store, sample_policy):
        """Test that policy update is logged."""
        store.create_policy(sample_policy)
        store.update_policy(sample_policy.id, {"name": "Updated"}, "admin")

        with store.connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM policy_audit WHERE policy_id = ? ORDER BY changed_at",
                (sample_policy.id,),
            )
            rows = cursor.fetchall()

        assert len(rows) == 2
        assert rows[1][2] == "update"

    def test_audit_log_on_delete(self, store, sample_policy):
        """Test that policy deletion is logged."""
        store.create_policy(sample_policy)
        store.delete_policy(sample_policy.id, "admin")

        with store.connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM policy_audit WHERE policy_id = ?",
                (sample_policy.id,),
            )
            rows = cursor.fetchall()

        # Should have both create and delete
        assert len(rows) == 2
        actions = [row[2] for row in rows]
        assert "create" in actions
        assert "delete" in actions
