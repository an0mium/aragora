"""
Tests for AuditAPI resource.

Tests cover:
- Preset management (list, get)
- Audit type registry (list)
- Finding workflow (get, update status, comment, assign, priority)
- Bulk finding operations
- Quick audit
- Finding search
- Model parsing
"""

from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock

import pytest

from aragora.client import AragoraClient
from aragora.client.models import (
    AuditFinding,
    AuditPreset,
    AuditPresetDetail,
    AuditType,
    AuditTypeCapabilities,
    AuditTypeInfo,
    FindingSeverity,
    FindingWorkflowData,
    FindingWorkflowEvent,
    FindingWorkflowStatus,
    QuickAuditResult,
)
from aragora.client.resources.audit import AuditAPI


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def client():
    """Create a basic client for testing."""
    return AragoraClient(base_url="http://test.example.com", api_key="test-key")


@pytest.fixture
def audit_api(client):
    """Create an AuditAPI instance for testing."""
    return client.audit


# ============================================================================
# Model Tests
# ============================================================================


class TestAuditPresetModel:
    """Tests for AuditPreset model."""

    def test_audit_preset_required_fields(self):
        """Test AuditPreset with required fields."""
        preset = AuditPreset(
            name="Legal Due Diligence",
            description="Audit preset for legal documents",
        )
        assert preset.name == "Legal Due Diligence"
        assert preset.audit_types == []
        assert preset.consensus_threshold == 0.8

    def test_audit_preset_all_fields(self):
        """Test AuditPreset with all fields."""
        preset = AuditPreset(
            name="Code Security",
            description="Security audit for code",
            audit_types=["security", "compliance"],
            custom_rules_count=5,
            consensus_threshold=0.9,
            agents=["claude", "gpt4"],
            parameters={"strict_mode": True},
        )
        assert preset.custom_rules_count == 5
        assert preset.agents == ["claude", "gpt4"]


class TestAuditPresetDetailModel:
    """Tests for AuditPresetDetail model."""

    def test_audit_preset_detail(self):
        """Test AuditPresetDetail with custom rules."""
        detail = AuditPresetDetail(
            name="Legal Due Diligence",
            description="Full legal audit",
            audit_types=["compliance", "consistency"],
            custom_rules=[
                {"name": "check_signatures", "pattern": ".*signature.*"},
            ],
            consensus_threshold=0.85,
        )
        assert len(detail.custom_rules) == 1
        assert detail.custom_rules[0]["name"] == "check_signatures"


class TestAuditTypeInfoModel:
    """Tests for AuditTypeInfo model."""

    def test_audit_type_info(self):
        """Test AuditTypeInfo model."""
        info = AuditTypeInfo(
            id="security",
            display_name="Security Audit",
            description="Detects security vulnerabilities",
            version="2.0.0",
            capabilities=AuditTypeCapabilities(
                supports_chunk_analysis=True,
                supports_cross_document=True,
                requires_llm=True,
            ),
        )
        assert info.id == "security"
        assert info.capabilities.supports_cross_document is True


class TestFindingWorkflowDataModel:
    """Tests for FindingWorkflowData model."""

    def test_workflow_data_defaults(self):
        """Test FindingWorkflowData with defaults."""
        data = FindingWorkflowData(finding_id="finding123")
        assert data.current_state == FindingWorkflowStatus.OPEN
        assert data.priority == 3
        assert data.assigned_to is None

    def test_workflow_data_with_history(self):
        """Test FindingWorkflowData with history."""
        now = datetime.now()
        data = FindingWorkflowData(
            finding_id="finding123",
            current_state=FindingWorkflowStatus.INVESTIGATING,
            assigned_to="user456",
            priority=2,
            history=[
                FindingWorkflowEvent(
                    id="evt1",
                    event_type="state_change",
                    timestamp=now,
                    user_id="user123",
                    from_state="open",
                    to_state="investigating",
                ),
            ],
        )
        assert data.current_state == FindingWorkflowStatus.INVESTIGATING
        assert len(data.history) == 1


class TestQuickAuditResultModel:
    """Tests for QuickAuditResult model."""

    def test_quick_audit_result(self):
        """Test QuickAuditResult model."""
        result = QuickAuditResult(
            session_id="sess123",
            preset_used="Code Security",
            document_count=10,
            total_findings=25,
            findings_by_severity={"critical": 2, "high": 5, "medium": 10, "low": 8},
        )
        assert result.total_findings == 25
        assert result.findings_by_severity["critical"] == 2


# ============================================================================
# Preset Management Tests
# ============================================================================


class TestListPresets:
    """Tests for list_presets method."""

    def test_list_presets(self, audit_api, client):
        """Test listing audit presets."""
        client._get = MagicMock(
            return_value={
                "presets": [
                    {
                        "name": "Legal Due Diligence",
                        "description": "Audit for legal documents",
                        "audit_types": ["compliance"],
                    },
                    {
                        "name": "Code Security",
                        "description": "Security audit",
                        "audit_types": ["security"],
                    },
                ],
            }
        )

        result = audit_api.list_presets()

        client._get.assert_called_once_with("/api/audit/presets")
        assert len(result) == 2
        assert all(isinstance(p, AuditPreset) for p in result)
        assert result[0].name == "Legal Due Diligence"

    def test_list_presets_direct_array(self, audit_api, client):
        """Test listing presets with direct array response."""
        client._get = MagicMock(
            return_value=[
                {"name": "Preset 1", "description": "Desc 1"},
            ]
        )

        result = audit_api.list_presets()

        assert len(result) == 1


class TestGetPreset:
    """Tests for get_preset method."""

    def test_get_preset(self, audit_api, client):
        """Test getting preset details."""
        client._get = MagicMock(
            return_value={
                "preset": {
                    "name": "Code Security",
                    "description": "Full security audit",
                    "audit_types": ["security", "compliance"],
                    "custom_rules": [{"name": "check_secrets"}],
                },
            }
        )

        result = audit_api.get_preset("Code Security")

        client._get.assert_called_once_with("/api/audit/presets/Code Security")
        assert isinstance(result, AuditPresetDetail)
        assert len(result.custom_rules) == 1

    def test_get_preset_direct_response(self, audit_api, client):
        """Test getting preset with direct response."""
        client._get = MagicMock(
            return_value={
                "name": "Code Security",
                "description": "Full security audit",
            }
        )

        result = audit_api.get_preset("Code Security")

        assert result.name == "Code Security"


# ============================================================================
# Audit Type Registry Tests
# ============================================================================


class TestListAuditTypes:
    """Tests for list_audit_types method."""

    def test_list_audit_types(self, audit_api, client):
        """Test listing audit types."""
        client._get = MagicMock(
            return_value={
                "audit_types": [
                    {
                        "id": "security",
                        "display_name": "Security",
                        "description": "Security vulnerabilities",
                    },
                    {
                        "id": "compliance",
                        "display_name": "Compliance",
                        "description": "Regulatory compliance",
                    },
                ],
            }
        )

        result = audit_api.list_audit_types()

        client._get.assert_called_once_with("/api/audit/types")
        assert len(result) == 2
        assert all(isinstance(t, AuditTypeInfo) for t in result)
        assert result[0].id == "security"


# ============================================================================
# Finding Workflow Tests
# ============================================================================


class TestGetFinding:
    """Tests for get_finding method."""

    def test_get_finding(self, audit_api, client):
        """Test getting a finding."""
        now = datetime.now()
        client._get = MagicMock(
            return_value={
                "id": "finding123",
                "session_id": "sess456",
                "document_id": "doc789",
                "audit_type": "security",
                "category": "credentials",
                "severity": "high",
                "title": "Exposed API Key",
                "description": "API key found in source code",
                "created_at": now.isoformat(),
            }
        )

        result = audit_api.get_finding("finding123")

        client._get.assert_called_once_with("/api/audit/findings/finding123")
        assert isinstance(result, AuditFinding)
        assert result.id == "finding123"
        assert result.severity == FindingSeverity.HIGH


class TestGetFindingWorkflow:
    """Tests for get_finding_workflow method."""

    def test_get_finding_workflow(self, audit_api, client):
        """Test getting finding workflow."""
        now = datetime.now()
        client._get = MagicMock(
            return_value={
                "finding_id": "finding123",
                "current_state": "investigating",
                "assigned_to": "user456",
                "priority": 2,
                "history": [
                    {
                        "id": "evt1",
                        "event_type": "state_change",
                        "timestamp": now.isoformat(),
                        "user_id": "user123",
                    },
                ],
            }
        )

        result = audit_api.get_finding_workflow("finding123")

        client._get.assert_called_once_with("/api/audit/findings/finding123/history")
        assert isinstance(result, FindingWorkflowData)
        assert result.current_state == FindingWorkflowStatus.INVESTIGATING


class TestUpdateFindingStatus:
    """Tests for update_finding_status method."""

    def test_update_status_with_string(self, audit_api, client):
        """Test updating status with string."""
        client._patch = MagicMock(
            return_value={
                "finding_id": "finding123",
                "current_state": "investigating",
            }
        )

        result = audit_api.update_finding_status("finding123", "investigating")

        client._patch.assert_called_once()
        call_args = client._patch.call_args
        assert call_args[0][0] == "/api/audit/findings/finding123/status"
        assert call_args[0][1]["status"] == "investigating"

    def test_update_status_with_enum(self, audit_api, client):
        """Test updating status with enum."""
        client._patch = MagicMock(
            return_value={
                "finding_id": "finding123",
                "current_state": "investigating",
            }
        )

        audit_api.update_finding_status(
            "finding123",
            FindingWorkflowStatus.INVESTIGATING,
        )

        call_args = client._patch.call_args
        assert call_args[0][1]["status"] == "investigating"

    def test_update_status_with_comment(self, audit_api, client):
        """Test updating status with comment."""
        client._patch = MagicMock(
            return_value={
                "finding_id": "finding123",
                "current_state": "resolved",
            }
        )

        audit_api.update_finding_status(
            "finding123",
            "resolved",
            comment="Fixed in PR #123",
        )

        call_args = client._patch.call_args
        assert call_args[0][1]["comment"] == "Fixed in PR #123"

    def test_update_status_with_user_id(self, audit_api, client):
        """Test updating status with user attribution."""
        client._patch = MagicMock(
            return_value={
                "finding_id": "finding123",
                "current_state": "triaging",
            }
        )

        audit_api.update_finding_status(
            "finding123",
            "triaging",
            user_id="user456",
        )

        call_args = client._patch.call_args
        assert call_args[1]["headers"]["X-User-ID"] == "user456"


class TestAddFindingComment:
    """Tests for add_finding_comment method."""

    def test_add_comment(self, audit_api, client):
        """Test adding a comment."""
        client._post = MagicMock(
            return_value={
                "finding_id": "finding123",
                "current_state": "open",
            }
        )

        result = audit_api.add_finding_comment(
            "finding123",
            "Need to verify with security team",
        )

        client._post.assert_called_once()
        call_args = client._post.call_args
        assert call_args[0][0] == "/api/audit/findings/finding123/comments"
        assert call_args[0][1]["comment"] == "Need to verify with security team"

    def test_add_comment_with_user_id(self, audit_api, client):
        """Test adding comment with user attribution."""
        client._post = MagicMock(
            return_value={
                "finding_id": "finding123",
            }
        )

        audit_api.add_finding_comment(
            "finding123",
            "Comment",
            user_id="user456",
        )

        call_args = client._post.call_args
        assert call_args[1]["headers"]["X-User-ID"] == "user456"


class TestAssignFinding:
    """Tests for assign_finding method."""

    def test_assign_finding(self, audit_api, client):
        """Test assigning a finding."""
        client._patch = MagicMock(
            return_value={
                "finding_id": "finding123",
                "assigned_to": "user456",
            }
        )

        result = audit_api.assign_finding("finding123", "user456")

        client._patch.assert_called_once()
        call_args = client._patch.call_args
        assert call_args[0][0] == "/api/audit/findings/finding123/assign"
        assert call_args[0][1]["user_id"] == "user456"


class TestSetFindingPriority:
    """Tests for set_finding_priority method."""

    def test_set_priority(self, audit_api, client):
        """Test setting finding priority."""
        client._patch = MagicMock(
            return_value={
                "finding_id": "finding123",
                "priority": 1,
            }
        )

        result = audit_api.set_finding_priority("finding123", 1)

        client._patch.assert_called_once()
        call_args = client._patch.call_args
        assert call_args[0][0] == "/api/audit/findings/finding123/priority"
        assert call_args[0][1]["priority"] == 1

    def test_set_priority_invalid_low(self, audit_api, client):
        """Test setting priority with value too low."""
        with pytest.raises(ValueError, match="Priority must be between 1"):
            audit_api.set_finding_priority("finding123", 0)

    def test_set_priority_invalid_high(self, audit_api, client):
        """Test setting priority with value too high."""
        with pytest.raises(ValueError, match="Priority must be between 1"):
            audit_api.set_finding_priority("finding123", 6)


class TestBulkUpdateFindings:
    """Tests for bulk_update_findings method."""

    def test_bulk_update_status(self, audit_api, client):
        """Test bulk status update."""
        client._post = MagicMock(
            return_value={
                "success_count": 3,
                "failure_count": 0,
            }
        )

        result = audit_api.bulk_update_findings(
            ["f1", "f2", "f3"],
            action="status",
            value="investigating",
        )

        client._post.assert_called_once()
        call_args = client._post.call_args
        assert call_args[0][0] == "/api/audit/findings/bulk-action"
        assert call_args[0][1]["finding_ids"] == ["f1", "f2", "f3"]
        assert call_args[0][1]["action"] == "status"
        assert result["success_count"] == 3

    def test_bulk_update_with_user_id(self, audit_api, client):
        """Test bulk update with user attribution."""
        client._post = MagicMock(return_value={"success_count": 2})

        audit_api.bulk_update_findings(
            ["f1", "f2"],
            action="assign",
            value="user123",
            user_id="admin456",
        )

        call_args = client._post.call_args
        assert call_args[1]["headers"]["X-User-ID"] == "admin456"


# ============================================================================
# Quick Audit Tests
# ============================================================================


class TestRunQuickAudit:
    """Tests for run_quick_audit method."""

    def test_run_quick_audit(self, audit_api, client):
        """Test running quick audit."""
        client._post = MagicMock(
            return_value={
                "session_id": "sess123",
                "preset_used": "Code Security",
                "document_count": 5,
                "total_findings": 15,
                "findings_by_severity": {"high": 3, "medium": 7, "low": 5},
            }
        )

        result = audit_api.run_quick_audit(
            ["doc1", "doc2", "doc3", "doc4", "doc5"],
        )

        client._post.assert_called_once()
        call_args = client._post.call_args
        assert call_args[0][0] == "/api/audit/quick"
        assert len(call_args[0][1]["document_ids"]) == 5
        assert call_args[0][1]["preset"] == "Code Security"
        assert isinstance(result, QuickAuditResult)
        assert result.total_findings == 15

    def test_run_quick_audit_custom_preset(self, audit_api, client):
        """Test running quick audit with custom preset."""
        client._post = MagicMock(
            return_value={
                "session_id": "sess123",
                "preset_used": "Legal Due Diligence",
                "document_count": 2,
                "total_findings": 10,
            }
        )

        audit_api.run_quick_audit(
            ["doc1", "doc2"],
            preset="Legal Due Diligence",
        )

        call_args = client._post.call_args
        assert call_args[0][1]["preset"] == "Legal Due Diligence"


# ============================================================================
# Finding Search Tests
# ============================================================================


class TestSearchFindings:
    """Tests for search_findings method."""

    def test_search_findings_basic(self, audit_api, client):
        """Test basic finding search."""
        client._get = MagicMock(
            return_value={
                "findings": [
                    {
                        "id": "f1",
                        "session_id": "s1",
                        "audit_type": "security",
                        "category": "credentials",
                        "severity": "high",
                        "title": "API Key Exposed",
                        "description": "Found in code",
                    },
                ],
            }
        )

        result = audit_api.search_findings()

        client._get.assert_called_once()
        call_args = client._get.call_args
        assert call_args[0][0] == "/api/audit/findings"
        assert call_args[1]["params"]["limit"] == 50
        assert call_args[1]["params"]["offset"] == 0
        assert len(result) == 1
        assert isinstance(result[0], AuditFinding)

    def test_search_findings_with_filters(self, audit_api, client):
        """Test finding search with filters."""
        client._get = MagicMock(return_value={"findings": []})

        audit_api.search_findings(
            query="API key",
            severity="critical",
            status="open",
            audit_type="security",
            assigned_to="user123",
            limit=100,
            offset=50,
        )

        call_args = client._get.call_args
        params = call_args[1]["params"]
        assert params["query"] == "API key"
        assert params["severity"] == "critical"
        assert params["status"] == "open"
        assert params["audit_type"] == "security"
        assert params["assigned_to"] == "user123"
        assert params["limit"] == 100
        assert params["offset"] == 50

    def test_search_findings_direct_array(self, audit_api, client):
        """Test finding search with direct array response."""
        client._get = MagicMock(
            return_value=[
                {
                    "id": "f1",
                    "session_id": "s1",
                    "audit_type": "security",
                    "category": "test",
                    "severity": "low",
                    "title": "Test",
                    "description": "Test finding",
                },
            ]
        )

        result = audit_api.search_findings()

        assert len(result) == 1


# ============================================================================
# Client Integration Tests
# ============================================================================


class TestAuditAPIIntegration:
    """Tests for AuditAPI integration with AragoraClient."""

    def test_audit_accessible_from_client(self):
        """Test audit API is accessible from client."""
        client = AragoraClient(base_url="http://test.example.com")
        assert hasattr(client, "audit")
        assert isinstance(client.audit, AuditAPI)

    def test_audit_shares_client(self):
        """Test audit API shares the same client."""
        client = AragoraClient(base_url="http://test.example.com")
        assert client.audit._client is client
