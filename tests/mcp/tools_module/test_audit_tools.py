"""Tests for MCP audit tools execution logic."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.mcp.tools_module.audit import (
    create_audit_session_tool,
    get_audit_findings_tool,
    get_audit_preset_tool,
    get_audit_status_tool,
    list_audit_presets_tool,
    list_audit_types_tool,
    run_audit_tool,
    run_quick_audit_tool,
    update_finding_status_tool,
)


class TestListAuditPresetsTool:
    """Tests for list_audit_presets_tool."""

    @pytest.mark.asyncio
    async def test_list_presets_success(self):
        """Test successful preset listing."""
        mock_registry = MagicMock()
        mock_preset = MagicMock()
        mock_preset.name = "Legal Due Diligence"
        mock_preset.description = "Legal document review"
        mock_preset.audit_types = ["compliance", "consistency"]
        mock_preset.custom_rules = [{"pattern": ".*"}]
        mock_preset.consensus_threshold = 0.7
        mock_registry.list_presets.return_value = [mock_preset]

        with patch(
            "aragora.mcp.tools_module.audit.audit_registry",
            mock_registry,
        ):
            result = await list_audit_presets_tool()

        assert result["success"] is True
        assert result["count"] == 1
        assert result["presets"][0]["name"] == "Legal Due Diligence"

    @pytest.mark.asyncio
    async def test_list_presets_error(self):
        """Test preset listing when registry fails."""
        with patch(
            "aragora.mcp.tools_module.audit.audit_registry",
            side_effect=Exception("Registry error"),
        ):
            result = await list_audit_presets_tool()

        assert result["success"] is False
        assert "error" in result


class TestListAuditTypesTool:
    """Tests for list_audit_types_tool."""

    @pytest.mark.asyncio
    async def test_list_types_success(self):
        """Test successful audit type listing."""
        mock_registry = MagicMock()
        mock_type = MagicMock()
        mock_type.id = "security"
        mock_type.display_name = "Security Audit"
        mock_type.description = "Finds security vulnerabilities"
        mock_type.version = "1.0"
        mock_type.capabilities = ["injection", "xss"]
        mock_registry.list_audit_types.return_value = [mock_type]

        with patch(
            "aragora.mcp.tools_module.audit.audit_registry",
            mock_registry,
        ):
            result = await list_audit_types_tool()

        assert result["success"] is True
        assert result["count"] == 1
        assert result["audit_types"][0]["id"] == "security"


class TestGetAuditPresetTool:
    """Tests for get_audit_preset_tool."""

    @pytest.mark.asyncio
    async def test_get_preset_success(self):
        """Test successful preset retrieval."""
        mock_registry = MagicMock()
        mock_preset = MagicMock()
        mock_preset.name = "Code Security"
        mock_preset.description = "Code vulnerability scanning"
        mock_preset.audit_types = ["security"]
        mock_preset.custom_rules = [
            {"pattern": "password", "severity": "high", "category": "secrets", "title": "Hardcoded password"}
        ]
        mock_preset.consensus_threshold = 0.8
        mock_preset.agents = ["claude", "gpt4"]
        mock_preset.parameters = {}
        mock_registry.get_preset.return_value = mock_preset

        with patch(
            "aragora.mcp.tools_module.audit.audit_registry",
            mock_registry,
        ):
            result = await get_audit_preset_tool(preset_name="Code Security")

        assert result["success"] is True
        assert result["preset"]["name"] == "Code Security"

    @pytest.mark.asyncio
    async def test_get_preset_not_found(self):
        """Test get for non-existent preset."""
        mock_registry = MagicMock()
        mock_registry.get_preset.return_value = None

        with patch(
            "aragora.mcp.tools_module.audit.audit_registry",
            mock_registry,
        ):
            result = await get_audit_preset_tool(preset_name="Nonexistent")

        assert result["success"] is False
        assert "not found" in result["error"].lower()


class TestCreateAuditSessionTool:
    """Tests for create_audit_session_tool."""

    @pytest.mark.asyncio
    async def test_create_empty_document_ids(self):
        """Test create with empty document IDs."""
        result = await create_audit_session_tool(document_ids="")
        assert result["success"] is False
        assert "No document IDs" in result["error"]

    @pytest.mark.asyncio
    async def test_create_success(self):
        """Test successful session creation."""
        mock_session = MagicMock()
        mock_session.id = "session-001"
        mock_session.name = "Audit-security,compliance"
        mock_session.status = MagicMock(value="created")

        mock_auditor = AsyncMock()
        mock_auditor.create_session.return_value = mock_session

        with patch(
            "aragora.mcp.tools_module.audit.DocumentAuditor",
            return_value=mock_auditor,
        ), patch("aragora.mcp.tools_module.audit.AuditConfig"):
            result = await create_audit_session_tool(
                document_ids="doc1,doc2",
                audit_types="security,compliance",
            )

        assert result["success"] is True
        assert result["session"]["id"] == "session-001"
        assert result["session"]["document_count"] == 2

    @pytest.mark.asyncio
    async def test_create_with_preset(self):
        """Test session creation with preset."""
        mock_registry = MagicMock()
        mock_preset_config = MagicMock()
        mock_preset_config.audit_types = ["security", "quality"]
        mock_registry.get_preset.return_value = mock_preset_config

        mock_session = MagicMock()
        mock_session.id = "session-002"
        mock_session.name = "Audit"
        mock_session.status = MagicMock(value="created")

        mock_auditor = AsyncMock()
        mock_auditor.create_session.return_value = mock_session

        with patch(
            "aragora.mcp.tools_module.audit.DocumentAuditor",
            return_value=mock_auditor,
        ), patch(
            "aragora.mcp.tools_module.audit.AuditConfig",
        ), patch(
            "aragora.mcp.tools_module.audit.audit_registry",
            mock_registry,
        ):
            result = await create_audit_session_tool(
                document_ids="doc1",
                preset="Code Security",
            )

        assert result["success"] is True


class TestRunAuditTool:
    """Tests for run_audit_tool."""

    @pytest.mark.asyncio
    async def test_run_success(self):
        """Test successful audit run."""
        mock_result = MagicMock()
        mock_result.status = MagicMock(value="completed")
        mock_result.findings = [MagicMock(), MagicMock()]

        mock_auditor = AsyncMock()
        mock_auditor.run_audit.return_value = mock_result

        with patch(
            "aragora.mcp.tools_module.audit.DocumentAuditor",
            return_value=mock_auditor,
        ):
            result = await run_audit_tool(session_id="session-001")

        assert result["success"] is True
        assert result["findings_count"] == 2

    @pytest.mark.asyncio
    async def test_run_error(self):
        """Test audit run with error."""
        mock_auditor = AsyncMock()
        mock_auditor.run_audit.side_effect = Exception("Audit failed")

        with patch(
            "aragora.mcp.tools_module.audit.DocumentAuditor",
            return_value=mock_auditor,
        ):
            result = await run_audit_tool(session_id="session-001")

        assert result["success"] is False


class TestGetAuditStatusTool:
    """Tests for get_audit_status_tool."""

    @pytest.mark.asyncio
    async def test_get_status_success(self):
        """Test successful status retrieval."""
        mock_session = MagicMock()
        mock_session.id = "session-001"
        mock_session.name = "Test Audit"
        mock_session.status = MagicMock(value="running")
        mock_session.progress = 0.5
        mock_session.current_phase = "analysis"
        mock_session.total_chunks = 10
        mock_session.processed_chunks = 5
        mock_session.findings = [MagicMock()]

        mock_auditor = MagicMock()
        mock_auditor.get_session.return_value = mock_session

        with patch(
            "aragora.mcp.tools_module.audit.DocumentAuditor",
            return_value=mock_auditor,
        ):
            result = await get_audit_status_tool(session_id="session-001")

        assert result["success"] is True
        assert result["session"]["progress"] == 0.5

    @pytest.mark.asyncio
    async def test_get_status_not_found(self):
        """Test status for non-existent session."""
        mock_auditor = MagicMock()
        mock_auditor.get_session.return_value = None

        with patch(
            "aragora.mcp.tools_module.audit.DocumentAuditor",
            return_value=mock_auditor,
        ):
            result = await get_audit_status_tool(session_id="nonexistent")

        assert result["success"] is False
        assert "not found" in result["error"].lower()


class TestGetAuditFindingsTool:
    """Tests for get_audit_findings_tool."""

    @pytest.mark.asyncio
    async def test_get_findings_success(self):
        """Test successful findings retrieval."""
        mock_finding = MagicMock()
        mock_finding.id = "f-001"
        mock_finding.title = "SQL Injection"
        mock_finding.description = "Potential SQL injection in query"
        mock_finding.severity = MagicMock(value="critical")
        mock_finding.status = MagicMock(value="open")
        mock_finding.audit_type = MagicMock(value="security")
        mock_finding.category = "injection"
        mock_finding.confidence = 0.9
        mock_finding.evidence_text = "SELECT * FROM users WHERE id = " + "'" * 100
        mock_finding.recommendation = "Use parameterized queries"
        mock_finding.document_id = "doc-001"

        mock_auditor = MagicMock()
        mock_auditor.get_findings.return_value = [mock_finding]

        with patch(
            "aragora.mcp.tools_module.audit.DocumentAuditor",
            return_value=mock_auditor,
        ):
            result = await get_audit_findings_tool(session_id="session-001")

        assert result["success"] is True
        assert result["count"] == 1
        assert result["findings"][0]["title"] == "SQL Injection"
        assert result["findings"][0]["severity"] == "critical"

    @pytest.mark.asyncio
    async def test_get_findings_with_severity_filter(self):
        """Test findings with severity filter."""
        mock_finding_critical = MagicMock()
        mock_finding_critical.severity = MagicMock(value="critical")
        mock_finding_critical.status = MagicMock(value="open")

        mock_finding_low = MagicMock()
        mock_finding_low.severity = MagicMock(value="low")
        mock_finding_low.status = MagicMock(value="open")

        mock_auditor = MagicMock()
        mock_auditor.get_findings.return_value = [mock_finding_critical, mock_finding_low]

        with patch(
            "aragora.mcp.tools_module.audit.DocumentAuditor",
            return_value=mock_auditor,
        ):
            result = await get_audit_findings_tool(
                session_id="session-001", severity="critical"
            )

        assert result["count"] == 1


class TestUpdateFindingStatusTool:
    """Tests for update_finding_status_tool."""

    @pytest.mark.asyncio
    async def test_update_success(self):
        """Test successful finding status update."""
        mock_event = MagicMock()
        mock_event.from_state = MagicMock(value="open")

        mock_workflow = MagicMock()
        mock_workflow.can_transition_to.return_value = True
        mock_workflow.transition_to.return_value = mock_event
        mock_workflow.state = MagicMock(value="triaging")

        with patch(
            "aragora.mcp.tools_module.audit.FindingWorkflow",
            return_value=mock_workflow,
        ), patch(
            "aragora.mcp.tools_module.audit.FindingWorkflowData",
        ), patch(
            "aragora.mcp.tools_module.audit.WorkflowState",
        ):
            result = await update_finding_status_tool(
                finding_id="f-001", status="triaging", comment="Starting triage"
            )

        assert result["success"] is True
        assert result["current_state"] == "triaging"

    @pytest.mark.asyncio
    async def test_update_invalid_transition(self):
        """Test update with invalid state transition."""
        mock_workflow = MagicMock()
        mock_workflow.can_transition_to.return_value = False
        mock_workflow.state = MagicMock(value="open")
        mock_workflow.get_valid_transitions.return_value = [MagicMock(value="triaging")]

        with patch(
            "aragora.mcp.tools_module.audit.FindingWorkflow",
            return_value=mock_workflow,
        ), patch(
            "aragora.mcp.tools_module.audit.FindingWorkflowData",
        ), patch(
            "aragora.mcp.tools_module.audit.WorkflowState",
        ):
            result = await update_finding_status_tool(
                finding_id="f-001", status="resolved"
            )

        assert result["success"] is False
        assert "Cannot transition" in result["error"]


class TestRunQuickAuditTool:
    """Tests for run_quick_audit_tool."""

    @pytest.mark.asyncio
    async def test_quick_audit_empty_docs(self):
        """Test quick audit with empty document IDs."""
        result = await run_quick_audit_tool(document_ids="")
        assert result["success"] is False

    @pytest.mark.asyncio
    async def test_quick_audit_success(self):
        """Test successful quick audit."""
        mock_session = MagicMock()
        mock_session.id = "session-001"
        mock_session.name = "Quick Audit"
        mock_session.status = MagicMock(value="created")

        mock_result = MagicMock()
        mock_result.status = MagicMock(value="completed")
        mock_result.findings = []

        mock_auditor_instance = AsyncMock()
        mock_auditor_instance.create_session.return_value = mock_session
        mock_auditor_instance.run_audit.return_value = mock_result

        mock_auditor_get_instance = MagicMock()
        mock_auditor_get_instance.get_findings.return_value = []

        # The function calls create_audit_session_tool, run_audit_tool, get_audit_findings_tool
        # which each create their own DocumentAuditor instance. Mock at the tool level.
        with patch(
            "aragora.mcp.tools_module.audit.create_audit_session_tool",
            return_value={
                "success": True,
                "session": {"id": "session-001", "name": "Quick", "document_count": 1, "audit_types": ["security"], "status": "created"},
            },
        ), patch(
            "aragora.mcp.tools_module.audit.run_audit_tool",
            return_value={"success": True, "session_id": "session-001", "status": "completed", "findings_count": 0},
        ), patch(
            "aragora.mcp.tools_module.audit.get_audit_findings_tool",
            return_value={"success": True, "findings": [], "count": 0},
        ):
            result = await run_quick_audit_tool(document_ids="doc1")

        assert result["success"] is True
        assert result["total_findings"] == 0
