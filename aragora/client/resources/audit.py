"""AuditAPI resource for the Aragora client.

Provides SDK methods for enterprise audit features:
- Preset management (industry-specific audit configurations)
- Audit type registry (registered auditor metadata)
- Finding workflow (triage, assignment, remediation tracking)
- Quick audit (convenience method for running audits with presets)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, List, Optional

from ..models import (
    AuditFinding,
    AuditPreset,
    AuditPresetDetail,
    AuditTypeInfo,
    FindingWorkflowData,
    FindingWorkflowStatus,
    QuickAuditResult,
)

if TYPE_CHECKING:
    from ..client import AragoraClient


class AuditAPI:
    """API interface for enterprise audit features.

    This class provides access to audit presets, audit type registry,
    finding workflow management, and quick audit functionality.

    For basic audit session management (create, start, pause, resume),
    use the DocumentsAPI class instead.
    """

    def __init__(self, client: "AragoraClient"):
        self._client = client

    # =========================================================================
    # Preset Management
    # =========================================================================

    def list_presets(self) -> List[AuditPreset]:
        """
        List available audit presets.

        Presets are pre-configured audit templates for specific industries:
        - Legal Due Diligence (contracts, compliance, jurisdiction)
        - Financial Audit (accounting irregularities, SOX compliance)
        - Code Security (vulnerabilities, secrets, licenses)

        Returns:
            List of AuditPreset objects.
        """
        response = self._client._get("/api/audit/presets")
        presets = response.get("presets", []) if isinstance(response, dict) else response
        return [AuditPreset(**p) for p in presets]

    async def list_presets_async(self) -> List[AuditPreset]:
        """Async version of list_presets()."""
        response = await self._client._get_async("/api/audit/presets")
        presets = response.get("presets", []) if isinstance(response, dict) else response
        return [AuditPreset(**p) for p in presets]

    def get_preset(self, preset_name: str) -> AuditPresetDetail:
        """
        Get detailed information about a specific preset.

        Args:
            preset_name: Name of the preset (e.g., "Legal Due Diligence").

        Returns:
            AuditPresetDetail with full configuration including custom rules.
        """
        response = self._client._get(f"/api/audit/presets/{preset_name}")
        preset_data = response.get("preset", response) if isinstance(response, dict) else response
        return AuditPresetDetail(**preset_data)

    async def get_preset_async(self, preset_name: str) -> AuditPresetDetail:
        """Async version of get_preset()."""
        response = await self._client._get_async(f"/api/audit/presets/{preset_name}")
        preset_data = response.get("preset", response) if isinstance(response, dict) else response
        return AuditPresetDetail(**preset_data)

    # =========================================================================
    # Audit Type Registry
    # =========================================================================

    def list_audit_types(self) -> List[AuditTypeInfo]:
        """
        List registered audit types with their capabilities.

        Returns information about available auditors:
        - security: Credentials, injection vulnerabilities, data exposure
        - compliance: GDPR, HIPAA, SOC2, contractual violations
        - consistency: Cross-document contradictions
        - quality: Ambiguity, completeness, documentation quality

        Returns:
            List of AuditTypeInfo objects.
        """
        response = self._client._get("/api/audit/types")
        types = response.get("audit_types", []) if isinstance(response, dict) else response
        return [AuditTypeInfo(**t) for t in types]

    async def list_audit_types_async(self) -> List[AuditTypeInfo]:
        """Async version of list_audit_types()."""
        response = await self._client._get_async("/api/audit/types")
        types = response.get("audit_types", []) if isinstance(response, dict) else response
        return [AuditTypeInfo(**t) for t in types]

    # =========================================================================
    # Finding Workflow
    # =========================================================================

    def get_finding(self, finding_id: str) -> AuditFinding:
        """
        Get a specific finding by ID.

        Args:
            finding_id: The finding ID.

        Returns:
            AuditFinding with full details.
        """
        response = self._client._get(f"/api/audit/findings/{finding_id}")
        return AuditFinding(**response)

    async def get_finding_async(self, finding_id: str) -> AuditFinding:
        """Async version of get_finding()."""
        response = await self._client._get_async(f"/api/audit/findings/{finding_id}")
        return AuditFinding(**response)

    def get_finding_workflow(self, finding_id: str) -> FindingWorkflowData:
        """
        Get workflow data and history for a finding.

        Args:
            finding_id: The finding ID.

        Returns:
            FindingWorkflowData with current state, assignment, and history.
        """
        response = self._client._get(f"/api/audit/findings/{finding_id}/history")
        return FindingWorkflowData(**response)

    async def get_finding_workflow_async(self, finding_id: str) -> FindingWorkflowData:
        """Async version of get_finding_workflow()."""
        response = await self._client._get_async(f"/api/audit/findings/{finding_id}/history")
        return FindingWorkflowData(**response)

    def update_finding_status(
        self,
        finding_id: str,
        status: FindingWorkflowStatus | str,
        comment: str = "",
        user_id: Optional[str] = None,
    ) -> FindingWorkflowData:
        """
        Update the workflow status of a finding.

        Valid transitions depend on current state:
        - open -> triaging, investigating, false_positive, duplicate
        - triaging -> investigating, false_positive, accepted_risk, duplicate, open
        - investigating -> remediating, false_positive, accepted_risk, triaging
        - remediating -> resolved, investigating, accepted_risk
        - resolved -> open (reopen)
        - false_positive, accepted_risk, duplicate -> open (reopen)

        Args:
            finding_id: The finding ID.
            status: New workflow status.
            comment: Optional comment explaining the change.
            user_id: Optional user ID for attribution.

        Returns:
            Updated FindingWorkflowData.
        """
        if isinstance(status, FindingWorkflowStatus):
            status = status.value

        headers = {}
        if user_id:
            headers["X-User-ID"] = user_id

        data = {"status": status}
        if comment:
            data["comment"] = comment

        response = self._client._patch(
            f"/api/audit/findings/{finding_id}/status",
            data,
            headers=headers,
        )
        return FindingWorkflowData(**response)

    async def update_finding_status_async(
        self,
        finding_id: str,
        status: FindingWorkflowStatus | str,
        comment: str = "",
        user_id: Optional[str] = None,
    ) -> FindingWorkflowData:
        """Async version of update_finding_status()."""
        if isinstance(status, FindingWorkflowStatus):
            status = status.value

        headers = {}
        if user_id:
            headers["X-User-ID"] = user_id

        data = {"status": status}
        if comment:
            data["comment"] = comment

        response = await self._client._patch_async(
            f"/api/audit/findings/{finding_id}/status",
            data,
            headers=headers,
        )
        return FindingWorkflowData(**response)

    def add_finding_comment(
        self,
        finding_id: str,
        comment: str,
        user_id: Optional[str] = None,
    ) -> FindingWorkflowData:
        """
        Add a comment to a finding's history.

        Args:
            finding_id: The finding ID.
            comment: The comment text.
            user_id: Optional user ID for attribution.

        Returns:
            Updated FindingWorkflowData with new comment in history.
        """
        headers = {}
        if user_id:
            headers["X-User-ID"] = user_id

        response = self._client._post(
            f"/api/audit/findings/{finding_id}/comments",
            {"comment": comment},
            headers=headers,
        )
        return FindingWorkflowData(**response)

    async def add_finding_comment_async(
        self,
        finding_id: str,
        comment: str,
        user_id: Optional[str] = None,
    ) -> FindingWorkflowData:
        """Async version of add_finding_comment()."""
        headers = {}
        if user_id:
            headers["X-User-ID"] = user_id

        response = await self._client._post_async(
            f"/api/audit/findings/{finding_id}/comments",
            {"comment": comment},
            headers=headers,
        )
        return FindingWorkflowData(**response)

    def assign_finding(
        self,
        finding_id: str,
        assignee_id: str,
        user_id: Optional[str] = None,
    ) -> FindingWorkflowData:
        """
        Assign a finding to a user.

        Args:
            finding_id: The finding ID.
            assignee_id: User ID to assign the finding to.
            user_id: Optional user ID for attribution.

        Returns:
            Updated FindingWorkflowData with assignment.
        """
        headers = {}
        if user_id:
            headers["X-User-ID"] = user_id

        response = self._client._patch(
            f"/api/audit/findings/{finding_id}/assign",
            {"user_id": assignee_id},
            headers=headers,
        )
        return FindingWorkflowData(**response)

    async def assign_finding_async(
        self,
        finding_id: str,
        assignee_id: str,
        user_id: Optional[str] = None,
    ) -> FindingWorkflowData:
        """Async version of assign_finding()."""
        headers = {}
        if user_id:
            headers["X-User-ID"] = user_id

        response = await self._client._patch_async(
            f"/api/audit/findings/{finding_id}/assign",
            {"user_id": assignee_id},
            headers=headers,
        )
        return FindingWorkflowData(**response)

    def set_finding_priority(
        self,
        finding_id: str,
        priority: int,
        user_id: Optional[str] = None,
    ) -> FindingWorkflowData:
        """
        Set the priority of a finding.

        Args:
            finding_id: The finding ID.
            priority: Priority level (1=Critical, 2=High, 3=Medium, 4=Low, 5=Lowest).
            user_id: Optional user ID for attribution.

        Returns:
            Updated FindingWorkflowData with new priority.
        """
        if priority < 1 or priority > 5:
            raise ValueError("Priority must be between 1 (Critical) and 5 (Lowest)")

        headers = {}
        if user_id:
            headers["X-User-ID"] = user_id

        response = self._client._patch(
            f"/api/audit/findings/{finding_id}/priority",
            {"priority": priority},
            headers=headers,
        )
        return FindingWorkflowData(**response)

    async def set_finding_priority_async(
        self,
        finding_id: str,
        priority: int,
        user_id: Optional[str] = None,
    ) -> FindingWorkflowData:
        """Async version of set_finding_priority()."""
        if priority < 1 or priority > 5:
            raise ValueError("Priority must be between 1 (Critical) and 5 (Lowest)")

        headers = {}
        if user_id:
            headers["X-User-ID"] = user_id

        response = await self._client._patch_async(
            f"/api/audit/findings/{finding_id}/priority",
            {"priority": priority},
            headers=headers,
        )
        return FindingWorkflowData(**response)

    def bulk_update_findings(
        self,
        finding_ids: List[str],
        action: str,
        value: Any = None,
        user_id: Optional[str] = None,
    ) -> dict[str, Any]:
        """
        Perform bulk action on multiple findings.

        Args:
            finding_ids: List of finding IDs to update.
            action: Action to perform ("status", "assign", "priority").
            value: Value for the action (status string, user_id, or priority int).
            user_id: Optional user ID for attribution.

        Returns:
            Dictionary with success count and any failures.
        """
        headers = {}
        if user_id:
            headers["X-User-ID"] = user_id

        response = self._client._post(
            "/api/audit/findings/bulk-action",
            {
                "finding_ids": finding_ids,
                "action": action,
                "value": value,
            },
            headers=headers,
        )
        return response

    async def bulk_update_findings_async(
        self,
        finding_ids: List[str],
        action: str,
        value: Any = None,
        user_id: Optional[str] = None,
    ) -> dict[str, Any]:
        """Async version of bulk_update_findings()."""
        headers = {}
        if user_id:
            headers["X-User-ID"] = user_id

        response = await self._client._post_async(
            "/api/audit/findings/bulk-action",
            {
                "finding_ids": finding_ids,
                "action": action,
                "value": value,
            },
            headers=headers,
        )
        return response

    # =========================================================================
    # Quick Audit
    # =========================================================================

    def run_quick_audit(
        self,
        document_ids: List[str],
        preset: str = "Code Security",
    ) -> QuickAuditResult:
        """
        Run a quick audit using a preset.

        This is a convenience method that creates a session, runs the audit,
        and returns a summary of findings. Use this for rapid assessment.

        For more control over the audit process, use DocumentsAPI methods:
        create_audit(), start_audit(), get_audit(), audit_findings().

        Args:
            document_ids: List of document IDs to audit.
            preset: Preset name to use (default: "Code Security").

        Returns:
            QuickAuditResult with findings summary and top issues.
        """
        response = self._client._post(
            "/api/audit/quick",
            {
                "document_ids": document_ids,
                "preset": preset,
            },
        )
        return QuickAuditResult(**response)

    async def run_quick_audit_async(
        self,
        document_ids: List[str],
        preset: str = "Code Security",
    ) -> QuickAuditResult:
        """Async version of run_quick_audit()."""
        response = await self._client._post_async(
            "/api/audit/quick",
            {
                "document_ids": document_ids,
                "preset": preset,
            },
        )
        return QuickAuditResult(**response)

    # =========================================================================
    # Cross-Session Finding Search
    # =========================================================================

    def search_findings(
        self,
        query: Optional[str] = None,
        severity: Optional[str] = None,
        status: Optional[str] = None,
        audit_type: Optional[str] = None,
        assigned_to: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> List[AuditFinding]:
        """
        Search findings across all audit sessions.

        Args:
            query: Text search in title/description.
            severity: Filter by severity (critical, high, medium, low, info).
            status: Filter by workflow status.
            audit_type: Filter by audit type.
            assigned_to: Filter by assigned user.
            limit: Maximum number of results.
            offset: Number of results to skip.

        Returns:
            List of matching AuditFinding objects.
        """
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        if query:
            params["query"] = query
        if severity:
            params["severity"] = severity
        if status:
            params["status"] = status
        if audit_type:
            params["audit_type"] = audit_type
        if assigned_to:
            params["assigned_to"] = assigned_to

        response = self._client._get("/api/audit/findings", params=params)
        findings = response.get("findings", []) if isinstance(response, dict) else response
        return [AuditFinding(**f) for f in findings]

    async def search_findings_async(
        self,
        query: Optional[str] = None,
        severity: Optional[str] = None,
        status: Optional[str] = None,
        audit_type: Optional[str] = None,
        assigned_to: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> List[AuditFinding]:
        """Async version of search_findings()."""
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        if query:
            params["query"] = query
        if severity:
            params["severity"] = severity
        if status:
            params["status"] = status
        if audit_type:
            params["audit_type"] = audit_type
        if assigned_to:
            params["assigned_to"] = assigned_to

        response = await self._client._get_async("/api/audit/findings", params=params)
        findings = response.get("findings", []) if isinstance(response, dict) else response
        return [AuditFinding(**f) for f in findings]
