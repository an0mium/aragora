"""
Compliance Namespace API

Provides methods for compliance and audit operations including
SOC 2 reporting, GDPR compliance, and audit trail verification.

Features:
- SOC 2 Type II report generation
- GDPR data export and right-to-be-forgotten
- Audit trail verification
- SIEM-compatible event export
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from ..client import AragoraAsyncClient, AragoraClient


AuditEventType = Literal[
    "authentication",
    "authorization",
    "data_access",
    "data_modification",
    "admin_action",
    "compliance",
]


class ComplianceAPI:
    """
    Synchronous Compliance API.

    Provides methods for compliance and audit operations:
    - SOC 2 reporting
    - GDPR compliance
    - Audit verification
    - Event export

    Example:
        >>> client = AragoraClient(base_url="https://api.aragora.ai", api_key="...")
        >>> status = client.compliance.get_status()
        >>> report = client.compliance.generate_soc2_report()
    """

    def __init__(self, client: AragoraClient):
        self._client = client

    # ===========================================================================
    # Compliance Status
    # ===========================================================================

    def get_status(self) -> dict[str, Any]:
        """
        Get overall compliance status.

        Returns:
            Dict with compliance status across frameworks (SOC 2, GDPR, etc.)
        """
        return self._client.request("GET", "/api/v2/compliance/status")

    # ===========================================================================
    # SOC 2 Compliance
    # ===========================================================================

    def generate_soc2_report(
        self,
        start_date: str | None = None,
        end_date: str | None = None,
        controls: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Generate SOC 2 compliance summary report.

        Args:
            start_date: Report period start (ISO date)
            end_date: Report period end (ISO date)
            controls: Specific controls to include (default: all)

        Returns:
            Dict with control assessments, findings, and evidence
        """
        params: dict[str, Any] = {}
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date
        if controls:
            params["controls"] = ",".join(controls)

        return self._client.request("GET", "/api/v2/compliance/soc2-report", params=params)

    # ===========================================================================
    # GDPR Compliance
    # ===========================================================================

    def gdpr_export(
        self,
        user_id: str,
        format: str = "json",
    ) -> dict[str, Any]:
        """
        Export user data for GDPR compliance (Article 15).

        Args:
            user_id: ID of the user whose data to export
            format: Export format (json, csv)

        Returns:
            Dict with user data export
        """
        return self._client.request(
            "GET",
            "/api/v2/compliance/gdpr-export",
            params={"user_id": user_id, "format": format},
        )

    def gdpr_right_to_be_forgotten(
        self,
        user_id: str,
        confirm: bool = True,
        reason: str | None = None,
    ) -> dict[str, Any]:
        """
        Execute GDPR right to erasure (Article 17).

        Args:
            user_id: ID of the user to erase
            confirm: Must be True to confirm deletion
            reason: Reason for deletion request

        Returns:
            Dict with deletion_id and status

        Note:
            Some data may be retained for legal compliance.
        """
        data: dict[str, Any] = {"user_id": user_id, "confirm": confirm}
        if reason:
            data["reason"] = reason

        return self._client.request(
            "POST",
            "/api/v2/compliance/gdpr/right-to-be-forgotten",
            json=data,
        )

    # ===========================================================================
    # Audit Trail
    # ===========================================================================

    def verify_audit_trail(
        self,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> dict[str, Any]:
        """
        Verify audit trail integrity.

        Args:
            start_date: Verification period start (ISO date)
            end_date: Verification period end (ISO date)

        Returns:
            Dict with verification results including any anomalies
        """
        data: dict[str, Any] = {}
        if start_date:
            data["start_date"] = start_date
        if end_date:
            data["end_date"] = end_date

        return self._client.request("POST", "/api/v2/compliance/audit-verify", json=data)

    def export_audit_events(
        self,
        start_date: str,
        end_date: str,
        event_types: list[AuditEventType] | None = None,
        format: str = "json",
        limit: int = 1000,
    ) -> dict[str, Any]:
        """
        Export audit events for SIEM integration.

        Args:
            start_date: Export period start (ISO date)
            end_date: Export period end (ISO date)
            event_types: Filter by event types
            format: Export format (json, elasticsearch)
            limit: Maximum events to export

        Returns:
            Dict with events array in requested format
        """
        params: dict[str, Any] = {
            "start_date": start_date,
            "end_date": end_date,
            "format": format,
            "limit": limit,
        }
        if event_types:
            params["event_types"] = ",".join(event_types)

        return self._client.request("GET", "/api/v2/compliance/audit-events", params=params)


class AsyncComplianceAPI:
    """
    Asynchronous Compliance API.

    Example:
        >>> async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
        ...     status = await client.compliance.get_status()
        ...     report = await client.compliance.generate_soc2_report()
    """

    def __init__(self, client: AragoraAsyncClient):
        self._client = client

    # ===========================================================================
    # Compliance Status
    # ===========================================================================

    async def get_status(self) -> dict[str, Any]:
        """Get overall compliance status."""
        return await self._client.request("GET", "/api/v2/compliance/status")

    # ===========================================================================
    # SOC 2 Compliance
    # ===========================================================================

    async def generate_soc2_report(
        self,
        start_date: str | None = None,
        end_date: str | None = None,
        controls: list[str] | None = None,
    ) -> dict[str, Any]:
        """Generate SOC 2 compliance summary report."""
        params: dict[str, Any] = {}
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date
        if controls:
            params["controls"] = ",".join(controls)

        return await self._client.request("GET", "/api/v2/compliance/soc2-report", params=params)

    # ===========================================================================
    # GDPR Compliance
    # ===========================================================================

    async def gdpr_export(
        self,
        user_id: str,
        format: str = "json",
    ) -> dict[str, Any]:
        """Export user data for GDPR compliance (Article 15)."""
        return await self._client.request(
            "GET",
            "/api/v2/compliance/gdpr-export",
            params={"user_id": user_id, "format": format},
        )

    async def gdpr_right_to_be_forgotten(
        self,
        user_id: str,
        confirm: bool = True,
        reason: str | None = None,
    ) -> dict[str, Any]:
        """Execute GDPR right to erasure (Article 17)."""
        data: dict[str, Any] = {"user_id": user_id, "confirm": confirm}
        if reason:
            data["reason"] = reason

        return await self._client.request(
            "POST",
            "/api/v2/compliance/gdpr/right-to-be-forgotten",
            json=data,
        )

    # ===========================================================================
    # Audit Trail
    # ===========================================================================

    async def verify_audit_trail(
        self,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> dict[str, Any]:
        """Verify audit trail integrity."""
        data: dict[str, Any] = {}
        if start_date:
            data["start_date"] = start_date
        if end_date:
            data["end_date"] = end_date

        return await self._client.request("POST", "/api/v2/compliance/audit-verify", json=data)

    async def export_audit_events(
        self,
        start_date: str,
        end_date: str,
        event_types: list[AuditEventType] | None = None,
        format: str = "json",
        limit: int = 1000,
    ) -> dict[str, Any]:
        """Export audit events for SIEM integration."""
        params: dict[str, Any] = {
            "start_date": start_date,
            "end_date": end_date,
            "format": format,
            "limit": limit,
        }
        if event_types:
            params["event_types"] = ",".join(event_types)

        return await self._client.request("GET", "/api/v2/compliance/audit-events", params=params)
