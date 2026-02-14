"""
Compliance Namespace API

Provides methods for compliance and audit operations including
SOC 2 reporting, GDPR compliance, CCPA, HIPAA, EU AI Act, and
audit trail verification.

Features:
- SOC 2 Type II report generation
- GDPR data export and right-to-be-forgotten
- CCPA consumer rights
- HIPAA PHI access logging and breach assessment
- EU AI Act risk classification and artifact bundles
- Audit trail verification
- SIEM-compatible event export
"""

from __future__ import annotations

from typing import Any, TYPE_CHECKING, Literal

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
    - CCPA consumer rights
    - HIPAA compliance
    - EU AI Act compliance
    - Audit verification
    - Event export

    Example:
        >>> client = AragoraClient(base_url="https://api.aragora.ai", api_key="...")
        >>> status = client.compliance.get_status()
        >>> report = client.compliance.generate_soc2_report()
        >>> classification = client.compliance.eu_ai_act_classify("AI for hiring")
    """

    def __init__(self, client: AragoraClient):
        self._client = client

    # ===========================================================================
    # Compliance Status

    def get_status(self) -> dict[str, Any]:
        """Get overall compliance status."""
        return self._client._get("/api/v2/compliance/status")

    def generate_soc2_report(self, **params: Any) -> dict[str, Any]:
        """Generate SOC 2 Type II compliance report."""
        return self._client._get("/api/v2/compliance/soc2-report", params=params)

    # ===========================================================================
    # EU AI Act

    def eu_ai_act_classify(self, description: str) -> dict[str, Any]:
        """Classify an AI use case by EU AI Act risk level.

        Args:
            description: Free-text description of the AI use case.

        Returns:
            Risk classification with level, rationale, and obligations.
        """
        return self._client._post(
            "/api/v2/compliance/eu-ai-act/classify",
            json={"description": description},
        )

    def eu_ai_act_audit(self, receipt: dict[str, Any]) -> dict[str, Any]:
        """Generate a conformity report from a decision receipt.

        Args:
            receipt: Decision receipt data.

        Returns:
            Conformity report with article-by-article assessment.
        """
        return self._client._post(
            "/api/v2/compliance/eu-ai-act/audit",
            json={"receipt": receipt},
        )

    def eu_ai_act_generate_bundle(
        self,
        receipt: dict[str, Any],
        *,
        provider_name: str | None = None,
        provider_contact: str | None = None,
        eu_representative: str | None = None,
        system_name: str | None = None,
        system_version: str | None = None,
    ) -> dict[str, Any]:
        """Generate a full EU AI Act compliance artifact bundle.

        Produces Articles 12 (Record-Keeping), 13 (Transparency), and
        14 (Human Oversight) artifacts bundled with a conformity report.

        Args:
            receipt: Decision receipt data.
            provider_name: Provider organization name.
            provider_contact: Provider contact email.
            eu_representative: EU representative name.
            system_name: AI system name.
            system_version: AI system version.

        Returns:
            Complete artifact bundle with integrity hash.
        """
        body: dict[str, Any] = {"receipt": receipt}
        if provider_name:
            body["provider_name"] = provider_name
        if provider_contact:
            body["provider_contact"] = provider_contact
        if eu_representative:
            body["eu_representative"] = eu_representative
        if system_name:
            body["system_name"] = system_name
        if system_version:
            body["system_version"] = system_version
        return self._client._post(
            "/api/v2/compliance/eu-ai-act/generate-bundle",
            json=body,
        )


class AsyncComplianceAPI:
    """
    Asynchronous Compliance API.

    Example:
        >>> async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
        ...     status = await client.compliance.get_status()
        ...     classification = await client.compliance.eu_ai_act_classify("AI for hiring")
    """

    def __init__(self, client: AragoraAsyncClient):
        self._client = client

    # ===========================================================================
    # Compliance Status

    async def get_status(self) -> dict[str, Any]:
        """Get overall compliance status."""
        return await self._client._get("/api/v2/compliance/status")

    async def generate_soc2_report(self, **params: Any) -> dict[str, Any]:
        """Generate SOC 2 Type II compliance report."""
        return await self._client._get("/api/v2/compliance/soc2-report", params=params)

    # ===========================================================================
    # EU AI Act

    async def eu_ai_act_classify(self, description: str) -> dict[str, Any]:
        """Classify an AI use case by EU AI Act risk level."""
        return await self._client._post(
            "/api/v2/compliance/eu-ai-act/classify",
            json={"description": description},
        )

    async def eu_ai_act_audit(self, receipt: dict[str, Any]) -> dict[str, Any]:
        """Generate a conformity report from a decision receipt."""
        return await self._client._post(
            "/api/v2/compliance/eu-ai-act/audit",
            json={"receipt": receipt},
        )

    async def eu_ai_act_generate_bundle(
        self,
        receipt: dict[str, Any],
        *,
        provider_name: str | None = None,
        provider_contact: str | None = None,
        eu_representative: str | None = None,
        system_name: str | None = None,
        system_version: str | None = None,
    ) -> dict[str, Any]:
        """Generate a full EU AI Act compliance artifact bundle."""
        body: dict[str, Any] = {"receipt": receipt}
        if provider_name:
            body["provider_name"] = provider_name
        if provider_contact:
            body["provider_contact"] = provider_contact
        if eu_representative:
            body["eu_representative"] = eu_representative
        if system_name:
            body["system_name"] = system_name
        if system_version:
            body["system_version"] = system_version
        return await self._client._post(
            "/api/v2/compliance/eu-ai-act/generate-bundle",
            json=body,
        )
