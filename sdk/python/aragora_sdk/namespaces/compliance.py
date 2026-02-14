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

