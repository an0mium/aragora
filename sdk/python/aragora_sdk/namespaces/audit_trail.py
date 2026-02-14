"""
Audit Trail Namespace API

Provides methods for audit trail access and verification:
- List and retrieve audit trails
- Export in multiple formats (json, csv, md)
- Verify integrity checksums
- Access v1 decision receipts
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from ..client import AragoraAsyncClient, AragoraClient

AuditExportFormat = Literal["json", "csv", "md"]

class AuditTrailAPI:
    """
    Synchronous Audit Trail API.

    Example:
        >>> client = AragoraClient(base_url="https://api.aragora.ai")
        >>> trails = client.audit_trail.list()
        >>> for trail in trails["trails"]:
        ...     print(trail["trail_id"], trail["verdict"])
    """

    def __init__(self, client: AragoraClient):
        self._client = client

    # -- Audit trails ---------------------------------------------------------

    # -- Decision receipts (v1) -----------------------------------------------


class AsyncAuditTrailAPI:
    """
    Asynchronous Audit Trail API.

    Example:
        >>> async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
        ...     trails = await client.audit_trail.list()
    """

    def __init__(self, client: AragoraAsyncClient):
        self._client = client

    # -- Audit trails ---------------------------------------------------------

    # -- Decision receipts (v1) -----------------------------------------------
