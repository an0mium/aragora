"""
Invoice Processing Namespace API.

Provides invoice processing and approval workflows.

Note: Accounting invoice endpoints are currently undergoing handler migration.
Methods will be re-added as handler routes are stabilized.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..client import AragoraAsyncClient, AragoraClient


class InvoiceProcessingAPI:
    """Synchronous Invoice Processing API."""

    def __init__(self, client: AragoraClient):
        self._client = client


class AsyncInvoiceProcessingAPI:
    """Asynchronous Invoice Processing API."""

    def __init__(self, client: AragoraAsyncClient):
        self._client = client
