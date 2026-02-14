"""
Accounting Namespace API.

Provides a namespaced interface for QuickBooks Online and Gusto payroll integration.

Note: The accounting backend uses direct route registration (app.router.add_*)
rather than the ROUTES class-variable pattern. SDK methods will be re-added once
the handler is migrated to the standard ROUTES pattern for parity tracking.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..client import AragoraAsyncClient, AragoraClient


class AccountingAPI:
    """Synchronous Accounting API."""

    def __init__(self, client: AragoraClient) -> None:
        self._client = client


class AsyncAccountingAPI:
    """Asynchronous Accounting API."""

    def __init__(self, client: AragoraAsyncClient) -> None:
        self._client = client
