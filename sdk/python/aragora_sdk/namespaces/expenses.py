"""
Expenses Namespace API.

Provides a namespaced interface for expense tracking and management.

Note: Accounting expense endpoints are currently undergoing handler migration.
Methods will be re-added as handler routes are stabilized.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..client import AragoraAsyncClient, AragoraClient


class ExpensesAPI:
    """Synchronous Expenses API."""

    def __init__(self, client: AragoraClient):
        self._client = client


class AsyncExpensesAPI:
    """Asynchronous Expenses API."""

    def __init__(self, client: AragoraAsyncClient):
        self._client = client
