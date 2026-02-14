"""
Costs Namespace API

Provides methods for cost tracking and management:
- View usage costs
- Manage budgets
- Generate cost reports
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..client import AragoraAsyncClient, AragoraClient

class CostsAPI:
    """Synchronous Costs API."""

    def __init__(self, client: AragoraClient):
        self._client = client

class AsyncCostsAPI:
    """Asynchronous Costs API."""

    def __init__(self, client: AragoraAsyncClient):
        self._client = client

