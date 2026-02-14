"""
Autonomous Namespace API

Provides methods for autonomous agent operations:
- Manage autonomous approvals
- Configure triggers
- Monitor autonomous executions
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..client import AragoraAsyncClient, AragoraClient

class AutonomousAPI:
    """Synchronous Autonomous API."""

    def __init__(self, client: AragoraClient):
        self._client = client

class AsyncAutonomousAPI:
    """Asynchronous Autonomous API."""

    def __init__(self, client: AragoraAsyncClient):
        self._client = client

