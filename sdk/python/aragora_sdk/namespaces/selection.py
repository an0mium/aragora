"""
Selection Namespace API

Provides methods for agent selection:
- Selection criteria management
- Agent matching
- Performance-based selection
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..client import AragoraAsyncClient, AragoraClient

class SelectionAPI:
    """Synchronous Selection API."""

    def __init__(self, client: AragoraClient):
        self._client = client

class AsyncSelectionAPI:
    """Asynchronous Selection API."""

    def __init__(self, client: AragoraAsyncClient):
        self._client = client

