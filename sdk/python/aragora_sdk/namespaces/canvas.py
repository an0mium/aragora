"""
Canvas Namespace API

Provides methods for visual collaboration canvas:
- Create and manage canvases
- Real-time collaboration
- Export and sharing
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..client import AragoraAsyncClient, AragoraClient

class CanvasAPI:
    """Synchronous Canvas API."""

    def __init__(self, client: AragoraClient):
        self._client = client

class AsyncCanvasAPI:
    """Asynchronous Canvas API."""

    def __init__(self, client: AragoraAsyncClient):
        self._client = client

