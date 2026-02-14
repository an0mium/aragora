"""
Workspace Settings Namespace API

Provides methods for workspace configuration:
- Workspace preferences
- Member settings
- Integration settings
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..client import AragoraAsyncClient, AragoraClient

class WorkspaceSettingsAPI:
    """Synchronous Workspace Settings API."""

    def __init__(self, client: AragoraClient):
        self._client = client

class AsyncWorkspaceSettingsAPI:
    """Asynchronous Workspace Settings API."""

    def __init__(self, client: AragoraAsyncClient):
        self._client = client

