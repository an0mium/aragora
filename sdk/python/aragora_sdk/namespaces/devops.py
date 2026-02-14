"""
DevOps Namespace API

Provides methods for DevOps integration:
- CI/CD pipelines
- Deployment management
- Infrastructure monitoring
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..client import AragoraAsyncClient, AragoraClient

class DevopsAPI:
    """Synchronous DevOps API."""

    def __init__(self, client: AragoraClient):
        self._client = client

class AsyncDevopsAPI:
    """Asynchronous DevOps API."""

    def __init__(self, client: AragoraAsyncClient):
        self._client = client

