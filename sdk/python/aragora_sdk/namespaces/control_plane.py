"""
Control Plane Namespace API

Provides methods for enterprise control plane operations.

Note: Control plane endpoints are currently undergoing handler migration.
Methods will be re-added as handler routes are stabilized.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..client import AragoraAsyncClient, AragoraClient


class ControlPlaneAPI:
    """Synchronous Control Plane API."""

    def __init__(self, client: AragoraClient):
        self._client = client


class AsyncControlPlaneAPI:
    """Asynchronous Control Plane API."""

    def __init__(self, client: AragoraAsyncClient):
        self._client = client
