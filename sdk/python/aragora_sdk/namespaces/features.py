"""
Features Namespace API

Provides access to feature flags and feature discovery.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..client import AragoraAsyncClient, AragoraClient


class FeaturesAPI:
    """Synchronous Features API."""

    def __init__(self, client: AragoraClient):
        self._client = client


class AsyncFeaturesAPI:
    """Asynchronous Features API."""

    def __init__(self, client: AragoraAsyncClient):
        self._client = client
