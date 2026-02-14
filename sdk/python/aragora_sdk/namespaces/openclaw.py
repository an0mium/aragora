"""OpenClaw Gateway namespace API.

Provides methods for Aragora's OpenClaw gateway endpoints:
- Session orchestration
- Action execution
- Policy and approvals
- Credential lifecycle
- Health, metrics, and audit
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..client import AragoraAsyncClient, AragoraClient


class OpenclawAPI:
    """Synchronous OpenClaw Gateway API."""

    def __init__(self, client: AragoraClient):
        self._client = client

    # -- Session management ---------------------------------------------------

    # -- Action management ----------------------------------------------------

    # -- Credential lifecycle -------------------------------------------------

    # -- Policy rules ---------------------------------------------------------

    # -- Approvals ------------------------------------------------------------

    # -- Service introspection ------------------------------------------------


class AsyncOpenclawAPI:
    """Asynchronous OpenClaw Gateway API."""

    def __init__(self, client: AragoraAsyncClient):
        self._client = client

    # -- Session management ---------------------------------------------------

    # -- Action management ----------------------------------------------------

    # -- Credential lifecycle -------------------------------------------------

    # -- Policy rules ---------------------------------------------------------

    # -- Approvals ------------------------------------------------------------

    # -- Service introspection ------------------------------------------------
