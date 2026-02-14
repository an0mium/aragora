"""
GitHub Namespace API

Provides methods for GitHub integration:
- Repository analysis
- PR review
- Issue tracking
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..client import AragoraAsyncClient, AragoraClient

class GithubAPI:
    """Synchronous GitHub API."""

    def __init__(self, client: AragoraClient):
        self._client = client

class AsyncGithubAPI:
    """Asynchronous GitHub API."""

    def __init__(self, client: AragoraAsyncClient):
        self._client = client

