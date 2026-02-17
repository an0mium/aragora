"""Bitbucket Connector.

Provides integration with Bitbucket for repos and pipelines.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from aragora.connectors.base import BaseConnector, Evidence
from aragora.reasoning.provenance import SourceType

logger = logging.getLogger(__name__)

CONFIG_ENV_VARS = ("BITBUCKET_USERNAME", "BITBUCKET_APP_PASSWORD")


class BitbucketConnector(BaseConnector):
    """Bitbucket connector for repository and pipeline data."""

    def __init__(self) -> None:
        super().__init__()
        self._configured = all(os.environ.get(v) for v in CONFIG_ENV_VARS)

    @property
    def name(self) -> str:
        return "bitbucket"

    @property
    def source_type(self) -> SourceType:
        return SourceType.CODE_ANALYSIS

    @property
    def is_configured(self) -> bool:
        return self._configured

    async def search(self, query: str, limit: int = 10, **kwargs: Any) -> list[Evidence]:
        """Search Bitbucket for repos and pull requests."""
        if not self._configured:
            logger.debug("Bitbucket connector not configured")
            return []
        # TODO: Implement Bitbucket API search
        return []

    async def fetch(self, evidence_id: str, **kwargs: Any) -> Evidence | None:
        """Fetch a specific repo or PR from Bitbucket."""
        if not self._configured:
            return None
        # TODO: Implement Bitbucket API fetch
        return None
