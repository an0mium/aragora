"""Instagram Connector.

Provides integration with Instagram for posts and media data.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from aragora.connectors.base import BaseConnector, Evidence
from aragora.reasoning.provenance import SourceType

logger = logging.getLogger(__name__)

CONFIG_ENV_VARS = ("INSTAGRAM_ACCESS_TOKEN",)


class InstagramConnector(BaseConnector):
    """Instagram connector for posts and media data."""

    def __init__(self) -> None:
        super().__init__()
        self._configured = all(os.environ.get(v) for v in CONFIG_ENV_VARS)

    @property
    def name(self) -> str:
        return "instagram"

    @property
    def source_type(self) -> SourceType:
        return SourceType.WEB_SEARCH

    @property
    def is_configured(self) -> bool:
        return self._configured

    async def search(self, query: str, limit: int = 10, **kwargs: Any) -> list[Evidence]:
        """Search Instagram for posts and media."""
        if not self._configured:
            logger.debug("Instagram connector not configured")
            return []
        # TODO: Implement Instagram Graph API search
        return []

    async def fetch(self, evidence_id: str, **kwargs: Any) -> Evidence | None:
        """Fetch a specific post or media from Instagram."""
        if not self._configured:
            return None
        # TODO: Implement Instagram Graph API fetch
        return None
