"""Instagram Connector.

Provides integration with Instagram for posts and media data.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any

import httpx

from aragora.connectors.base import BaseConnector, Evidence
from aragora.reasoning.provenance import SourceType

logger = logging.getLogger(__name__)

CONFIG_ENV_VARS = ("INSTAGRAM_ACCESS_TOKEN",)

_SAFE_QUERY_RE = re.compile(r"[^\w\s@.\-+]")
MAX_QUERY_LENGTH = 500

_IG_API_BASE = "https://graph.instagram.com"


def _sanitize_query(query: str) -> str:
    """Sanitize query to prevent injection."""
    query = query[:MAX_QUERY_LENGTH]
    return _SAFE_QUERY_RE.sub("", query)


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

    def _get_token(self) -> str:
        return os.environ.get("INSTAGRAM_ACCESS_TOKEN", "")

    async def search(self, query: str, limit: int = 10, **kwargs: Any) -> list[Evidence]:
        """Search Instagram for posts and media.

        Raises:
            NotImplementedError: Instagram connector is not yet implemented.
        """
        raise NotImplementedError(
            "InstagramConnector.search() is not yet implemented. "
            "Configure INSTAGRAM_ACCESS_TOKEN and contribute an implementation."
        )

    async def fetch(self, evidence_id: str, **kwargs: Any) -> Evidence | None:
        """Fetch a specific post or media from Instagram.

        Raises:
            NotImplementedError: Instagram connector is not yet implemented.
        """
        raise NotImplementedError(
            "InstagramConnector.fetch() is not yet implemented. "
            "Configure INSTAGRAM_ACCESS_TOKEN and contribute an implementation."
        )
