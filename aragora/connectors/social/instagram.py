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
        """Search Instagram for posts and media."""
        if not self._configured:
            logger.debug("Instagram connector not configured")
            return []

        sanitized = _sanitize_query(query)
        if not sanitized.strip():
            return []

        token = self._get_token()

        async def _do_request() -> Any:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.get(
                    f"{_IG_API_BASE}/me/media",
                    params={
                        "fields": "id,caption,media_type,timestamp,permalink",
                        "limit": limit,
                        "access_token": token,
                    },
                )
                resp.raise_for_status()
                return resp.json()

        try:
            data = await self._request_with_retry(_do_request, "search")
        except Exception:
            logger.warning("Instagram search failed", exc_info=True)
            return []

        results: list[Evidence] = []
        media_items = data.get("data", [])
        query_lower = sanitized.lower()
        for item in media_items[:limit]:
            caption = item.get("caption", "") or ""
            if query_lower and query_lower not in caption.lower():
                continue
            media_id = item.get("id", "")
            media_type = item.get("media_type", "")
            permalink = item.get("permalink", "")
            timestamp = item.get("timestamp", "")
            results.append(
                Evidence(
                    id=f"ig_media_{media_id}",
                    source_type=self.source_type,
                    source_id=f"instagram://media/{media_id}",
                    content=caption or f"Instagram {media_type} post",
                    title=caption[:100] if caption else f"Post {media_id}",
                    url=permalink,
                    created_at=timestamp,
                    confidence=0.6,
                    freshness=1.0,
                    authority=0.5,
                    metadata={
                        "media_id": media_id,
                        "media_type": media_type,
                        "permalink": permalink,
                    },
                )
            )
        return results

    async def fetch(self, evidence_id: str, **kwargs: Any) -> Evidence | None:
        """Fetch a specific post or media from Instagram."""
        if not self._configured:
            return None

        token = self._get_token()

        async def _do_request() -> Any:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.get(
                    f"{_IG_API_BASE}/{evidence_id}",
                    params={
                        "fields": "id,caption,media_type,timestamp,permalink",
                        "access_token": token,
                    },
                )
                resp.raise_for_status()
                return resp.json()

        try:
            data = await self._request_with_retry(_do_request, "fetch")
        except Exception:
            logger.warning("Instagram fetch failed", exc_info=True)
            return None

        media_id = data.get("id", evidence_id)
        caption = data.get("caption", "") or ""
        media_type = data.get("media_type", "")
        permalink = data.get("permalink", "")
        timestamp = data.get("timestamp", "")
        return Evidence(
            id=f"ig_media_{media_id}",
            source_type=self.source_type,
            source_id=f"instagram://media/{media_id}",
            content=caption or f"Instagram {media_type} post",
            title=caption[:100] if caption else f"Post {media_id}",
            url=permalink,
            created_at=timestamp,
            confidence=0.6,
            freshness=1.0,
            authority=0.5,
            metadata={
                "media_id": media_id,
                "media_type": media_type,
                "permalink": permalink,
            },
        )
