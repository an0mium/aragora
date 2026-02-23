"""Instagram Connector.

Provides integration with the Instagram Graph API for media and comments.

Searches:
- User media (posts, reels, stories published by the authenticated user)
- Comments on media

The Instagram Graph API does not offer a free-text search endpoint, so
``search()`` retrieves recent media from the authenticated user's account
and filters client-side by caption/hashtag matching.

Environment Variables:
- ``INSTAGRAM_ACCESS_TOKEN`` -- Long-lived Instagram Graph API token.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any

import httpx

from aragora.connectors.base import BaseConnector, Evidence
from aragora.connectors.exceptions import ConnectorError
from aragora.reasoning.provenance import SourceType

logger = logging.getLogger(__name__)

CONFIG_ENV_VARS = ("INSTAGRAM_ACCESS_TOKEN",)

_SAFE_QUERY_RE = re.compile(r"[^\w\s@.\-+#]")
MAX_QUERY_LENGTH = 500

_IG_API_BASE = "https://graph.instagram.com"


def _sanitize_query(query: str) -> str:
    """Sanitize query to prevent injection."""
    query = query[:MAX_QUERY_LENGTH]
    return _SAFE_QUERY_RE.sub("", query)


class InstagramConnector(BaseConnector):
    """Instagram connector for media and comment data."""

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
        """Search Instagram for media or comments.

        Retrieves recent media from the authenticated user's account via
        the ``/me/media`` endpoint and filters client-side by caption
        matching. Pass ``search_type="comments"`` with a ``media_id``
        kwarg to search comments on a specific post instead.

        Args:
            query: Text to match in captions (or comment text).
            limit: Maximum results (capped at 50).
            **kwargs: Optional ``search_type`` ("media" or "comments") and
                      ``media_id`` (required when search_type is "comments").

        Returns:
            List of Evidence objects from matching Instagram records.
        """
        if not self._configured:
            logger.debug("Instagram connector not configured")
            return []

        sanitized = _sanitize_query(query)
        if not sanitized.strip():
            return []

        search_type = kwargs.get("search_type", "media")
        capped_limit = min(limit, 50)

        if search_type == "comments":
            media_id = kwargs.get("media_id", "")
            if not media_id:
                return []
            return await self._search_comments(sanitized, media_id, capped_limit)
        return await self._search_media(sanitized, capped_limit)

    async def _search_media(self, query: str, limit: int) -> list[Evidence]:
        """Fetch recent media and filter client-side by caption."""
        token = self._get_token()
        fields = "id,caption,media_type,media_url,permalink,timestamp,like_count,comments_count"

        async def _do_request() -> Any:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.get(
                    f"{_IG_API_BASE}/me/media",
                    params={
                        "fields": fields,
                        "limit": min(limit * 3, 100),  # over-fetch for client filter
                        "access_token": token,
                    },
                )
                resp.raise_for_status()
                return resp.json()

        try:
            data = await self._request_with_retry(_do_request, "search_media")
        except (ConnectorError, httpx.HTTPError, OSError, ValueError):
            logger.warning("Instagram media search failed", exc_info=True)
            return []

        results: list[Evidence] = []
        query_lower = query.lower()
        for media in data.get("data", []):
            caption = media.get("caption", "")
            if query_lower and query_lower not in caption.lower():
                continue

            media_id = media.get("id", "")
            media_type = media.get("media_type", "")
            permalink = media.get("permalink", "")
            timestamp = media.get("timestamp", "")
            like_count = media.get("like_count", 0)
            comments_count = media.get("comments_count", 0)

            results.append(
                Evidence(
                    id=f"ig_media_{media_id}",
                    source_type=self.source_type,
                    source_id=f"instagram://media/{media_id}",
                    content=f"{media_type}: {caption[:1000]}" if caption else f"{media_type} (no caption)",
                    title=f"Instagram {media_type.lower()}: {caption[:80]}" if caption else f"Instagram {media_type.lower()}",
                    url=permalink,
                    created_at=timestamp,
                    confidence=0.65,
                    freshness=self.calculate_freshness(timestamp) if timestamp else 1.0,
                    authority=0.5,
                    metadata={
                        "type": "media",
                        "media_id": media_id,
                        "media_type": media_type,
                        "like_count": like_count,
                        "comments_count": comments_count,
                        "permalink": permalink,
                    },
                )
            )

            if len(results) >= limit:
                break

        return results

    async def _search_comments(self, query: str, media_id: str, limit: int) -> list[Evidence]:
        """Fetch comments on a specific media and filter by text."""
        token = self._get_token()

        async def _do_request() -> Any:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.get(
                    f"{_IG_API_BASE}/{media_id}/comments",
                    params={
                        "fields": "id,text,timestamp,username,like_count",
                        "limit": min(limit * 3, 100),
                        "access_token": token,
                    },
                )
                resp.raise_for_status()
                return resp.json()

        try:
            data = await self._request_with_retry(_do_request, "search_comments")
        except (ConnectorError, httpx.HTTPError, OSError, ValueError):
            logger.warning("Instagram comment search failed", exc_info=True)
            return []

        results: list[Evidence] = []
        query_lower = query.lower()
        for comment in data.get("data", []):
            text = comment.get("text", "")
            if query_lower and query_lower not in text.lower():
                continue

            comment_id = comment.get("id", "")
            username = comment.get("username", "")
            timestamp = comment.get("timestamp", "")
            like_count = comment.get("like_count", 0)

            results.append(
                Evidence(
                    id=f"ig_comment_{comment_id}",
                    source_type=self.source_type,
                    source_id=f"instagram://comments/{comment_id}",
                    content=f"Comment by @{username}: {text[:1000]}",
                    title=f"Comment by @{username} on media {media_id}",
                    url=f"https://www.instagram.com/p/{media_id}/",
                    author=username,
                    created_at=timestamp,
                    confidence=0.6,
                    freshness=self.calculate_freshness(timestamp) if timestamp else 1.0,
                    authority=0.4,
                    metadata={
                        "type": "comment",
                        "comment_id": comment_id,
                        "media_id": media_id,
                        "username": username,
                        "like_count": like_count,
                    },
                )
            )

            if len(results) >= limit:
                break

        return results

    async def fetch(self, evidence_id: str, **kwargs: Any) -> Evidence | None:
        """Fetch a specific media post or comment from Instagram.

        The ``evidence_id`` should be in one of the following formats:
        - ``ig_media_<id>`` -- fetches a media post
        - ``ig_comment_<id>`` -- fetches a comment
        """
        if not self._configured:
            return None

        cached = self._cache_get(evidence_id)
        if cached is not None:
            return cached

        if evidence_id.startswith("ig_comment_"):
            return await self._fetch_comment(evidence_id[len("ig_comment_"):], evidence_id)
        elif evidence_id.startswith("ig_media_"):
            return await self._fetch_media(evidence_id[len("ig_media_"):], evidence_id)

        return None

    async def _fetch_media(self, media_id: str, evidence_id: str) -> Evidence | None:
        """Fetch a single media post by ID."""
        token = self._get_token()
        fields = "id,caption,media_type,media_url,permalink,timestamp,like_count,comments_count"

        async def _do_request() -> Any:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.get(
                    f"{_IG_API_BASE}/{media_id}",
                    params={"fields": fields, "access_token": token},
                )
                resp.raise_for_status()
                return resp.json()

        try:
            media = await self._request_with_retry(_do_request, "fetch_media")
        except (ConnectorError, httpx.HTTPError, OSError, ValueError):
            logger.warning("Instagram media fetch failed for %s", evidence_id, exc_info=True)
            return None

        caption = media.get("caption", "")
        media_type = media.get("media_type", "")
        permalink = media.get("permalink", "")
        timestamp = media.get("timestamp", "")
        like_count = media.get("like_count", 0)
        comments_count = media.get("comments_count", 0)

        evidence = Evidence(
            id=evidence_id,
            source_type=self.source_type,
            source_id=f"instagram://media/{media_id}",
            content=f"{media_type}: {caption[:2000]}" if caption else f"{media_type} (no caption)",
            title=f"Instagram {media_type.lower()}: {caption[:80]}" if caption else f"Instagram {media_type.lower()}",
            url=permalink,
            created_at=timestamp,
            confidence=0.7,
            freshness=self.calculate_freshness(timestamp) if timestamp else 1.0,
            authority=0.5,
            metadata={
                "type": "media",
                "media_id": media_id,
                "media_type": media_type,
                "like_count": like_count,
                "comments_count": comments_count,
                "permalink": permalink,
            },
        )
        self._cache_put(evidence.id, evidence)
        return evidence

    async def _fetch_comment(self, comment_id: str, evidence_id: str) -> Evidence | None:
        """Fetch a single comment by ID."""
        token = self._get_token()

        async def _do_request() -> Any:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.get(
                    f"{_IG_API_BASE}/{comment_id}",
                    params={
                        "fields": "id,text,timestamp,username,like_count",
                        "access_token": token,
                    },
                )
                resp.raise_for_status()
                return resp.json()

        try:
            comment = await self._request_with_retry(_do_request, "fetch_comment")
        except (ConnectorError, httpx.HTTPError, OSError, ValueError):
            logger.warning("Instagram comment fetch failed for %s", evidence_id, exc_info=True)
            return None

        text = comment.get("text", "")
        username = comment.get("username", "")
        timestamp = comment.get("timestamp", "")
        like_count = comment.get("like_count", 0)

        evidence = Evidence(
            id=evidence_id,
            source_type=self.source_type,
            source_id=f"instagram://comments/{comment_id}",
            content=f"Comment by @{username}: {text[:2000]}",
            title=f"Comment by @{username}",
            url="",
            author=username,
            created_at=timestamp,
            confidence=0.65,
            freshness=self.calculate_freshness(timestamp) if timestamp else 1.0,
            authority=0.4,
            metadata={
                "type": "comment",
                "comment_id": comment_id,
                "username": username,
                "like_count": like_count,
            },
        )
        self._cache_put(evidence.id, evidence)
        return evidence
