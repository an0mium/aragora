"""Instagram Connector.

Provides production-quality integration with the Instagram Graph API for
media and comments.

Searches:
- User media (posts, reels, stories published by the authenticated user)
- Comments on media
- Hashtag media (via the hashtag search + recent-media endpoints)

The Instagram Graph API does not offer a free-text search endpoint, so
``search()`` retrieves recent media from the authenticated user's account
and filters client-side by caption/hashtag matching.

Features:
- Exponential backoff with jitter via ``_request_with_retry``
- Circuit breaker integration for failure protection
- Rate limiting between consecutive API calls
- LRU cache with configurable TTL
- Health check against the ``/me`` endpoint
- Query sanitization to prevent injection

Environment Variables:
- ``INSTAGRAM_ACCESS_TOKEN`` -- Long-lived Instagram Graph API token.
"""

from __future__ import annotations

__all__ = [
    "InstagramConnector",
    "CONFIG_ENV_VARS",
    "MAX_QUERY_LENGTH",
]

import asyncio
import logging
import os
import re
import time
from typing import Any

import httpx

from aragora.connectors.base import BaseConnector, ConnectorCapabilities, Evidence
from aragora.connectors.exceptions import (
    ConnectorAuthError,
    ConnectorError,
)
from aragora.reasoning.provenance import SourceType

logger = logging.getLogger(__name__)

CONFIG_ENV_VARS = ("INSTAGRAM_ACCESS_TOKEN",)

_SAFE_QUERY_RE = re.compile(r"[^\w\s@.\-+#]")
MAX_QUERY_LENGTH = 500

_IG_API_BASE = "https://graph.instagram.com"

# Instagram Graph API rate limit: 200 calls per user per hour.
# Conservative per-request delay to avoid bursts.
_DEFAULT_RATE_LIMIT_DELAY = 0.2  # seconds between requests


def _sanitize_query(query: str) -> str:
    """Sanitize query to prevent injection."""
    query = query[:MAX_QUERY_LENGTH]
    return _SAFE_QUERY_RE.sub("", query)


class InstagramConnector(BaseConnector):
    """Instagram connector for media and comment data.

    Uses the Instagram Graph API (v18.0+) which requires a valid long-lived
    access token.  All external calls go through the base-class
    ``_request_with_retry`` helper which provides exponential backoff,
    circuit-breaker recording, and structured error mapping.
    """

    def __init__(
        self,
        *,
        rate_limit_delay: float = _DEFAULT_RATE_LIMIT_DELAY,
        max_retries: int = BaseConnector.DEFAULT_MAX_RETRIES,
        base_delay: float = BaseConnector.DEFAULT_BASE_DELAY,
        max_delay: float = BaseConnector.DEFAULT_MAX_DELAY,
        max_cache_entries: int = 500,
        cache_ttl_seconds: float = 3600.0,
        enable_circuit_breaker: bool = True,
    ) -> None:
        super().__init__(
            max_retries=max_retries,
            base_delay=base_delay,
            max_delay=max_delay,
            max_cache_entries=max_cache_entries,
            cache_ttl_seconds=cache_ttl_seconds,
            enable_circuit_breaker=enable_circuit_breaker,
        )
        self._configured = all(os.environ.get(v) for v in CONFIG_ENV_VARS)
        self.rate_limit_delay = rate_limit_delay
        self._last_request_time: float = 0.0

    # ------------------------------------------------------------------
    # BaseConnector abstract / override properties
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "instagram"

    @property
    def source_type(self) -> SourceType:
        return SourceType.WEB_SEARCH

    @property
    def is_configured(self) -> bool:
        return self._configured

    @property
    def is_available(self) -> bool:
        """httpx is a hard dependency imported at module level."""
        return True

    def capabilities(self) -> ConnectorCapabilities:
        return ConnectorCapabilities(
            can_search=True,
            can_sync=False,
            can_send=False,
            can_receive=False,
            requires_auth=True,
            supports_oauth=True,
            supports_retry=True,
            has_circuit_breaker=self._enable_circuit_breaker,
            max_requests_per_second=5.0,
            platform_features=["media", "comments", "hashtags", "reels"],
        )

    # ------------------------------------------------------------------
    # Health check
    # ------------------------------------------------------------------

    async def _perform_health_check(self, timeout: float) -> bool:
        """Lightweight check against ``/me``."""
        if not self._configured:
            return False
        token = self._get_token()
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                resp = await client.get(
                    f"{_IG_API_BASE}/me",
                    params={"fields": "id", "access_token": token},
                )
                return resp.status_code == 200
        except httpx.TimeoutException:
            logger.debug("Instagram health check timed out")
            return False
        except httpx.RequestError as exc:
            logger.debug("Instagram health check failed: %s", exc)
            return False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_token(self) -> str:
        return os.environ.get("INSTAGRAM_ACCESS_TOKEN", "")

    async def _rate_limit(self) -> None:
        """Enforce minimum delay between consecutive requests."""
        now = time.time()
        elapsed = now - self._last_request_time
        if elapsed < self.rate_limit_delay:
            await asyncio.sleep(self.rate_limit_delay - elapsed)
        self._last_request_time = time.time()

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

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
        await self._rate_limit()

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
        except ConnectorAuthError:
            logger.warning("Instagram media search auth failure")
            return []
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
                    content=f"{media_type}: {caption[:1000]}"
                    if caption
                    else f"{media_type} (no caption)",
                    title=f"Instagram {media_type.lower()}: {caption[:80]}"
                    if caption
                    else f"Instagram {media_type.lower()}",
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
        await self._rate_limit()

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
        except ConnectorAuthError:
            logger.warning("Instagram comment search auth failure")
            return []
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

    # ------------------------------------------------------------------
    # Fetch
    # ------------------------------------------------------------------

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
            return await self._fetch_comment(evidence_id[len("ig_comment_") :], evidence_id)
        elif evidence_id.startswith("ig_media_"):
            return await self._fetch_media(evidence_id[len("ig_media_") :], evidence_id)

        return None

    async def _fetch_media(self, media_id: str, evidence_id: str) -> Evidence | None:
        """Fetch a single media post by ID."""
        token = self._get_token()
        fields = "id,caption,media_type,media_url,permalink,timestamp,like_count,comments_count"
        await self._rate_limit()

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
        except ConnectorAuthError:
            logger.warning("Instagram media fetch auth failure for %s", evidence_id)
            return None
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
            title=f"Instagram {media_type.lower()}: {caption[:80]}"
            if caption
            else f"Instagram {media_type.lower()}",
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
        await self._rate_limit()

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
        except ConnectorAuthError:
            logger.warning("Instagram comment fetch auth failure for %s", evidence_id)
            return None
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

    # ------------------------------------------------------------------
    # Hashtag media (production feature)
    # ------------------------------------------------------------------

    async def get_hashtag_media(
        self,
        hashtag: str,
        *,
        limit: int = 25,
        user_id: str | None = None,
    ) -> list[Evidence]:
        """Retrieve recent media for a hashtag.

        Uses the Instagram Graph API two-step flow:
        1. ``GET /ig_hashtag_search`` to resolve the hashtag name to an ID.
        2. ``GET /{hashtag-id}/recent_media`` to fetch recent posts.

        Args:
            hashtag: Hashtag name (without leading ``#``).
            limit: Maximum results (capped at 50).
            user_id: Instagram Business/Creator user ID.  If ``None``,
                     the connector fetches it from ``/me``.

        Returns:
            List of Evidence objects for recent media with this hashtag.
        """
        if not self._configured:
            return []

        token = self._get_token()
        capped_limit = min(limit, 50)

        # Step 0: resolve user ID if not provided
        if not user_id:
            user_id = await self._resolve_user_id()
            if not user_id:
                logger.warning("Instagram: could not resolve user ID for hashtag search")
                return []

        # Step 1: resolve hashtag name -> ID
        await self._rate_limit()

        async def _resolve_hashtag() -> Any:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.get(
                    f"{_IG_API_BASE}/ig_hashtag_search",
                    params={
                        "q": hashtag.lstrip("#")[:100],
                        "user_id": user_id,
                        "access_token": token,
                    },
                )
                resp.raise_for_status()
                return resp.json()

        try:
            ht_data = await self._request_with_retry(_resolve_hashtag, "resolve_hashtag")
        except (ConnectorError, httpx.HTTPError, OSError, ValueError):
            logger.warning("Instagram hashtag resolution failed for #%s", hashtag, exc_info=True)
            return []

        ht_list = ht_data.get("data", [])
        if not ht_list:
            return []
        hashtag_id = ht_list[0].get("id", "")
        if not hashtag_id:
            return []

        # Step 2: fetch recent media
        await self._rate_limit()

        async def _fetch_recent() -> Any:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.get(
                    f"{_IG_API_BASE}/{hashtag_id}/recent_media",
                    params={
                        "user_id": user_id,
                        "fields": "id,caption,media_type,permalink,timestamp,like_count,comments_count",
                        "limit": capped_limit,
                        "access_token": token,
                    },
                )
                resp.raise_for_status()
                return resp.json()

        try:
            media_data = await self._request_with_retry(_fetch_recent, "hashtag_recent_media")
        except (ConnectorError, httpx.HTTPError, OSError, ValueError):
            logger.warning("Instagram hashtag media fetch failed for #%s", hashtag, exc_info=True)
            return []

        results: list[Evidence] = []
        for media in media_data.get("data", [])[:capped_limit]:
            media_id = media.get("id", "")
            caption = media.get("caption", "")
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
                    content=f"#{hashtag} - {media_type}: {caption[:1000]}"
                    if caption
                    else f"#{hashtag} - {media_type} (no caption)",
                    title=f"#{hashtag} {media_type.lower()}: {caption[:60]}"
                    if caption
                    else f"#{hashtag} {media_type.lower()}",
                    url=permalink,
                    created_at=timestamp,
                    confidence=0.6,
                    freshness=self.calculate_freshness(timestamp) if timestamp else 1.0,
                    authority=0.45,
                    metadata={
                        "type": "hashtag_media",
                        "media_id": media_id,
                        "media_type": media_type,
                        "hashtag": hashtag,
                        "like_count": like_count,
                        "comments_count": comments_count,
                        "permalink": permalink,
                    },
                )
            )
        return results

    async def _resolve_user_id(self) -> str | None:
        """Resolve the authenticated user's IG user ID via ``/me``."""
        token = self._get_token()
        await self._rate_limit()

        async def _do_request() -> Any:
            async with httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.get(
                    f"{_IG_API_BASE}/me",
                    params={"fields": "id", "access_token": token},
                )
                resp.raise_for_status()
                return resp.json()

        try:
            data = await self._request_with_retry(_do_request, "resolve_user_id")
            return data.get("id")
        except (ConnectorError, httpx.HTTPError, OSError, ValueError):
            logger.warning("Instagram user ID resolution failed", exc_info=True)
            return None
