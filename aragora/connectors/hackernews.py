"""
HackerNews Connector - Tech discussion evidence for aragora agents.

Provides access to HackerNews via the Algolia Search API:
- Search stories by query
- Fetch story details and comments
- Get top/best/new stories

The Algolia HN API is free and requires no authentication.
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Optional

from aragora.connectors.base import BaseConnector, Evidence
from aragora.reasoning.provenance import ProvenanceManager, SourceType

logger = logging.getLogger(__name__)

# Try to import optional dependencies
try:
    import httpx

    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False


# Algolia HN Search API endpoints
HN_SEARCH_URL = "https://hn.algolia.com/api/v1/search"
HN_SEARCH_BY_DATE_URL = "https://hn.algolia.com/api/v1/search_by_date"
HN_ITEM_URL = "https://hn.algolia.com/api/v1/items"

# HackerNews item URL template
HN_STORY_URL = "https://news.ycombinator.com/item?id={}"


class HackerNewsConnector(BaseConnector):
    """
    Connector for HackerNews via Algolia Search API.

    Enables agents to:
    - Search HN stories and comments
    - Get top/recent stories
    - Retrieve story details and discussions
    - Track tech community evidence

    Example:
        connector = HackerNewsConnector()
        results = await connector.search("rust vs go performance")
        for evidence in results:
            print(f"{evidence.title} ({evidence.metadata['points']} points)")
    """

    def __init__(
        self,
        provenance: Optional[ProvenanceManager] = None,
        default_confidence: float = 0.7,
        timeout: int = 30,
        rate_limit_delay: float = 0.5,  # Algolia allows higher rates
        max_cache_entries: int = 500,
        cache_ttl_seconds: float = 3600.0,  # 1 hour cache
    ):
        """
        Initialize HackerNewsConnector.

        Args:
            provenance: Optional provenance manager for tracking
            default_confidence: Base confidence for HN sources
            timeout: HTTP request timeout in seconds
            rate_limit_delay: Delay between API requests
            max_cache_entries: Maximum cached entries
            cache_ttl_seconds: Cache TTL in seconds
        """
        super().__init__(
            provenance=provenance,
            default_confidence=default_confidence,
            max_cache_entries=max_cache_entries,
            cache_ttl_seconds=cache_ttl_seconds,
        )
        self.timeout = timeout
        self.rate_limit_delay = rate_limit_delay
        self._last_request_time: float = 0.0

    @property
    def source_type(self) -> SourceType:
        """HackerNews is external API data."""
        return SourceType.EXTERNAL_API

    @property
    def name(self) -> str:
        """Human-readable connector name."""
        return "HackerNews"

    @property
    def is_available(self) -> bool:
        """Check if httpx is available for making requests."""
        return HTTPX_AVAILABLE

    async def _rate_limit(self) -> None:
        """Enforce rate limiting between requests."""
        import time

        now = time.time()
        elapsed = now - self._last_request_time
        if elapsed < self.rate_limit_delay:
            await asyncio.sleep(self.rate_limit_delay - elapsed)
        self._last_request_time = time.time()

    async def search(
        self,
        query: str,
        limit: int = 10,
        tags: Optional[str] = None,
        sort_by: str = "relevance",
        **kwargs,
    ) -> list[Evidence]:
        """
        Search HackerNews for stories matching query.

        Args:
            query: Search query
            limit: Maximum results to return (max 100)
            tags: Filter by type - "story", "comment", "poll", "ask_hn", "show_hn"
            sort_by: "relevance" (default) or "date"
            **kwargs: Additional API parameters

        Returns:
            List of Evidence objects with story content

        Examples:
            - Search stories: search("AI safety", tags="story")
            - Search comments: search("kubernetes", tags="comment")
            - Show HN posts: search("startup", tags="show_hn")
        """
        if not HTTPX_AVAILABLE:
            logger.warning("httpx not available, cannot search HackerNews")
            return []

        # Clamp limit
        limit = min(limit, 100)

        # Choose endpoint based on sort
        url = HN_SEARCH_BY_DATE_URL if sort_by == "date" else HN_SEARCH_URL

        # Build params
        params = {
            "query": query,
            "hitsPerPage": limit,
        }

        if tags:
            params["tags"] = tags

        # Rate limiting
        await self._rate_limit()

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(url, params=params)
                response.raise_for_status()

            data = response.json()
            results = self._parse_search_results(data)
            logger.info(f"HackerNews search '{query}' returned {len(results)} results")
            return results

        except httpx.TimeoutException:
            logger.warning(f"HackerNews search timeout for query: {query}")
            return []
        except httpx.HTTPStatusError as e:
            logger.error(f"HackerNews API error: {e.response.status_code}")
            return []
        except httpx.ConnectError as e:
            logger.error(f"HackerNews connection error: {e}")
            return []
        except httpx.RequestError as e:
            logger.error(f"HackerNews request error: {e}")
            return []
        except json.JSONDecodeError as e:
            logger.error(f"HackerNews JSON parsing error: {e}")
            return []
        except Exception as e:
            logger.error(f"HackerNews search failed unexpectedly ({type(e).__name__}): {e}")
            return []

    async def fetch(self, evidence_id: str) -> Optional[Evidence]:
        """
        Fetch a specific HackerNews item by ID.

        Args:
            evidence_id: HN item ID (e.g., "hn:12345" or just "12345")

        Returns:
            Evidence object or None if not found
        """
        # Check cache first
        cached = self._cache_get(evidence_id)
        if cached:
            return cached

        if not HTTPX_AVAILABLE:
            logger.warning("httpx not available, cannot fetch HackerNews item")
            return None

        # Extract numeric ID
        item_id = evidence_id.replace("hn:", "").strip()

        await self._rate_limit()

        try:
            url = f"{HN_ITEM_URL}/{item_id}"
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(url)
                response.raise_for_status()

            data = response.json()
            evidence = self._parse_item(data)
            if evidence:
                self._cache_put(evidence_id, evidence)
            return evidence

        except httpx.TimeoutException:
            logger.warning(f"HackerNews fetch timeout for {evidence_id}")
            return None
        except httpx.HTTPStatusError as e:
            logger.error(f"HackerNews API error fetching {evidence_id}: {e.response.status_code}")
            return None
        except httpx.ConnectError as e:
            logger.error(f"HackerNews connection error for {evidence_id}: {e}")
            return None
        except httpx.RequestError as e:
            logger.error(f"HackerNews request error for {evidence_id}: {e}")
            return None
        except json.JSONDecodeError as e:
            logger.error(f"HackerNews JSON parsing error for {evidence_id}: {e}")
            return None
        except Exception as e:
            logger.error(
                f"HackerNews fetch failed unexpectedly for {evidence_id} ({type(e).__name__}): {e}"
            )
            return None

    def _parse_search_results(self, data: dict) -> list[Evidence]:
        """Parse Algolia search results into Evidence objects."""
        results = []
        hits = data.get("hits", [])

        for hit in hits:
            try:
                evidence = self._parse_hit(hit)
                if evidence:
                    results.append(evidence)
            except Exception as e:
                logger.debug(f"Error parsing HN hit: {e}")
                continue

        return results

    def _parse_hit(self, hit: dict) -> Optional[Evidence]:
        """Parse a single Algolia hit into Evidence."""
        object_id = hit.get("objectID")
        if not object_id:
            return None

        # Determine content type
        story_id = hit.get("story_id") or object_id
        is_comment = hit.get("story_id") is not None

        # Build content
        if is_comment:
            # Comment - use comment_text
            content = hit.get("comment_text", "")
            title = f"Comment on: {hit.get('story_title', 'Unknown')}"
        else:
            # Story - use title + optional URL
            title = hit.get("title", "Untitled")
            story_url = hit.get("url", "")
            content = title
            if story_url:
                content = f"{title}\n\nURL: {story_url}"

        if not content:
            return None

        # Author
        author = hit.get("author", "unknown")

        # Created time
        created_at = hit.get("created_at")

        # Points and comments
        points = hit.get("points") or 0
        num_comments = hit.get("num_comments") or 0

        # Calculate confidence based on engagement
        # More points/comments = higher confidence in community validation
        engagement_score = min(1.0, (points + num_comments) / 500)
        confidence = self.default_confidence + (engagement_score * 0.2)

        # Freshness
        freshness = self.calculate_freshness(created_at) if created_at else 0.5

        # HN story URL
        url = HN_STORY_URL.format(story_id)

        return Evidence(
            id=f"hn:{object_id}",
            source_type=self.source_type,
            source_id=object_id,
            content=content,
            title=title,
            created_at=created_at,
            author=author,
            url=url,
            confidence=min(confidence, 0.95),
            freshness=freshness,
            authority=0.7,  # Tech community, but not peer-reviewed
            metadata={
                "points": points,
                "num_comments": num_comments,
                "story_id": story_id,
                "is_comment": is_comment,
                "story_url": hit.get("url"),
                "tags": hit.get("_tags", []),
            },
        )

    def _parse_item(self, data: dict) -> Optional[Evidence]:
        """Parse a full item response into Evidence."""
        item_id = data.get("id")
        if not item_id:
            return None

        item_type = data.get("type", "story")

        if item_type == "comment":
            title = f"Comment on: {data.get('title', 'Unknown')}"
            content = data.get("text", "")
        else:
            title = data.get("title", "Untitled")
            content = title
            if data.get("url"):
                content = f"{title}\n\nURL: {data['url']}"
            if data.get("text"):
                content += f"\n\n{data['text']}"

        # Created time (Unix timestamp)
        created_at_unix = data.get("created_at_i")
        created_at = None
        if created_at_unix:
            created_at = datetime.fromtimestamp(created_at_unix).isoformat()

        points = data.get("points") or 0
        children = data.get("children", [])
        num_comments = len(children)

        engagement_score = min(1.0, (points + num_comments) / 500)
        confidence = self.default_confidence + (engagement_score * 0.2)
        freshness = self.calculate_freshness(created_at) if created_at else 0.5

        return Evidence(
            id=f"hn:{item_id}",
            source_type=self.source_type,
            source_id=str(item_id),
            content=content,
            title=title,
            created_at=created_at,
            author=data.get("author", "unknown"),
            url=HN_STORY_URL.format(item_id),
            confidence=min(confidence, 0.95),
            freshness=freshness,
            authority=0.7,
            metadata={
                "points": points,
                "num_comments": num_comments,
                "type": item_type,
                "story_url": data.get("url"),
            },
        )

    async def get_top_stories(self, limit: int = 10) -> list[Evidence]:
        """
        Get current top stories from HackerNews.

        Args:
            limit: Maximum stories to return

        Returns:
            List of Evidence objects for top stories
        """
        return await self.search("", limit=limit, tags="story", sort_by="relevance")

    async def get_recent_stories(
        self,
        limit: int = 10,
        tags: str = "story",
    ) -> list[Evidence]:
        """
        Get recent stories from HackerNews.

        Args:
            limit: Maximum stories to return
            tags: Filter by type (story, show_hn, ask_hn)

        Returns:
            List of Evidence objects sorted by date
        """
        return await self.search("", limit=limit, tags=tags, sort_by="date")

    async def search_show_hn(self, query: str = "", limit: int = 10) -> list[Evidence]:
        """
        Search Show HN posts (project launches).

        Args:
            query: Optional search query
            limit: Maximum results

        Returns:
            List of Evidence objects
        """
        return await self.search(query, limit=limit, tags="show_hn")

    async def search_ask_hn(self, query: str = "", limit: int = 10) -> list[Evidence]:
        """
        Search Ask HN posts (community questions).

        Args:
            query: Optional search query
            limit: Maximum results

        Returns:
            List of Evidence objects
        """
        return await self.search(query, limit=limit, tags="ask_hn")


__all__ = ["HackerNewsConnector"]
