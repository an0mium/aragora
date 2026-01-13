"""
Reddit Connector - Community discussion evidence for aragora agents.

Provides access to Reddit via the public JSON API:
- Search posts by query
- Fetch post details and comments
- Get top/hot/new posts from subreddits

The Reddit JSON API is free and requires no authentication for read-only access.
Simply append .json to any Reddit URL.
"""

import asyncio
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


# Reddit JSON API endpoints
REDDIT_SEARCH_URL = "https://www.reddit.com/search.json"
REDDIT_SUBREDDIT_URL = "https://www.reddit.com/r/{subreddit}.json"
REDDIT_POST_URL = "https://www.reddit.com/comments/{post_id}.json"

# User-Agent required by Reddit API guidelines
REDDIT_USER_AGENT = "aragora-connector/1.0 (evidence collection for AI debate)"


class RedditConnector(BaseConnector):
    """
    Connector for Reddit via public JSON API.

    Enables agents to:
    - Search Reddit posts and comments
    - Get top/hot/new posts from subreddits
    - Retrieve post details and discussions
    - Track community sentiment on topics

    Example:
        connector = RedditConnector()
        results = await connector.search("AI safety")
        for evidence in results:
            print(f"{evidence.title} ({evidence.metadata['score']} upvotes)")
    """

    def __init__(
        self,
        provenance: Optional[ProvenanceManager] = None,
        default_confidence: float = 0.6,
        timeout: int = 30,
        rate_limit_delay: float = 2.0,  # Reddit requires >= 1 second between requests
        max_cache_entries: int = 500,
        cache_ttl_seconds: float = 3600.0,  # 1 hour cache
    ):
        """
        Initialize RedditConnector.

        Args:
            provenance: Optional provenance manager for tracking
            default_confidence: Base confidence for Reddit sources (lower than HN due to less moderation)
            timeout: HTTP request timeout in seconds
            rate_limit_delay: Delay between API requests (Reddit requires >= 1s)
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
        """Reddit is external API data."""
        return SourceType.EXTERNAL_API

    @property
    def name(self) -> str:
        """Human-readable connector name."""
        return "Reddit"

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

    def _get_headers(self) -> dict:
        """Get headers for Reddit API requests."""
        return {
            "User-Agent": REDDIT_USER_AGENT,
            "Accept": "application/json",
        }

    async def search(
        self,
        query: str,
        limit: int = 25,
        subreddit: Optional[str] = None,
        sort: str = "relevance",
        time_filter: str = "all",
        **kwargs,
    ) -> list[Evidence]:
        """
        Search Reddit for posts matching query.

        Args:
            query: Search query
            limit: Maximum results to return (max 100)
            subreddit: Optional subreddit to search within
            sort: Sort order - "relevance", "hot", "top", "new", "comments"
            time_filter: Time filter - "hour", "day", "week", "month", "year", "all"
            **kwargs: Additional API parameters

        Returns:
            List of Evidence objects with post content

        Examples:
            - Search all of Reddit: search("machine learning")
            - Search specific subreddit: search("rust", subreddit="programming")
            - Get recent posts: search("AI", sort="new", time_filter="week")
        """
        if not HTTPX_AVAILABLE:
            logger.warning("httpx not available, cannot search Reddit")
            return []

        # Clamp limit
        limit = min(limit, 100)

        # Build URL and params
        if subreddit:
            url = f"https://www.reddit.com/r/{subreddit}/search.json"
            params = {
                "q": query,
                "limit": limit,
                "sort": sort,
                "t": time_filter,
                "restrict_sr": "on",  # Restrict to subreddit
            }
        else:
            url = REDDIT_SEARCH_URL
            params = {
                "q": query,
                "limit": limit,
                "sort": sort,
                "t": time_filter,
            }

        # Rate limiting
        await self._rate_limit()

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(url, params=params, headers=self._get_headers())
                response.raise_for_status()

            data = response.json()
            results = self._parse_listing(data)
            logger.info(f"Reddit search '{query}' returned {len(results)} results")
            return results

        except httpx.TimeoutException:
            logger.warning(f"Reddit search timeout for query: {query}")
            return []
        except httpx.HTTPStatusError as e:
            logger.error(f"Reddit API error: {e.response.status_code}")
            return []
        except Exception as e:
            logger.error(f"Reddit search failed: {e}")
            return []

    async def fetch(self, evidence_id: str) -> Optional[Evidence]:
        """
        Fetch a specific Reddit post by ID.

        Args:
            evidence_id: Reddit post ID (e.g., "reddit:abc123" or just "abc123")

        Returns:
            Evidence object or None if not found
        """
        # Check cache first
        cached = self._cache_get(evidence_id)
        if cached:
            return cached

        if not HTTPX_AVAILABLE:
            logger.warning("httpx not available, cannot fetch Reddit post")
            return None

        # Extract post ID
        post_id = evidence_id.replace("reddit:", "").strip()

        await self._rate_limit()

        try:
            url = REDDIT_POST_URL.format(post_id=post_id)
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(url, headers=self._get_headers())
                response.raise_for_status()

            data = response.json()
            evidence = self._parse_post_detail(data)
            if evidence:
                self._cache_put(evidence_id, evidence)
            return evidence

        except Exception as e:
            logger.error(f"Reddit fetch failed for {evidence_id}: {e}")
            return None

    def _parse_listing(self, data: dict) -> list[Evidence]:
        """Parse Reddit listing response into Evidence objects."""
        results = []
        listing = data.get("data", {})
        children = listing.get("children", [])

        for child in children:
            try:
                if child.get("kind") == "t3":  # t3 = post/link
                    evidence = self._parse_post(child.get("data", {}))
                    if evidence:
                        results.append(evidence)
            except Exception as e:
                logger.debug(f"Error parsing Reddit post: {e}")
                continue

        return results

    def _parse_post(self, post: dict) -> Optional[Evidence]:
        """Parse a single Reddit post into Evidence."""
        post_id = post.get("id")
        if not post_id:
            return None

        # Build content
        title = post.get("title", "Untitled")
        selftext = post.get("selftext", "")
        url = post.get("url", "")

        # Build full content
        content_parts = [title]
        if selftext:
            content_parts.append(f"\n\n{selftext}")
        if url and not url.startswith("https://www.reddit.com"):
            content_parts.append(f"\n\nLink: {url}")
        content = "".join(content_parts)

        if not content:
            return None

        # Author
        author = post.get("author", "[deleted]")

        # Created time (Unix timestamp)
        created_utc = post.get("created_utc")
        created_at = None
        if created_utc:
            created_at = datetime.utcfromtimestamp(created_utc).isoformat()

        # Engagement metrics
        score = post.get("score", 0)
        num_comments = post.get("num_comments", 0)
        upvote_ratio = post.get("upvote_ratio", 0.5)

        # Subreddit info
        subreddit = post.get("subreddit", "unknown")
        subreddit_subscribers = post.get("subreddit_subscribers", 0)

        # Calculate confidence based on engagement and upvote ratio
        # Higher score and upvote ratio = higher community validation
        engagement_score = min(1.0, (score + num_comments) / 1000)
        ratio_score = upvote_ratio  # 0-1
        confidence = self.default_confidence + (engagement_score * 0.15) + (ratio_score * 0.1)

        # Freshness
        freshness = self.calculate_freshness(created_at) if created_at else 0.5

        # Authority based on subreddit size (larger = more diverse opinions but more moderation)
        if subreddit_subscribers > 1_000_000:
            authority = 0.6
        elif subreddit_subscribers > 100_000:
            authority = 0.65
        elif subreddit_subscribers > 10_000:
            authority = 0.55
        else:
            authority = 0.5

        # Post permalink
        permalink = post.get("permalink", f"/comments/{post_id}")
        post_url = f"https://www.reddit.com{permalink}"

        return Evidence(
            id=f"reddit:{post_id}",
            source_type=self.source_type,
            source_id=post_id,
            content=content,
            title=title,
            created_at=created_at,
            author=author,
            url=post_url,
            confidence=min(confidence, 0.85),  # Cap below HN due to less expert moderation
            freshness=freshness,
            authority=authority,
            metadata={
                "score": score,
                "num_comments": num_comments,
                "upvote_ratio": upvote_ratio,
                "subreddit": subreddit,
                "subreddit_subscribers": subreddit_subscribers,
                "is_self": post.get("is_self", False),
                "link_url": url if not post.get("is_self") else None,
                "over_18": post.get("over_18", False),
                "spoiler": post.get("spoiler", False),
                "stickied": post.get("stickied", False),
                "distinguished": post.get("distinguished"),
                "flair": post.get("link_flair_text"),
            },
        )

    def _parse_post_detail(self, data: list) -> Optional[Evidence]:
        """Parse a full post response (includes comments) into Evidence."""
        if not data or len(data) < 1:
            return None

        # First element is the post listing
        post_listing = data[0].get("data", {}).get("children", [])
        if not post_listing:
            return None

        post_data = post_listing[0].get("data", {})
        evidence = self._parse_post(post_data)

        # Optionally include top comments in metadata
        if evidence and len(data) > 1:
            comment_listing = data[1].get("data", {}).get("children", [])
            top_comments = []
            for child in comment_listing[:5]:  # Top 5 comments
                if child.get("kind") == "t1":  # t1 = comment
                    comment_data = child.get("data", {})
                    comment_body = comment_data.get("body", "")
                    comment_author = comment_data.get("author", "[deleted]")
                    comment_score = comment_data.get("score", 0)
                    if comment_body and comment_author != "[deleted]":
                        top_comments.append({
                            "author": comment_author,
                            "body": comment_body[:500],  # Truncate
                            "score": comment_score,
                        })
            evidence.metadata["top_comments"] = top_comments

        return evidence

    async def get_subreddit(
        self,
        subreddit: str,
        sort: str = "hot",
        limit: int = 25,
        time_filter: str = "all",
    ) -> list[Evidence]:
        """
        Get posts from a specific subreddit.

        Args:
            subreddit: Subreddit name (without r/)
            sort: Sort order - "hot", "new", "top", "rising"
            limit: Maximum posts to return
            time_filter: Time filter for "top" sort - "hour", "day", "week", "month", "year", "all"

        Returns:
            List of Evidence objects
        """
        if not HTTPX_AVAILABLE:
            logger.warning("httpx not available, cannot get subreddit")
            return []

        limit = min(limit, 100)

        url = f"https://www.reddit.com/r/{subreddit}/{sort}.json"
        params = {"limit": limit}
        if sort == "top":
            params["t"] = time_filter

        await self._rate_limit()

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(url, params=params, headers=self._get_headers())
                response.raise_for_status()

            data = response.json()
            results = self._parse_listing(data)
            logger.info(f"Reddit r/{subreddit}/{sort} returned {len(results)} results")
            return results

        except Exception as e:
            logger.error(f"Reddit get_subreddit failed for r/{subreddit}: {e}")
            return []

    async def get_hot(self, subreddit: str = "all", limit: int = 25) -> list[Evidence]:
        """
        Get hot posts from a subreddit or r/all.

        Args:
            subreddit: Subreddit name (default "all" for front page)
            limit: Maximum posts to return

        Returns:
            List of Evidence objects
        """
        return await self.get_subreddit(subreddit, sort="hot", limit=limit)

    async def get_top(
        self,
        subreddit: str = "all",
        limit: int = 25,
        time_filter: str = "day",
    ) -> list[Evidence]:
        """
        Get top posts from a subreddit.

        Args:
            subreddit: Subreddit name (default "all")
            limit: Maximum posts to return
            time_filter: Time period - "hour", "day", "week", "month", "year", "all"

        Returns:
            List of Evidence objects
        """
        return await self.get_subreddit(subreddit, sort="top", limit=limit, time_filter=time_filter)

    async def get_new(self, subreddit: str = "all", limit: int = 25) -> list[Evidence]:
        """
        Get new posts from a subreddit.

        Args:
            subreddit: Subreddit name (default "all")
            limit: Maximum posts to return

        Returns:
            List of Evidence objects
        """
        return await self.get_subreddit(subreddit, sort="new", limit=limit)


__all__ = ["RedditConnector"]
