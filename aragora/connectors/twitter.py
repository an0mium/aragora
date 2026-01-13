"""
Twitter/X Connector - Social media evidence for aragora agents.

Provides access to Twitter/X via the API v2:
- Search recent tweets by query
- Fetch tweet details
- Get user timeline tweets

Requires TWITTER_BEARER_TOKEN environment variable for API access.
The free tier allows read-only access with limited rate limits.
"""

import asyncio
import logging
import os
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


# Twitter API v2 endpoints
TWITTER_SEARCH_URL = "https://api.twitter.com/2/tweets/search/recent"
TWITTER_TWEET_URL = "https://api.twitter.com/2/tweets"
TWITTER_USER_TWEETS_URL = "https://api.twitter.com/2/users/{user_id}/tweets"

# Tweet URL template
TWEET_URL_TEMPLATE = "https://twitter.com/i/status/{tweet_id}"


class TwitterConnector(BaseConnector):
    """
    Connector for Twitter/X via API v2.

    Enables agents to:
    - Search recent tweets (last 7 days)
    - Fetch tweet details with engagement metrics
    - Get tweets from specific users
    - Track public discourse on topics

    Requires TWITTER_BEARER_TOKEN environment variable.

    Example:
        connector = TwitterConnector()
        if connector.is_available and connector.is_configured:
            results = await connector.search("AI safety")
            for evidence in results:
                print(f"{evidence.title} ({evidence.metadata['retweet_count']} RTs)")
    """

    def __init__(
        self,
        bearer_token: Optional[str] = None,
        provenance: Optional[ProvenanceManager] = None,
        default_confidence: float = 0.5,  # Lower than Reddit due to less fact-checking
        timeout: int = 30,
        rate_limit_delay: float = 1.0,  # Twitter has strict rate limits
        max_cache_entries: int = 500,
        cache_ttl_seconds: float = 1800.0,  # 30 min cache (tweets change quickly)
    ):
        """
        Initialize TwitterConnector.

        Args:
            bearer_token: Twitter API v2 Bearer token. If not provided,
                         uses TWITTER_BEARER_TOKEN environment variable.
            provenance: Optional provenance manager for tracking
            default_confidence: Base confidence for Twitter sources
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
        self.bearer_token = bearer_token or os.environ.get("TWITTER_BEARER_TOKEN", "")
        self.timeout = timeout
        self.rate_limit_delay = rate_limit_delay
        self._last_request_time: float = 0.0

    @property
    def source_type(self) -> SourceType:
        """Twitter is external API data."""
        return SourceType.EXTERNAL_API

    @property
    def name(self) -> str:
        """Human-readable connector name."""
        return "Twitter"

    @property
    def is_available(self) -> bool:
        """Check if httpx is available for making requests."""
        return HTTPX_AVAILABLE

    @property
    def is_configured(self) -> bool:
        """Check if Twitter API credentials are configured."""
        return bool(self.bearer_token)

    async def _rate_limit(self) -> None:
        """Enforce rate limiting between requests."""
        import time

        now = time.time()
        elapsed = now - self._last_request_time
        if elapsed < self.rate_limit_delay:
            await asyncio.sleep(self.rate_limit_delay - elapsed)
        self._last_request_time = time.time()

    def _get_headers(self) -> dict:
        """Get headers for Twitter API requests."""
        return {
            "Authorization": f"Bearer {self.bearer_token}",
            "Content-Type": "application/json",
        }

    async def search(
        self,
        query: str,
        limit: int = 10,
        sort_order: str = "relevancy",
        **kwargs,
    ) -> list[Evidence]:
        """
        Search recent tweets matching query.

        Note: Free tier only searches tweets from the last 7 days.

        Args:
            query: Search query (supports Twitter search operators)
            limit: Maximum results to return (max 100)
            sort_order: "relevancy" (default) or "recency"
            **kwargs: Additional API parameters

        Returns:
            List of Evidence objects with tweet content

        Query operators:
            - "exact phrase" - Exact match
            - from:username - From specific user
            - -keyword - Exclude keyword
            - lang:en - Language filter
            - is:retweet / -is:retweet - Include/exclude retweets
            - has:links - Only tweets with links
        """
        if not HTTPX_AVAILABLE:
            logger.warning("httpx not available, cannot search Twitter")
            return []

        if not self.is_configured:
            logger.warning("Twitter Bearer token not configured, cannot search")
            return []

        # Clamp limit (API max is 100)
        limit = min(limit, 100)

        # Build params with expansions for full data
        params: dict[str, str | int] = {
            "query": query,
            "max_results": max(10, limit),  # API requires min 10
            "sort_order": sort_order,
            "tweet.fields": "created_at,author_id,public_metrics,lang,source",
            "expansions": "author_id",
            "user.fields": "username,name,verified,public_metrics",
        }

        # Rate limiting
        await self._rate_limit()

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(
                    TWITTER_SEARCH_URL,
                    params=params,
                    headers=self._get_headers(),
                )

                if response.status_code == 401:
                    logger.error("Twitter API authentication failed - check TWITTER_BEARER_TOKEN")
                    return []
                elif response.status_code == 429:
                    logger.warning("Twitter API rate limit exceeded")
                    return []

                response.raise_for_status()

            data = response.json()
            results = self._parse_search_results(data)
            logger.info(f"Twitter search '{query[:50]}...' returned {len(results)} results")
            return results[:limit]

        except httpx.TimeoutException:
            logger.warning(f"Twitter search timeout for query: {query}")
            return []
        except httpx.HTTPStatusError as e:
            logger.error(f"Twitter API error: {e.response.status_code}")
            return []
        except Exception as e:
            logger.error(f"Twitter search failed: {e}")
            return []

    async def fetch(self, evidence_id: str) -> Optional[Evidence]:
        """
        Fetch a specific tweet by ID.

        Args:
            evidence_id: Tweet ID (e.g., "twitter:123456" or just "123456")

        Returns:
            Evidence object or None if not found
        """
        # Check cache first
        cached = self._cache_get(evidence_id)
        if cached:
            return cached

        if not HTTPX_AVAILABLE:
            logger.warning("httpx not available, cannot fetch tweet")
            return None

        if not self.is_configured:
            logger.warning("Twitter Bearer token not configured, cannot fetch")
            return None

        # Extract tweet ID
        tweet_id = evidence_id.replace("twitter:", "").strip()

        await self._rate_limit()

        try:
            url = f"{TWITTER_TWEET_URL}/{tweet_id}"
            params = {
                "tweet.fields": "created_at,author_id,public_metrics,lang,source,conversation_id",
                "expansions": "author_id",
                "user.fields": "username,name,verified,public_metrics",
            }

            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(url, params=params, headers=self._get_headers())

                if response.status_code == 404:
                    logger.debug(f"Tweet not found: {tweet_id}")
                    return None

                response.raise_for_status()

            data = response.json()
            evidence = self._parse_tweet(data.get("data", {}), data.get("includes", {}))
            if evidence:
                self._cache_put(evidence_id, evidence)
            return evidence

        except Exception as e:
            logger.error(f"Twitter fetch failed for {evidence_id}: {e}")
            return None

    def _parse_search_results(self, data: dict) -> list[Evidence]:
        """Parse Twitter search API response into Evidence objects."""
        results = []
        tweets = data.get("data", [])
        includes = data.get("includes", {})

        # Build user lookup for author info
        users = {u["id"]: u for u in includes.get("users", [])}

        for tweet in tweets:
            try:
                evidence = self._parse_tweet(tweet, includes, users)
                if evidence:
                    results.append(evidence)
            except Exception as e:
                logger.debug(f"Error parsing tweet: {e}")
                continue

        return results

    def _parse_tweet(
        self,
        tweet: dict,
        includes: dict,
        users: Optional[dict] = None,
    ) -> Optional[Evidence]:
        """Parse a single tweet into Evidence."""
        tweet_id = tweet.get("id")
        if not tweet_id:
            return None

        # Get content
        text = tweet.get("text", "")
        if not text:
            return None

        # Get author info
        author_id = tweet.get("author_id")
        if users is None:
            users = {u["id"]: u for u in includes.get("users", [])}

        author_info = users.get(author_id, {})
        username = author_info.get("username", "unknown")
        author_name = author_info.get("name", username)
        is_verified = author_info.get("verified", False)

        # Author metrics
        author_metrics = author_info.get("public_metrics", {})
        followers_count = author_metrics.get("followers_count", 0)

        # Created time
        created_at = tweet.get("created_at")

        # Engagement metrics
        metrics = tweet.get("public_metrics", {})
        retweet_count = metrics.get("retweet_count", 0)
        reply_count = metrics.get("reply_count", 0)
        like_count = metrics.get("like_count", 0)
        quote_count = metrics.get("quote_count", 0)

        # Total engagement
        total_engagement = retweet_count + reply_count + like_count + quote_count

        # Calculate confidence based on engagement and author credibility
        engagement_score = min(1.0, total_engagement / 10000)  # Normalize
        follower_score = min(1.0, followers_count / 1_000_000)  # Normalize

        # Verified accounts get bonus
        verified_bonus = 0.1 if is_verified else 0

        confidence = (
            self.default_confidence
            + (engagement_score * 0.15)
            + (follower_score * 0.1)
            + verified_bonus
        )

        # Freshness (tweets decay quickly in relevance)
        freshness = self.calculate_freshness(created_at) if created_at else 0.5

        # Authority based on follower count and verification
        if is_verified and followers_count > 100_000:
            authority = 0.7
        elif followers_count > 100_000:
            authority = 0.6
        elif followers_count > 10_000:
            authority = 0.55
        else:
            authority = 0.45

        # Tweet URL
        tweet_url = TWEET_URL_TEMPLATE.format(tweet_id=tweet_id)

        # Build title
        title = f"@{username}: {text[:50]}{'...' if len(text) > 50 else ''}"

        return Evidence(
            id=f"twitter:{tweet_id}",
            source_type=self.source_type,
            source_id=tweet_id,
            content=text,
            title=title,
            created_at=created_at,
            author=f"@{username}",
            url=tweet_url,
            confidence=min(confidence, 0.80),  # Cap lower than Reddit
            freshness=freshness,
            authority=authority,
            metadata={
                "retweet_count": retweet_count,
                "reply_count": reply_count,
                "like_count": like_count,
                "quote_count": quote_count,
                "total_engagement": total_engagement,
                "author_id": author_id,
                "author_name": author_name,
                "username": username,
                "is_verified": is_verified,
                "followers_count": followers_count,
                "lang": tweet.get("lang"),
                "source": tweet.get("source"),
                "conversation_id": tweet.get("conversation_id"),
            },
        )

    async def get_user_tweets(
        self,
        user_id: str,
        limit: int = 10,
        exclude_replies: bool = True,
        exclude_retweets: bool = True,
    ) -> list[Evidence]:
        """
        Get recent tweets from a specific user.

        Args:
            user_id: Twitter user ID (numeric)
            limit: Maximum tweets to return
            exclude_replies: Exclude reply tweets
            exclude_retweets: Exclude retweets

        Returns:
            List of Evidence objects
        """
        if not HTTPX_AVAILABLE:
            logger.warning("httpx not available, cannot get user tweets")
            return []

        if not self.is_configured:
            logger.warning("Twitter Bearer token not configured")
            return []

        limit = min(limit, 100)

        url = TWITTER_USER_TWEETS_URL.format(user_id=user_id)
        params: dict[str, str | int] = {
            "max_results": max(5, limit),
            "tweet.fields": "created_at,author_id,public_metrics,lang,source",
        }

        exclude = []
        if exclude_replies:
            exclude.append("replies")
        if exclude_retweets:
            exclude.append("retweets")
        if exclude:
            params["exclude"] = ",".join(exclude)

        await self._rate_limit()

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(url, params=params, headers=self._get_headers())
                response.raise_for_status()

            data = response.json()
            tweets = data.get("data", [])

            results = []
            for tweet in tweets[:limit]:
                evidence = self._parse_tweet(tweet, {})
                if evidence:
                    results.append(evidence)

            logger.info(f"Twitter user {user_id} returned {len(results)} tweets")
            return results

        except Exception as e:
            logger.error(f"Twitter get_user_tweets failed for {user_id}: {e}")
            return []

    async def search_hashtag(self, hashtag: str, limit: int = 10) -> list[Evidence]:
        """
        Search tweets with a specific hashtag.

        Args:
            hashtag: Hashtag without # (e.g., "ArtificialIntelligence")
            limit: Maximum results

        Returns:
            List of Evidence objects
        """
        # Ensure hashtag has # prefix for search
        if not hashtag.startswith("#"):
            hashtag = f"#{hashtag}"

        return await self.search(hashtag, limit=limit)

    async def search_from_user(
        self,
        username: str,
        query: str = "",
        limit: int = 10,
    ) -> list[Evidence]:
        """
        Search tweets from a specific user, optionally with query.

        Args:
            username: Twitter username (without @)
            query: Optional additional search query
            limit: Maximum results

        Returns:
            List of Evidence objects
        """
        search_query = f"from:{username}"
        if query:
            search_query = f"{query} {search_query}"

        return await self.search(search_query, limit=limit)


__all__ = ["TwitterConnector"]
