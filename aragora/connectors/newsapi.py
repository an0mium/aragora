"""
NewsAPI Connector - News article search for aragora agents.

Provides access to NewsAPI.org for:
- Breaking news headlines
- Article search by keyword, source, or date
- Multi-language news coverage

Requires NEWSAPI_KEY environment variable for API access.
Free tier: 100 requests/day, 1 month historical data.
"""

import asyncio
import logging
import os
from datetime import datetime, timedelta
from typing import Optional
from urllib.parse import quote_plus

from aragora.connectors.base import BaseConnector, Evidence
from aragora.reasoning.provenance import ProvenanceManager, SourceType

logger = logging.getLogger(__name__)

# Try to import optional dependencies
try:
    import httpx

    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False


# NewsAPI endpoints
NEWSAPI_EVERYTHING_URL = "https://newsapi.org/v2/everything"
NEWSAPI_HEADLINES_URL = "https://newsapi.org/v2/top-headlines"
NEWSAPI_SOURCES_URL = "https://newsapi.org/v2/top-headlines/sources"

# Source credibility tiers (subjective but useful for weighting)
HIGH_CREDIBILITY_SOURCES = {
    "reuters", "associated-press", "bbc-news", "the-economist",
    "financial-times", "the-wall-street-journal", "bloomberg",
    "the-washington-post", "the-new-york-times", "npr",
    "abc-news", "cbs-news", "nbc-news", "pbs",
}

MEDIUM_CREDIBILITY_SOURCES = {
    "cnn", "fox-news", "msnbc", "politico", "the-hill",
    "usa-today", "time", "newsweek", "business-insider",
    "techcrunch", "wired", "ars-technica", "the-verge",
    "engadget", "mashable", "vice-news",
}


class NewsAPIConnector(BaseConnector):
    """
    Connector for NewsAPI.org news aggregation service.

    Enables agents to:
    - Search news articles by keyword
    - Get top headlines by country/category
    - Filter by source, language, date range
    - Track news source provenance

    Requires NEWSAPI_KEY environment variable.

    Example:
        connector = NewsAPIConnector()
        if connector.is_configured:
            results = await connector.search("artificial intelligence")
            for evidence in results:
                print(f"{evidence.title} - {evidence.author}")
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        provenance: Optional[ProvenanceManager] = None,
        default_confidence: float = 0.65,
        timeout: int = 30,
        rate_limit_delay: float = 0.5,
        max_cache_entries: int = 500,
        cache_ttl_seconds: float = 1800.0,  # 30 min cache (news changes quickly)
    ):
        """
        Initialize NewsAPIConnector.

        Args:
            api_key: NewsAPI API key. If not provided,
                    uses NEWSAPI_KEY environment variable.
            provenance: Optional provenance manager for tracking
            default_confidence: Base confidence for news sources
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
        self.api_key = api_key or os.environ.get("NEWSAPI_KEY", "")
        self.timeout = timeout
        self.rate_limit_delay = rate_limit_delay
        self._last_request_time: float = 0.0

    @property
    def source_type(self) -> SourceType:
        """News articles are external API data."""
        return SourceType.EXTERNAL_API

    @property
    def name(self) -> str:
        """Human-readable connector name."""
        return "NewsAPI"

    @property
    def is_available(self) -> bool:
        """Check if httpx is available for making requests."""
        return HTTPX_AVAILABLE

    @property
    def is_configured(self) -> bool:
        """Check if NewsAPI key is configured."""
        return bool(self.api_key)

    async def _rate_limit(self) -> None:
        """Enforce rate limiting between requests."""
        import time

        now = time.time()
        elapsed = now - self._last_request_time
        if elapsed < self.rate_limit_delay:
            await asyncio.sleep(self.rate_limit_delay - elapsed)
        self._last_request_time = time.time()

    def _get_headers(self) -> dict:
        """Get headers for NewsAPI requests."""
        return {
            "X-Api-Key": self.api_key,
            "User-Agent": "Aragora/1.0 (debate-platform)",
        }

    async def search(
        self,
        query: str,
        limit: int = 10,
        language: str = "en",
        sort_by: str = "relevancy",
        from_date: Optional[str] = None,
        to_date: Optional[str] = None,
        sources: Optional[list[str]] = None,
        domains: Optional[list[str]] = None,
        **kwargs,
    ) -> list[Evidence]:
        """
        Search news articles matching query.

        Args:
            query: Search query (supports AND, OR, NOT operators)
            limit: Maximum results to return (max 100)
            language: Language code (e.g., "en", "es", "fr")
            sort_by: "relevancy", "popularity", or "publishedAt"
            from_date: Start date (YYYY-MM-DD or ISO 8601)
            to_date: End date (YYYY-MM-DD or ISO 8601)
            sources: List of source IDs to filter by
            domains: List of domains to filter by (e.g., ["bbc.co.uk"])
            **kwargs: Additional API parameters

        Returns:
            List of Evidence objects with article information

        Query operators:
            - "exact phrase" - Exact match
            - keyword1 AND keyword2 - Both required
            - keyword1 OR keyword2 - Either matches
            - keyword1 NOT keyword2 - Exclude second
        """
        if not HTTPX_AVAILABLE:
            logger.warning("httpx not available, cannot search NewsAPI")
            return []

        if not self.is_configured:
            logger.warning("NewsAPI key not configured, cannot search")
            return []

        # Clamp limit (API max is 100)
        limit = min(limit, 100)

        # Build params
        params: dict[str, str | int] = {
            "q": query,
            "pageSize": limit,
            "language": language,
            "sortBy": sort_by,
        }

        if from_date:
            params["from"] = from_date
        if to_date:
            params["to"] = to_date
        if sources:
            params["sources"] = ",".join(sources)
        if domains:
            params["domains"] = ",".join(domains)

        await self._rate_limit()

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(
                    NEWSAPI_EVERYTHING_URL,
                    params=params,
                    headers=self._get_headers(),
                )

                if response.status_code == 401:
                    logger.error("NewsAPI authentication failed - check NEWSAPI_KEY")
                    return []
                elif response.status_code == 426:
                    logger.warning("NewsAPI requires paid plan for this request")
                    return []
                elif response.status_code == 429:
                    logger.warning("NewsAPI rate limit exceeded")
                    return []

                response.raise_for_status()

            data = response.json()

            if data.get("status") != "ok":
                logger.error(f"NewsAPI error: {data.get('message', 'Unknown error')}")
                return []

            results = self._parse_articles(data.get("articles", []))
            logger.info(f"NewsAPI search '{query[:50]}...' returned {len(results)} results")
            return results[:limit]

        except httpx.TimeoutException:
            logger.warning(f"NewsAPI search timeout for query: {query}")
            return []
        except httpx.HTTPStatusError as e:
            logger.error(f"NewsAPI HTTP error: {e.response.status_code}")
            return []
        except Exception as e:
            logger.error(f"NewsAPI search failed: {e}")
            return []

    async def get_headlines(
        self,
        country: str = "us",
        category: Optional[str] = None,
        sources: Optional[list[str]] = None,
        query: Optional[str] = None,
        limit: int = 10,
    ) -> list[Evidence]:
        """
        Get top headlines.

        Args:
            country: 2-letter country code (e.g., "us", "gb", "de")
            category: Category filter: business, entertainment, general,
                     health, science, sports, technology
            sources: Source IDs (cannot be mixed with country/category)
            query: Optional keyword filter
            limit: Maximum results

        Returns:
            List of Evidence objects
        """
        if not HTTPX_AVAILABLE or not self.is_configured:
            return []

        limit = min(limit, 100)

        params: dict[str, str | int] = {"pageSize": limit}

        if sources:
            params["sources"] = ",".join(sources)
        else:
            params["country"] = country
            if category:
                params["category"] = category

        if query:
            params["q"] = query

        await self._rate_limit()

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(
                    NEWSAPI_HEADLINES_URL,
                    params=params,
                    headers=self._get_headers(),
                )
                response.raise_for_status()

            data = response.json()

            if data.get("status") != "ok":
                return []

            results = self._parse_articles(data.get("articles", []))
            logger.info(f"NewsAPI headlines returned {len(results)} results")
            return results

        except Exception as e:
            logger.error(f"NewsAPI headlines failed: {e}")
            return []

    async def fetch(self, evidence_id: str) -> Optional[Evidence]:
        """
        Fetch is not directly supported by NewsAPI.
        Returns cached evidence if available.

        Args:
            evidence_id: Evidence ID (newsapi:hash)

        Returns:
            Cached Evidence or None
        """
        return self._cache_get(evidence_id)

    async def get_sources(
        self,
        category: Optional[str] = None,
        language: str = "en",
        country: Optional[str] = None,
    ) -> list[dict]:
        """
        Get available news sources.

        Args:
            category: Filter by category
            language: Filter by language code
            country: Filter by country code

        Returns:
            List of source dictionaries with id, name, description, etc.
        """
        if not HTTPX_AVAILABLE or not self.is_configured:
            return []

        params: dict[str, str] = {"language": language}
        if category:
            params["category"] = category
        if country:
            params["country"] = country

        await self._rate_limit()

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(
                    NEWSAPI_SOURCES_URL,
                    params=params,
                    headers=self._get_headers(),
                )
                response.raise_for_status()

            data = response.json()

            if data.get("status") != "ok":
                return []

            return data.get("sources", [])

        except Exception as e:
            logger.error(f"NewsAPI get_sources failed: {e}")
            return []

    def _parse_articles(self, articles: list[dict]) -> list[Evidence]:
        """Parse NewsAPI articles into Evidence objects."""
        results = []

        for article in articles:
            try:
                evidence = self._parse_article(article)
                if evidence:
                    results.append(evidence)
                    self._cache_put(evidence.id, evidence)
            except Exception as e:
                logger.debug(f"Error parsing article: {e}")
                continue

        return results

    def _parse_article(self, article: dict) -> Optional[Evidence]:
        """Parse a single article into Evidence."""
        title = article.get("title")
        if not title or title == "[Removed]":
            return None

        # Get source info
        source = article.get("source", {})
        source_id = source.get("id") or "unknown"
        source_name = source.get("name") or "Unknown Source"

        # Content (may be truncated by API)
        content = article.get("content") or article.get("description") or ""
        if not content:
            return None

        # URL for evidence ID
        url = article.get("url", "")
        import hashlib
        evidence_id = f"newsapi:{hashlib.md5(url.encode(), usedforsecurity=False).hexdigest()[:12]}"

        # Published date
        published_at = article.get("publishedAt")

        # Calculate credibility based on source
        confidence, authority = self._calculate_source_credibility(source_id, source_name)

        # Freshness
        freshness = self.calculate_freshness(published_at) if published_at else 0.7

        return Evidence(
            id=evidence_id,
            source_type=self.source_type,
            source_id=url or source_id,
            content=content,
            title=title,
            created_at=published_at,
            author=article.get("author") or source_name,
            url=url,
            confidence=confidence,
            freshness=freshness,
            authority=authority,
            metadata={
                "source_id": source_id,
                "source_name": source_name,
                "description": article.get("description", ""),
                "image_url": article.get("urlToImage"),
                "connector": "newsapi",
            },
        )

    def _calculate_source_credibility(
        self, source_id: str, source_name: str
    ) -> tuple[float, float]:
        """
        Calculate confidence and authority based on news source.

        Returns:
            Tuple of (confidence, authority)
        """
        source_lower = (source_id or "").lower()
        name_lower = (source_name or "").lower()

        # Check against known credibility tiers
        if source_lower in HIGH_CREDIBILITY_SOURCES or any(
            s in name_lower for s in ["reuters", "associated press", "bbc", "npr"]
        ):
            return (0.80, 0.85)

        if source_lower in MEDIUM_CREDIBILITY_SOURCES or any(
            s in name_lower for s in ["cnn", "techcrunch", "wired", "verge"]
        ):
            return (0.70, 0.70)

        # Default for unknown sources
        return (self.default_confidence, 0.55)

    async def search_recent(
        self,
        query: str,
        days: int = 7,
        limit: int = 10,
    ) -> list[Evidence]:
        """
        Search articles from the last N days.

        Args:
            query: Search query
            days: Number of days to look back (max 30 for free tier)
            limit: Maximum results

        Returns:
            List of Evidence objects
        """
        from_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        return await self.search(
            query,
            limit=limit,
            from_date=from_date,
            sort_by="publishedAt",
        )

    async def search_tech_news(
        self,
        query: str = "",
        limit: int = 10,
    ) -> list[Evidence]:
        """
        Search tech-focused news sources.

        Args:
            query: Optional search query
            limit: Maximum results

        Returns:
            List of Evidence objects from tech sources
        """
        tech_sources = [
            "techcrunch", "wired", "ars-technica", "the-verge",
            "engadget", "hacker-news", "recode",
        ]

        if query:
            return await self.search(query, limit=limit, sources=tech_sources)
        else:
            return await self.get_headlines(sources=tech_sources, limit=limit)


__all__ = [
    "NewsAPIConnector",
    "HIGH_CREDIBILITY_SOURCES",
    "MEDIUM_CREDIBILITY_SOURCES",
]
