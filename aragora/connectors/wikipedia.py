"""
Wikipedia Connector - Encyclopedia search for aragora agents.

Provides access to Wikipedia's knowledge base for:
- Article search and summaries
- Reference information
- Factual grounding

The Wikipedia API is free and requires no authentication.
"""

import asyncio
import hashlib
import logging
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


# Wikipedia API endpoints
WIKIPEDIA_API_URL = "https://en.wikipedia.org/w/api.php"
WIKIPEDIA_REST_URL = "https://en.wikipedia.org/api/rest_v1"


class WikipediaConnector(BaseConnector):
    """
    Connector for Wikipedia encyclopedia.

    Enables agents to:
    - Search for encyclopedia articles
    - Retrieve article summaries
    - Access factual reference information
    - Track knowledge source provenance

    Example:
        connector = WikipediaConnector()
        results = await connector.search("machine learning")
        for evidence in results:
            print(f"{evidence.title}")
            print(f"  {evidence.content[:200]}...")
    """

    def __init__(
        self,
        provenance: Optional[ProvenanceManager] = None,
        default_confidence: float = 0.75,  # Wikipedia is generally reliable but crowd-sourced
        timeout: int = 30,
        rate_limit_delay: float = 0.5,  # Wikipedia is generous with rate limits
        max_cache_entries: int = 500,
        cache_ttl_seconds: float = 3600.0,  # 1 hour cache (Wikipedia can change)
        language: str = "en",
    ):
        """
        Initialize WikipediaConnector.

        Args:
            provenance: Optional provenance manager for tracking
            default_confidence: Base confidence for Wikipedia sources
            timeout: HTTP request timeout in seconds
            rate_limit_delay: Delay between API requests
            max_cache_entries: Maximum cached entries
            cache_ttl_seconds: Cache TTL in seconds
            language: Wikipedia language code (default: "en")
        """
        super().__init__(
            provenance=provenance,
            default_confidence=default_confidence,
            max_cache_entries=max_cache_entries,
            cache_ttl_seconds=cache_ttl_seconds,
        )
        self.timeout = timeout
        self.rate_limit_delay = rate_limit_delay
        self.language = language
        self._last_request_time: float = 0.0

        # Update URLs for language
        self.api_url = f"https://{language}.wikipedia.org/w/api.php"
        self.rest_url = f"https://{language}.wikipedia.org/api/rest_v1"

    @property
    def source_type(self) -> SourceType:
        """Wikipedia is document-based knowledge."""
        return SourceType.DOCUMENT

    @property
    def name(self) -> str:
        """Human-readable connector name."""
        return "Wikipedia"

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
        include_summary: bool = True,
        **kwargs,
    ) -> list[Evidence]:
        """
        Search Wikipedia for articles matching query.

        Args:
            query: Search query
            limit: Maximum results to return (max 50)
            include_summary: Whether to fetch article summaries
            **kwargs: Additional parameters

        Returns:
            List of Evidence objects with article information
        """
        if not HTTPX_AVAILABLE:
            logger.warning("httpx not available, cannot search Wikipedia")
            return []

        # Clamp limit
        limit = min(limit, 50)

        # Use opensearch API for quick search
        params: dict[str, str | int] = {
            "action": "opensearch",
            "search": query,
            "limit": limit,
            "namespace": 0,  # Main namespace only
            "format": "json",
        }

        await self._rate_limit()

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(self.api_url, params=params)
                response.raise_for_status()
                data = response.json()

            # opensearch returns [query, [titles], [descriptions], [urls]]
            if len(data) < 4:
                return []

            titles = data[1]
            descriptions = data[2]
            urls = data[3]

            results = []
            for i, title in enumerate(titles):
                # Generate evidence ID from title
                evidence_id = f"wiki:{hashlib.md5(title.encode(), usedforsecurity=False).hexdigest()[:12]}"

                # Get description if available
                description = descriptions[i] if i < len(descriptions) else ""
                url = (
                    urls[i]
                    if i < len(urls)
                    else f"https://{self.language}.wikipedia.org/wiki/{quote_plus(title.replace(' ', '_'))}"
                )

                # If include_summary and description is empty, fetch summary
                content = description
                if include_summary and not description:
                    summary = await self._fetch_summary(title)
                    if summary:
                        content = summary

                evidence = Evidence(
                    id=evidence_id,
                    source_type=self.source_type,
                    source_id=title,
                    content=content or f"Wikipedia article: {title}",
                    title=title,
                    url=url,
                    author="Wikipedia contributors",
                    confidence=self.default_confidence,
                    freshness=0.8,  # Wikipedia is regularly updated
                    authority=0.7,  # Collaborative encyclopedia
                    metadata={
                        "language": self.language,
                        "connector": "wikipedia",
                    },
                )

                results.append(evidence)
                self._cache_put(evidence_id, evidence)

            logger.info(f"Wikipedia search '{query}' returned {len(results)} results")
            return results

        except httpx.TimeoutException:
            logger.warning(f"Wikipedia search timeout for query: {query}")
            return []
        except httpx.HTTPStatusError as e:
            logger.error(f"Wikipedia API error: {e.response.status_code}")
            return []
        except Exception as e:
            logger.error(f"Wikipedia search failed: {e}")
            return []

    async def fetch(self, evidence_id: str) -> Optional[Evidence]:
        """
        Fetch a specific article by title or evidence ID.

        Args:
            evidence_id: Wikipedia article title or evidence ID

        Returns:
            Evidence object or None if not found
        """
        # Check cache first
        cached = self._cache_get(evidence_id)
        if cached:
            return cached

        if not HTTPX_AVAILABLE:
            logger.warning("httpx not available, cannot fetch Wikipedia article")
            return None

        # Handle different ID formats
        title = evidence_id
        if evidence_id.startswith("wiki:"):
            # Need to search for it
            return None

        await self._rate_limit()

        try:
            # Use the REST API for detailed article info
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                # Get page summary
                summary_url = f"{self.rest_url}/page/summary/{quote_plus(title.replace(' ', '_'))}"
                response = await client.get(summary_url)

                if response.status_code == 404:
                    logger.info(f"Wikipedia article not found: {title}")
                    return None

                response.raise_for_status()
                data = response.json()

            # Parse response
            evidence = self._parse_summary_response(data)
            if evidence:
                self._cache_put(evidence_id, evidence)
            return evidence

        except httpx.TimeoutException:
            logger.warning(f"Wikipedia fetch timeout for: {title}")
            return None
        except Exception as e:
            logger.error(f"Wikipedia fetch failed for {title}: {e}")
            return None

    async def _fetch_summary(self, title: str) -> Optional[str]:
        """Fetch article summary using REST API."""
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                summary_url = f"{self.rest_url}/page/summary/{quote_plus(title.replace(' ', '_'))}"
                response = await client.get(summary_url)

                if response.status_code == 200:
                    data = response.json()
                    return data.get("extract", "")
                return None
        except httpx.TimeoutException:
            logger.debug(f"Timeout fetching Wikipedia summary for: {title}")
            return None
        except httpx.HTTPError as e:
            logger.debug(f"HTTP error fetching Wikipedia summary: {e}")
            return None
        except (ValueError, KeyError) as e:
            logger.debug(f"Parse error in Wikipedia summary: {e}")
            return None

    def _parse_summary_response(self, data: dict) -> Optional[Evidence]:
        """Parse Wikipedia REST API summary response."""
        try:
            title = data.get("title", "Unknown")
            extract = data.get("extract", "")
            page_id = data.get("pageid", 0)

            # Generate evidence ID
            evidence_id = (
                f"wiki:{page_id}"
                if page_id
                else f"wiki:{hashlib.md5(title.encode(), usedforsecurity=False).hexdigest()[:12]}"
            )

            # Get URL
            url = data.get("content_urls", {}).get("desktop", {}).get("page", "")
            if not url:
                url = f"https://{self.language}.wikipedia.org/wiki/{quote_plus(title.replace(' ', '_'))}"

            # Get timestamp if available
            timestamp = data.get("timestamp")
            created_at = None
            freshness = 0.8
            if timestamp:
                try:
                    created_at = timestamp
                    freshness = self.calculate_freshness(timestamp)
                except (ValueError, TypeError) as e:
                    logger.debug(f"Could not parse Wikipedia timestamp: {e}")
                    # Keep default freshness

            return Evidence(
                id=evidence_id,
                source_type=self.source_type,
                source_id=title,
                content=extract or f"Wikipedia article: {title}",
                title=title,
                url=url,
                author="Wikipedia contributors",
                created_at=created_at,
                confidence=self.default_confidence,
                freshness=freshness,
                authority=0.7,
                metadata={
                    "page_id": page_id,
                    "language": self.language,
                    "connector": "wikipedia",
                    "description": data.get("description", ""),
                },
            )
        except Exception as e:
            logger.error(f"Failed to parse Wikipedia response: {e}")
            return None

    async def get_article_sections(self, title: str) -> list[dict]:
        """
        Get the section structure of an article.

        Args:
            title: Article title

        Returns:
            List of section dictionaries with titles and content
        """
        if not HTTPX_AVAILABLE:
            return []

        await self._rate_limit()

        try:
            params = {
                "action": "parse",
                "page": title,
                "prop": "sections",
                "format": "json",
            }

            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(self.api_url, params=params)
                response.raise_for_status()
                data = response.json()

            parse_data = data.get("parse", {})
            sections = parse_data.get("sections", [])

            return [
                {
                    "index": s.get("index"),
                    "level": s.get("level"),
                    "title": s.get("line"),
                    "anchor": s.get("anchor"),
                }
                for s in sections
            ]
        except Exception as e:
            logger.error(f"Failed to get sections for {title}: {e}")
            return []

    async def get_related_articles(self, title: str, limit: int = 10) -> list[str]:
        """
        Get articles related to the given article (via links).

        Args:
            title: Article title
            limit: Maximum related articles to return

        Returns:
            List of related article titles
        """
        if not HTTPX_AVAILABLE:
            return []

        await self._rate_limit()

        try:
            params: dict[str, str | int] = {
                "action": "query",
                "titles": title,
                "prop": "links",
                "pllimit": limit,
                "plnamespace": 0,  # Main namespace
                "format": "json",
            }

            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(self.api_url, params=params)
                response.raise_for_status()
                data = response.json()

            pages = data.get("query", {}).get("pages", {})
            for page_data in pages.values():
                links = page_data.get("links", [])
                return [link.get("title") for link in links if link.get("title")]

            return []
        except Exception as e:
            logger.error(f"Failed to get related articles for {title}: {e}")
            return []
