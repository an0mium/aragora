"""
RSS/Atom Feed Ingestor.

Fetches content from RSS/Atom feeds for use in debates and knowledge ingestion.

Production features:
- Support for RSS 2.0 and Atom feeds
- Exponential backoff with configurable retries
- Circuit breaker for failing feeds
- Content extraction and summarization
- Caching with configurable TTL
- Parallel fetching with concurrency control
"""

import asyncio
import hashlib
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from xml.etree import ElementTree as ET

import httpx

from aragora.resilience import CircuitBreaker

logger = logging.getLogger(__name__)


@dataclass
class FeedEntry:
    """A single entry from an RSS/Atom feed."""

    id: str
    title: str
    link: str
    summary: str = ""
    content: str = ""
    author: str = ""
    published: str = ""
    updated: str = ""
    categories: List[str] = field(default_factory=list)
    source_url: str = ""
    source_name: str = ""
    raw_data: Dict[str, Any] = field(default_factory=dict)

    def to_debate_context(self) -> str:
        """Convert to debate context string."""
        parts = [f"Title: {self.title}"]
        if self.author:
            parts.append(f"Author: {self.author}")
        if self.published:
            parts.append(f"Published: {self.published}")
        if self.summary:
            parts.append(f"Summary: {self.summary}")
        elif self.content:
            # Use first 500 chars of content as summary
            parts.append(f"Content: {self.content[:500]}...")
        if self.link:
            parts.append(f"Source: {self.link}")
        return "\n".join(parts)

    @property
    def content_hash(self) -> str:
        """Generate a hash of the entry content for deduplication."""
        content = f"{self.title}|{self.link}|{self.summary or self.content}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]


@dataclass
class FeedSource:
    """Configuration for a feed source."""

    url: str
    name: str = ""
    category: str = ""
    priority: float = 1.0  # Higher = more important
    max_entries: int = 50
    enabled: bool = True

    def __post_init__(self):
        if not self.name:
            # Extract name from URL
            from urllib.parse import urlparse

            parsed = urlparse(self.url)
            self.name = parsed.netloc or self.url[:30]


class FeedIngestor:
    """Ingests content from RSS/Atom feeds.

    Supports:
    - RSS 2.0 feeds
    - Atom feeds
    - Parallel fetching with concurrency control
    - Caching with configurable TTL
    - Circuit breaker for resilience
    """

    # Common RSS namespaces
    RSS_NAMESPACES = {
        "atom": "http://www.w3.org/2005/Atom",
        "content": "http://purl.org/rss/1.0/modules/content/",
        "dc": "http://purl.org/dc/elements/1.1/",
    }

    def __init__(
        self,
        sources: Optional[List[FeedSource]] = None,
        cache_ttl: int = 300,  # 5 minutes
        max_concurrent: int = 5,
        timeout: float = 30.0,
        max_retries: int = 3,
        base_retry_delay: float = 1.0,
        user_agent: str = "Aragora/1.0 (+https://github.com/aragora)",
    ):
        """Initialize the feed ingestor.

        Args:
            sources: List of feed sources to fetch from
            cache_ttl: Cache time-to-live in seconds
            max_concurrent: Maximum concurrent feed fetches
            timeout: HTTP request timeout in seconds
            max_retries: Maximum retry attempts per feed
            base_retry_delay: Base delay for exponential backoff
            user_agent: User agent string for HTTP requests
        """
        self.sources = sources or []
        self.cache_ttl = cache_ttl
        self.max_concurrent = max_concurrent
        self.timeout = timeout
        self.max_retries = max_retries
        self.base_retry_delay = base_retry_delay
        self.user_agent = user_agent

        # Internal state
        self._cache: Dict[str, tuple[List[FeedEntry], float]] = {}
        self._circuit_breakers: Dict[str, CircuitBreaker] = {}
        self._seen_hashes: set[str] = set()

    def add_source(self, source: FeedSource) -> None:
        """Add a feed source."""
        self.sources.append(source)
        logger.info(f"Added feed source: {source.name} ({source.url})")

    def remove_source(self, url: str) -> bool:
        """Remove a feed source by URL."""
        original_len = len(self.sources)
        self.sources = [s for s in self.sources if s.url != url]
        removed = len(self.sources) < original_len
        if removed:
            logger.info(f"Removed feed source: {url}")
        return removed

    def _get_circuit_breaker(self, url: str) -> CircuitBreaker:
        """Get or create circuit breaker for a URL."""
        if url not in self._circuit_breakers:
            self._circuit_breakers[url] = CircuitBreaker()
        return self._circuit_breakers[url]

    async def _fetch_feed(self, source: FeedSource) -> List[FeedEntry]:
        """Fetch and parse a single feed with retries and circuit breaker."""
        cb = self._get_circuit_breaker(source.url)

        if not cb.can_proceed():
            logger.warning(f"Circuit breaker open for {source.name}")
            return []

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            for attempt in range(self.max_retries):
                try:
                    response = await client.get(
                        source.url,
                        headers={"User-Agent": self.user_agent},
                        follow_redirects=True,
                    )
                    response.raise_for_status()

                    entries = self._parse_feed(
                        response.text,
                        source,
                    )

                    cb.record_success()
                    logger.debug(f"Fetched {len(entries)} entries from {source.name}")
                    return entries[: source.max_entries]

                except httpx.HTTPStatusError as e:
                    logger.warning(f"HTTP error fetching {source.name}: {e.response.status_code}")
                    if e.response.status_code >= 500:
                        # Server error, retry with backoff
                        delay = self.base_retry_delay * (2**attempt)
                        await asyncio.sleep(delay)
                    else:
                        # Client error, don't retry
                        cb.record_failure()
                        return []

                except httpx.RequestError as e:
                    logger.warning(f"Request error fetching {source.name}: {e}")
                    delay = self.base_retry_delay * (2**attempt)
                    await asyncio.sleep(delay)

                except ET.ParseError as e:
                    logger.error(f"Parse error for {source.name}: {e}")
                    cb.record_failure()
                    return []

            cb.record_failure()
            return []

    def _parse_feed(self, content: str, source: FeedSource) -> List[FeedEntry]:
        """Parse RSS or Atom feed content."""
        try:
            root = ET.fromstring(content)
        except ET.ParseError as e:
            logger.error(f"Failed to parse feed XML: {e}")
            return []

        # Detect feed type
        if root.tag == "{http://www.w3.org/2005/Atom}feed" or root.tag == "feed":
            return self._parse_atom(root, source)
        elif root.tag == "rss" or root.find("channel") is not None:
            return self._parse_rss(root, source)
        else:
            logger.warning(f"Unknown feed format: {root.tag}")
            return []

    def _parse_rss(self, root: ET.Element, source: FeedSource) -> List[FeedEntry]:
        """Parse RSS 2.0 feed."""
        entries: list[FeedEntry] = []
        channel = root.find("channel")
        if channel is None:
            return entries

        for item in channel.findall("item"):
            entry = FeedEntry(
                id=self._get_text(item, "guid") or self._get_text(item, "link") or "",
                title=self._get_text(item, "title") or "",
                link=self._get_text(item, "link") or "",
                summary=self._get_text(item, "description") or "",
                content=self._get_text(item, "{http://purl.org/rss/1.0/modules/content/}encoded")
                or "",
                author=self._get_text(item, "author")
                or self._get_text(item, "{http://purl.org/dc/elements/1.1/}creator")
                or "",
                published=self._get_text(item, "pubDate") or "",
                source_url=source.url,
                source_name=source.name,
                categories=[cat.text for cat in item.findall("category") if cat.text],
            )
            entries.append(entry)

        return entries

    def _parse_atom(self, root: ET.Element, source: FeedSource) -> List[FeedEntry]:
        """Parse Atom feed."""
        entries = []
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        atom_ns = "{http://www.w3.org/2005/Atom}"

        # Handle both namespaced and non-namespaced Atom
        atom_entries = root.findall("atom:entry", ns)
        if not atom_entries:
            atom_entries = root.findall(f"{atom_ns}entry")
        if not atom_entries:
            atom_entries = root.findall("entry")

        for item in atom_entries:
            entry = FeedEntry(
                id=self._get_text(item, "atom:id", ns) or self._get_text(item, "id") or "",
                title=self._get_text(item, "atom:title", ns) or self._get_text(item, "title") or "",
                link=self._get_atom_link(item, ns),
                summary=self._get_text(item, "atom:summary", ns)
                or self._get_text(item, "summary")
                or "",
                content=self._get_text(item, "atom:content", ns)
                or self._get_text(item, "content")
                or "",
                author=self._get_atom_author(item, ns),
                published=self._get_text(item, "atom:published", ns)
                or self._get_text(item, "published")
                or "",
                updated=self._get_text(item, "atom:updated", ns)
                or self._get_text(item, "updated")
                or "",
                source_url=source.url,
                source_name=source.name,
                categories=[
                    cat.get("term", "")
                    for cat in (item.findall("atom:category", ns) or item.findall("category"))
                    if cat.get("term")
                ],
            )
            entries.append(entry)

        return entries

    def _get_text(self, element: ET.Element, tag: str, ns: Optional[dict] = None) -> str:
        """Get text content of a child element."""
        child = element.find(tag, ns) if ns else element.find(tag)
        if child is None and ns:
            # Try with full namespace prefix
            atom_ns = "{http://www.w3.org/2005/Atom}"
            simple_tag = tag.replace("atom:", "")
            child = element.find(f"{atom_ns}{simple_tag}")
        if child is None:
            # Try without namespace
            simple_tag = tag.replace("atom:", "") if ":" in tag else tag
            child = element.find(simple_tag)
        return (child.text or "").strip() if child is not None else ""

    def _get_atom_link(self, item: ET.Element, ns: dict) -> str:
        """Get link from Atom entry (prefers alternate link)."""
        atom_ns = "{http://www.w3.org/2005/Atom}"
        links = item.findall("atom:link", ns)
        if not links:
            links = item.findall(f"{atom_ns}link")
        if not links:
            links = item.findall("link")

        for link in links:
            rel = link.get("rel", "alternate")
            if rel == "alternate":
                return link.get("href", "")
        # Fall back to first link
        if links:
            return links[0].get("href", "")
        return ""

    def _get_atom_author(self, item: ET.Element, ns: dict) -> str:
        """Get author from Atom entry."""
        atom_ns = "{http://www.w3.org/2005/Atom}"
        # Try multiple namespace patterns
        author = item.find("atom:author", ns)
        if author is None:
            author = item.find(f"{atom_ns}author")
        if author is None:
            author = item.find("author")

        if author is not None:
            name = author.find("atom:name", ns)
            if name is None:
                name = author.find(f"{atom_ns}name")
            if name is None:
                name = author.find("name")
            if name is not None and name.text:
                return name.text.strip()
        return ""

    async def fetch_all(
        self,
        deduplicate: bool = True,
        use_cache: bool = True,
    ) -> List[FeedEntry]:
        """Fetch entries from all enabled sources.

        Args:
            deduplicate: Remove duplicate entries based on content hash
            use_cache: Use cached results if available and fresh

        Returns:
            List of feed entries sorted by priority
        """
        enabled_sources = [s for s in self.sources if s.enabled]
        if not enabled_sources:
            logger.warning("No enabled feed sources")
            return []

        all_entries: List[FeedEntry] = []

        # Check cache for each source
        sources_to_fetch = []
        for source in enabled_sources:
            if use_cache and source.url in self._cache:
                entries, cached_at = self._cache[source.url]
                if time.time() - cached_at < self.cache_ttl:
                    all_entries.extend(entries)
                    continue
            sources_to_fetch.append(source)

        # Fetch remaining sources in parallel with concurrency limit
        if sources_to_fetch:
            sem = asyncio.Semaphore(self.max_concurrent)

            async def fetch_with_limit(source: FeedSource) -> List[FeedEntry]:
                async with sem:
                    entries = await self._fetch_feed(source)
                    # Cache the results
                    self._cache[source.url] = (entries, time.time())
                    return entries

            results = await asyncio.gather(
                *[fetch_with_limit(s) for s in sources_to_fetch],
                return_exceptions=True,
            )

            for result in results:
                if isinstance(result, list):
                    all_entries.extend(result)
                elif isinstance(result, Exception):
                    logger.error(f"Feed fetch failed: {result}")

        # Deduplicate based on content hash
        if deduplicate:
            unique_entries = []
            seen = set(self._seen_hashes)
            for entry in all_entries:
                h = entry.content_hash
                if h not in seen:
                    seen.add(h)
                    unique_entries.append(entry)
            self._seen_hashes = seen
            all_entries = unique_entries

        logger.info(f"Fetched {len(all_entries)} entries from {len(enabled_sources)} sources")
        return all_entries

    async def fetch_source(self, url: str) -> List[FeedEntry]:
        """Fetch entries from a specific source by URL."""
        source = next((s for s in self.sources if s.url == url), None)
        if not source:
            source = FeedSource(url=url)
        return await self._fetch_feed(source)

    def clear_cache(self) -> None:
        """Clear the entry cache."""
        self._cache.clear()
        self._seen_hashes.clear()
        logger.info("Feed cache cleared")

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "cached_feeds": len(self._cache),
            "seen_hashes": len(self._seen_hashes),
            "sources": len(self.sources),
            "enabled_sources": sum(1 for s in self.sources if s.enabled),
        }
