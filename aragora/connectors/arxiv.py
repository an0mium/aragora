"""
ArXiv Connector - Academic paper search for aragora agents.

Provides access to ArXiv's preprint repository for:
- Scientific paper search
- Abstract and metadata retrieval
- Author and citation information

The ArXiv API is free and requires no authentication.
"""

import asyncio
import logging
import re
import xml.etree.ElementTree as ET
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


# ArXiv API endpoint
ARXIV_API_URL = "http://export.arxiv.org/api/query"

# ArXiv category codes for filtering
ARXIV_CATEGORIES = {
    "cs": "Computer Science",
    "cs.AI": "Artificial Intelligence",
    "cs.CL": "Computation and Language",
    "cs.CV": "Computer Vision",
    "cs.LG": "Machine Learning",
    "cs.NE": "Neural and Evolutionary Computing",
    "stat.ML": "Machine Learning (Statistics)",
    "math": "Mathematics",
    "physics": "Physics",
    "q-bio": "Quantitative Biology",
    "q-fin": "Quantitative Finance",
    "econ": "Economics",
    "eess": "Electrical Engineering and Systems Science",
}


class ArXivConnector(BaseConnector):
    """
    Connector for ArXiv preprint repository.

    Enables agents to:
    - Search for academic papers by query
    - Filter by category (cs.AI, cs.CL, etc.)
    - Retrieve paper abstracts and metadata
    - Track academic source provenance

    Example:
        connector = ArXivConnector()
        results = await connector.search("transformer attention mechanism")
        for evidence in results:
            print(f"{evidence.title} by {evidence.author}")
            print(f"  {evidence.content[:200]}...")
    """

    def __init__(
        self,
        provenance: Optional[ProvenanceManager] = None,
        default_confidence: float = 0.85,
        timeout: int = 30,
        rate_limit_delay: float = 3.0,  # ArXiv recommends 3s between requests
        max_cache_entries: int = 500,
        cache_ttl_seconds: float = 86400.0,  # 24 hour cache for papers
    ):
        """
        Initialize ArXivConnector.

        Args:
            provenance: Optional provenance manager for tracking
            default_confidence: Base confidence for ArXiv sources (high by default)
            timeout: HTTP request timeout in seconds
            rate_limit_delay: Delay between API requests (ArXiv recommends 3s)
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
        """ArXiv papers are external API data."""
        return SourceType.EXTERNAL_API

    @property
    def name(self) -> str:
        """Human-readable connector name."""
        return "ArXiv"

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
        category: Optional[str] = None,
        sort_by: str = "relevance",
        sort_order: str = "descending",
        **kwargs,
    ) -> list[Evidence]:
        """
        Search ArXiv for papers matching query.

        Args:
            query: Search query (supports ArXiv query syntax)
            limit: Maximum results to return (max 100)
            category: Optional category filter (e.g., "cs.AI", "cs.CL")
            sort_by: Sort method - "relevance", "lastUpdatedDate", "submittedDate"
            sort_order: "ascending" or "descending"
            **kwargs: Additional ArXiv API parameters

        Returns:
            List of Evidence objects with paper abstracts

        ArXiv Query Syntax Examples:
            - Simple: "transformer attention"
            - Author: "au:Vaswani"
            - Title: "ti:attention"
            - Abstract: "abs:neural network"
            - Category: "cat:cs.AI"
            - Combined: "all:transformer AND au:Vaswani"
        """
        if not HTTPX_AVAILABLE:
            logger.warning("httpx not available, cannot search ArXiv")
            return []

        # Clamp limit to ArXiv max
        limit = min(limit, 100)

        # Build search query
        search_query = query
        if category:
            search_query = f"cat:{category} AND ({query})"

        # Construct API URL
        params: dict[str, str | int] = {
            "search_query": search_query,
            "start": 0,
            "max_results": limit,
            "sortBy": sort_by,
            "sortOrder": sort_order,
        }

        # Rate limiting
        await self._rate_limit()

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(ARXIV_API_URL, params=params)
                response.raise_for_status()

            # Parse XML response
            results = self._parse_arxiv_response(response.text)
            logger.info(f"ArXiv search '{query}' returned {len(results)} results")
            return results

        except httpx.TimeoutException:
            logger.warning(f"ArXiv search timeout for query: {query}")
            return []
        except httpx.HTTPStatusError as e:
            logger.error(f"ArXiv API error: {e.response.status_code}")
            return []
        except httpx.ConnectError as e:
            logger.error(f"ArXiv connection error: {e}")
            return []
        except httpx.RequestError as e:
            logger.error(f"ArXiv request error: {e}")
            return []
        except ET.ParseError as e:
            logger.error(f"ArXiv XML parsing error: {e}")
            return []
        except Exception as e:
            logger.error(f"ArXiv search failed unexpectedly ({type(e).__name__}): {e}")
            return []

    async def fetch(self, evidence_id: str) -> Optional[Evidence]:
        """
        Fetch a specific paper by ArXiv ID.

        Args:
            evidence_id: ArXiv paper ID (e.g., "2309.12345", "1706.03762")

        Returns:
            Evidence object or None if not found
        """
        # Check cache first
        cached = self._cache_get(evidence_id)
        if cached:
            return cached

        if not HTTPX_AVAILABLE:
            logger.warning("httpx not available, cannot fetch ArXiv paper")
            return None

        # ArXiv IDs can be with or without version
        arxiv_id = evidence_id.replace("arxiv:", "").replace("arXiv:", "")

        # Build query for specific paper
        params: dict[str, str | int] = {
            "id_list": arxiv_id,
            "max_results": 1,
        }

        await self._rate_limit()

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(ARXIV_API_URL, params=params)
                response.raise_for_status()

            results = self._parse_arxiv_response(response.text)
            if results:
                evidence = results[0]
                self._cache_put(evidence_id, evidence)
                return evidence
            return None

        except httpx.TimeoutException:
            logger.warning(f"ArXiv fetch timeout for {evidence_id}")
            return None
        except httpx.HTTPStatusError as e:
            logger.error(f"ArXiv API error fetching {evidence_id}: {e.response.status_code}")
            return None
        except httpx.ConnectError as e:
            logger.error(f"ArXiv connection error for {evidence_id}: {e}")
            return None
        except httpx.RequestError as e:
            logger.error(f"ArXiv request error for {evidence_id}: {e}")
            return None
        except ET.ParseError as e:
            logger.error(f"ArXiv XML parsing error for {evidence_id}: {e}")
            return None
        except Exception as e:
            logger.error(f"ArXiv fetch failed unexpectedly for {evidence_id} ({type(e).__name__}): {e}")
            return None

    def _parse_arxiv_response(self, xml_text: str) -> list[Evidence]:
        """Parse ArXiv Atom XML response into Evidence objects."""
        results = []

        try:
            # ArXiv uses Atom namespace
            namespaces = {
                "atom": "http://www.w3.org/2005/Atom",
                "arxiv": "http://arxiv.org/schemas/atom",
            }

            root = ET.fromstring(xml_text)

            for entry in root.findall("atom:entry", namespaces):
                try:
                    evidence = self._parse_entry(entry, namespaces)
                    if evidence:
                        results.append(evidence)
                except Exception as e:
                    logger.debug(f"Error parsing ArXiv entry: {e}")
                    continue

        except ET.ParseError as e:
            logger.error(f"Failed to parse ArXiv XML: {e}")

        return results

    def _parse_entry(self, entry: ET.Element, namespaces: dict) -> Optional[Evidence]:
        """Parse a single ArXiv entry into Evidence."""
        # Extract ArXiv ID from entry ID URL
        entry_id = entry.find("atom:id", namespaces)
        if entry_id is None or entry_id.text is None:
            return None

        # ArXiv ID URL format: http://arxiv.org/abs/2309.12345v1
        arxiv_url = entry_id.text
        arxiv_id = arxiv_url.split("/")[-1]  # Extract ID from URL

        # Title
        title_elem = entry.find("atom:title", namespaces)
        title = (
            title_elem.text.strip() if title_elem is not None and title_elem.text else "Untitled"
        )
        # Clean up whitespace in title
        title = re.sub(r"\s+", " ", title)

        # Abstract (summary)
        summary_elem = entry.find("atom:summary", namespaces)
        abstract = (
            summary_elem.text.strip() if summary_elem is not None and summary_elem.text else ""
        )
        # Clean up whitespace in abstract
        abstract = re.sub(r"\s+", " ", abstract)

        # Authors
        authors = []
        for author in entry.findall("atom:author", namespaces):
            name_elem = author.find("atom:name", namespaces)
            if name_elem is not None and name_elem.text:
                authors.append(name_elem.text)
        author_str = ", ".join(authors[:5])  # Limit to first 5 authors
        if len(authors) > 5:
            author_str += f" et al. ({len(authors)} authors)"

        # Published date
        published_elem = entry.find("atom:published", namespaces)
        published = (
            published_elem.text if published_elem is not None and published_elem.text else None
        )

        # Categories
        categories = []
        for category in entry.findall("arxiv:primary_category", namespaces):
            term = category.get("term")
            if term:
                categories.append(term)
        for category in entry.findall("atom:category", namespaces):
            term = category.get("term")
            if term and term not in categories:
                categories.append(term)

        # PDF link
        pdf_link = None
        for link in entry.findall("atom:link", namespaces):
            if link.get("title") == "pdf":
                pdf_link = link.get("href")
                break

        # Create evidence ID
        evidence_id = f"arxiv:{arxiv_id}"

        # Calculate freshness
        freshness = self.calculate_freshness(published) if published else 0.7

        return Evidence(
            id=evidence_id,
            source_type=self.source_type,
            source_id=arxiv_id,
            content=abstract,
            title=title,
            created_at=published,
            author=author_str,
            url=arxiv_url,
            confidence=self.default_confidence,
            freshness=freshness,
            authority=0.9,  # Academic preprints have high authority
            metadata={
                "categories": categories,
                "pdf_url": pdf_link,
                "arxiv_id": arxiv_id,
                "author_count": len(authors),
                "all_authors": authors,
            },
        )

    async def search_by_author(
        self,
        author: str,
        limit: int = 10,
    ) -> list[Evidence]:
        """
        Search for papers by a specific author.

        Args:
            author: Author name to search for
            limit: Maximum results

        Returns:
            List of Evidence objects
        """
        query = f"au:{quote_plus(author)}"
        return await self.search(query, limit=limit, sort_by="submittedDate")

    async def search_recent(
        self,
        category: str = "cs.AI",
        limit: int = 10,
    ) -> list[Evidence]:
        """
        Get recent papers in a category.

        Args:
            category: ArXiv category code
            limit: Maximum results

        Returns:
            List of Evidence objects sorted by submission date
        """
        # Use wildcard query to get all papers in category
        return await self.search(
            "*",
            limit=limit,
            category=category,
            sort_by="submittedDate",
            sort_order="descending",
        )

    def get_categories(self) -> dict[str, str]:
        """Return available ArXiv category codes and descriptions."""
        return ARXIV_CATEGORIES.copy()


__all__ = ["ArXivConnector", "ARXIV_CATEGORIES"]
