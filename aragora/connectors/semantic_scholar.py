"""
Semantic Scholar Connector - Academic paper search for Aragora agents.

Uses Semantic Scholar Graph API to fetch paper metadata.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
from typing import Any

from aragora.connectors.base import BaseConnector, Evidence
from aragora.reasoning.provenance import ProvenanceManager, SourceType

logger = logging.getLogger(__name__)

try:
    import httpx

    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False

SEMANTIC_SCHOLAR_API = "https://api.semanticscholar.org/graph/v1"


class SemanticScholarConnector(BaseConnector):
    """Connector for Semantic Scholar search and metadata."""

    def __init__(
        self,
        provenance: ProvenanceManager | None = None,
        default_confidence: float = 0.85,
        timeout: int = 30,
        rate_limit_delay: float = 0.5,
        max_cache_entries: int = 500,
        cache_ttl_seconds: float = 86400.0,
        api_key: str | None = None,
    ) -> None:
        super().__init__(
            provenance=provenance,
            default_confidence=default_confidence,
            max_cache_entries=max_cache_entries,
            cache_ttl_seconds=cache_ttl_seconds,
        )
        self.timeout = timeout
        self.rate_limit_delay = rate_limit_delay
        self._last_request_time: float = 0.0
        self.api_key = (
            api_key or os.environ.get("SEMANTIC_SCHOLAR_API_KEY") or os.environ.get("S2_API_KEY")
        )

    @property
    def source_type(self) -> SourceType:
        return SourceType.DOCUMENT

    @property
    def name(self) -> str:
        return "Semantic Scholar"

    @property
    def is_available(self) -> bool:
        return HTTPX_AVAILABLE

    async def _rate_limit(self) -> None:
        import time

        now = time.time()
        elapsed = now - self._last_request_time
        if elapsed < self.rate_limit_delay:
            await asyncio.sleep(self.rate_limit_delay - elapsed)
        self._last_request_time = time.time()

    def _headers(self) -> dict[str, str]:
        headers = {"User-Agent": "Aragora/1.0"}
        if self.api_key:
            headers["x-api-key"] = self.api_key
        return headers

    async def search(
        self,
        query: str,
        limit: int = 10,
        fields: list[str] | None = None,
        **kwargs: Any,
    ) -> list[Evidence]:
        if not HTTPX_AVAILABLE:
            logger.warning("httpx not available, cannot search Semantic Scholar")
            return []

        limit = min(limit, 50)
        fields = fields or [
            "title",
            "abstract",
            "authors",
            "year",
            "venue",
            "url",
            "citationCount",
            "paperId",
            "externalIds",
            "isOpenAccess",
        ]

        params = {
            "query": query,
            "limit": limit,
            "fields": ",".join(fields),
        }

        await self._rate_limit()

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(
                    f"{SEMANTIC_SCHOLAR_API}/paper/search",
                    params=params,
                    headers=self._headers(),
                )
                response.raise_for_status()
                data = response.json()

            results = data.get("data", [])
            return [self._paper_to_evidence(item) for item in results if item]
        except httpx.HTTPError as e:
            logger.warning("Semantic Scholar search failed: %s", e)
            return []
        except (ValueError, KeyError, TypeError) as e:
            logger.warning("Semantic Scholar response parse error: %s", e)
            return []

    def _paper_to_evidence(self, item: dict[str, Any]) -> Evidence:
        paper_id = item.get("paperId") or ""
        title = (item.get("title") or "").strip()
        abstract = (item.get("abstract") or "").strip()
        url = item.get("url") or ""
        venue = item.get("venue") or ""
        year = item.get("year")
        authors = [a.get("name", "") for a in item.get("authors", []) if a.get("name")]
        external_ids = item.get("externalIds") or {}
        doi = external_ids.get("DOI") or external_ids.get("doi")

        content_parts = []
        if title:
            content_parts.append(title)
        if abstract:
            content_parts.append(abstract)
        content = "\n\n".join(content_parts) if content_parts else title

        evidence_id = (
            f"s2:{paper_id}" if paper_id else hashlib.sha256(title.encode()).hexdigest()[:16]
        )
        source_id = paper_id or doi or title

        return Evidence(
            id=evidence_id,
            source_type=SourceType.DOCUMENT,
            source_id=source_id,
            content=content,
            title=title,
            url=url or (f"https://doi.org/{doi}" if doi else None),
            author=venue,
            created_at=str(year) if year else None,
            confidence=self.default_confidence,
            authority=0.85,
            freshness=1.0,
            metadata={
                "paper_id": paper_id,
                "authors": authors,
                "venue": venue,
                "year": year,
                "doi": doi,
                "citation_count": item.get("citationCount"),
                "is_open_access": item.get("isOpenAccess"),
            },
        )

    async def fetch(self, evidence_id: str) -> Evidence | None:
        if not HTTPX_AVAILABLE:
            return None
        paper_id = evidence_id.replace("s2:", "") if evidence_id.startswith("s2:") else evidence_id
        if not paper_id:
            return None
        fields = [
            "title",
            "abstract",
            "authors",
            "year",
            "venue",
            "url",
            "citationCount",
            "paperId",
            "externalIds",
            "isOpenAccess",
        ]
        params = {"fields": ",".join(fields)}
        await self._rate_limit()
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(
                    f"{SEMANTIC_SCHOLAR_API}/paper/{paper_id}",
                    params=params,
                    headers=self._headers(),
                )
                response.raise_for_status()
                data = response.json()
            return self._paper_to_evidence(data)
        except httpx.HTTPError as e:
            logger.warning("Semantic Scholar fetch failed: %s", e)
            return None
