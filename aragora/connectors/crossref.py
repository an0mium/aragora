"""
Crossref Connector - Citation metadata lookup for Aragora agents.

Uses the Crossref public API to search DOI metadata and publication info.
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

CROSSREF_API = "https://api.crossref.org/works"


class CrossRefConnector(BaseConnector):
    """Connector for Crossref DOI metadata."""

    def __init__(
        self,
        provenance: ProvenanceManager | None = None,
        default_confidence: float = 0.85,
        timeout: int = 30,
        rate_limit_delay: float = 0.5,
        max_cache_entries: int = 500,
        cache_ttl_seconds: float = 86400.0,
        mailto: str | None = None,
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
        self.mailto = mailto or os.environ.get("CROSSREF_MAILTO") or os.environ.get("NCBI_EMAIL")

    @property
    def source_type(self) -> SourceType:
        return SourceType.DOCUMENT

    @property
    def name(self) -> str:
        return "Crossref"

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
        return {"User-Agent": "Aragora/1.0"}

    async def search(
        self,
        query: str,
        limit: int = 10,
        **kwargs: Any,
    ) -> list[Evidence]:
        if not HTTPX_AVAILABLE:
            logger.warning("httpx not available, cannot search Crossref")
            return []

        limit = min(limit, 50)
        params: dict[str, Any] = {"query": query, "rows": limit}
        if self.mailto:
            params["mailto"] = self.mailto

        await self._rate_limit()

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(CROSSREF_API, params=params, headers=self._headers())
                response.raise_for_status()
                data = response.json()

            items = data.get("message", {}).get("items", [])
            return [self._item_to_evidence(item) for item in items if item]
        except httpx.HTTPError as e:
            logger.warning("Crossref search failed: %s", e)
            return []
        except (ValueError, KeyError, TypeError) as e:
            logger.warning("Crossref response parse error: %s", e)
            return []

    def _item_to_evidence(self, item: dict[str, Any]) -> Evidence:
        doi = item.get("DOI") or item.get("doi") or ""
        title = (
            (item.get("title") or [""])[0]
            if isinstance(item.get("title"), list)
            else item.get("title")
        )
        title = (title or "").strip()
        container = (
            (item.get("container-title") or [""])[0]
            if isinstance(item.get("container-title"), list)
            else item.get("container-title")
        )
        issued = item.get("issued", {}).get("date-parts", [])
        year = None
        if issued and issued[0]:
            year = issued[0][0]
        author_list = []
        for author in item.get("author", []) or []:
            given = author.get("given", "").strip()
            family = author.get("family", "").strip()
            name = " ".join([p for p in [given, family] if p])
            if name:
                author_list.append(name)

        url = f"https://doi.org/{doi}" if doi else item.get("URL")
        content_parts = []
        if title:
            content_parts.append(title)
        if container or year:
            content_parts.append(
                " | ".join([p for p in [container, str(year) if year else ""] if p])
            )
        if author_list:
            content_parts.append(f"Authors: {', '.join(author_list[:6])}")
        content = "\n".join(content_parts)

        evidence_id = f"crossref:{doi}" if doi else hashlib.sha256(title.encode()).hexdigest()[:16]
        source_id = doi or title

        return Evidence(
            id=evidence_id,
            source_type=SourceType.DOCUMENT,
            source_id=source_id,
            content=content,
            title=title,
            url=url,
            author=container,
            created_at=str(year) if year else None,
            confidence=self.default_confidence,
            authority=0.85,
            freshness=1.0,
            metadata={
                "doi": doi,
                "container_title": container,
                "year": year,
                "authors": author_list,
                "is_referenced_by_count": item.get("is-referenced-by-count"),
                "type": item.get("type"),
            },
        )

    async def fetch(self, evidence_id: str) -> Evidence | None:
        if not HTTPX_AVAILABLE:
            return None
        doi = (
            evidence_id.replace("crossref:", "")
            if evidence_id.startswith("crossref:")
            else evidence_id
        )
        if not doi:
            return None
        params: dict[str, Any] = {}
        if self.mailto:
            params["mailto"] = self.mailto
        await self._rate_limit()
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(
                    f"{CROSSREF_API}/{doi}", params=params, headers=self._headers()
                )
                response.raise_for_status()
                data = response.json()
            item = data.get("message", {})
            if not item:
                return None
            return self._item_to_evidence(item)
        except httpx.HTTPError as e:
            logger.warning("Crossref fetch failed: %s", e)
            return None
