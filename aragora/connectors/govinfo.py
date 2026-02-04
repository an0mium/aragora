"""
GovInfo Connector - US government documents and statutes search.

Uses the GovInfo Search Service API to retrieve federal documents
including U.S. Code, statutes, regulations, and court documents.
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

SEARCH_URL = "https://api.govinfo.gov/search"
PACKAGE_URL = "https://api.govinfo.gov/packages"


class GovInfoConnector(BaseConnector):
    """Connector for GovInfo Search Service."""

    def __init__(
        self,
        provenance: ProvenanceManager | None = None,
        default_confidence: float = 0.8,
        timeout: int = 30,
        rate_limit_delay: float = 0.4,
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
        self.api_key = api_key or os.environ.get("GOVINFO_API_KEY") or os.environ.get("GOVINFO_KEY")

    @property
    def source_type(self) -> SourceType:
        return SourceType.DOCUMENT

    @property
    def name(self) -> str:
        return "GovInfo"

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

    async def search(
        self,
        query: str,
        limit: int = 10,
        collection: str | None = None,
        sort_field: str = "relevance",
        sort_order: str = "DESC",
        **kwargs: Any,
    ) -> list[Evidence]:
        if not HTTPX_AVAILABLE:
            logger.warning("httpx not available, cannot search GovInfo")
            return []

        query = (query or "").strip()
        if collection:
            collection_filter = f"collection:({collection})"
            query = f"{collection_filter} {query}".strip()

        if not query:
            return []

        limit = min(limit, 50)

        payload: dict[str, Any] = {
            "query": query,
            "pageSize": limit,
            "offsetMark": "*",
            "sorts": [{"field": sort_field, "sortOrder": sort_order}],
        }

        params = {}
        if self.api_key:
            params["api_key"] = self.api_key

        await self._rate_limit()

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    SEARCH_URL,
                    params=params,
                    json=payload,
                    headers={"Accept": "application/json", "User-Agent": "Aragora/1.0"},
                )
                response.raise_for_status()
                data = response.json()
        except httpx.HTTPError as e:
            logger.warning("GovInfo search failed: %s", e)
            return []
        except Exception as e:  # noqa: BLE001
            logger.warning("GovInfo search error: %s", e)
            return []

        results = data.get("results", []) if isinstance(data, dict) else []
        return [self._item_to_evidence(item) for item in results if item]

    def _item_to_evidence(self, item: dict[str, Any]) -> Evidence:
        title = (item.get("title") or item.get("packageTitle") or "").strip()
        package_id = item.get("packageId") or item.get("package_id") or ""
        granule_id = item.get("granuleId") or item.get("granule_id") or ""
        date_issued = item.get("dateIssued") or item.get("date_issued") or ""
        last_modified = item.get("lastModified") or item.get("last_modified") or ""
        collection = item.get("collectionName") or item.get("collectionCode") or ""

        url = (
            item.get("detailsLink")
            or item.get("pdfLink")
            or item.get("htmlLink")
            or item.get("modsLink")
            or ""
        )

        content_parts = [title] if title else []
        meta_line = " | ".join([p for p in [collection, date_issued] if p])
        if meta_line:
            content_parts.append(meta_line)
        if granule_id:
            content_parts.append(f"Granule: {granule_id}")
        if package_id:
            content_parts.append(f"Package: {package_id}")
        if last_modified:
            content_parts.append(f"Last modified: {last_modified}")

        content = "\n".join([p for p in content_parts if p])
        evidence_id = (
            f"govinfo:{package_id}:{granule_id}"
            if package_id
            else hashlib.sha256((title or content).encode()).hexdigest()[:16]
        )

        return Evidence(
            id=evidence_id,
            source_type=SourceType.DOCUMENT,
            source_id=package_id or title,
            content=content or title or "GovInfo result",
            title=title,
            url=url or None,
            author=collection or None,
            created_at=str(date_issued) if date_issued else None,
            confidence=self.default_confidence,
            authority=0.85,
            freshness=1.0,
            metadata={
                "package_id": package_id,
                "granule_id": granule_id,
                "collection": collection,
                "date_issued": date_issued,
                "last_modified": last_modified,
            },
        )

    async def fetch(self, evidence_id: str) -> Evidence | None:
        if not HTTPX_AVAILABLE:
            return None

        raw = evidence_id.replace("govinfo:", "")
        if not raw:
            return None

        package_id = raw.split(":")[0] if ":" in raw else raw
        if not package_id:
            return None

        params = {}
        if self.api_key:
            params["api_key"] = self.api_key

        await self._rate_limit()

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(
                    f"{PACKAGE_URL}/{package_id}/summary",
                    params=params,
                    headers={"Accept": "application/json", "User-Agent": "Aragora/1.0"},
                )
                if response.status_code == 404:
                    return None
                response.raise_for_status()
                data = response.json()
        except Exception as e:  # noqa: BLE001
            logger.warning("GovInfo fetch failed: %s", e)
            return None

        if not isinstance(data, dict):
            return None

        return self._item_to_evidence(data)
