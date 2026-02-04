"""
CourtListener Connector - Case law search for Aragora agents.

Uses the CourtListener REST API to search court opinions and dockets.
Authentication is optional but recommended for higher rate limits.
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

API_BASE = "https://www.courtlistener.com/api/rest/v4"
WEB_BASE = "https://www.courtlistener.com"


class CourtListenerConnector(BaseConnector):
    """Connector for CourtListener case law search."""

    def __init__(
        self,
        provenance: ProvenanceManager | None = None,
        default_confidence: float = 0.85,
        timeout: int = 30,
        rate_limit_delay: float = 0.34,
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
        self.api_key = api_key or os.environ.get("COURTLISTENER_API_KEY")

    @property
    def source_type(self) -> SourceType:
        return SourceType.DOCUMENT

    @property
    def name(self) -> str:
        return "CourtListener"

    @property
    def is_available(self) -> bool:
        return HTTPX_AVAILABLE

    def _headers(self) -> dict[str, str]:
        headers = {"Accept": "application/json", "User-Agent": "Aragora/1.0"}
        if self.api_key:
            headers["Authorization"] = f"Token {self.api_key}"
        return headers

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
        search_type: str = "o",
        order_by: str | None = None,
        court: str | None = None,
        **kwargs: Any,
    ) -> list[Evidence]:
        if not HTTPX_AVAILABLE:
            logger.warning("httpx not available, cannot search CourtListener")
            return []

        query = (query or "").strip()
        if not query:
            return []

        limit = min(limit, 50)

        params: dict[str, Any] = {
            "q": query,
            "type": search_type,
            "page_size": limit,
        }

        optional: dict[str, Any] = {}
        if order_by:
            optional["order_by"] = order_by
        if court:
            optional["court"] = court
        params.update(optional)

        await self._rate_limit()

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(
                    f"{API_BASE}/search/",
                    params=params,
                    headers=self._headers(),
                )
                response.raise_for_status()
                data = response.json()
        except httpx.HTTPStatusError as e:
            if optional and e.response.status_code == 400:
                # Retry without optional filters for better compatibility.
                try:
                    async with httpx.AsyncClient(timeout=self.timeout) as client:
                        response = await client.get(
                            f"{API_BASE}/search/",
                            params={"q": query, "type": search_type, "page_size": limit},
                            headers=self._headers(),
                        )
                        response.raise_for_status()
                        data = response.json()
                except Exception as inner:
                    logger.warning("CourtListener search failed after retry: %s", inner)
                    return []
            else:
                logger.warning("CourtListener search failed: %s", e)
                return []
        except Exception as e:  # noqa: BLE001
            logger.warning("CourtListener search error: %s", e)
            return []

        results = data.get("results", []) if isinstance(data, dict) else []
        return [self._item_to_evidence(item) for item in results if item]

    def _item_to_evidence(self, item: dict[str, Any]) -> Evidence:
        raw_id = item.get("id") or item.get("opinion_id") or item.get("docket_id")
        title = (item.get("caseName") or item.get("case_name") or item.get("caption") or "").strip()
        court = item.get("court") or item.get("court_id") or item.get("court_name") or ""
        date_filed = item.get("dateFiled") or item.get("date_filed") or item.get("date") or ""
        docket_number = item.get("docketNumber") or item.get("docket_number") or ""
        citation = item.get("citation") or item.get("citations") or ""
        snippet = item.get("snippet") or item.get("text") or ""

        url = item.get("absolute_url") or item.get("absoluteUrl") or item.get("permalink") or ""
        if url and isinstance(url, str) and url.startswith("/"):
            url = f"{WEB_BASE}{url}"

        content_parts = [title] if title else []
        meta_line = " | ".join([p for p in [court, date_filed, docket_number] if p])
        if meta_line:
            content_parts.append(meta_line)
        if citation:
            content_parts.append(f"Citation: {citation}")
        if snippet:
            content_parts.append(snippet)
        content = "\n".join([p for p in content_parts if p])

        evidence_id = (
            f"courtlistener:{raw_id}"
            if raw_id
            else hashlib.sha256((title or content).encode()).hexdigest()[:16]
        )

        return Evidence(
            id=evidence_id,
            source_type=SourceType.DOCUMENT,
            source_id=str(raw_id or title or url),
            content=content or title or "CourtListener result",
            title=title,
            url=url or None,
            author=court or None,
            created_at=str(date_filed) if date_filed else None,
            confidence=self.default_confidence,
            authority=0.8,
            freshness=1.0,
            metadata={
                "court": court,
                "date_filed": date_filed,
                "docket_number": docket_number,
                "citation": citation,
            },
        )

    async def fetch(self, evidence_id: str) -> Evidence | None:
        if not HTTPX_AVAILABLE:
            return None

        raw_id = evidence_id.replace("courtlistener:", "")
        if not raw_id:
            return None

        await self._rate_limit()

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(
                    f"{API_BASE}/opinions/{raw_id}/",
                    headers=self._headers(),
                )
                if response.status_code == 404:
                    return None
                response.raise_for_status()
                data = response.json()
        except Exception as e:  # noqa: BLE001
            logger.warning("CourtListener fetch failed: %s", e)
            return None

        if not isinstance(data, dict):
            return None

        return self._item_to_evidence(data)
