"""
Generic Tax Connector - Multi-jurisdiction tax guidance via proxy endpoints.

Environment variable pattern:
  TAX_{JURISDICTION}_API_BASE
  TAX_{JURISDICTION}_SEARCH_URL
  TAX_{JURISDICTION}_API_KEY (optional)
  TAX_{JURISDICTION}_SEARCH_METHOD (GET/POST)
  TAX_{JURISDICTION}_SEARCH_QUERY_PARAM
  TAX_{JURISDICTION}_SEARCH_LIMIT_PARAM
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


class GenericTaxConnector(BaseConnector):
    """Connector for non-US tax guidance (proxy-ready)."""

    def __init__(
        self,
        jurisdiction: str,
        provenance: ProvenanceManager | None = None,
        default_confidence: float = 0.75,
        timeout: int = 30,
        rate_limit_delay: float = 0.4,
        max_cache_entries: int = 500,
        cache_ttl_seconds: float = 86400.0,
    ) -> None:
        super().__init__(
            provenance=provenance,
            default_confidence=default_confidence,
            max_cache_entries=max_cache_entries,
            cache_ttl_seconds=cache_ttl_seconds,
        )
        self.jurisdiction = (jurisdiction or "").upper()
        self.timeout = timeout
        self.rate_limit_delay = rate_limit_delay
        self._last_request_time: float = 0.0

    @property
    def source_type(self) -> SourceType:
        return SourceType.DOCUMENT

    @property
    def name(self) -> str:
        return f"Tax:{self.jurisdiction}"

    @property
    def is_available(self) -> bool:
        return HTTPX_AVAILABLE

    @property
    def is_configured(self) -> bool:
        return bool(self._search_url() or self._base_url())

    def _env_prefix(self) -> str:
        return f"TAX_{self.jurisdiction}_"

    def _base_url(self) -> str | None:
        return os.environ.get(f"{self._env_prefix()}API_BASE")

    def _search_url(self) -> str | None:
        return os.environ.get(f"{self._env_prefix()}SEARCH_URL")

    def _api_key(self) -> str | None:
        return os.environ.get(f"{self._env_prefix()}API_KEY")

    def _resolve_search_url(self) -> str | None:
        search = self._search_url()
        if search:
            return search
        base = self._base_url()
        if base:
            return f"{base.rstrip('/')}/search"
        return None

    def _resolve_method(self) -> str:
        return (os.environ.get(f"{self._env_prefix()}SEARCH_METHOD", "GET") or "GET").upper()

    def _resolve_param_names(self) -> tuple[str, str]:
        query_key = os.environ.get(f"{self._env_prefix()}SEARCH_QUERY_PARAM", "q")
        limit_key = os.environ.get(f"{self._env_prefix()}SEARCH_LIMIT_PARAM", "limit")
        return query_key, limit_key

    def _headers(self) -> dict[str, str]:
        headers = {"Accept": "application/json", "User-Agent": "Aragora/1.0"}
        api_key = self._api_key()
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
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
        **kwargs: Any,
    ) -> list[Evidence]:
        if not HTTPX_AVAILABLE:
            logger.warning("httpx not available, cannot search tax connector")
            return []

        query = (query or "").strip()
        if not query:
            return []

        search_url = self._resolve_search_url()
        if not search_url:
            logger.warning("Tax connector not configured for %s", self.jurisdiction)
            return []

        limit = min(limit, 50)
        query_key, limit_key = self._resolve_param_names()
        params = {query_key: query, limit_key: limit}
        method = self._resolve_method()

        await self._rate_limit()

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                if method == "POST":
                    response = await client.post(search_url, json=params, headers=self._headers())
                else:
                    response = await client.get(search_url, params=params, headers=self._headers())
                response.raise_for_status()
                data = response.json()
        except Exception as e:  # noqa: BLE001
            logger.warning("Tax search failed (%s): %s", self.jurisdiction, e)
            return []

        results = _extract_results(data)
        return [_item_to_evidence(item, self.jurisdiction) for item in results]

    async def fetch(self, evidence_id: str) -> Evidence | None:
        return None


def _extract_results(data: Any) -> list[dict[str, Any]]:
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    if isinstance(data, dict):
        for key in ("results", "items", "documents"):
            items = data.get(key)
            if isinstance(items, list):
                return [item for item in items if isinstance(item, dict)]
    return []


def _item_to_evidence(item: dict[str, Any], jurisdiction: str) -> Evidence:
    title = (item.get("title") or item.get("name") or "").strip()
    summary = item.get("summary") or item.get("abstract") or item.get("snippet") or ""
    url = item.get("url") or item.get("link") or ""
    issued = item.get("issued") or item.get("published") or item.get("date") or ""
    doc_type = item.get("type") or item.get("document_type") or ""

    content_parts = [title] if title else []
    if summary:
        content_parts.append(summary)
    if doc_type:
        content_parts.append(f"Type: {doc_type}")
    content = "\n\n".join(content_parts) if content_parts else title

    source_id = item.get("id") or item.get("doc_id") or title or url
    evidence_id = (
        f"tax:{jurisdiction}:{source_id}"
        if source_id
        else hashlib.sha256((title or content).encode()).hexdigest()[:16]
    )

    return Evidence(
        id=evidence_id,
        source_type=SourceType.DOCUMENT,
        source_id=str(source_id),
        content=content or title or "Tax guidance result",
        title=title,
        url=url or None,
        author=jurisdiction,
        created_at=str(issued) if issued else None,
        confidence=0.75,
        authority=0.8,
        freshness=1.0,
        metadata={"doc_type": doc_type, "jurisdiction": jurisdiction},
    )
