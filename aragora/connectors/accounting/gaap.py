"""
FASB GAAP Connector - Licensed GAAP standards search.

This connector is designed for licensed GAAP content. It expects a
configured API endpoint (FASB or an enterprise KB proxy) and an API key.

Production quality: circuit breaker, retry with backoff, query sanitization.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import re
from typing import Any

from aragora.connectors.base import BaseConnector, Evidence
from aragora.connectors.exceptions import ConnectorError
from aragora.reasoning.provenance import ProvenanceManager, SourceType

logger = logging.getLogger(__name__)

try:
    import httpx

    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False

CONFIG_ENV_VARS = ("FASB_API_BASE", "FASB_SEARCH_URL")
OPTIONAL_ENV_VARS = ("FASB_API_KEY",)

_SAFE_QUERY_RE = re.compile(r"[^\w\s@.\-+/:]")
MAX_QUERY_LENGTH = 500


def _sanitize_query(query: str) -> str:
    """Sanitize query to prevent injection."""
    query = query[:MAX_QUERY_LENGTH]
    return _SAFE_QUERY_RE.sub("", query).strip()


def get_config_status() -> dict[str, Any]:
    """Return configuration status for the FASB connector."""
    base = os.environ.get("FASB_API_BASE")
    search = os.environ.get("FASB_SEARCH_URL")
    configured = bool(base or search)
    missing_required = []
    if not configured:
        missing_required = list(CONFIG_ENV_VARS)
    missing_optional = [key for key in OPTIONAL_ENV_VARS if not os.environ.get(key)]
    return {
        "configured": configured,
        "required": list(CONFIG_ENV_VARS),
        "optional": list(OPTIONAL_ENV_VARS),
        "missing_required": missing_required,
        "missing_optional": missing_optional,
        "notes": "Requires licensed FASB content or internal proxy",
    }


class FASBConnector(BaseConnector):
    """Connector for licensed GAAP content (FASB or internal proxy)."""

    def __init__(
        self,
        provenance: ProvenanceManager | None = None,
        default_confidence: float = 0.85,
        timeout: int = 30,
        rate_limit_delay: float = 0.4,
        max_cache_entries: int = 500,
        cache_ttl_seconds: float = 86400.0,
        api_key: str | None = None,
        base_url: str | None = None,
        search_url: str | None = None,
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
        self.api_key = api_key or os.environ.get("FASB_API_KEY")
        self.base_url = base_url or os.environ.get("FASB_API_BASE")
        self.search_url = search_url or os.environ.get("FASB_SEARCH_URL")

    @property
    def source_type(self) -> SourceType:
        return SourceType.DOCUMENT

    @property
    def name(self) -> str:
        return "FASB GAAP"

    @property
    def is_available(self) -> bool:
        return HTTPX_AVAILABLE

    @property
    def is_configured(self) -> bool:
        return bool(self.search_url or self.base_url)

    async def _rate_limit(self) -> None:
        import time

        now = time.time()
        elapsed = now - self._last_request_time
        if elapsed < self.rate_limit_delay:
            await asyncio.sleep(self.rate_limit_delay - elapsed)
        self._last_request_time = time.time()

    def _headers(self) -> dict[str, str]:
        headers = {"Accept": "application/json", "User-Agent": "Aragora/1.0"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _resolve_search_url(self) -> str | None:
        if self.search_url:
            return self.search_url
        if self.base_url:
            return f"{self.base_url.rstrip('/')}/search"
        return None

    def _resolve_method(self) -> str:
        return (os.environ.get("FASB_SEARCH_METHOD", "GET") or "GET").upper()

    def _resolve_param_names(self) -> tuple[str, str]:
        query_key = os.environ.get("FASB_SEARCH_QUERY_PARAM", "q")
        limit_key = os.environ.get("FASB_SEARCH_LIMIT_PARAM", "limit")
        return query_key, limit_key

    async def search(
        self,
        query: str,
        limit: int = 10,
        **kwargs: Any,
    ) -> list[Evidence]:
        if not HTTPX_AVAILABLE:
            logger.warning("httpx not available, cannot search FASB")
            return []

        query = _sanitize_query((query or "").strip())
        if not query:
            return []

        search_url = self._resolve_search_url()
        if not search_url:
            logger.warning("FASB connector not configured (missing base/search URL)")
            return []

        limit = min(limit, 50)
        query_key, limit_key = self._resolve_param_names()
        params: dict[str, Any] = {query_key: query, limit_key: limit}
        method = self._resolve_method()

        await self._rate_limit()

        async def _do_request() -> Any:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                if method == "POST":
                    response = await client.post(
                        search_url,
                        json=params,
                        headers=self._headers(),
                    )
                else:
                    response = await client.get(
                        search_url,
                        params=params,
                        headers=self._headers(),
                    )
                response.raise_for_status()
                return response.json()

        try:
            data = await self._request_with_retry(_do_request, "search")
        except (ConnectorError, httpx.HTTPError, OSError, ValueError) as e:
            logger.warning("FASB search failed: %s", e)
            return []

        results = _extract_results(data)
        return [_item_to_evidence(item) for item in results]

    async def fetch(self, evidence_id: str) -> Evidence | None:
        # Optional: implement if licensed API provides a detail endpoint.
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


def _item_to_evidence(item: dict[str, Any]) -> Evidence:
    title = (item.get("title") or item.get("name") or "").strip()
    summary = item.get("summary") or item.get("abstract") or item.get("snippet") or ""
    url = item.get("url") or item.get("link") or ""
    issued = item.get("issued") or item.get("published") or item.get("date") or ""
    section = item.get("section") or item.get("topic") or ""

    content_parts = [title] if title else []
    if summary:
        content_parts.append(summary)
    if section:
        content_parts.append(f"Section: {section}")
    content = "\n\n".join(content_parts) if content_parts else title

    source_id = item.get("id") or item.get("doc_id") or title or url
    evidence_id = (
        f"fasb:{source_id}"
        if source_id
        else hashlib.sha256((title or content).encode()).hexdigest()[:16]
    )

    return Evidence(
        id=evidence_id,
        source_type=SourceType.DOCUMENT,
        source_id=str(source_id),
        content=content or title or "FASB GAAP result",
        title=title,
        url=url or None,
        author="FASB",
        created_at=str(issued) if issued else None,
        confidence=0.85,
        authority=0.9,
        freshness=1.0,
        metadata={"section": section},
    )
