"""
NICE Guidance Connector - Clinical guidelines search for Aragora agents.

Uses the NICE API to list and filter clinical guidance documents.
Requires an API key (NICE developer portal).
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

GUIDANCE_INDEX = "https://api.nice.org.uk/services/guidance/index"


class NICEGuidanceConnector(BaseConnector):
    """Connector for NICE clinical guidance."""

    def __init__(
        self,
        provenance: ProvenanceManager | None = None,
        default_confidence: float = 0.8,
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
        self.api_key = api_key or os.environ.get("NICE_API_KEY")

    @property
    def source_type(self) -> SourceType:
        return SourceType.DOCUMENT

    @property
    def name(self) -> str:
        return "NICE Guidance"

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
        headers = {"Accept": "application/json", "User-Agent": "Aragora/1.0"}
        if self.api_key:
            headers["API-Key"] = self.api_key
        return headers

    async def search(
        self,
        query: str,
        limit: int = 10,
        **kwargs: Any,
    ) -> list[Evidence]:
        if not HTTPX_AVAILABLE:
            logger.warning("httpx not available, cannot search NICE guidance")
            return []

        query = (query or "").strip()
        limit = min(limit, 50)

        await self._rate_limit()

        url = f"{GUIDANCE_INDEX}.json"
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(url, headers=self._headers())
                response.raise_for_status()
                data = response.json()
        except httpx.HTTPStatusError as e:
            logger.warning("NICE guidance search failed: %s", e)
            return []
        except (httpx.RequestError, OSError, ValueError) as e:
            logger.warning("NICE guidance search error: %s", e)
            return []

        entries = _extract_guidance_entries(data)
        if not entries:
            return []

        if query:
            query_lower = query.lower()
            entries = [
                entry
                for entry in entries
                if query_lower in (entry.get("title", "") or entry.get("shortTitle", "")).lower()
                or query_lower in (entry.get("summary", "") or entry.get("description", "")).lower()
            ]

        evidence_items: list[Evidence] = []
        for entry in entries[:limit]:
            evidence_items.append(_entry_to_evidence(entry))
        return evidence_items

    async def fetch(self, evidence_id: str) -> Evidence | None:
        # The NICE API does not expose a generic fetch by ID without additional
        # endpoint knowledge. Use search() to retrieve entries.
        return None


def _extract_guidance_entries(data: Any) -> list[dict[str, Any]]:
    """Extract guidance entries from a NICE API response."""
    entries: list[dict[str, Any]] = []

    def _visit(node: Any) -> None:
        if isinstance(node, dict):
            if _looks_like_guidance(node):
                entries.append(node)
            for value in node.values():
                _visit(value)
        elif isinstance(node, list):
            for item in node:
                _visit(item)

    _visit(data)

    # Deduplicate by title or id
    seen: set[str] = set()
    deduped: list[dict[str, Any]] = []
    for entry in entries:
        key = str(entry.get("id") or entry.get("guidanceId") or entry.get("title") or "")
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(entry)
    return deduped


def _looks_like_guidance(node: dict[str, Any]) -> bool:
    if not isinstance(node, dict):
        return False
    has_title = "title" in node or "shortTitle" in node
    has_link = any(key in node for key in ("url", "link", "href", "guidanceUrl"))
    return has_title and has_link


def _entry_to_evidence(entry: dict[str, Any]) -> Evidence:
    title = (entry.get("title") or entry.get("shortTitle") or "").strip()
    summary = entry.get("summary") or entry.get("description") or ""
    url = (
        entry.get("url")
        or entry.get("guidanceUrl")
        or entry.get("href")
        or _extract_link(entry.get("link"))
        or ""
    )
    published = entry.get("published") or entry.get("publicationDate") or entry.get("date") or ""
    guidance_id = entry.get("id") or entry.get("guidanceId") or ""

    content_parts = [title] if title else []
    if summary:
        content_parts.append(summary)
    content = "\n\n".join(content_parts) if content_parts else title

    evidence_id = (
        f"nice:{guidance_id}"
        if guidance_id
        else hashlib.sha256((title or content).encode()).hexdigest()[:16]
    )

    return Evidence(
        id=evidence_id,
        source_type=SourceType.DOCUMENT,
        source_id=str(guidance_id or title),
        content=content or title or "NICE guidance",
        title=title,
        url=url or None,
        author="NICE",
        created_at=str(published) if published else None,
        confidence=0.8,
        authority=0.85,
        freshness=1.0,
        metadata={
            "guidance_id": guidance_id,
            "published": published,
        },
    )


def _extract_link(link_field: Any) -> str:
    if isinstance(link_field, str):
        return link_field
    if isinstance(link_field, dict):
        return link_field.get("href") or link_field.get("url") or ""
    if isinstance(link_field, list) and link_field:
        for item in link_field:
            url = _extract_link(item)
            if url:
                return url
    return ""
