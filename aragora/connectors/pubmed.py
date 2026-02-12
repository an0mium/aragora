"""
PubMed Connector - Biomedical literature search for Aragora agents.

Uses NCBI E-utilities to search PubMed articles and return metadata.
No authentication required; optional NCBI API key speeds requests.
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

EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
PUBMED_URL = "https://pubmed.ncbi.nlm.nih.gov"


class PubMedConnector(BaseConnector):
    """Connector for PubMed biomedical literature."""

    def __init__(
        self,
        provenance: ProvenanceManager | None = None,
        default_confidence: float = 0.9,
        timeout: int = 30,
        rate_limit_delay: float = 0.34,  # ~3 req/sec
        max_cache_entries: int = 500,
        cache_ttl_seconds: float = 86400.0,
        api_key: str | None = None,
        email: str | None = None,
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
        self.api_key = api_key or os.environ.get("NCBI_API_KEY")
        self.email = email or os.environ.get("NCBI_EMAIL")

    @property
    def source_type(self) -> SourceType:
        return SourceType.DOCUMENT

    @property
    def name(self) -> str:
        return "PubMed"

    @property
    def is_available(self) -> bool:
        return HTTPX_AVAILABLE

    def _base_params(self) -> dict[str, Any]:
        params: dict[str, Any] = {"tool": "aragora"}
        if self.api_key:
            params["api_key"] = self.api_key
        if self.email:
            params["email"] = self.email
        return params

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
        sort: str = "relevance",
        **kwargs: Any,
    ) -> list[Evidence]:
        if not HTTPX_AVAILABLE:
            logger.warning("httpx not available, cannot search PubMed")
            return []

        limit = min(limit, 50)

        params: dict[str, Any] = {
            "db": "pubmed",
            "term": query,
            "retmax": limit,
            "retmode": "json",
            "sort": sort,
        }
        params.update(self._base_params())

        await self._rate_limit()

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(f"{EUTILS_BASE}/esearch.fcgi", params=params)
                response.raise_for_status()
                data = response.json()

            ids = data.get("esearchresult", {}).get("idlist", [])
            if not ids:
                return []

            summaries = await self._fetch_summaries(ids)
            return summaries
        except httpx.HTTPError as e:
            logger.warning("PubMed search failed: %s", e)
            return []
        except (ValueError, KeyError, TypeError) as e:
            logger.warning("PubMed response parse error: %s", e)
            return []

    async def _fetch_summaries(self, ids: list[str]) -> list[Evidence]:
        params = {
            "db": "pubmed",
            "id": ",".join(ids),
            "retmode": "json",
        }
        params.update(self._base_params())

        await self._rate_limit()

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.get(f"{EUTILS_BASE}/esummary.fcgi", params=params)
            response.raise_for_status()
            data = response.json()

        result = data.get("result", {})
        uids = result.get("uids", [])
        evidence_items: list[Evidence] = []
        for uid in uids:
            summary = result.get(str(uid), {})
            if not summary:
                continue
            evidence_items.append(self._summary_to_evidence(summary))
        return evidence_items

    def _summary_to_evidence(self, summary: dict[str, Any]) -> Evidence:
        pmid = str(summary.get("uid", ""))
        title = summary.get("title", "").strip()
        pubdate = summary.get("pubdate", "")
        source = summary.get("source") or summary.get("fulljournalname") or ""
        authors = [a.get("name", "") for a in summary.get("authors", []) if a.get("name")]
        doi = None
        for aid in summary.get("articleids", []) or []:
            if aid.get("idtype") == "doi":
                doi = aid.get("value")
                break

        url = f"{PUBMED_URL}/{pmid}/" if pmid else ""
        content_parts = []
        if title:
            content_parts.append(title)
        if source or pubdate:
            content_parts.append(" | ".join([p for p in [source, pubdate] if p]))
        if authors:
            content_parts.append(f"Authors: {', '.join(authors[:6])}")
        if doi:
            content_parts.append(f"DOI: {doi}")
        content = "\n".join(content_parts)

        evidence_id = f"pubmed:{pmid}" if pmid else hashlib.sha256(title.encode()).hexdigest()[:16]

        return Evidence(
            id=evidence_id,
            source_type=SourceType.DOCUMENT,
            source_id=pmid or title,
            content=content,
            title=title,
            url=url,
            author=source,
            created_at=summary.get("sortpubdate") or summary.get("pubdate"),
            confidence=self.default_confidence,
            authority=0.9,
            freshness=1.0,
            metadata={
                "pmid": pmid,
                "source": source,
                "pubdate": pubdate,
                "authors": authors,
                "doi": doi,
            },
        )

    async def fetch(self, evidence_id: str) -> Evidence | None:
        if not HTTPX_AVAILABLE:
            return None
        pmid = (
            evidence_id.replace("pubmed:", "") if evidence_id.startswith("pubmed:") else evidence_id
        )
        if not pmid:
            return None
        try:
            summaries = await self._fetch_summaries([pmid])
            return summaries[0] if summaries else None
        except httpx.HTTPError as e:
            logger.warning("PubMed fetch failed: %s", e)
            return None
