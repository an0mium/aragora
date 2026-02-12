"""
Clinical Tables Connector - ICD-10 lookup for Aragora agents.

Uses NLM ClinicalTables API to search ICD-10-CM codes.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
from typing import Any

from aragora.connectors.base import BaseConnector, Evidence
from aragora.reasoning.provenance import ProvenanceManager, SourceType

logger = logging.getLogger(__name__)

try:
    import httpx

    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False

CLINICAL_TABLES_API = "https://clinicaltables.nlm.nih.gov/api/icd10cm/v3/search"


class ClinicalTablesConnector(BaseConnector):
    """Connector for ICD-10-CM code lookup."""

    def __init__(
        self,
        provenance: ProvenanceManager | None = None,
        default_confidence: float = 0.9,
        timeout: int = 30,
        rate_limit_delay: float = 0.2,
        max_cache_entries: int = 500,
        cache_ttl_seconds: float = 86400.0,
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

    @property
    def source_type(self) -> SourceType:
        return SourceType.DOCUMENT

    @property
    def name(self) -> str:
        return "ClinicalTables"

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
        **kwargs: Any,
    ) -> list[Evidence]:
        if not HTTPX_AVAILABLE:
            logger.warning("httpx not available, cannot search ClinicalTables")
            return []

        limit = min(limit, 50)
        params: dict[str, Any] = {
            "terms": query,
            "maxList": limit,
            "sf": "code,name",
        }

        await self._rate_limit()

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(CLINICAL_TABLES_API, params=params)
                response.raise_for_status()
                data = response.json()

            # Response format: [term, codes, names, extras]
            if not isinstance(data, list) or len(data) < 3:
                return []
            codes = data[1] or []
            names = data[2] or []

            results = []
            for idx, code in enumerate(codes):
                name = names[idx] if idx < len(names) else ""
                results.append(self._entry_to_evidence(code, name))
            return results
        except httpx.HTTPError as e:
            logger.warning("ClinicalTables search failed: %s", e)
            return []
        except (ValueError, KeyError, TypeError) as e:
            logger.warning("ClinicalTables response parse error: %s", e)
            return []

    def _entry_to_evidence(self, code: str, name: str) -> Evidence:
        content = f"{code} - {name}" if name else code
        evidence_id = f"icd10:{code}" if code else hashlib.sha256(content.encode()).hexdigest()[:16]
        return Evidence(
            id=evidence_id,
            source_type=SourceType.DOCUMENT,
            source_id=code or name,
            content=content,
            title=code,
            url=None,
            author="NLM ClinicalTables",
            created_at=None,
            confidence=self.default_confidence,
            authority=0.9,
            freshness=1.0,
            metadata={"code": code, "name": name},
        )

    async def fetch(self, evidence_id: str) -> Evidence | None:
        # No direct fetch endpoint; treat evidence_id as code and return from search.
        code = (
            evidence_id.replace("icd10:", "") if evidence_id.startswith("icd10:") else evidence_id
        )
        if not code:
            return None
        results = await self.search(code, limit=1)
        return results[0] if results else None
