"""
RxNav Connector - Drug lookup for Aragora agents.

Uses NIH RxNav REST API for drug name and interaction lookup.
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

RXNAV_API = "https://rxnav.nlm.nih.gov/REST"


class RxNavConnector(BaseConnector):
    """Connector for RxNav drug lookup."""

    def __init__(
        self,
        provenance: ProvenanceManager | None = None,
        default_confidence: float = 0.85,
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
        return "RxNav"

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
            logger.warning("httpx not available, cannot search RxNav")
            return []

        limit = min(limit, 20)
        params = {"name": query}
        await self._rate_limit()

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(f"{RXNAV_API}/drugs.json", params=params)
                response.raise_for_status()
                data = response.json()

            group = data.get("drugGroup", {})
            concepts = []
            for group_item in group.get("conceptGroup", []) or []:
                for concept in group_item.get("conceptProperties", []) or []:
                    concepts.append(concept)

            results: list[Evidence] = []
            for concept in concepts[:limit]:
                results.append(self._concept_to_evidence(concept))
            return results
        except httpx.HTTPError as e:
            logger.warning("RxNav search failed: %s", e)
            return []
        except (ValueError, KeyError, TypeError) as e:
            logger.warning("RxNav response parse error: %s", e)
            return []

    def _concept_to_evidence(self, concept: dict[str, Any]) -> Evidence:
        rxcui = concept.get("rxcui", "")
        name = concept.get("name", "")
        tty = concept.get("tty", "")
        content = f"{name} ({tty})" if tty else name
        evidence_id = (
            f"rxnav:{rxcui}" if rxcui else hashlib.sha256(content.encode()).hexdigest()[:16]
        )
        return Evidence(
            id=evidence_id,
            source_type=SourceType.DOCUMENT,
            source_id=rxcui or name,
            content=content,
            title=name,
            url=None,
            author="RxNav",
            created_at=None,
            confidence=self.default_confidence,
            authority=0.85,
            freshness=1.0,
            metadata={"rxcui": rxcui, "tty": tty},
        )

    async def fetch(self, evidence_id: str) -> Evidence | None:
        if not HTTPX_AVAILABLE:
            return None
        rxcui = (
            evidence_id.replace("rxnav:", "") if evidence_id.startswith("rxnav:") else evidence_id
        )
        if not rxcui:
            return None
        await self._rate_limit()
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(f"{RXNAV_API}/rxcui/{rxcui}/properties.json")
                response.raise_for_status()
                data = response.json()
            props = data.get("properties", {})
            concept = {
                "rxcui": rxcui,
                "name": props.get("name"),
                "tty": props.get("tty"),
            }
            return self._concept_to_evidence(concept)
        except httpx.HTTPError as e:
            logger.warning("RxNav fetch failed: %s", e)
            return None

    async def fetch_interactions(self, rxcui: str) -> dict[str, Any]:
        """Fetch drug interactions for a given RxCUI."""
        if not HTTPX_AVAILABLE:
            return {"interactions": [], "error": "httpx not available"}
        if not rxcui:
            return {"interactions": [], "error": "rxcui is required"}

        await self._rate_limit()
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(
                    f"{RXNAV_API}/interaction/interaction.json",
                    params={"rxcui": rxcui},
                )
                response.raise_for_status()
                data = response.json()

            interactions = []
            for group in data.get("interactionTypeGroup", []) or []:
                for interaction_type in group.get("interactionType", []) or []:
                    for pair in interaction_type.get("interactionPair", []) or []:
                        description = pair.get("description")
                        interaction = {
                            "description": description,
                            "severity": pair.get("severity"),
                            "interaction_concepts": pair.get("interactionConcept", []),
                        }
                        interactions.append(interaction)

            return {"interactions": interactions}
        except httpx.HTTPError as e:
            logger.warning("RxNav interaction fetch failed: %s", e)
            return {"interactions": [], "error": str(e)}
