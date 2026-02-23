"""Trello Connector.

Provides integration with Trello for boards, cards, and lists.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any

import httpx

from aragora.connectors.base import BaseConnector, Evidence
from aragora.reasoning.provenance import SourceType

logger = logging.getLogger(__name__)

CONFIG_ENV_VARS = ("TRELLO_API_KEY", "TRELLO_TOKEN")

_SAFE_QUERY_RE = re.compile(r"[^\w\s@.\-+]")
MAX_QUERY_LENGTH = 500

_TRELLO_API_BASE = "https://api.trello.com/1"


def _sanitize_query(query: str) -> str:
    """Sanitize query to prevent injection."""
    query = query[:MAX_QUERY_LENGTH]
    return _SAFE_QUERY_RE.sub("", query)


class TrelloConnector(BaseConnector):
    """Trello connector for board, card, and list management."""

    def __init__(self) -> None:
        super().__init__()
        self._configured = all(os.environ.get(v) for v in CONFIG_ENV_VARS)

    @property
    def name(self) -> str:
        return "trello"

    @property
    def source_type(self) -> SourceType:
        return SourceType.EXTERNAL_API

    @property
    def is_configured(self) -> bool:
        return self._configured

    def _get_auth_params(self) -> dict[str, str]:
        return {
            "key": os.environ.get("TRELLO_API_KEY", ""),
            "token": os.environ.get("TRELLO_TOKEN", ""),
        }

    async def search(self, query: str, limit: int = 10, **kwargs: Any) -> list[Evidence]:
        """Search Trello for boards, cards, and lists."""
        if not self._configured:
            logger.debug("Trello connector not configured")
            return []

        sanitized = _sanitize_query(query)
        if not sanitized.strip():
            return []

        auth_params = self._get_auth_params()

        async def _do_request() -> Any:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.get(
                    f"{_TRELLO_API_BASE}/search",
                    params={
                        "query": sanitized,
                        "modelTypes": "cards",
                        "cards_limit": limit,
                        **auth_params,
                    },
                )
                resp.raise_for_status()
                return resp.json()

        try:
            data = await self._request_with_retry(_do_request, "search")
        except Exception:
            logger.warning("Trello search failed", exc_info=True)
            return []

        results: list[Evidence] = []
        cards = data.get("cards", [])
        for card in cards[:limit]:
            card_id = card.get("id", "")
            card_name = card.get("name", "")
            card_desc = card.get("desc", "")
            card_url = card.get("shortUrl", "")
            board_name = card.get("board", {}).get("name", "") if isinstance(card.get("board"), dict) else ""
            results.append(
                Evidence(
                    id=f"trello_card_{card_id}",
                    source_type=self.source_type,
                    source_id=f"trello://cards/{card_id}",
                    content=card_desc or card_name,
                    title=card_name,
                    url=card_url,
                    confidence=0.7,
                    freshness=1.0,
                    authority=0.6,
                    metadata={
                        "card_id": card_id,
                        "board_name": board_name,
                    },
                )
            )
        return results

    async def fetch(self, evidence_id: str, **kwargs: Any) -> Evidence | None:
        """Fetch a specific card or board from Trello."""
        if not self._configured:
            return None

        auth_params = self._get_auth_params()

        async def _do_request() -> Any:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.get(
                    f"{_TRELLO_API_BASE}/cards/{evidence_id}",
                    params=auth_params,
                )
                resp.raise_for_status()
                return resp.json()

        try:
            data = await self._request_with_retry(_do_request, "fetch")
        except Exception:
            logger.warning("Trello fetch failed", exc_info=True)
            return None

        card_id = data.get("id", evidence_id)
        card_name = data.get("name", "")
        card_desc = data.get("desc", "")
        card_url = data.get("shortUrl", "")
        return Evidence(
            id=f"trello_card_{card_id}",
            source_type=self.source_type,
            source_id=f"trello://cards/{card_id}",
            content=card_desc or card_name,
            title=card_name,
            url=card_url,
            confidence=0.7,
            freshness=1.0,
            authority=0.6,
            metadata={"card_id": card_id},
        )
