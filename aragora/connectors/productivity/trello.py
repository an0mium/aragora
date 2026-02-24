"""Trello Connector.

Provides production-quality integration with the Trello REST API for cards
and boards.

Searches:
- Cards (via the Trello search endpoint)
- Boards (via the Trello search endpoint)
- Cards by board (via ``/boards/{id}/cards``)
- Cards by list (via ``/lists/{id}/cards``)

Features:
- Exponential backoff with jitter via ``_request_with_retry``
- Circuit breaker integration for failure protection
- Rate limiting between consecutive API calls
- LRU cache with configurable TTL
- Health check against the Trello ``/members/me`` endpoint
- Query sanitization to prevent injection

Environment Variables:
- ``TRELLO_API_KEY`` -- Trello application API key.
- ``TRELLO_TOKEN`` -- Trello member token with read access.
"""

from __future__ import annotations

__all__ = [
    "TrelloConnector",
    "CONFIG_ENV_VARS",
    "MAX_QUERY_LENGTH",
]

import asyncio
import logging
import os
import re
import time
from typing import Any

import httpx

from aragora.connectors.base import BaseConnector, ConnectorCapabilities, Evidence
from aragora.connectors.exceptions import (
    ConnectorAuthError,
    ConnectorError,
)
from aragora.reasoning.provenance import SourceType

logger = logging.getLogger(__name__)

CONFIG_ENV_VARS = ("TRELLO_API_KEY", "TRELLO_TOKEN")

_SAFE_QUERY_RE = re.compile(r"[^\w\s@.\-+]")
MAX_QUERY_LENGTH = 500

_TRELLO_API_BASE = "https://api.trello.com/1"

# Trello REST API rate limit: 100 requests per 10 seconds per token.
# We use a conservative per-request delay so bursts stay under the limit.
_DEFAULT_RATE_LIMIT_DELAY = 0.15  # seconds between requests


def _sanitize_query(query: str) -> str:
    """Sanitize query to prevent injection."""
    query = query[:MAX_QUERY_LENGTH]
    return _SAFE_QUERY_RE.sub("", query)


class TrelloConnector(BaseConnector):
    """Trello connector for card and board search/fetch.

    Provides search across cards and boards, individual card/board fetch,
    and board-level card listing.  All external calls go through the
    base-class ``_request_with_retry`` helper which provides exponential
    backoff, circuit-breaker recording, and structured error mapping.
    """

    def __init__(
        self,
        *,
        rate_limit_delay: float = _DEFAULT_RATE_LIMIT_DELAY,
        max_retries: int = BaseConnector.DEFAULT_MAX_RETRIES,
        base_delay: float = BaseConnector.DEFAULT_BASE_DELAY,
        max_delay: float = BaseConnector.DEFAULT_MAX_DELAY,
        max_cache_entries: int = 500,
        cache_ttl_seconds: float = 3600.0,
        enable_circuit_breaker: bool = True,
    ) -> None:
        super().__init__(
            max_retries=max_retries,
            base_delay=base_delay,
            max_delay=max_delay,
            max_cache_entries=max_cache_entries,
            cache_ttl_seconds=cache_ttl_seconds,
            enable_circuit_breaker=enable_circuit_breaker,
        )
        self._configured = all(os.environ.get(v) for v in CONFIG_ENV_VARS)
        self.rate_limit_delay = rate_limit_delay
        self._last_request_time: float = 0.0

    # ------------------------------------------------------------------
    # BaseConnector abstract / override properties
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "trello"

    @property
    def source_type(self) -> SourceType:
        return SourceType.EXTERNAL_API

    @property
    def is_configured(self) -> bool:
        return self._configured

    @property
    def is_available(self) -> bool:
        """httpx is a hard dependency imported at module level."""
        return True

    def capabilities(self) -> ConnectorCapabilities:
        return ConnectorCapabilities(
            can_search=True,
            can_sync=False,
            can_send=False,
            can_receive=False,
            requires_auth=True,
            supports_oauth=True,
            supports_retry=True,
            has_circuit_breaker=self._enable_circuit_breaker,
            max_requests_per_second=10.0,
            platform_features=["cards", "boards", "lists", "labels", "comments"],
        )

    # ------------------------------------------------------------------
    # Health check
    # ------------------------------------------------------------------

    async def _perform_health_check(self, timeout: float) -> bool:
        """Lightweight check against ``/members/me``."""
        if not self._configured:
            return False
        auth = self._get_auth_params()
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                resp = await client.get(
                    f"{_TRELLO_API_BASE}/members/me",
                    params={**auth, "fields": "id"},
                )
                return resp.status_code == 200
        except httpx.TimeoutException:
            logger.debug("Trello health check timed out")
            return False
        except httpx.RequestError as exc:
            logger.debug("Trello health check failed: %s", exc)
            return False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_auth_params(self) -> dict[str, str]:
        return {
            "key": os.environ.get("TRELLO_API_KEY", ""),
            "token": os.environ.get("TRELLO_TOKEN", ""),
        }

    async def _rate_limit(self) -> None:
        """Enforce minimum delay between consecutive requests."""
        now = time.time()
        elapsed = now - self._last_request_time
        if elapsed < self.rate_limit_delay:
            await asyncio.sleep(self.rate_limit_delay - elapsed)
        self._last_request_time = time.time()

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    async def search(self, query: str, limit: int = 10, **kwargs: Any) -> list[Evidence]:
        """Search Trello for cards or boards.

        Uses the Trello ``/search`` endpoint which supports full-text
        search across cards, boards, and organizations.

        Args:
            query: Search term.
            limit: Maximum results (capped at 50).
            **kwargs: Optional ``search_type`` ("cards" or "boards"),
                      ``board_id`` (filter cards to a specific board).

        Returns:
            List of Evidence objects from matching Trello records.
        """
        if not self._configured:
            logger.debug("Trello connector not configured")
            return []

        sanitized = _sanitize_query(query)
        if not sanitized.strip():
            return []

        search_type = kwargs.get("search_type", "cards")
        capped_limit = min(limit, 50)

        if search_type == "boards":
            return await self._search_boards(sanitized, capped_limit)
        return await self._search_cards(sanitized, capped_limit, board_id=kwargs.get("board_id"))

    async def _search_cards(
        self,
        query: str,
        limit: int,
        *,
        board_id: str | None = None,
    ) -> list[Evidence]:
        """Search cards via the Trello Search API."""
        auth = self._get_auth_params()
        await self._rate_limit()

        async def _do_request() -> Any:
            params: dict[str, Any] = {
                **auth,
                "query": query,
                "modelTypes": "cards",
                "cards_limit": limit,
                "card_fields": "name,desc,url,dateLastActivity,idBoard,idList,labels,closed",
            }
            if board_id:
                params["idBoards"] = board_id
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.get(f"{_TRELLO_API_BASE}/search", params=params)
                resp.raise_for_status()
                return resp.json()

        try:
            data = await self._request_with_retry(_do_request, "search_cards")
        except ConnectorAuthError:
            logger.warning("Trello card search auth failure")
            return []
        except (ConnectorError, httpx.HTTPError, OSError, ValueError):
            logger.warning("Trello card search failed", exc_info=True)
            return []

        results: list[Evidence] = []
        for card in data.get("cards", [])[:limit]:
            card_id = card.get("id", "")
            card_name = card.get("name", "")
            card_desc = card.get("desc", "")
            card_url = card.get("url", "")
            date_activity = card.get("dateLastActivity", "")
            board_id_val = card.get("idBoard", "")
            list_id = card.get("idList", "")
            labels = [lbl.get("name", "") for lbl in card.get("labels", []) if lbl.get("name")]
            closed = card.get("closed", False)

            results.append(
                Evidence(
                    id=f"trello_card_{card_id}",
                    source_type=self.source_type,
                    source_id=f"trello://cards/{card_id}",
                    content=f"# {card_name}\n\n{card_desc[:2000]}" if card_desc else card_name,
                    title=f"Card: {card_name}",
                    url=card_url,
                    created_at=date_activity,
                    confidence=0.7,
                    freshness=self.calculate_freshness(date_activity) if date_activity else 1.0,
                    authority=0.6,
                    metadata={
                        "type": "card",
                        "card_id": card_id,
                        "board_id": board_id_val,
                        "list_id": list_id,
                        "labels": labels,
                        "closed": closed,
                    },
                )
            )
        return results

    async def _search_boards(self, query: str, limit: int) -> list[Evidence]:
        """Search boards via the Trello Search API."""
        auth = self._get_auth_params()
        await self._rate_limit()

        async def _do_request() -> Any:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.get(
                    f"{_TRELLO_API_BASE}/search",
                    params={
                        **auth,
                        "query": query,
                        "modelTypes": "boards",
                        "boards_limit": limit,
                        "board_fields": "name,desc,url,dateLastActivity,closed",
                    },
                )
                resp.raise_for_status()
                return resp.json()

        try:
            data = await self._request_with_retry(_do_request, "search_boards")
        except ConnectorAuthError:
            logger.warning("Trello board search auth failure")
            return []
        except (ConnectorError, httpx.HTTPError, OSError, ValueError):
            logger.warning("Trello board search failed", exc_info=True)
            return []

        results: list[Evidence] = []
        for board in data.get("boards", [])[:limit]:
            board_id = board.get("id", "")
            board_name = board.get("name", "")
            board_desc = board.get("desc", "")
            board_url = board.get("url", "")
            date_activity = board.get("dateLastActivity", "")
            closed = board.get("closed", False)

            results.append(
                Evidence(
                    id=f"trello_board_{board_id}",
                    source_type=self.source_type,
                    source_id=f"trello://boards/{board_id}",
                    content=f"# {board_name}\n\n{board_desc[:2000]}" if board_desc else board_name,
                    title=f"Board: {board_name}",
                    url=board_url,
                    created_at=date_activity,
                    confidence=0.7,
                    freshness=self.calculate_freshness(date_activity) if date_activity else 1.0,
                    authority=0.65,
                    metadata={
                        "type": "board",
                        "board_id": board_id,
                        "closed": closed,
                    },
                )
            )
        return results

    # ------------------------------------------------------------------
    # Fetch
    # ------------------------------------------------------------------

    async def fetch(self, evidence_id: str, **kwargs: Any) -> Evidence | None:
        """Fetch a specific card or board from Trello.

        The ``evidence_id`` should be in one of the following formats:
        - ``trello_card_<id>`` -- fetches a card
        - ``trello_board_<id>`` -- fetches a board
        """
        if not self._configured:
            return None

        cached = self._cache_get(evidence_id)
        if cached is not None:
            return cached

        if evidence_id.startswith("trello_board_"):
            return await self._fetch_board(evidence_id[len("trello_board_"):], evidence_id)
        elif evidence_id.startswith("trello_card_"):
            return await self._fetch_card(evidence_id[len("trello_card_"):], evidence_id)

        return None

    async def _fetch_card(self, card_id: str, evidence_id: str) -> Evidence | None:
        """Fetch a single card by ID."""
        auth = self._get_auth_params()
        await self._rate_limit()

        async def _do_request() -> Any:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.get(
                    f"{_TRELLO_API_BASE}/cards/{card_id}",
                    params={
                        **auth,
                        "fields": "name,desc,url,dateLastActivity,idBoard,idList,labels,closed,due,idMembers",
                        "actions": "commentCard",
                        "actions_limit": 5,
                    },
                )
                resp.raise_for_status()
                return resp.json()

        try:
            card = await self._request_with_retry(_do_request, "fetch_card")
        except ConnectorAuthError:
            logger.warning("Trello card fetch auth failure for %s", evidence_id)
            return None
        except (ConnectorError, httpx.HTTPError, OSError, ValueError):
            logger.warning("Trello card fetch failed for %s", evidence_id, exc_info=True)
            return None

        card_name = card.get("name", "")
        card_desc = card.get("desc", "")
        card_url = card.get("url", "")
        date_activity = card.get("dateLastActivity", "")
        board_id = card.get("idBoard", "")
        list_id = card.get("idList", "")
        labels = [lbl.get("name", "") for lbl in card.get("labels", []) if lbl.get("name")]
        closed = card.get("closed", False)
        due = card.get("due", "")
        members = card.get("idMembers", [])

        # Include recent comments
        actions = card.get("actions", [])
        comments_text = "\n\n---\n\n".join(
            f"**{a.get('memberCreator', {}).get('username', 'unknown')}**: "
            f"{a.get('data', {}).get('text', '')[:300]}"
            for a in actions[:5]
        )

        content = f"# {card_name}\n\n{card_desc[:3000]}"
        if comments_text:
            content += f"\n\n## Comments\n\n{comments_text}"

        evidence = Evidence(
            id=evidence_id,
            source_type=self.source_type,
            source_id=f"trello://cards/{card_id}",
            content=content[:5000],
            title=f"Card: {card_name}",
            url=card_url,
            created_at=date_activity,
            confidence=0.75,
            freshness=self.calculate_freshness(date_activity) if date_activity else 1.0,
            authority=0.6,
            metadata={
                "type": "card",
                "card_id": card_id,
                "board_id": board_id,
                "list_id": list_id,
                "labels": labels,
                "closed": closed,
                "due": due,
                "member_count": len(members),
                "comment_count": len(actions),
            },
        )
        self._cache_put(evidence.id, evidence)
        return evidence

    async def _fetch_board(self, board_id: str, evidence_id: str) -> Evidence | None:
        """Fetch a single board by ID."""
        auth = self._get_auth_params()
        await self._rate_limit()

        async def _do_request() -> Any:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.get(
                    f"{_TRELLO_API_BASE}/boards/{board_id}",
                    params={
                        **auth,
                        "fields": "name,desc,url,dateLastActivity,closed,idOrganization",
                        "lists": "open",
                        "list_fields": "name",
                    },
                )
                resp.raise_for_status()
                return resp.json()

        try:
            board = await self._request_with_retry(_do_request, "fetch_board")
        except ConnectorAuthError:
            logger.warning("Trello board fetch auth failure for %s", evidence_id)
            return None
        except (ConnectorError, httpx.HTTPError, OSError, ValueError):
            logger.warning("Trello board fetch failed for %s", evidence_id, exc_info=True)
            return None

        board_name = board.get("name", "")
        board_desc = board.get("desc", "")
        board_url = board.get("url", "")
        date_activity = board.get("dateLastActivity", "")
        closed = board.get("closed", False)
        org_id = board.get("idOrganization", "")
        lists = board.get("lists", [])
        list_names = [lst.get("name", "") for lst in lists]

        content = f"# {board_name}\n\n{board_desc[:3000]}"
        if list_names:
            content += "\n\n## Lists\n\n" + "\n".join(f"- {n}" for n in list_names)

        evidence = Evidence(
            id=evidence_id,
            source_type=self.source_type,
            source_id=f"trello://boards/{board_id}",
            content=content[:5000],
            title=f"Board: {board_name}",
            url=board_url,
            created_at=date_activity,
            confidence=0.75,
            freshness=self.calculate_freshness(date_activity) if date_activity else 1.0,
            authority=0.65,
            metadata={
                "type": "board",
                "board_id": board_id,
                "closed": closed,
                "organization_id": org_id,
                "list_count": len(lists),
                "list_names": list_names,
            },
        )
        self._cache_put(evidence.id, evidence)
        return evidence

    # ------------------------------------------------------------------
    # Board-level card listing (production feature)
    # ------------------------------------------------------------------

    async def get_board_cards(
        self,
        board_id: str,
        *,
        limit: int = 50,
        filter_: str = "open",
    ) -> list[Evidence]:
        """Retrieve cards belonging to a board.

        Uses ``GET /boards/{id}/cards/{filter}`` which returns cards
        directly without the search overhead.

        Args:
            board_id: Trello board ID.
            limit: Maximum cards to return (capped at 100).
            filter_: Card filter -- "open", "closed", "all".

        Returns:
            List of Evidence objects for matching cards.
        """
        if not self._configured:
            return []

        auth = self._get_auth_params()
        capped_limit = min(limit, 100)
        filter_value = filter_ if filter_ in ("open", "closed", "all") else "open"
        await self._rate_limit()

        async def _do_request() -> Any:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.get(
                    f"{_TRELLO_API_BASE}/boards/{board_id}/cards/{filter_value}",
                    params={
                        **auth,
                        "fields": "name,desc,url,dateLastActivity,idBoard,idList,labels,closed",
                    },
                )
                resp.raise_for_status()
                return resp.json()

        try:
            cards = await self._request_with_retry(_do_request, "get_board_cards")
        except ConnectorAuthError:
            logger.warning("Trello board cards auth failure for board %s", board_id)
            return []
        except (ConnectorError, httpx.HTTPError, OSError, ValueError):
            logger.warning("Trello board cards fetch failed for board %s", board_id, exc_info=True)
            return []

        results: list[Evidence] = []
        for card in cards[:capped_limit]:
            card_id = card.get("id", "")
            card_name = card.get("name", "")
            card_desc = card.get("desc", "")
            card_url = card.get("url", "")
            date_activity = card.get("dateLastActivity", "")
            list_id = card.get("idList", "")
            labels = [lbl.get("name", "") for lbl in card.get("labels", []) if lbl.get("name")]
            closed = card.get("closed", False)

            results.append(
                Evidence(
                    id=f"trello_card_{card_id}",
                    source_type=self.source_type,
                    source_id=f"trello://cards/{card_id}",
                    content=f"# {card_name}\n\n{card_desc[:2000]}" if card_desc else card_name,
                    title=f"Card: {card_name}",
                    url=card_url,
                    created_at=date_activity,
                    confidence=0.7,
                    freshness=self.calculate_freshness(date_activity) if date_activity else 1.0,
                    authority=0.6,
                    metadata={
                        "type": "card",
                        "card_id": card_id,
                        "board_id": board_id,
                        "list_id": list_id,
                        "labels": labels,
                        "closed": closed,
                    },
                )
            )
        return results
