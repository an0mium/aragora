"""Trello Connector.

Provides integration with the Trello REST API for cards and boards.

Searches:
- Cards (via the Trello search endpoint)
- Boards (via the Trello search endpoint)

Environment Variables:
- ``TRELLO_API_KEY`` -- Trello application API key.
- ``TRELLO_TOKEN`` -- Trello member token with read access.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any

import httpx

from aragora.connectors.base import BaseConnector, Evidence
from aragora.connectors.exceptions import ConnectorError
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
    """Trello connector for card and board search/fetch."""

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
        """Search Trello for cards or boards.

        Uses the Trello ``/search`` endpoint which supports full-text
        search across cards, boards, and organizations. Pass
        ``search_type="boards"`` to search only boards.

        Args:
            query: Search term.
            limit: Maximum results (capped at 50).
            **kwargs: Optional ``search_type`` ("cards" or "boards").
                      Defaults to "cards".

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
        return await self._search_cards(sanitized, capped_limit)

    async def _search_cards(self, query: str, limit: int) -> list[Evidence]:
        """Search cards via the Trello Search API."""
        auth = self._get_auth_params()

        async def _do_request() -> Any:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.get(
                    f"{_TRELLO_API_BASE}/search",
                    params={
                        **auth,
                        "query": query,
                        "modelTypes": "cards",
                        "cards_limit": limit,
                        "card_fields": "name,desc,url,dateLastActivity,idBoard,idList,labels,closed",
                    },
                )
                resp.raise_for_status()
                return resp.json()

        try:
            data = await self._request_with_retry(_do_request, "search_cards")
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
            board_id = card.get("idBoard", "")
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

    async def _search_boards(self, query: str, limit: int) -> list[Evidence]:
        """Search boards via the Trello Search API."""
        auth = self._get_auth_params()

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
            return await self._fetch_board(evidence_id[len("trello_board_") :], evidence_id)
        elif evidence_id.startswith("trello_card_"):
            return await self._fetch_card(evidence_id[len("trello_card_") :], evidence_id)

        return None

    async def _fetch_card(self, card_id: str, evidence_id: str) -> Evidence | None:
        """Fetch a single card by ID."""
        auth = self._get_auth_params()

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
