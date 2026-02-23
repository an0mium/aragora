"""Trello Connector.

Provides integration with Trello for boards, cards, and lists.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any

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
        """Search Trello for boards, cards, and lists.

        Raises:
            NotImplementedError: Trello connector is not yet implemented.
        """
        raise NotImplementedError(
            "TrelloConnector.search() is not yet implemented. "
            "Configure TRELLO_API_KEY and TRELLO_TOKEN and contribute an implementation."
        )

    async def fetch(self, evidence_id: str, **kwargs: Any) -> Evidence | None:
        """Fetch a specific card or board from Trello.

        Raises:
            NotImplementedError: Trello connector is not yet implemented.
        """
        raise NotImplementedError(
            "TrelloConnector.fetch() is not yet implemented. "
            "Configure TRELLO_API_KEY and TRELLO_TOKEN and contribute an implementation."
        )
