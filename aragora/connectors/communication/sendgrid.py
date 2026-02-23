"""SendGrid Connector.

Provides integration with SendGrid for email delivery and templates.
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

CONFIG_ENV_VARS = ("SENDGRID_API_KEY",)

_SAFE_QUERY_RE = re.compile(r"[^\w\s@.\-+]")
MAX_QUERY_LENGTH = 500

_SG_API_BASE = "https://api.sendgrid.com/v3"


def _sanitize_query(query: str) -> str:
    """Sanitize query to prevent injection."""
    query = query[:MAX_QUERY_LENGTH]
    return _SAFE_QUERY_RE.sub("", query)


class SendGridConnector(BaseConnector):
    """SendGrid connector for email delivery and template management."""

    def __init__(self) -> None:
        super().__init__()
        self._configured = all(os.environ.get(v) for v in CONFIG_ENV_VARS)

    @property
    def name(self) -> str:
        return "sendgrid"

    @property
    def source_type(self) -> SourceType:
        return SourceType.EXTERNAL_API

    @property
    def is_configured(self) -> bool:
        return self._configured

    def _get_headers(self) -> dict[str, str]:
        api_key = os.environ.get("SENDGRID_API_KEY", "")
        return {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

    async def search(self, query: str, limit: int = 10, **kwargs: Any) -> list[Evidence]:
        """Search SendGrid activity for relevant email data.

        Raises:
            NotImplementedError: SendGrid connector is not yet implemented.
        """
        raise NotImplementedError(
            "SendGridConnector.search() is not yet implemented. "
            "Configure SENDGRID_API_KEY and contribute an implementation."
        )

    async def fetch(self, evidence_id: str, **kwargs: Any) -> Evidence | None:
        """Fetch a specific email or template from SendGrid.

        Raises:
            NotImplementedError: SendGrid connector is not yet implemented.
        """
        raise NotImplementedError(
            "SendGridConnector.fetch() is not yet implemented. "
            "Configure SENDGRID_API_KEY and contribute an implementation."
        )
