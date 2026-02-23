"""Twilio Connector.

Provides integration with Twilio for SMS and voice communications.
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

CONFIG_ENV_VARS = ("TWILIO_ACCOUNT_SID", "TWILIO_AUTH_TOKEN")

_SAFE_QUERY_RE = re.compile(r"[^\w\s@.\-+]")
MAX_QUERY_LENGTH = 500

_TWILIO_API_BASE = "https://api.twilio.com/2010-04-01"


def _sanitize_query(query: str) -> str:
    """Sanitize query to prevent injection."""
    query = query[:MAX_QUERY_LENGTH]
    return _SAFE_QUERY_RE.sub("", query)


class TwilioConnector(BaseConnector):
    """Twilio connector for SMS and voice communications."""

    def __init__(self) -> None:
        super().__init__()
        self._configured = all(os.environ.get(v) for v in CONFIG_ENV_VARS)

    @property
    def name(self) -> str:
        return "twilio"

    @property
    def source_type(self) -> SourceType:
        return SourceType.EXTERNAL_API

    @property
    def is_configured(self) -> bool:
        return self._configured

    def _get_auth(self) -> tuple[str, str]:
        account_sid = os.environ.get("TWILIO_ACCOUNT_SID", "")
        auth_token = os.environ.get("TWILIO_AUTH_TOKEN", "")
        return (account_sid, auth_token)

    def _get_account_sid(self) -> str:
        return os.environ.get("TWILIO_ACCOUNT_SID", "")

    async def search(self, query: str, limit: int = 10, **kwargs: Any) -> list[Evidence]:
        """Search Twilio message logs for relevant data.

        Raises:
            NotImplementedError: Twilio connector is not yet implemented.
        """
        raise NotImplementedError(
            "TwilioConnector.search() is not yet implemented. "
            "Configure TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN and contribute an implementation."
        )

    async def fetch(self, evidence_id: str, **kwargs: Any) -> Evidence | None:
        """Fetch a specific message from Twilio.

        Raises:
            NotImplementedError: Twilio connector is not yet implemented.
        """
        raise NotImplementedError(
            "TwilioConnector.fetch() is not yet implemented. "
            "Configure TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN and contribute an implementation."
        )
