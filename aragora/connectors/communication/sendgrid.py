"""SendGrid Connector.

Provides integration with SendGrid for email delivery and templates.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from aragora.connectors.base import BaseConnector, Evidence
from aragora.reasoning.provenance import SourceType

logger = logging.getLogger(__name__)

CONFIG_ENV_VARS = ("SENDGRID_API_KEY",)


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

    async def search(self, query: str, limit: int = 10, **kwargs: Any) -> list[Evidence]:
        """Search SendGrid activity for relevant email data."""
        if not self._configured:
            logger.debug("SendGrid connector not configured")
            return []
        # TODO: Implement SendGrid API search
        return []

    async def fetch(self, evidence_id: str, **kwargs: Any) -> Evidence | None:
        """Fetch a specific email or template from SendGrid."""
        if not self._configured:
            return None
        # TODO: Implement SendGrid API fetch
        return None
