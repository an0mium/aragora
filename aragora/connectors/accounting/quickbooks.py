"""QuickBooks Connector.

Provides integration with QuickBooks for invoices, payments, and reports.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from aragora.connectors.base import BaseConnector, Evidence
from aragora.reasoning.provenance import SourceType

logger = logging.getLogger(__name__)

CONFIG_ENV_VARS = ("QUICKBOOKS_CLIENT_ID", "QUICKBOOKS_CLIENT_SECRET")


class QuickBooksConnector(BaseConnector):
    """QuickBooks connector for accounting, invoices, and payments."""

    def __init__(self) -> None:
        super().__init__()
        self._configured = all(os.environ.get(v) for v in CONFIG_ENV_VARS)

    @property
    def name(self) -> str:
        return "quickbooks"

    @property
    def source_type(self) -> SourceType:
        return SourceType.EXTERNAL_API

    @property
    def is_configured(self) -> bool:
        return self._configured

    async def search(self, query: str, limit: int = 10, **kwargs: Any) -> list[Evidence]:
        """Search QuickBooks for invoices, payments, and reports."""
        if not self._configured:
            logger.debug("QuickBooks connector not configured")
            return []
        # TODO: Implement QuickBooks API search
        return []

    async def fetch(self, evidence_id: str, **kwargs: Any) -> Evidence | None:
        """Fetch a specific invoice or payment from QuickBooks."""
        if not self._configured:
            return None
        # TODO: Implement QuickBooks API fetch
        return None
