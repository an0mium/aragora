"""Supabase Connector.

Provides integration with Supabase for database, auth, and storage.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from aragora.connectors.base import BaseConnector, Evidence
from aragora.reasoning.provenance import SourceType

logger = logging.getLogger(__name__)

CONFIG_ENV_VARS = ("SUPABASE_URL", "SUPABASE_KEY")


class SupabaseConnector(BaseConnector):
    """Supabase connector for database, auth, and storage operations."""

    def __init__(self) -> None:
        super().__init__()
        self._configured = all(os.environ.get(v) for v in CONFIG_ENV_VARS)

    @property
    def name(self) -> str:
        return "supabase"

    @property
    def source_type(self) -> SourceType:
        return SourceType.DATABASE

    @property
    def is_configured(self) -> bool:
        return self._configured

    async def search(self, query: str, limit: int = 10, **kwargs: Any) -> list[Evidence]:
        """Search Supabase tables for relevant data."""
        if not self._configured:
            logger.debug("Supabase connector not configured")
            return []
        # TODO: Implement Supabase API search
        return []

    async def fetch(self, evidence_id: str, **kwargs: Any) -> Evidence | None:
        """Fetch a specific record from Supabase."""
        if not self._configured:
            return None
        # TODO: Implement Supabase API fetch
        return None
