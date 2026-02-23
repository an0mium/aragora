"""QuickBooks Connector (DEPRECATED).

This module is superseded by ``aragora.connectors.accounting.qbo`` which
provides a full QuickBooks Online integration with OAuth 2.0, transaction
sync, multi-company support, and more.

Import ``QuickBooksConnector`` from the package or from ``qbo`` instead::

    from aragora.connectors.accounting import QuickBooksConnector
    # or
    from aragora.connectors.accounting.qbo import QuickBooksConnector
"""

from __future__ import annotations

import logging
import os
import re
import warnings
from typing import Any

import httpx

from aragora.connectors.base import BaseConnector, Evidence
from aragora.reasoning.provenance import SourceType

logger = logging.getLogger(__name__)

CONFIG_ENV_VARS = (
    "QUICKBOOKS_CLIENT_ID",
    "QUICKBOOKS_CLIENT_SECRET",
    "QUICKBOOKS_ACCESS_TOKEN",
    "QUICKBOOKS_REALM_ID",
)

_SAFE_QUERY_RE = re.compile(r"[^\w\s@.\-+]")
MAX_QUERY_LENGTH = 500

_QB_API_BASE = "https://quickbooks.api.intuit.com"


def _sanitize_query(query: str) -> str:
    """Sanitize query to prevent injection."""
    query = query[:MAX_QUERY_LENGTH]
    return _SAFE_QUERY_RE.sub("", query)


class QuickBooksConnector(BaseConnector):
    """QuickBooks connector for accounting, invoices, and payments.

    .. deprecated::
        This class is superseded by
        ``aragora.connectors.accounting.qbo.QuickBooksConnector``.
        Use ``from aragora.connectors.accounting import QuickBooksConnector``
        to get the canonical (qbo) implementation.
    """

    def __init__(self) -> None:
        super().__init__()
        self._configured = all(os.environ.get(v) for v in CONFIG_ENV_VARS)
        warnings.warn(
            "aragora.connectors.accounting.quickbooks.QuickBooksConnector is "
            "deprecated. Use aragora.connectors.accounting.qbo.QuickBooksConnector "
            "instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        logger.warning(
            "Using deprecated quickbooks.QuickBooksConnector. "
            "Migrate to aragora.connectors.accounting.qbo.QuickBooksConnector."
        )

    @property
    def name(self) -> str:
        return "quickbooks"

    @property
    def source_type(self) -> SourceType:
        return SourceType.EXTERNAL_API

    @property
    def is_configured(self) -> bool:
        return self._configured

    def _get_headers(self) -> dict[str, str]:
        token = os.environ.get("QUICKBOOKS_ACCESS_TOKEN", "")
        return {
            "Authorization": f"Bearer {token}",
            "Accept": "application/json",
        }

    def _get_realm_id(self) -> str:
        return os.environ.get("QUICKBOOKS_REALM_ID", "")

    async def search(self, query: str, limit: int = 10, **kwargs: Any) -> list[Evidence]:
        """Search QuickBooks for invoices, payments, and reports."""
        if not self._configured:
            logger.debug("QuickBooks connector not configured")
            return []

        sanitized = _sanitize_query(query)
        if not sanitized.strip():
            return []

        realm_id = self._get_realm_id()
        sql_query = f"SELECT * FROM Invoice WHERE DocNumber LIKE '%{sanitized}%' MAXRESULTS {limit}"

        async def _do_request() -> Any:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.get(
                    f"{_QB_API_BASE}/v3/company/{realm_id}/query",
                    headers=self._get_headers(),
                    params={"query": sql_query},
                )
                resp.raise_for_status()
                return resp.json()

        try:
            data = await self._request_with_retry(_do_request, "search")
        except Exception:
            logger.warning("QuickBooks search failed", exc_info=True)
            return []

        results: list[Evidence] = []
        query_response = data.get("QueryResponse", {})
        invoices = query_response.get("Invoice", [])
        for inv in invoices[:limit]:
            inv_id = inv.get("Id", "")
            doc_num = inv.get("DocNumber", "")
            total = inv.get("TotalAmt", 0)
            customer = inv.get("CustomerRef", {}).get("name", "Unknown")
            results.append(
                Evidence(
                    id=f"qb_inv_{inv_id}",
                    source_type=self.source_type,
                    source_id=f"quickbooks://invoices/{inv_id}",
                    content=f"Invoice #{doc_num}: ${total} from {customer}",
                    title=f"Invoice #{doc_num}",
                    url=f"{_QB_API_BASE}/v3/company/{realm_id}/invoice/{inv_id}",
                    confidence=0.7,
                    freshness=1.0,
                    authority=0.6,
                    metadata={"invoice_id": inv_id, "amount": total, "customer": customer},
                )
            )
        return results

    async def fetch(self, evidence_id: str, **kwargs: Any) -> Evidence | None:
        """Fetch a specific invoice or payment from QuickBooks."""
        if not self._configured:
            return None

        realm_id = self._get_realm_id()

        async def _do_request() -> Any:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.get(
                    f"{_QB_API_BASE}/v3/company/{realm_id}/invoice/{evidence_id}",
                    headers=self._get_headers(),
                )
                resp.raise_for_status()
                return resp.json()

        try:
            data = await self._request_with_retry(_do_request, "fetch")
        except Exception:
            logger.warning("QuickBooks fetch failed", exc_info=True)
            return None

        inv = data.get("Invoice", {})
        inv_id = inv.get("Id", evidence_id)
        doc_num = inv.get("DocNumber", "")
        total = inv.get("TotalAmt", 0)
        customer = inv.get("CustomerRef", {}).get("name", "Unknown")
        return Evidence(
            id=f"qb_inv_{inv_id}",
            source_type=self.source_type,
            source_id=f"quickbooks://invoices/{inv_id}",
            content=f"Invoice #{doc_num}: ${total} from {customer}",
            title=f"Invoice #{doc_num}",
            url=f"{_QB_API_BASE}/v3/company/{realm_id}/invoice/{inv_id}",
            confidence=0.7,
            freshness=1.0,
            authority=0.6,
            metadata={"invoice_id": inv_id, "amount": total, "customer": customer},
        )
