# mypy: ignore-errors
"""
CRM Deal Operations - Mixin for Deal-Related API Endpoints.

This module provides deal management functionality for the CRM handler:
- List deals (all platforms or single platform)
- Get individual deal
- Create deal
- Fetch platform deals (internal helper)

Stability: STABLE
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, TYPE_CHECKING

from aragora.server.handlers.base import HandlerResult
from aragora.server.validation.query_params import safe_query_int

from .circuit_breaker import get_crm_circuit_breaker
from .validation import (
    validate_platform_id,
    validate_resource_id,
    validate_string_field,
    validate_amount,
    MAX_DEAL_NAME_LENGTH,
    MAX_STAGE_LENGTH,
    MAX_PIPELINE_LENGTH,
)

if TYPE_CHECKING:
    from .handler import CRMHandler

logger = logging.getLogger(__name__)


class DealOperationsMixin:
    """Mixin providing deal operations for CRMHandler."""

    async def _list_all_deals(self: "CRMHandler", request: Any) -> HandlerResult:
        """List deals from all connected platforms."""
        # Check circuit breaker
        if err := self._check_circuit_breaker():
            return err

        limit = safe_query_int(request.query, "limit", default=100, max_val=1000)
        stage = request.query.get("stage")

        # Validate stage if provided
        if stage:
            valid, err = validate_string_field(stage, "Stage", MAX_STAGE_LENGTH)
            if not valid:
                return self._error_response(400, err)

        all_deals: list[dict[str, Any]] = []
        cb = get_crm_circuit_breaker()

        from .handler import _platform_credentials
        from .models import SUPPORTED_PLATFORMS

        tasks = []
        for platform in _platform_credentials.keys():
            if not SUPPORTED_PLATFORMS.get(platform, {}).get("coming_soon"):
                tasks.append(self._fetch_platform_deals(platform, limit, stage))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        has_failure = False
        for platform, result in zip(_platform_credentials.keys(), results):
            if isinstance(result, BaseException):
                logger.error(f"Error fetching deals from {platform}: {result}")
                has_failure = True
                continue
            all_deals.extend(result)

        if has_failure:
            cb.record_failure()
        elif _platform_credentials:
            cb.record_success()

        return self._json_response(
            200,
            {
                "deals": all_deals[:limit],
                "total": len(all_deals),
            },
        )

    async def _fetch_platform_deals(
        self: "CRMHandler",
        platform: str,
        limit: int = 100,
        stage: str | None = None,
    ) -> list[dict[str, Any]]:
        """Fetch deals from a specific platform."""
        connector = await self._get_connector(platform)
        if not connector:
            return []

        cb = get_crm_circuit_breaker()

        try:
            if platform == "hubspot":
                deals = await connector.get_deals(limit=limit)
                cb.record_success()
                normalized = [self._normalize_hubspot_deal(d) for d in deals]
                if stage:
                    normalized = [d for d in normalized if d.get("stage") == stage]
                return normalized

        except Exception as e:
            logger.error(f"Error fetching {platform} deals: {e}")
            cb.record_failure()

        return []

    async def _list_platform_deals(
        self: "CRMHandler", request: Any, platform: str
    ) -> HandlerResult:
        """List deals from a specific platform."""
        # Check circuit breaker
        if err := self._check_circuit_breaker():
            return err

        # Validate platform ID
        valid, err = validate_platform_id(platform)
        if not valid:
            return self._error_response(400, err)

        from .handler import _platform_credentials

        if platform not in _platform_credentials:
            return self._error_response(404, f"Platform {platform} is not connected")

        limit = safe_query_int(request.query, "limit", default=100, max_val=1000)
        stage = request.query.get("stage")

        # Validate stage if provided
        if stage:
            valid, err = validate_string_field(stage, "Stage", MAX_STAGE_LENGTH)
            if not valid:
                return self._error_response(400, err)

        deals = await self._fetch_platform_deals(platform, limit, stage)

        return self._json_response(
            200,
            {
                "deals": deals,
                "total": len(deals),
                "platform": platform,
            },
        )

    async def _get_deal(
        self: "CRMHandler", request: Any, platform: str, deal_id: str
    ) -> HandlerResult:
        """Get a specific deal."""
        # Check circuit breaker
        if err := self._check_circuit_breaker():
            return err

        # Validate platform and deal ID
        valid, err = validate_platform_id(platform)
        if not valid:
            return self._error_response(400, err)

        valid, err = validate_resource_id(deal_id, "Deal ID")
        if not valid:
            return self._error_response(400, err)

        from .handler import _platform_credentials

        if platform not in _platform_credentials:
            return self._error_response(404, f"Platform {platform} is not connected")

        connector = await self._get_connector(platform)
        if not connector:
            return self._error_response(500, f"Could not initialize {platform} connector")

        cb = get_crm_circuit_breaker()

        try:
            if platform == "hubspot":
                deal = await connector.get_deal(deal_id)
                cb.record_success()
                return self._json_response(200, self._normalize_hubspot_deal(deal))

        except Exception as e:
            logger.warning("CRM get_deal failed for %s/%s: %s", platform, deal_id, e)
            cb.record_failure()
            return self._error_response(404, f"Deal not found: {e}")

        return self._error_response(400, "Unsupported platform")

    async def _create_deal(self: "CRMHandler", request: Any, platform: str) -> HandlerResult:
        """Create a new deal."""
        # Check circuit breaker
        if err := self._check_circuit_breaker():
            return err

        # Validate platform ID
        valid, err = validate_platform_id(platform)
        if not valid:
            return self._error_response(400, err)

        from .handler import _platform_credentials

        if platform not in _platform_credentials:
            return self._error_response(404, f"Platform {platform} is not connected")

        try:
            body = await self._get_json_body(request)
        except Exception as e:
            logger.warning("CRM create_deal: invalid JSON body: %s", e)
            return self._error_response(400, f"Invalid JSON body: {e}")

        # Validate deal fields
        name = body.get("name")
        valid, err = validate_string_field(name, "Deal name", MAX_DEAL_NAME_LENGTH, required=True)
        if not valid:
            return self._error_response(400, err)

        stage = body.get("stage")
        valid, err = validate_string_field(stage, "Stage", MAX_STAGE_LENGTH, required=True)
        if not valid:
            return self._error_response(400, err)

        pipeline = body.get("pipeline")
        valid, err = validate_string_field(pipeline, "Pipeline", MAX_PIPELINE_LENGTH)
        if not valid:
            return self._error_response(400, err)

        # Validate amount if provided
        amount = body.get("amount")
        valid, err, _ = validate_amount(amount)
        if not valid:
            return self._error_response(400, err)

        connector = await self._get_connector(platform)
        if not connector:
            return self._error_response(500, f"Could not initialize {platform} connector")

        cb = get_crm_circuit_breaker()

        try:
            if platform == "hubspot":
                properties = {
                    "dealname": name,
                    "amount": amount,
                    "dealstage": stage,
                    "pipeline": pipeline or "default",
                    "closedate": body.get("close_date"),
                }
                properties = {k: v for k, v in properties.items() if v is not None}

                deal = await connector.create_deal(properties)
                cb.record_success()
                return self._json_response(201, self._normalize_hubspot_deal(deal))

        except Exception as e:
            logger.error("CRM create_deal failed for %s: %s", platform, e, exc_info=True)
            cb.record_failure()
            return self._error_response(500, f"Failed to create deal: {e}")

        return self._error_response(400, "Unsupported platform")


__all__ = ["DealOperationsMixin"]
