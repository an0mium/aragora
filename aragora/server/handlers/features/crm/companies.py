# mypy: ignore-errors
"""
CRM Company Operations - Mixin for Company-Related API Endpoints.

This module provides company management functionality for the CRM handler:
- List companies (all platforms or single platform)
- Get individual company
- Create company
- Fetch platform companies (internal helper)

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
    MAX_COMPANY_NAME_LENGTH,
    MAX_DOMAIN_LENGTH,
)

if TYPE_CHECKING:
    from .handler import CRMHandler

logger = logging.getLogger(__name__)


class CompanyOperationsMixin:
    """Mixin providing company operations for CRMHandler."""

    async def _list_all_companies(self: CRMHandler, request: Any) -> HandlerResult:
        """List companies from all connected platforms."""
        # Check circuit breaker
        if err := self._check_circuit_breaker():
            return err

        limit = safe_query_int(request.query, "limit", default=100, max_val=1000)

        all_companies: list[dict[str, Any]] = []
        cb = get_crm_circuit_breaker()

        from .handler import _platform_credentials
        from .models import SUPPORTED_PLATFORMS

        tasks = []
        for platform in _platform_credentials.keys():
            if not SUPPORTED_PLATFORMS.get(platform, {}).get("coming_soon"):
                tasks.append(self._fetch_platform_companies(platform, limit))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        has_failure = False
        for platform, result in zip(_platform_credentials.keys(), results):
            if isinstance(result, BaseException):
                logger.error("Error fetching companies from %s: %s", platform, result)
                has_failure = True
                continue
            all_companies.extend(result)

        if has_failure:
            cb.record_failure()
        elif _platform_credentials:
            cb.record_success()

        return self._json_response(
            200,
            {
                "companies": all_companies[:limit],
                "total": len(all_companies),
            },
        )

    async def _fetch_platform_companies(
        self: CRMHandler,
        platform: str,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Fetch companies from a specific platform."""
        connector = await self._get_connector(platform)
        if not connector:
            return []

        cb = get_crm_circuit_breaker()

        try:
            if platform == "hubspot":
                companies = await connector.get_companies(limit=limit)
                cb.record_success()
                return [self._normalize_hubspot_company(c) for c in companies]

        except (ConnectionError, TimeoutError, OSError, ValueError) as e:
            logger.error("Error fetching %s companies: %s", platform, e)
            cb.record_failure()

        return []

    async def _list_platform_companies(
        self: CRMHandler, request: Any, platform: str
    ) -> HandlerResult:
        """List companies from a specific platform."""
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
        companies = await self._fetch_platform_companies(platform, limit)

        return self._json_response(
            200,
            {
                "companies": companies,
                "total": len(companies),
                "platform": platform,
            },
        )

    async def _get_company(
        self: CRMHandler, request: Any, platform: str, company_id: str
    ) -> HandlerResult:
        """Get a specific company."""
        # Check circuit breaker
        if err := self._check_circuit_breaker():
            return err

        # Validate platform and company ID
        valid, err = validate_platform_id(platform)
        if not valid:
            return self._error_response(400, err)

        valid, err = validate_resource_id(company_id, "Company ID")
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
                company = await connector.get_company(company_id)
                cb.record_success()
                return self._json_response(200, self._normalize_hubspot_company(company))

        except (ConnectionError, TimeoutError, OSError, ValueError) as e:
            logger.warning("CRM get_company failed for %s/%s: %s", platform, company_id, e)
            cb.record_failure()
            return self._error_response(404, "Company not found")

        return self._error_response(400, "Unsupported platform")

    async def _create_company(self: CRMHandler, request: Any, platform: str) -> HandlerResult:
        """Create a new company."""
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
        except (ValueError, KeyError, TypeError) as e:
            logger.warning("CRM create_company: invalid JSON body: %s", e)
            return self._error_response(400, "Invalid request body")

        # Validate company fields
        name = body.get("name")
        valid, err = validate_string_field(
            name, "Company name", MAX_COMPANY_NAME_LENGTH, required=True
        )
        if not valid:
            return self._error_response(400, err)

        domain = body.get("domain")
        valid, err = validate_string_field(domain, "Domain", MAX_DOMAIN_LENGTH)
        if not valid:
            return self._error_response(400, err)

        connector = await self._get_connector(platform)
        if not connector:
            return self._error_response(500, f"Could not initialize {platform} connector")

        cb = get_crm_circuit_breaker()

        try:
            if platform == "hubspot":
                properties = {
                    "name": name,
                    "domain": domain,
                    "industry": body.get("industry"),
                    "numberofemployees": body.get("employee_count"),
                    "annualrevenue": body.get("annual_revenue"),
                }
                properties = {k: v for k, v in properties.items() if v is not None}

                company = await connector.create_company(properties)
                cb.record_success()
                return self._json_response(201, self._normalize_hubspot_company(company))

        except (ConnectionError, TimeoutError, OSError, ValueError) as e:
            logger.error("CRM create_company failed for %s: %s", platform, e, exc_info=True)
            cb.record_failure()
            return self._error_response(500, "Company creation failed")

        return self._error_response(400, "Unsupported platform")


__all__ = ["CompanyOperationsMixin"]
