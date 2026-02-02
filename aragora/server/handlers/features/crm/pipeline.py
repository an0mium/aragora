# mypy: ignore-errors
"""
CRM Pipeline Operations - Mixin for Pipeline, Lead Sync, Enrichment, and Search.

This module provides additional CRM functionality:
- Pipeline summary and deal stages
- Lead synchronization from external sources
- Contact enrichment
- Cross-platform search

Stability: STABLE
"""

from __future__ import annotations

import logging
from typing import Any, TYPE_CHECKING

from aragora.server.handlers.base import HandlerResult

from .circuit_breaker import get_crm_circuit_breaker
from .validation import (
    validate_platform_id,
    validate_email,
    validate_string_field,
    MAX_SEARCH_QUERY_LENGTH,
)

if TYPE_CHECKING:
    from .handler import CRMHandler

logger = logging.getLogger(__name__)


class PipelineOperationsMixin:
    """Mixin providing pipeline, lead sync, enrichment and search operations."""

    async def _get_pipeline(self: "CRMHandler", request: Any) -> HandlerResult:
        """Get sales pipeline summary."""
        # Check circuit breaker
        if err := self._check_circuit_breaker():
            return err

        platform = request.query.get("platform")

        # Validate platform if provided
        if platform:
            valid, err = validate_platform_id(platform)
            if not valid:
                return self._error_response(400, err)

        from .handler import _platform_credentials
        from .models import SUPPORTED_PLATFORMS

        if platform and platform not in _platform_credentials:
            return self._error_response(404, f"Platform {platform} is not connected")

        pipelines: list[dict[str, Any]] = []
        cb = get_crm_circuit_breaker()

        platforms_to_query = [platform] if platform else list(_platform_credentials.keys())

        for p in platforms_to_query:
            if SUPPORTED_PLATFORMS.get(p, {}).get("coming_soon"):
                continue

            connector = await self._get_connector(p)
            if not connector:
                continue

            try:
                if p == "hubspot":
                    pipeline_data = await connector.get_pipelines()
                    cb.record_success()
                    for pipe in pipeline_data:
                        pipelines.append(
                            {
                                "id": pipe.id,
                                "platform": p,
                                "name": pipe.label,
                                "stages": [
                                    {
                                        "id": s.id,
                                        "name": s.label,
                                        "display_order": s.display_order,
                                        "probability": (
                                            s.metadata.get("probability")
                                            if hasattr(s, "metadata")
                                            else None
                                        ),
                                    }
                                    for s in (pipe.stages if hasattr(pipe, "stages") else [])
                                ],
                            }
                        )

            except Exception as e:
                logger.error(f"Error fetching {p} pipelines: {e}")
                cb.record_failure()

        # Get deal summary by stage
        all_deals = await self._list_all_deals(request)
        deals = all_deals.get("body", {}).get("deals", []) if isinstance(all_deals, dict) else []

        stage_summary: dict[str, dict[str, Any]] = {}
        for deal in deals:
            stage = deal.get("stage", "unknown")
            if stage not in stage_summary:
                stage_summary[stage] = {"count": 0, "total_value": 0}
            stage_summary[stage]["count"] += 1
            stage_summary[stage]["total_value"] += deal.get("amount") or 0

        return self._json_response(
            200,
            {
                "pipelines": pipelines,
                "stage_summary": stage_summary,
                "total_deals": len(deals),
                "total_pipeline_value": sum(s["total_value"] for s in stage_summary.values()),
            },
        )

    async def _sync_lead(self: "CRMHandler", request: Any) -> HandlerResult:
        """Sync a lead from an external source (e.g., LinkedIn Ads, form submission)."""
        # Check circuit breaker
        if err := self._check_circuit_breaker():
            return err

        try:
            body = await self._get_json_body(request)
        except Exception as e:
            logger.warning("CRM sync_lead: invalid JSON body: %s", e)
            return self._error_response(400, f"Invalid JSON body: {e}")

        target_platform = body.get("platform", "hubspot")

        # Validate platform ID
        valid, err = validate_platform_id(target_platform)
        if not valid:
            return self._error_response(400, err)

        from .handler import _platform_credentials

        if target_platform not in _platform_credentials:
            return self._error_response(404, f"Platform {target_platform} is not connected")

        source = body.get("source", "api")
        lead_data = body.get("lead", {})

        # Validate lead email
        lead_email = lead_data.get("email")
        valid, err = validate_email(lead_email, required=True)
        if not valid:
            return self._error_response(400, err)

        connector = await self._get_connector(target_platform)
        if not connector:
            return self._error_response(500, f"Could not initialize {target_platform} connector")

        cb = get_crm_circuit_breaker()

        try:
            # Check if contact exists
            existing = None
            try:
                existing = await connector.get_contact_by_email(lead_email)
            except (ConnectionError, TimeoutError, OSError) as e:
                # Network errors - log and proceed with create (may cause duplicate)
                logger.warning(f"Contact lookup failed, proceeding with create: {e}")

            if existing:
                # Update existing contact
                properties = self._map_lead_to_hubspot(lead_data, source)
                contact = await connector.update_contact(existing.id, properties)
                action = "updated"
            else:
                # Create new contact
                properties = self._map_lead_to_hubspot(lead_data, source)
                contact = await connector.create_contact(properties)
                action = "created"

            cb.record_success()
            return self._json_response(
                200,
                {
                    "action": action,
                    "contact": self._normalize_hubspot_contact(contact),
                    "source": source,
                },
            )

        except Exception as e:
            logger.error("CRM sync_lead failed: %s", e, exc_info=True)
            cb.record_failure()
            return self._error_response(500, f"Failed to sync lead: {e}")

    async def _enrich_contact(self: "CRMHandler", request: Any) -> HandlerResult:
        """Enrich contact data using available sources."""
        try:
            body = await self._get_json_body(request)
        except Exception as e:
            logger.warning("CRM enrich_contact: invalid JSON body: %s", e)
            return self._error_response(400, f"Invalid JSON body: {e}")

        email = body.get("email")
        valid, err = validate_email(email, required=True)
        if not valid:
            return self._error_response(400, err)

        # Placeholder for enrichment logic
        # In production, this would integrate with services like Clearbit, ZoomInfo, etc.
        enriched_data = {
            "email": email,
            "enriched": False,
            "message": "Enrichment service integration pending",
            "available_providers": ["clearbit", "zoominfo", "apollo"],
        }

        return self._json_response(200, enriched_data)

    async def _search_crm(self: "CRMHandler", request: Any) -> HandlerResult:
        """Search across CRM data."""
        # Check circuit breaker
        if err := self._check_circuit_breaker():
            return err

        try:
            body = await self._get_json_body(request)
        except Exception as e:
            logger.warning("CRM search: invalid JSON body: %s", e)
            return self._error_response(400, f"Invalid JSON body: {e}")

        query = body.get("query", "")

        # Validate query length
        valid, err = validate_string_field(query, "Search query", MAX_SEARCH_QUERY_LENGTH)
        if not valid:
            return self._error_response(400, err)

        object_types = body.get("types", ["contacts", "companies", "deals"])

        # Validate object types
        valid_types = {"contacts", "companies", "deals"}
        for obj_type in object_types:
            if obj_type not in valid_types:
                return self._error_response(400, f"Invalid object type: {obj_type}")

        limit = body.get("limit", 20)
        if not isinstance(limit, int) or limit < 1 or limit > 100:
            return self._error_response(400, "Limit must be an integer between 1 and 100")

        results: dict[str, list[dict[str, Any]]] = {}
        cb = get_crm_circuit_breaker()

        from .handler import _platform_credentials
        from .models import SUPPORTED_PLATFORMS

        for platform in _platform_credentials.keys():
            if SUPPORTED_PLATFORMS.get(platform, {}).get("coming_soon"):
                continue

            connector = await self._get_connector(platform)
            if not connector:
                continue

            try:
                if platform == "hubspot":
                    if "contacts" in object_types:
                        contacts = await connector.search_contacts(query, limit=limit)
                        results.setdefault("contacts", []).extend(
                            [self._normalize_hubspot_contact(c) for c in contacts]
                        )

                    if "companies" in object_types:
                        companies = await connector.search_companies(query, limit=limit)
                        results.setdefault("companies", []).extend(
                            [self._normalize_hubspot_company(c) for c in companies]
                        )

                    if "deals" in object_types:
                        deals = await connector.search_deals(query, limit=limit)
                        results.setdefault("deals", []).extend(
                            [self._normalize_hubspot_deal(d) for d in deals]
                        )

                cb.record_success()

            except Exception as e:
                logger.error(f"Error searching {platform}: {e}")
                cb.record_failure()

        return self._json_response(
            200,
            {
                "query": query,
                "results": results,
                "total": sum(len(v) for v in results.values()),
            },
        )


__all__ = ["PipelineOperationsMixin"]
