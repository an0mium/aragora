"""
CRM Contact Operations - Mixin for Contact-Related API Endpoints.

This module provides contact management functionality for the CRM handler:
- List contacts (all platforms or single platform)
- Get individual contact
- Create contact
- Update contact
- Fetch platform contacts (internal helper)

Stability: STABLE
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, TYPE_CHECKING

from aragora.server.validation.query_params import safe_query_int

from .circuit_breaker import get_crm_circuit_breaker
from .validation import (
    validate_platform_id,
    validate_resource_id,
    validate_email,
    validate_string_field,
    MAX_NAME_LENGTH,
    MAX_PHONE_LENGTH,
    MAX_COMPANY_NAME_LENGTH,
    MAX_JOB_TITLE_LENGTH,
)

if TYPE_CHECKING:
    from .handler import CRMHandler

logger = logging.getLogger(__name__)


class ContactOperationsMixin:
    """Mixin providing contact operations for CRMHandler."""

    async def _list_all_contacts(self: "CRMHandler", request: Any) -> dict[str, Any]:
        """List contacts from all connected platforms."""
        # Check circuit breaker
        if err := self._check_circuit_breaker():
            return err

        limit = safe_query_int(request.query, "limit", default=100, max_val=1000)
        email = request.query.get("email")

        # Validate email format if provided
        if email:
            valid, err = validate_email(email)
            if not valid:
                return self._error_response(400, err)

        all_contacts: list[dict[str, Any]] = []
        cb = get_crm_circuit_breaker()

        from .handler import _platform_credentials
        from .models import SUPPORTED_PLATFORMS

        tasks = []
        for platform in _platform_credentials.keys():
            if not SUPPORTED_PLATFORMS.get(platform, {}).get("coming_soon"):
                tasks.append(self._fetch_platform_contacts(platform, limit, email))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        has_failure = False
        for platform, result in zip(_platform_credentials.keys(), results):
            if isinstance(result, BaseException):
                logger.error(f"Error fetching contacts from {platform}: {result}")
                has_failure = True
                continue
            all_contacts.extend(result)

        # Record circuit breaker status
        if has_failure:
            cb.record_failure()
        elif _platform_credentials:
            cb.record_success()

        return self._json_response(
            200,
            {
                "contacts": all_contacts[:limit],
                "total": len(all_contacts),
                "platforms_queried": list(_platform_credentials.keys()),
            },
        )

    async def _fetch_platform_contacts(
        self: "CRMHandler",
        platform: str,
        limit: int = 100,
        email: str | None = None,
    ) -> list[dict[str, Any]]:
        """Fetch contacts from a specific platform."""
        connector = await self._get_connector(platform)
        if not connector:
            return []

        cb = get_crm_circuit_breaker()

        try:
            if platform == "hubspot":
                if email:
                    contact = await connector.get_contact_by_email(email)
                    cb.record_success()
                    if contact:
                        return [self._normalize_hubspot_contact(contact)]
                    return []
                else:
                    contacts = await connector.get_contacts(limit=limit)
                    cb.record_success()
                    return [self._normalize_hubspot_contact(c) for c in contacts]

        except Exception as e:
            logger.error(f"Error fetching {platform} contacts: {e}")
            cb.record_failure()

        return []

    async def _list_platform_contacts(
        self: "CRMHandler",
        request: Any,
        platform: str,
    ) -> dict[str, Any]:
        """List contacts from a specific platform."""
        # Check circuit breaker
        if err := self._check_circuit_breaker():
            return err

        # Validate platform ID format
        valid, err = validate_platform_id(platform)
        if not valid:
            return self._error_response(400, err)

        from .handler import _platform_credentials

        if platform not in _platform_credentials:
            return self._error_response(404, f"Platform {platform} is not connected")

        limit = safe_query_int(request.query, "limit", default=100, max_val=1000)
        email = request.query.get("email")

        # Validate email format if provided
        if email:
            valid, err = validate_email(email)
            if not valid:
                return self._error_response(400, err)

        contacts = await self._fetch_platform_contacts(platform, limit, email)

        return self._json_response(
            200,
            {
                "contacts": contacts,
                "total": len(contacts),
                "platform": platform,
            },
        )

    async def _get_contact(
        self: "CRMHandler",
        request: Any,
        platform: str,
        contact_id: str,
    ) -> dict[str, Any]:
        """Get a specific contact."""
        # Check circuit breaker
        if err := self._check_circuit_breaker():
            return err

        # Validate platform ID format
        valid, err = validate_platform_id(platform)
        if not valid:
            return self._error_response(400, err)

        # Validate contact ID format
        valid, err = validate_resource_id(contact_id, "Contact ID")
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
                contact = await connector.get_contact(contact_id)
                cb.record_success()
                return self._json_response(200, self._normalize_hubspot_contact(contact))

        except Exception as e:
            logger.warning("CRM get_contact failed for %s/%s: %s", platform, contact_id, e)
            cb.record_failure()
            return self._error_response(404, f"Contact not found: {e}")

        return self._error_response(400, "Unsupported platform")

    async def _create_contact(
        self: "CRMHandler",
        request: Any,
        platform: str,
    ) -> dict[str, Any]:
        """Create a new contact."""
        # Check circuit breaker
        if err := self._check_circuit_breaker():
            return err

        # Validate platform ID format
        valid, err = validate_platform_id(platform)
        if not valid:
            return self._error_response(400, err)

        from .handler import _platform_credentials

        if platform not in _platform_credentials:
            return self._error_response(404, f"Platform {platform} is not connected")

        try:
            body = await self._get_json_body(request)
        except Exception as e:
            logger.warning("CRM create_contact: invalid JSON body: %s", e)
            return self._error_response(400, f"Invalid JSON body: {e}")

        # Validate contact fields
        email = body.get("email")
        valid, err = validate_email(email)
        if not valid:
            return self._error_response(400, err)

        first_name = body.get("first_name")
        valid, err = validate_string_field(first_name, "First name", MAX_NAME_LENGTH)
        if not valid:
            return self._error_response(400, err)

        last_name = body.get("last_name")
        valid, err = validate_string_field(last_name, "Last name", MAX_NAME_LENGTH)
        if not valid:
            return self._error_response(400, err)

        phone = body.get("phone")
        valid, err = validate_string_field(phone, "Phone", MAX_PHONE_LENGTH)
        if not valid:
            return self._error_response(400, err)

        company = body.get("company")
        valid, err = validate_string_field(company, "Company", MAX_COMPANY_NAME_LENGTH)
        if not valid:
            return self._error_response(400, err)

        job_title = body.get("job_title")
        valid, err = validate_string_field(job_title, "Job title", MAX_JOB_TITLE_LENGTH)
        if not valid:
            return self._error_response(400, err)

        connector = await self._get_connector(platform)
        if not connector:
            return self._error_response(500, f"Could not initialize {platform} connector")

        cb = get_crm_circuit_breaker()

        try:
            if platform == "hubspot":
                properties = {
                    "email": email,
                    "firstname": first_name,
                    "lastname": last_name,
                    "phone": phone,
                    "company": company,
                    "jobtitle": job_title,
                }
                # Remove None values
                properties = {k: v for k, v in properties.items() if v is not None}

                contact = await connector.create_contact(properties)
                cb.record_success()
                return self._json_response(201, self._normalize_hubspot_contact(contact))

        except Exception as e:
            logger.error("CRM create_contact failed for %s: %s", platform, e, exc_info=True)
            cb.record_failure()
            return self._error_response(500, f"Failed to create contact: {e}")

        return self._error_response(400, "Unsupported platform")

    async def _update_contact(
        self: "CRMHandler",
        request: Any,
        platform: str,
        contact_id: str,
    ) -> dict[str, Any]:
        """Update an existing contact."""
        # Check circuit breaker
        if err := self._check_circuit_breaker():
            return err

        # Validate platform ID format
        valid, err = validate_platform_id(platform)
        if not valid:
            return self._error_response(400, err)

        # Validate contact ID format
        valid, err = validate_resource_id(contact_id, "Contact ID")
        if not valid:
            return self._error_response(400, err)

        from .handler import _platform_credentials

        if platform not in _platform_credentials:
            return self._error_response(404, f"Platform {platform} is not connected")

        try:
            body = await self._get_json_body(request)
        except Exception as e:
            logger.warning("CRM update_contact: invalid JSON body: %s", e)
            return self._error_response(400, f"Invalid JSON body: {e}")

        # Validate contact fields if provided
        if "email" in body:
            valid, err = validate_email(body["email"])
            if not valid:
                return self._error_response(400, err)

        if "first_name" in body:
            valid, err = validate_string_field(body["first_name"], "First name", MAX_NAME_LENGTH)
            if not valid:
                return self._error_response(400, err)

        if "last_name" in body:
            valid, err = validate_string_field(body["last_name"], "Last name", MAX_NAME_LENGTH)
            if not valid:
                return self._error_response(400, err)

        if "phone" in body:
            valid, err = validate_string_field(body["phone"], "Phone", MAX_PHONE_LENGTH)
            if not valid:
                return self._error_response(400, err)

        if "company" in body:
            valid, err = validate_string_field(body["company"], "Company", MAX_COMPANY_NAME_LENGTH)
            if not valid:
                return self._error_response(400, err)

        if "job_title" in body:
            valid, err = validate_string_field(body["job_title"], "Job title", MAX_JOB_TITLE_LENGTH)
            if not valid:
                return self._error_response(400, err)

        connector = await self._get_connector(platform)
        if not connector:
            return self._error_response(500, f"Could not initialize {platform} connector")

        cb = get_crm_circuit_breaker()

        try:
            if platform == "hubspot":
                properties = {}
                field_mapping = {
                    "email": "email",
                    "first_name": "firstname",
                    "last_name": "lastname",
                    "phone": "phone",
                    "company": "company",
                    "job_title": "jobtitle",
                }
                for api_field, hubspot_field in field_mapping.items():
                    if api_field in body:
                        properties[hubspot_field] = body[api_field]

                contact = await connector.update_contact(contact_id, properties)
                cb.record_success()
                return self._json_response(200, self._normalize_hubspot_contact(contact))

        except Exception as e:
            logger.error(
                "CRM update_contact failed for %s/%s: %s", platform, contact_id, e, exc_info=True
            )
            cb.record_failure()
            return self._error_response(500, f"Failed to update contact: {e}")

        return self._error_response(400, "Unsupported platform")
