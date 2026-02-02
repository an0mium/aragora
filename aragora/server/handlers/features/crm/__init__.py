"""
CRM Handler Package - Customer Relationship Management API Handlers.

This package provides API endpoints for CRM platform integration:
- HubSpot (contacts, companies, deals, marketing)
- Salesforce - planned
- Pipedrive - planned

The package is structured as follows:
- validation.py: Input validation constants and functions
- circuit_breaker.py: Circuit breaker for resilient platform access
- models.py: Unified data models (UnifiedContact, UnifiedCompany, UnifiedDeal)
- contacts.py: Contact operations mixin
- companies.py: Company operations mixin
- deals.py: Deal operations mixin
- pipeline.py: Pipeline, lead sync, enrichment, and search operations
- handler.py: Main CRMHandler class

For backwards compatibility, all public exports are available directly from this package.

Stability: STABLE

Example usage:
    from aragora.server.handlers.features.crm import CRMHandler
    from aragora.server.handlers.features.crm import CRMCircuitBreaker
    from aragora.server.handlers.features.crm import get_crm_circuit_breaker

Usage:
    GET    /api/v1/crm/platforms                  - List connected platforms
    POST   /api/v1/crm/connect                    - Connect a platform
    DELETE /api/v1/crm/{platform}                 - Disconnect platform

    GET    /api/v1/crm/contacts                   - List contacts (cross-platform)
    GET    /api/v1/crm/{platform}/contacts        - Platform contacts
    POST   /api/v1/crm/{platform}/contacts        - Create contact
    PUT    /api/v1/crm/{platform}/contacts/{id}   - Update contact

    GET    /api/v1/crm/companies                  - List companies
    GET    /api/v1/crm/deals                      - List deals/opportunities
    GET    /api/v1/crm/pipeline                   - Get sales pipeline

    POST   /api/v1/crm/sync-lead                  - Sync lead from external source
    POST   /api/v1/crm/enrich                     - Enrich contact data
"""

from __future__ import annotations

# Import main handler class
from .handler import CRMHandler, _platform_credentials, _platform_connectors

# Import circuit breaker
from .circuit_breaker import (
    CRMCircuitBreaker,
    get_crm_circuit_breaker,
    reset_crm_circuit_breaker,
)

# Import models
from .models import (
    SUPPORTED_PLATFORMS,
    UnifiedContact,
    UnifiedCompany,
    UnifiedDeal,
)

# Import validation utilities
from .validation import (
    # Constants
    SAFE_PLATFORM_PATTERN,
    SAFE_RESOURCE_ID_PATTERN,
    EMAIL_PATTERN,
    MAX_EMAIL_LENGTH,
    MAX_NAME_LENGTH,
    MAX_PHONE_LENGTH,
    MAX_COMPANY_NAME_LENGTH,
    MAX_JOB_TITLE_LENGTH,
    MAX_DOMAIN_LENGTH,
    MAX_DEAL_NAME_LENGTH,
    MAX_STAGE_LENGTH,
    MAX_PIPELINE_LENGTH,
    MAX_CREDENTIAL_VALUE_LENGTH,
    MAX_SEARCH_QUERY_LENGTH,
    # Functions (with underscore prefix for backwards compatibility)
    validate_platform_id as _validate_platform_id,
    validate_resource_id as _validate_resource_id,
    validate_email as _validate_email,
    validate_string_field as _validate_string_field,
    validate_amount as _validate_amount,
    validate_probability as _validate_probability,
)

__all__ = [
    # Main handler
    "CRMHandler",
    # Circuit breaker
    "CRMCircuitBreaker",
    "get_crm_circuit_breaker",
    "reset_crm_circuit_breaker",
    # Models
    "SUPPORTED_PLATFORMS",
    "UnifiedContact",
    "UnifiedCompany",
    "UnifiedDeal",
    # Internal state (for testing)
    "_platform_credentials",
    "_platform_connectors",
    # Validation constants
    "SAFE_PLATFORM_PATTERN",
    "SAFE_RESOURCE_ID_PATTERN",
    "EMAIL_PATTERN",
    "MAX_EMAIL_LENGTH",
    "MAX_NAME_LENGTH",
    "MAX_PHONE_LENGTH",
    "MAX_COMPANY_NAME_LENGTH",
    "MAX_JOB_TITLE_LENGTH",
    "MAX_DOMAIN_LENGTH",
    "MAX_DEAL_NAME_LENGTH",
    "MAX_STAGE_LENGTH",
    "MAX_PIPELINE_LENGTH",
    "MAX_CREDENTIAL_VALUE_LENGTH",
    "MAX_SEARCH_QUERY_LENGTH",
    # Validation functions (with underscore prefix for backwards compatibility)
    "_validate_platform_id",
    "_validate_resource_id",
    "_validate_email",
    "_validate_string_field",
    "_validate_amount",
    "_validate_probability",
]
