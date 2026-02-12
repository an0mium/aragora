"""SDK missing endpoints — DEPRECATED, all routes now have handler implementations.

This module previously contained OpenAPI schema stubs for planned-but-unimplemented
endpoints. As of Feb 2026, all routes have been implemented with proper handlers:

- Analytics: handlers/analytics_performance.py, handlers/_analytics_impl.py
- Accounting (AP/AR/invoices/expenses): handlers/accounting.py, ap_automation.py,
  ar_automation.py, invoices.py, expenses.py
- Knowledge Mound: handlers/knowledge_base/mound/
- Personas: handlers/persona.py
- Skills: handlers/skills.py
- Users/Auth: handlers/auth/
- Connectors: handlers/connectors/management.py
- SCIM: handlers/scim_handler.py

Previously deleted (earlier cleanup):
- sdk_missing_compliance.py → handlers/compliance/
- sdk_missing_integration.py → integrations.py, webhooks.py
- sdk_missing_debates.py → debates.py, agents.py, replays.py
- sdk_missing_media.py → pulse.py, handlers/social/

The empty dict is retained for backward compatibility with any code
that imports SDK_MISSING_ENDPOINTS.
"""

SDK_MISSING_ENDPOINTS: dict = {}

__all__ = ["SDK_MISSING_ENDPOINTS"]
