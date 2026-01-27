"""Webhook handlers for external integrations."""

from aragora.server.handlers.webhooks.github_app import (
    GITHUB_APP_ROUTES,
    handle_github_webhook,
)

# Re-export WebhookHandler from the sibling webhooks.py module
# This resolves the naming conflict between webhooks/ directory and webhooks.py file
import importlib.util
import os as _os

_webhooks_file = _os.path.join(_os.path.dirname(__file__), "..", "webhooks.py")
_spec = importlib.util.spec_from_file_location("webhooks_module", _webhooks_file)
if _spec and _spec.loader:
    _webhooks_module = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_webhooks_module)
    WebhookHandler = _webhooks_module.WebhookHandler
    WebhookStore = _webhooks_module.WebhookStore
    WebhookConfig = _webhooks_module.WebhookConfig
    get_webhook_store = _webhooks_module.get_webhook_store
    generate_signature = _webhooks_module.generate_signature
    verify_signature = _webhooks_module.verify_signature
    WEBHOOK_EVENTS = _webhooks_module.WEBHOOK_EVENTS
    # RBAC exports
    RBAC_AVAILABLE = _webhooks_module.RBAC_AVAILABLE
    check_permission = _webhooks_module.check_permission
    validate_webhook_url = _webhooks_module.validate_webhook_url
else:
    raise ImportError("Could not load webhooks.py module")

__all__ = [
    "GITHUB_APP_ROUTES",
    "handle_github_webhook",
    # Re-exports from webhooks.py
    "WebhookHandler",
    "WebhookStore",
    "WebhookConfig",
    "get_webhook_store",
    "generate_signature",
    "verify_signature",
    "WEBHOOK_EVENTS",
    # RBAC exports
    "RBAC_AVAILABLE",
    "check_permission",
    "validate_webhook_url",
]
