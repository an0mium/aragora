"""Webhook handlers for management APIs and external integrations."""

import importlib as importlib  # Compatibility: historically a public package attribute.
import warnings as _warnings

from aragora.server.handlers import webhook_management as _webhooks_module
from aragora.server.handlers.webhooks.github_app import (
    GITHUB_APP_ROUTES,
    handle_github_webhook,
)

# Preserve the package's historical exports and private module hook while
# loading the implementation through its one canonical module identity.
WebhookHandler = _webhooks_module.WebhookHandler
WebhookStore = _webhooks_module.WebhookStore
WebhookConfig = _webhooks_module.WebhookConfig
get_webhook_store = _webhooks_module.get_webhook_store
generate_signature = _webhooks_module.generate_signature
verify_signature = _webhooks_module.verify_signature
WEBHOOK_EVENTS = _webhooks_module.WEBHOOK_EVENTS
RBAC_AVAILABLE = _webhooks_module.RBAC_AVAILABLE
check_permission = _webhooks_module.check_permission
validate_webhook_url = _webhooks_module.validate_webhook_url

_warnings.warn(
    "aragora.server.handlers.webhooks is deprecated as the webhook management "
    "implementation home; use aragora.server.handlers.webhook_management instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "GITHUB_APP_ROUTES",
    "handle_github_webhook",
    "WebhookHandler",
    "WebhookStore",
    "WebhookConfig",
    "get_webhook_store",
    "generate_signature",
    "verify_signature",
    "WEBHOOK_EVENTS",
    "RBAC_AVAILABLE",
    "check_permission",
    "validate_webhook_url",
]
