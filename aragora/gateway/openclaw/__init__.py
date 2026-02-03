"""
OpenClaw Secure Gateway Adapter.

Provides enterprise-grade security wrapper for OpenClaw AI assistant:
- RBAC permission enforcement on all operations
- Audit logging with HMAC signing
- Capability filtering (allow/block/require-approval)
- Action filtering (allowlist/denylist with risk assessment)
- Sandbox isolation with resource limits
- Protocol translation (Aragora <-> OpenClaw)
- Channel mapping (WhatsApp, Telegram, Slack, Discord, etc.)
- Session management with tenant/user context
- Secure credential storage with AES-256-GCM encryption

Usage:
    from aragora.gateway.openclaw import (
        OpenClawAdapter,
        OpenClawMessage,
        OpenClawAction,
        OpenClawSession,
        ChannelMapping,
    )

    adapter = OpenClawAdapter(
        openclaw_endpoint="http://localhost:8081",
        rbac_checker=checker,
    )

    # Create session
    session = await adapter.create_session(
        user_id="user-123",
        channel="telegram",
        tenant_id="tenant-456",
    )

    # Execute action
    result = await adapter.execute_action(
        session_id=session.session_id,
        action=OpenClawAction(
            action_type="browser_navigate",
            parameters={"url": "https://example.com"},
        ),
    )

Legacy Gateway (backward compatible):
    from aragora.gateway.openclaw import OpenClawGatewayAdapter, SandboxConfig

    adapter = OpenClawGatewayAdapter(
        openclaw_endpoint="http://localhost:8081",
        rbac_checker=checker,
        audit_logger=logger,
    )

    result = await adapter.execute_task(
        task=OpenClawTask(type="text_generation", prompt="..."),
        auth_context=ctx,
    )

Action Filtering:
    from aragora.gateway.openclaw import ActionFilter, RiskLevel

    filter = ActionFilter(
        tenant_id="tenant_123",
        allowed_actions={"browser.navigate", "browser.click"},
    )

    decision = filter.check_action("filesystem.write")
    if decision.allowed:
        # Proceed
        ...

Credential Vault:
    from aragora.gateway.openclaw import (
        CredentialVault,
        CredentialType,
        RotationPolicy,
        get_credential_vault,
    )

    vault = get_credential_vault()
    cred_id = await vault.store_credential(
        tenant_id="acme",
        framework="openai",
        credential_type=CredentialType.API_KEY,
        value="sk-...",
        auth_context=ctx,
    )
"""

from __future__ import annotations

from .action_filter import (
    ActionCategory,
    ActionCategoryType,
    ActionFilter,
    ActionRule,
    FilterDecision,
    RiskLevel,
)
from .adapter import OpenClawAdapter
from .formatters import (
    ChannelFormatter,
    DiscordFormatter,
    SlackFormatter,
    TelegramFormatter,
    WhatsAppFormatter,
)
from .gateway_adapter import OpenClawGatewayAdapter
from .models import (
    ActionResult,
    ActionStatus,
    ChannelMapping,
    DeviceHandle,
    DeviceRegistration,
    GatewayResult,
    OpenClawAction,
    OpenClawActionType,
    OpenClawChannel,
    OpenClawMessage,
    OpenClawSession,
    PluginInstallRequest,
    SessionState,
)
from .protocols import (
    ApprovalGateProtocol,
    AuditLoggerProtocol,
    RBACCheckerProtocol,
)
from .audit import OpenClawAuditEvents
from .capabilities import CapabilityCategory, CapabilityFilter
from .credential_vault import (
    CredentialAuditEvent,
    CredentialFramework,
    CredentialMetadata,
    CredentialRateLimiter,
    CredentialType,
    CredentialVault,
    RotationPolicy,
    StoredCredential,
    get_credential_vault,
    init_credential_vault,
    reset_credential_vault,
)
from .protocol import OpenClawProtocolTranslator
from .sandbox import OpenClawSandbox, SandboxConfig

__all__ = [
    # New adapter
    "OpenClawAdapter",
    # Enums
    "OpenClawChannel",
    "OpenClawActionType",
    "SessionState",
    "ActionStatus",
    # Core dataclasses
    "OpenClawMessage",
    "OpenClawAction",
    "OpenClawSession",
    "ChannelMapping",
    "ActionResult",
    # Formatters
    "ChannelFormatter",
    "WhatsAppFormatter",
    "TelegramFormatter",
    "SlackFormatter",
    "DiscordFormatter",
    # Protocol interfaces
    "RBACCheckerProtocol",
    "AuditLoggerProtocol",
    "ApprovalGateProtocol",
    # Action filtering
    "ActionFilter",
    "ActionRule",
    "ActionCategory",
    "ActionCategoryType",
    "FilterDecision",
    "RiskLevel",
    # Legacy gateway adapter
    "OpenClawGatewayAdapter",
    "GatewayResult",
    "DeviceRegistration",
    "DeviceHandle",
    "PluginInstallRequest",
    "OpenClawAuditEvents",
    # Capability filtering
    "CapabilityCategory",
    "CapabilityFilter",
    # Protocol
    "OpenClawProtocolTranslator",
    # Sandbox
    "OpenClawSandbox",
    "SandboxConfig",
    # Credential vault
    "CredentialVault",
    "StoredCredential",
    "CredentialMetadata",
    "RotationPolicy",
    "CredentialType",
    "CredentialFramework",
    "CredentialAuditEvent",
    "CredentialRateLimiter",
    "get_credential_vault",
    "init_credential_vault",
    "reset_credential_vault",
]
