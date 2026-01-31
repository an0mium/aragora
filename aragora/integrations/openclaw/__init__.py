"""
OpenClaw Integration Package.

Provides enterprise integration with OpenClaw instances:
- OpenClawClient: API client for OpenClaw backend
- OpenClawAuditBridge: Audit logging to Knowledge Mound
- Configuration and deployment utilities

This package enables Aragora to act as an enterprise security
gateway for OpenClaw deployments.
"""

from aragora.integrations.openclaw.audit_bridge import OpenClawAuditBridge
from aragora.integrations.openclaw.client import OpenClawClient, OpenClawConfig

__all__ = [
    "OpenClawClient",
    "OpenClawConfig",
    "OpenClawAuditBridge",
]
