"""
SCIM 2.0 Protocol Implementation for Aragora.

Implements RFC 7643 (SCIM Core Schema) and RFC 7644 (SCIM Protocol)
for enterprise user provisioning from Identity Providers.

Supports:
- Okta
- Azure AD
- OneLogin
- JumpCloud
- Google Workspace

Usage:
    from aragora.auth.scim import SCIMServer, SCIMConfig

    config = SCIMConfig(
        bearer_token="your-scim-token",
        tenant_id="tenant-123",
    )
    server = SCIMServer(config)

    # Mount in FastAPI
    app.include_router(server.router, prefix="/scim/v2")
"""

from .schemas import (
    SCIMError,
    SCIMListResponse,
    SCIMMeta,
    SCIMResource,
    SCIMUser,
    SCIMGroup,
    SCIMGroupMember,
    SCIMName,
    SCIMEmail,
    SCIMPhoneNumber,
    SCIMAddress,
    SCIMEnterprise,
    SCIMPatchOperation,
    SCIMPatchRequest,
)
from .server import SCIMServer, SCIMConfig
from .filters import SCIMFilterParser, SCIMFilter

__all__ = [
    # Schemas
    "SCIMError",
    "SCIMListResponse",
    "SCIMMeta",
    "SCIMResource",
    "SCIMUser",
    "SCIMGroup",
    "SCIMGroupMember",
    "SCIMName",
    "SCIMEmail",
    "SCIMPhoneNumber",
    "SCIMAddress",
    "SCIMEnterprise",
    "SCIMPatchOperation",
    "SCIMPatchRequest",
    # Server
    "SCIMServer",
    "SCIMConfig",
    # Filters
    "SCIMFilterParser",
    "SCIMFilter",
]
