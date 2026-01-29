"""
SCIM 2.0 Resource Schemas.

Implements RFC 7643 (SCIM Core Schema) with support for:
- User resources (urn:ietf:params:scim:schemas:core:2.0:User)
- Group resources (urn:ietf:params:scim:schemas:core:2.0:Group)
- Enterprise User extension (urn:ietf:params:scim:schemas:extension:enterprise:2.0:User)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


# =============================================================================
# SCIM Schema URIs
# =============================================================================

SCHEMA_USER = "urn:ietf:params:scim:schemas:core:2.0:User"
SCHEMA_GROUP = "urn:ietf:params:scim:schemas:core:2.0:Group"
SCHEMA_ENTERPRISE_USER = "urn:ietf:params:scim:schemas:extension:enterprise:2.0:User"
SCHEMA_LIST_RESPONSE = "urn:ietf:params:scim:api:messages:2.0:ListResponse"
SCHEMA_ERROR = "urn:ietf:params:scim:api:messages:2.0:Error"
SCHEMA_PATCH_OP = "urn:ietf:params:scim:api:messages:2.0:PatchOp"


# =============================================================================
# SCIM Error Types
# =============================================================================


class SCIMErrorType(str, Enum):
    """SCIM error types per RFC 7644."""

    INVALID_FILTER = "invalidFilter"
    TOO_MANY = "tooMany"
    UNIQUENESS = "uniqueness"
    MUTABILITY = "mutability"
    INVALID_SYNTAX = "invalidSyntax"
    INVALID_PATH = "invalidPath"
    NO_TARGET = "noTarget"
    INVALID_VALUE = "invalidValue"
    INVALID_VERS = "invalidVers"
    SENSITIVE = "sensitive"


@dataclass
class SCIMError:
    """
    SCIM Error Response.

    Per RFC 7644 Section 3.12.
    """

    status: int
    detail: str
    scim_type: SCIMErrorType | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to SCIM error response format."""
        result: dict[str, Any] = {
            "schemas": [SCHEMA_ERROR],
            "status": str(self.status),
            "detail": self.detail,
        }
        if self.scim_type:
            result["scimType"] = self.scim_type.value
        return result


# =============================================================================
# SCIM Meta
# =============================================================================


@dataclass
class SCIMMeta:
    """
    Resource metadata.

    Per RFC 7643 Section 3.1.
    """

    resource_type: str
    created: datetime | None = None
    last_modified: datetime | None = None
    location: str | None = None
    version: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to SCIM meta format."""
        result: dict[str, Any] = {"resourceType": self.resource_type}
        if self.created:
            result["created"] = self.created.isoformat()
        if self.last_modified:
            result["lastModified"] = self.last_modified.isoformat()
        if self.location:
            result["location"] = self.location
        if self.version:
            result["version"] = self.version
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SCIMMeta:
        """Create from dictionary."""
        return cls(
            resource_type=data.get("resourceType", ""),
            created=datetime.fromisoformat(data["created"]) if data.get("created") else None,
            last_modified=(
                datetime.fromisoformat(data["lastModified"]) if data.get("lastModified") else None
            ),
            location=data.get("location"),
            version=data.get("version"),
        )


# =============================================================================
# SCIM Base Resource
# =============================================================================


@dataclass
class SCIMResource:
    """Base SCIM resource with common attributes."""

    id: str
    schemas: list[str]
    meta: SCIMMeta | None = None
    external_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to SCIM resource format."""
        result: dict[str, Any] = {
            "schemas": self.schemas,
            "id": self.id,
        }
        if self.external_id:
            result["externalId"] = self.external_id
        if self.meta:
            result["meta"] = self.meta.to_dict()
        return result


# =============================================================================
# User Resource Components
# =============================================================================


@dataclass
class SCIMName:
    """User name component."""

    formatted: str | None = None
    family_name: str | None = None
    given_name: str | None = None
    middle_name: str | None = None
    honorific_prefix: str | None = None
    honorific_suffix: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to SCIM name format."""
        result: dict[str, Any] = {}
        if self.formatted:
            result["formatted"] = self.formatted
        if self.family_name:
            result["familyName"] = self.family_name
        if self.given_name:
            result["givenName"] = self.given_name
        if self.middle_name:
            result["middleName"] = self.middle_name
        if self.honorific_prefix:
            result["honorificPrefix"] = self.honorific_prefix
        if self.honorific_suffix:
            result["honorificSuffix"] = self.honorific_suffix
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SCIMName:
        """Create from dictionary."""
        return cls(
            formatted=data.get("formatted"),
            family_name=data.get("familyName"),
            given_name=data.get("givenName"),
            middle_name=data.get("middleName"),
            honorific_prefix=data.get("honorificPrefix"),
            honorific_suffix=data.get("honorificSuffix"),
        )


@dataclass
class SCIMEmail:
    """User email."""

    value: str
    type: str | None = None
    primary: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convert to SCIM email format."""
        result: dict[str, Any] = {"value": self.value}
        if self.type:
            result["type"] = self.type
        if self.primary:
            result["primary"] = self.primary
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SCIMEmail:
        """Create from dictionary."""
        return cls(
            value=data["value"],
            type=data.get("type"),
            primary=data.get("primary", False),
        )


@dataclass
class SCIMPhoneNumber:
    """User phone number."""

    value: str
    type: str | None = None
    primary: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convert to SCIM phone format."""
        result: dict[str, Any] = {"value": self.value}
        if self.type:
            result["type"] = self.type
        if self.primary:
            result["primary"] = self.primary
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SCIMPhoneNumber:
        """Create from dictionary."""
        return cls(
            value=data["value"],
            type=data.get("type"),
            primary=data.get("primary", False),
        )


@dataclass
class SCIMAddress:
    """User address."""

    formatted: str | None = None
    street_address: str | None = None
    locality: str | None = None
    region: str | None = None
    postal_code: str | None = None
    country: str | None = None
    type: str | None = None
    primary: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convert to SCIM address format."""
        result: dict[str, Any] = {}
        if self.formatted:
            result["formatted"] = self.formatted
        if self.street_address:
            result["streetAddress"] = self.street_address
        if self.locality:
            result["locality"] = self.locality
        if self.region:
            result["region"] = self.region
        if self.postal_code:
            result["postalCode"] = self.postal_code
        if self.country:
            result["country"] = self.country
        if self.type:
            result["type"] = self.type
        if self.primary:
            result["primary"] = self.primary
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SCIMAddress:
        """Create from dictionary."""
        return cls(
            formatted=data.get("formatted"),
            street_address=data.get("streetAddress"),
            locality=data.get("locality"),
            region=data.get("region"),
            postal_code=data.get("postalCode"),
            country=data.get("country"),
            type=data.get("type"),
            primary=data.get("primary", False),
        )


@dataclass
class SCIMEnterprise:
    """Enterprise user extension."""

    employee_number: str | None = None
    cost_center: str | None = None
    organization: str | None = None
    division: str | None = None
    department: str | None = None
    manager: dict[str, str] | None = None  # {"value": manager_id, "$ref": uri}

    def to_dict(self) -> dict[str, Any]:
        """Convert to SCIM enterprise format."""
        result: dict[str, Any] = {}
        if self.employee_number:
            result["employeeNumber"] = self.employee_number
        if self.cost_center:
            result["costCenter"] = self.cost_center
        if self.organization:
            result["organization"] = self.organization
        if self.division:
            result["division"] = self.division
        if self.department:
            result["department"] = self.department
        if self.manager:
            result["manager"] = self.manager
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SCIMEnterprise:
        """Create from dictionary."""
        return cls(
            employee_number=data.get("employeeNumber"),
            cost_center=data.get("costCenter"),
            organization=data.get("organization"),
            division=data.get("division"),
            department=data.get("department"),
            manager=data.get("manager"),
        )


# =============================================================================
# User Resource
# =============================================================================


@dataclass
class SCIMUser(SCIMResource):
    """
    SCIM User Resource.

    Per RFC 7643 Section 4.1.
    """

    user_name: str = ""
    name: SCIMName | None = None
    display_name: str | None = None
    nick_name: str | None = None
    profile_url: str | None = None
    title: str | None = None
    user_type: str | None = None
    preferred_language: str | None = None
    locale: str | None = None
    timezone: str | None = None
    active: bool = True
    password: str | None = None  # Write-only
    emails: list[SCIMEmail] = field(default_factory=list)
    phone_numbers: list[SCIMPhoneNumber] = field(default_factory=list)
    addresses: list[SCIMAddress] = field(default_factory=list)
    groups: list[dict[str, str]] = field(default_factory=list)  # Read-only
    roles: list[str] = field(default_factory=list)
    enterprise: SCIMEnterprise | None = None

    def __post_init__(self):
        """Ensure User schema is in schemas list."""
        if SCHEMA_USER not in self.schemas:
            self.schemas = [SCHEMA_USER] + self.schemas
        if self.enterprise and SCHEMA_ENTERPRISE_USER not in self.schemas:
            self.schemas.append(SCHEMA_ENTERPRISE_USER)

    def to_dict(self) -> dict[str, Any]:
        """Convert to SCIM user format."""
        result = super().to_dict()
        result["userName"] = self.user_name

        if self.name:
            result["name"] = self.name.to_dict()
        if self.display_name:
            result["displayName"] = self.display_name
        if self.nick_name:
            result["nickName"] = self.nick_name
        if self.profile_url:
            result["profileUrl"] = self.profile_url
        if self.title:
            result["title"] = self.title
        if self.user_type:
            result["userType"] = self.user_type
        if self.preferred_language:
            result["preferredLanguage"] = self.preferred_language
        if self.locale:
            result["locale"] = self.locale
        if self.timezone:
            result["timezone"] = self.timezone

        result["active"] = self.active

        if self.emails:
            result["emails"] = [e.to_dict() for e in self.emails]
        if self.phone_numbers:
            result["phoneNumbers"] = [p.to_dict() for p in self.phone_numbers]
        if self.addresses:
            result["addresses"] = [a.to_dict() for a in self.addresses]
        if self.groups:
            result["groups"] = self.groups
        if self.roles:
            result["roles"] = [{"value": r} for r in self.roles]

        if self.enterprise:
            result[SCHEMA_ENTERPRISE_USER] = self.enterprise.to_dict()

        # Never include password in response
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SCIMUser:
        """Create from dictionary."""
        # Parse name
        name = None
        if "name" in data:
            name = SCIMName.from_dict(data["name"])

        # Parse emails
        emails = []
        for email_data in data.get("emails", []):
            emails.append(SCIMEmail.from_dict(email_data))

        # Parse phone numbers
        phones = []
        for phone_data in data.get("phoneNumbers", []):
            phones.append(SCIMPhoneNumber.from_dict(phone_data))

        # Parse addresses
        addresses = []
        for addr_data in data.get("addresses", []):
            addresses.append(SCIMAddress.from_dict(addr_data))

        # Parse enterprise extension
        enterprise = None
        if SCHEMA_ENTERPRISE_USER in data:
            enterprise = SCIMEnterprise.from_dict(data[SCHEMA_ENTERPRISE_USER])

        # Parse meta
        meta = None
        if "meta" in data:
            meta = SCIMMeta.from_dict(data["meta"])

        # Parse roles
        roles = []
        for role_data in data.get("roles", []):
            if isinstance(role_data, dict):
                roles.append(role_data.get("value", ""))
            else:
                roles.append(str(role_data))

        return cls(
            id=data.get("id", ""),
            schemas=data.get("schemas", [SCHEMA_USER]),
            external_id=data.get("externalId"),
            meta=meta,
            user_name=data.get("userName", ""),
            name=name,
            display_name=data.get("displayName"),
            nick_name=data.get("nickName"),
            profile_url=data.get("profileUrl"),
            title=data.get("title"),
            user_type=data.get("userType"),
            preferred_language=data.get("preferredLanguage"),
            locale=data.get("locale"),
            timezone=data.get("timezone"),
            active=data.get("active", True),
            password=data.get("password"),
            emails=emails,
            phone_numbers=phones,
            addresses=addresses,
            groups=data.get("groups", []),
            roles=roles,
            enterprise=enterprise,
        )

    def get_primary_email(self) -> str | None:
        """Get the primary email address."""
        for email in self.emails:
            if email.primary:
                return email.value
        return self.emails[0].value if self.emails else None


# =============================================================================
# Group Resource
# =============================================================================


@dataclass
class SCIMGroupMember:
    """Group member reference."""

    value: str  # User ID
    ref: str | None = None  # URI to user resource
    display: str | None = None  # User display name
    type: str = "User"  # User or Group

    def to_dict(self) -> dict[str, Any]:
        """Convert to SCIM member format."""
        result: dict[str, Any] = {"value": self.value}
        if self.ref:
            result["$ref"] = self.ref
        if self.display:
            result["display"] = self.display
        result["type"] = self.type
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SCIMGroupMember:
        """Create from dictionary."""
        return cls(
            value=data["value"],
            ref=data.get("$ref"),
            display=data.get("display"),
            type=data.get("type", "User"),
        )


@dataclass
class SCIMGroup(SCIMResource):
    """
    SCIM Group Resource.

    Per RFC 7643 Section 4.2.
    """

    display_name: str = ""
    members: list[SCIMGroupMember] = field(default_factory=list)

    def __post_init__(self):
        """Ensure Group schema is in schemas list."""
        if SCHEMA_GROUP not in self.schemas:
            self.schemas = [SCHEMA_GROUP] + self.schemas

    def to_dict(self) -> dict[str, Any]:
        """Convert to SCIM group format."""
        result = super().to_dict()
        result["displayName"] = self.display_name
        if self.members:
            result["members"] = [m.to_dict() for m in self.members]
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SCIMGroup:
        """Create from dictionary."""
        # Parse members
        members = []
        for member_data in data.get("members", []):
            members.append(SCIMGroupMember.from_dict(member_data))

        # Parse meta
        meta = None
        if "meta" in data:
            meta = SCIMMeta.from_dict(data["meta"])

        return cls(
            id=data.get("id", ""),
            schemas=data.get("schemas", [SCHEMA_GROUP]),
            external_id=data.get("externalId"),
            meta=meta,
            display_name=data.get("displayName", ""),
            members=members,
        )


# =============================================================================
# PATCH Operation
# =============================================================================


class SCIMPatchOp(str, Enum):
    """SCIM PATCH operation types."""

    ADD = "add"
    REMOVE = "remove"
    REPLACE = "replace"


@dataclass
class SCIMPatchOperation:
    """A single PATCH operation."""

    op: SCIMPatchOp
    path: str | None = None
    value: Any = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SCIMPatchOperation:
        """Create from dictionary."""
        return cls(
            op=SCIMPatchOp(data["op"].lower()),
            path=data.get("path"),
            value=data.get("value"),
        )


@dataclass
class SCIMPatchRequest:
    """SCIM PATCH request."""

    operations: list[SCIMPatchOperation]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SCIMPatchRequest:
        """Create from dictionary."""
        operations = []
        for op_data in data.get("Operations", []):
            operations.append(SCIMPatchOperation.from_dict(op_data))
        return cls(operations=operations)


# =============================================================================
# List Response
# =============================================================================


@dataclass
class SCIMListResponse:
    """
    SCIM List Response.

    Per RFC 7644 Section 3.4.2.
    """

    total_results: int
    resources: list[SCIMResource]
    start_index: int = 1
    items_per_page: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to SCIM list response format."""
        result: dict[str, Any] = {
            "schemas": [SCHEMA_LIST_RESPONSE],
            "totalResults": self.total_results,
            "startIndex": self.start_index,
            "Resources": [r.to_dict() for r in self.resources],
        }
        if self.items_per_page is not None:
            result["itemsPerPage"] = self.items_per_page
        return result
