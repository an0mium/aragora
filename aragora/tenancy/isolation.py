"""
Tenant data isolation enforcement.

Ensures that data access is always scoped to the current tenant,
preventing cross-tenant data leakage.

Usage:
    from aragora.tenancy import TenantDataIsolation, TenantIsolationConfig

    isolation = TenantDataIsolation(config)

    # Wrap queries to add tenant filtering
    filtered_query = isolation.filter_query(query, "debates")

    # Validate data belongs to tenant
    isolation.validate_ownership(debate, "tenant_id")
"""

from __future__ import annotations

import hashlib
import logging
import secrets
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable, Optional, TypeVar

from aragora.tenancy.context import (
    TenantNotSetError,
    get_current_tenant_id,
    require_tenant_id,
)

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

T = TypeVar("T")


class IsolationLevel(Enum):
    """Level of tenant isolation."""

    NONE = "none"  # No isolation (single-tenant mode)
    SOFT = "soft"  # Tenant filtering but shared resources
    STRICT = "strict"  # Full isolation with encryption
    COMPLETE = "complete"  # Separate databases/namespaces


class IsolationViolation(Exception):
    """Raised when tenant isolation is violated."""

    def __init__(self, message: str, tenant_id: Optional[str] = None):
        self.tenant_id = tenant_id
        super().__init__(message)


# Alias for backwards compatibility with tests
TenantIsolationError = IsolationViolation


@dataclass
class TenantIsolationConfig:
    """Configuration for tenant isolation."""

    # Isolation level
    level: IsolationLevel = IsolationLevel.STRICT
    """Isolation level to enforce."""

    # Filtering
    tenant_column: str = "tenant_id"
    """Default column name for tenant filtering."""

    auto_filter: bool = True
    """Automatically add tenant filters to queries."""

    # Encryption
    encrypt_at_rest: bool = True
    """Encrypt tenant data at rest."""

    per_tenant_keys: bool = False
    """Use separate encryption keys per tenant."""

    # Namespacing
    namespace_prefix: bool = True
    """Prefix resources with tenant ID."""

    namespace_separator: str = "_"
    """Separator for namespace prefix."""

    # Validation
    strict_validation: bool = True
    """Raise exceptions on validation failures."""

    audit_access: bool = True
    """Log all cross-tenant access attempts."""

    # Resources that bypass isolation
    shared_resources: list[str] = field(default_factory=list)
    """Resource types that are shared across tenants."""


@dataclass
class IsolationAuditEntry:
    """Audit entry for isolation events."""

    timestamp: datetime
    tenant_id: str
    resource_type: str
    action: str
    allowed: bool
    reason: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)


class TenantDataIsolation:
    """
    Enforces tenant data isolation.

    Provides methods for filtering queries, validating ownership,
    and namespacing resources to prevent cross-tenant data access.
    """

    def __init__(self, config: Optional[TenantIsolationConfig] = None):
        """Initialize isolation enforcement."""
        self.config = config or TenantIsolationConfig()
        self._audit_log: list[IsolationAuditEntry] = []
        self._encryption_keys: dict[str, bytes] = {}

    def get_tenant_filter(
        self,
        tenant_id: Optional[str] = None,
        column: Optional[str] = None,
    ) -> dict[str, str]:
        """
        Get a filter dict for tenant queries.

        Args:
            tenant_id: Tenant ID (uses current context if not provided)
            column: Column name (uses config default if not provided)

        Returns:
            Filter dictionary for query
        """
        if self.config.level == IsolationLevel.NONE:
            return {}

        tid = tenant_id or get_current_tenant_id()
        if tid is None:
            if self.config.strict_validation:
                raise TenantNotSetError("No tenant context for filtering")
            return {}

        col = column or self.config.tenant_column
        return {col: tid}

    def filter_query(
        self,
        query: dict[str, Any],
        resource_type: str,
        tenant_id: Optional[str] = None,
    ) -> dict[str, Any]:
        """
        Add tenant filtering to a query.

        Args:
            query: Original query dict
            resource_type: Type of resource being queried
            tenant_id: Tenant ID (uses current context if not provided)

        Returns:
            Query with tenant filter added
        """
        if self.config.level == IsolationLevel.NONE:
            return query

        if resource_type in self.config.shared_resources:
            return query

        tenant_filter = self.get_tenant_filter(tenant_id)
        if not tenant_filter:
            return query

        # Merge tenant filter with existing query
        return {**query, **tenant_filter}

    # Alias for backwards compatibility with tests
    def apply_tenant_filter(
        self,
        base_sql: str,
        resource_type: str = "default",
        tenant_id: Optional[str] = None,
        table_alias: Optional[str] = None,
    ) -> tuple[str, dict[str, Any]]:
        """Alias for filter_sql for backwards compatibility."""
        return self.filter_sql(base_sql, resource_type, tenant_id, table_alias)

    def filter_sql(
        self,
        base_sql: str,
        resource_type: str,
        tenant_id: Optional[str] = None,
        table_alias: Optional[str] = None,
    ) -> tuple[str, dict[str, Any]]:
        """
        Add tenant filtering to SQL query.

        Args:
            base_sql: Base SQL query
            resource_type: Type of resource being queried
            tenant_id: Tenant ID (uses current context if not provided)
            table_alias: Table alias for the tenant column

        Returns:
            Tuple of (modified SQL, parameters)
        """
        if self.config.level == IsolationLevel.NONE:
            return base_sql, {}

        if resource_type in self.config.shared_resources:
            return base_sql, {}

        tid = tenant_id or get_current_tenant_id()
        if tid is None:
            if self.config.strict_validation:
                raise TenantNotSetError("No tenant context for SQL filtering")
            return base_sql, {}

        # Build column reference
        col = self.config.tenant_column
        if table_alias:
            col = f"{table_alias}.{col}"

        # Check if query has WHERE clause
        sql_upper = base_sql.upper()
        if "WHERE" in sql_upper:
            # Add to existing WHERE
            modified_sql = base_sql.replace("WHERE", f"WHERE {col} = :tenant_id AND", 1)
        else:
            # Find insertion point (before ORDER BY, LIMIT, etc.)
            insert_keywords = ["ORDER BY", "GROUP BY", "LIMIT", "OFFSET", ";"]
            insert_pos = len(base_sql)
            for keyword in insert_keywords:
                pos = sql_upper.find(keyword)
                if pos != -1 and pos < insert_pos:
                    insert_pos = pos

            modified_sql = (
                base_sql[:insert_pos].rstrip()
                + f" WHERE {col} = :tenant_id "
                + base_sql[insert_pos:]
            )

        return modified_sql, {"tenant_id": tid}

    def validate_ownership(
        self,
        resource: Any,
        tenant_field: Optional[str] = None,
        expected_tenant: Optional[str] = None,
    ) -> bool:
        """
        Validate that a resource belongs to the current tenant.

        Args:
            resource: Resource to validate
            tenant_field: Field containing tenant ID
            expected_tenant: Expected tenant (uses current context if not provided)

        Returns:
            True if valid, False otherwise

        Raises:
            IsolationViolation: If strict validation enabled and ownership invalid
        """
        if self.config.level == IsolationLevel.NONE:
            return True

        field_name = tenant_field or self.config.tenant_column
        expected = expected_tenant or get_current_tenant_id()

        if expected is None:
            if self.config.strict_validation:
                raise TenantNotSetError("No tenant context for validation")
            return True

        # Get resource tenant ID
        resource_tenant = None
        if isinstance(resource, dict):
            resource_tenant = resource.get(field_name)
        elif hasattr(resource, field_name):
            resource_tenant = getattr(resource, field_name)

        if resource_tenant is None:
            self._audit_access(
                expected,
                "unknown",
                "validate",
                False,
                "Resource has no tenant field",
            )
            if self.config.strict_validation:
                raise IsolationViolation(
                    f"Resource has no {field_name} field",
                    tenant_id=expected,
                )
            return False

        if resource_tenant != expected:
            self._audit_access(
                expected,
                "unknown",
                "validate",
                False,
                f"Tenant mismatch: {resource_tenant} != {expected}",
            )
            if self.config.strict_validation:
                raise IsolationViolation(
                    f"Resource belongs to tenant {resource_tenant}, " f"not {expected}",
                    tenant_id=expected,
                )
            return False

        return True

    def namespace_key(
        self,
        key: str,
        tenant_id: Optional[str] = None,
    ) -> str:
        """
        Add tenant namespace prefix to a key.

        Args:
            key: Original key
            tenant_id: Tenant ID (uses current context if not provided)

        Returns:
            Namespaced key
        """
        if not self.config.namespace_prefix:
            return key

        tid = tenant_id or get_current_tenant_id()
        if tid is None:
            return key

        return f"{tid}{self.config.namespace_separator}{key}"

    def extract_tenant_from_key(self, namespaced_key: str) -> tuple[Optional[str], str]:
        """
        Extract tenant ID from a namespaced key.

        Args:
            namespaced_key: Key with potential namespace prefix

        Returns:
            Tuple of (tenant_id, original_key)
        """
        if not self.config.namespace_prefix:
            return None, namespaced_key

        if self.config.namespace_separator not in namespaced_key:
            return None, namespaced_key

        parts = namespaced_key.split(self.config.namespace_separator, 1)
        if len(parts) == 2:
            return parts[0], parts[1]
        return None, namespaced_key

    def get_encryption_key(self, tenant_id: Optional[str] = None) -> bytes:
        """
        Get encryption key for tenant data.

        Args:
            tenant_id: Tenant ID (uses current context if not provided)

        Returns:
            Encryption key bytes
        """
        if not self.config.encrypt_at_rest:
            raise ValueError("Encryption not enabled")

        if not self.config.per_tenant_keys:
            # Use shared key
            if "_shared" not in self._encryption_keys:
                self._encryption_keys["_shared"] = secrets.token_bytes(32)
            return self._encryption_keys["_shared"]

        tid = tenant_id or require_tenant_id()

        if tid not in self._encryption_keys:
            # Generate deterministic key from tenant ID
            # In production, this would come from a key management service
            seed = f"aragora_tenant_key_{tid}".encode()
            self._encryption_keys[tid] = hashlib.sha256(seed).digest()

        return self._encryption_keys[tid]

    def set_encryption_key(self, tenant_id: str, key: bytes) -> None:
        """
        Set encryption key for a tenant.

        Args:
            tenant_id: Tenant ID
            key: 32-byte encryption key
        """
        if len(key) != 32:
            raise ValueError("Encryption key must be 32 bytes")
        self._encryption_keys[tenant_id] = key

    def _audit_access(
        self,
        tenant_id: str,
        resource_type: str,
        action: str,
        allowed: bool,
        reason: Optional[str] = None,
    ) -> None:
        """Record access attempt in audit log."""
        if not self.config.audit_access:
            return

        entry = IsolationAuditEntry(
            timestamp=datetime.now(),
            tenant_id=tenant_id,
            resource_type=resource_type,
            action=action,
            allowed=allowed,
            reason=reason,
        )
        self._audit_log.append(entry)

        if not allowed:
            logger.warning(
                f"Isolation violation: tenant={tenant_id} "
                f"resource={resource_type} action={action} reason={reason}"
            )

    def get_audit_log(
        self,
        tenant_id: Optional[str] = None,
        limit: int = 100,
    ) -> list[IsolationAuditEntry]:
        """
        Get audit log entries.

        Args:
            tenant_id: Filter by tenant (None = all)
            limit: Maximum entries to return

        Returns:
            List of audit entries
        """
        entries = self._audit_log
        if tenant_id:
            entries = [e for e in entries if e.tenant_id == tenant_id]
        return entries[-limit:]

    def clear_audit_log(self) -> None:
        """Clear the audit log."""
        self._audit_log.clear()


class IsolatedResource:
    """
    Decorator for classes that require tenant isolation.

    Usage:
        @IsolatedResource(tenant_field="org_id")
        class MyModel:
            org_id: str
            data: dict
    """

    def __init__(
        self,
        tenant_field: str = "tenant_id",
        validate_on_access: bool = True,
    ):
        self.tenant_field = tenant_field
        self.validate_on_access = validate_on_access
        self._isolation = TenantDataIsolation()

    def __call__(self, cls: type[T]) -> type[T]:
        """Wrap class with isolation checks."""
        original_init = cls.__init__

        tenant_field = self.tenant_field
        isolation = self._isolation
        validate = self.validate_on_access

        def new_init(self, *args, **kwargs):
            original_init(self, *args, **kwargs)

            # Auto-set tenant if not provided
            if not hasattr(self, tenant_field) or getattr(self, tenant_field) is None:
                current = get_current_tenant_id()
                if current:
                    setattr(self, tenant_field, current)

            # Validate on creation
            if validate:
                isolation.validate_ownership(self, tenant_field)

        cls.__init__ = new_init
        return cls


def ensure_tenant_scope(func: Callable[..., T]) -> Callable[..., T]:
    """
    Decorator that ensures function runs in tenant scope.

    Validates tenant context exists before execution.
    """
    from functools import wraps

    @wraps(func)
    def wrapper(*args, **kwargs):
        tenant_id = get_current_tenant_id()
        if tenant_id is None:
            raise TenantNotSetError(f"Function {func.__name__} requires tenant scope")
        return func(*args, **kwargs)

    @wraps(func)
    async def async_wrapper(*args, **kwargs):
        tenant_id = get_current_tenant_id()
        if tenant_id is None:
            raise TenantNotSetError(f"Function {func.__name__} requires tenant scope")
        return await func(*args, **kwargs)

    import asyncio

    if asyncio.iscoroutinefunction(func):
        return async_wrapper  # type: ignore
    return wrapper  # type: ignore
