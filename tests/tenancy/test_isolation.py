"""Tests for tenant data isolation enforcement."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from aragora.tenancy.context import (
    TenantContext,
    TenantNotSetError,
)
from aragora.tenancy.isolation import (
    IsolatedResource,
    IsolationLevel,
    IsolationViolation,
    TenantDataIsolation,
    TenantIsolationConfig,
    ensure_tenant_scope,
)


class TestTenantIsolationConfig:
    """Tests for TenantIsolationConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = TenantIsolationConfig()

        assert config.level == IsolationLevel.STRICT
        assert config.tenant_column == "tenant_id"
        assert config.auto_filter is True
        assert config.encrypt_at_rest is True
        assert config.per_tenant_keys is False
        assert config.namespace_prefix is True
        assert config.namespace_separator == "_"
        assert config.strict_validation is True
        assert config.audit_access is True
        assert config.shared_resources == frozenset()

    def test_custom_config(self):
        """Test custom configuration."""
        config = TenantIsolationConfig(
            level=IsolationLevel.SOFT,
            tenant_column="org_id",
            auto_filter=False,
            shared_resources=frozenset({"system_config", "feature_flags"}),
        )

        assert config.level == IsolationLevel.SOFT
        assert config.tenant_column == "org_id"
        assert config.auto_filter is False
        assert config.shared_resources == frozenset({"system_config", "feature_flags"})


class TestIsolationLevel:
    """Tests for IsolationLevel enum."""

    def test_isolation_levels(self):
        """Test all isolation levels exist."""
        assert IsolationLevel.NONE.value == "none"
        assert IsolationLevel.SOFT.value == "soft"
        assert IsolationLevel.STRICT.value == "strict"
        assert IsolationLevel.COMPLETE.value == "complete"


class TestTenantDataIsolation:
    """Tests for TenantDataIsolation class."""

    def test_init_default(self):
        """Test default initialization."""
        isolation = TenantDataIsolation()
        assert isolation.config is not None
        assert isolation.config.level == IsolationLevel.STRICT

    def test_init_custom_config(self):
        """Test initialization with custom config."""
        config = TenantIsolationConfig(level=IsolationLevel.SOFT)
        isolation = TenantDataIsolation(config)
        assert isolation.config.level == IsolationLevel.SOFT

    def test_get_tenant_filter_with_context(self):
        """Test tenant filter with tenant context set."""
        isolation = TenantDataIsolation()

        with TenantContext(tenant_id="tenant_123"):
            result = isolation.get_tenant_filter()

        assert result == {"tenant_id": "tenant_123"}

    def test_get_tenant_filter_with_explicit_tenant(self):
        """Test tenant filter with explicit tenant ID."""
        isolation = TenantDataIsolation()
        result = isolation.get_tenant_filter(tenant_id="explicit_tenant")
        assert result == {"tenant_id": "explicit_tenant"}

    def test_get_tenant_filter_custom_column(self):
        """Test tenant filter with custom column name."""
        isolation = TenantDataIsolation()

        with TenantContext(tenant_id="tenant_123"):
            result = isolation.get_tenant_filter(column="org_id")

        assert result == {"org_id": "tenant_123"}

    def test_get_tenant_filter_no_isolation(self):
        """Test tenant filter with NONE isolation level."""
        config = TenantIsolationConfig(level=IsolationLevel.NONE)
        isolation = TenantDataIsolation(config)

        result = isolation.get_tenant_filter(tenant_id="ignored")
        assert result == {}

    def test_get_tenant_filter_no_context_strict(self):
        """Test tenant filter raises error without context in strict mode."""
        isolation = TenantDataIsolation()

        with pytest.raises(TenantNotSetError):
            isolation.get_tenant_filter()

    def test_get_tenant_filter_no_context_non_strict(self):
        """Test tenant filter returns empty without context in non-strict mode."""
        config = TenantIsolationConfig(strict_validation=False)
        isolation = TenantDataIsolation(config)

        result = isolation.get_tenant_filter()
        assert result == {}

    def test_filter_query_adds_tenant(self):
        """Test filter_query adds tenant filter to query."""
        isolation = TenantDataIsolation()
        query = {"status": "active", "type": "debate"}

        with TenantContext(tenant_id="tenant_abc"):
            result = isolation.filter_query(query, "debates")

        assert result == {"status": "active", "type": "debate", "tenant_id": "tenant_abc"}

    def test_filter_query_shared_resource(self):
        """Test filter_query skips shared resources."""
        config = TenantIsolationConfig(shared_resources=frozenset({"system_config"}))
        isolation = TenantDataIsolation(config)
        query = {"type": "global"}

        with TenantContext(tenant_id="tenant_abc"):
            result = isolation.filter_query(query, "system_config")

        # Query should be unchanged for shared resources
        assert result == {"type": "global"}

    def test_filter_sql_with_where(self):
        """Test filter_sql adds to existing WHERE clause."""
        isolation = TenantDataIsolation()
        base_sql = "SELECT * FROM debates WHERE status = 'active'"

        with TenantContext(tenant_id="tenant_123"):
            sql, params = isolation.filter_sql(base_sql, "debates")

        assert "WHERE tenant_id = :tenant_id AND status = 'active'" in sql
        assert params == {"tenant_id": "tenant_123"}

    def test_filter_sql_without_where(self):
        """Test filter_sql adds WHERE clause."""
        isolation = TenantDataIsolation()
        base_sql = "SELECT * FROM debates"

        with TenantContext(tenant_id="tenant_123"):
            sql, params = isolation.filter_sql(base_sql, "debates")

        assert "WHERE tenant_id = :tenant_id" in sql
        assert params == {"tenant_id": "tenant_123"}

    def test_filter_sql_with_order_by(self):
        """Test filter_sql inserts before ORDER BY."""
        isolation = TenantDataIsolation()
        base_sql = "SELECT * FROM debates ORDER BY created_at DESC"

        with TenantContext(tenant_id="tenant_123"):
            sql, params = isolation.filter_sql(base_sql, "debates")

        assert "WHERE tenant_id = :tenant_id" in sql
        assert "ORDER BY created_at DESC" in sql
        assert sql.index("WHERE") < sql.index("ORDER BY")

    def test_filter_sql_with_table_alias(self):
        """Test filter_sql uses table alias."""
        isolation = TenantDataIsolation()
        base_sql = "SELECT * FROM debates d"

        with TenantContext(tenant_id="tenant_123"):
            sql, params = isolation.filter_sql(base_sql, "debates", table_alias="d")

        assert "d.tenant_id = :tenant_id" in sql

    def test_validate_ownership_dict_valid(self):
        """Test validate_ownership with valid dict resource."""
        isolation = TenantDataIsolation()
        resource = {"id": "123", "tenant_id": "tenant_abc", "name": "Test"}

        with TenantContext(tenant_id="tenant_abc"):
            result = isolation.validate_ownership(resource)

        assert result is True

    def test_validate_ownership_dict_invalid(self):
        """Test validate_ownership with invalid dict resource."""
        isolation = TenantDataIsolation()
        resource = {"id": "123", "tenant_id": "other_tenant", "name": "Test"}

        with TenantContext(tenant_id="tenant_abc"):
            with pytest.raises(IsolationViolation) as exc_info:
                isolation.validate_ownership(resource)

        assert "other_tenant" in str(exc_info.value)

    def test_validate_ownership_object_valid(self):
        """Test validate_ownership with valid object resource."""

        class Resource:
            tenant_id = "tenant_xyz"

        isolation = TenantDataIsolation()
        resource = Resource()

        with TenantContext(tenant_id="tenant_xyz"):
            result = isolation.validate_ownership(resource)

        assert result is True

    def test_validate_ownership_missing_field(self):
        """Test validate_ownership with missing tenant field."""
        isolation = TenantDataIsolation()
        resource = {"id": "123", "name": "Test"}  # No tenant_id

        with TenantContext(tenant_id="tenant_abc"):
            with pytest.raises(IsolationViolation):
                isolation.validate_ownership(resource)

    def test_validate_ownership_custom_field(self):
        """Test validate_ownership with custom tenant field."""
        isolation = TenantDataIsolation()
        resource = {"id": "123", "org_id": "org_xyz", "name": "Test"}

        with TenantContext(tenant_id="org_xyz"):
            result = isolation.validate_ownership(resource, tenant_field="org_id")

        assert result is True

    def test_validate_ownership_no_isolation(self):
        """Test validate_ownership always passes with NONE isolation."""
        config = TenantIsolationConfig(level=IsolationLevel.NONE)
        isolation = TenantDataIsolation(config)
        resource = {"id": "123", "tenant_id": "any_tenant"}

        # Should pass even without context
        result = isolation.validate_ownership(resource)
        assert result is True

    def test_namespace_key(self):
        """Test namespace_key adds tenant prefix."""
        isolation = TenantDataIsolation()

        with TenantContext(tenant_id="tenant_123"):
            result = isolation.namespace_key("cache:debates")

        assert result == "tenant_123_cache:debates"

    def test_namespace_key_custom_separator(self):
        """Test namespace_key with custom separator."""
        config = TenantIsolationConfig(namespace_separator="::")
        isolation = TenantDataIsolation(config)

        with TenantContext(tenant_id="tenant_123"):
            result = isolation.namespace_key("cache:debates")

        assert result == "tenant_123::cache:debates"

    def test_namespace_key_disabled(self):
        """Test namespace_key returns original when disabled."""
        config = TenantIsolationConfig(namespace_prefix=False)
        isolation = TenantDataIsolation(config)

        with TenantContext(tenant_id="tenant_123"):
            result = isolation.namespace_key("cache:debates")

        assert result == "cache:debates"

    def test_extract_tenant_from_key(self):
        """Test extracting tenant ID from namespaced key."""
        # Default separator is '_', so split happens on first '_'
        isolation = TenantDataIsolation()

        tenant_id, key = isolation.extract_tenant_from_key("tenant_cache:debates")

        assert tenant_id == "tenant"
        assert key == "cache:debates"

    def test_extract_tenant_from_key_custom_separator(self):
        """Test extracting tenant ID with custom separator."""
        config = TenantIsolationConfig(namespace_separator="::")
        isolation = TenantDataIsolation(config)

        tenant_id, key = isolation.extract_tenant_from_key("tenant_123::cache:debates")

        assert tenant_id == "tenant_123"
        assert key == "cache:debates"

    def test_extract_tenant_from_key_no_prefix(self):
        """Test extracting from non-namespaced key."""
        isolation = TenantDataIsolation()

        tenant_id, key = isolation.extract_tenant_from_key("cache:debates")

        assert tenant_id is None
        assert key == "cache:debates"

    def test_encryption_key_shared(self):
        """Test encryption key generation - shared mode."""
        config = TenantIsolationConfig(per_tenant_keys=False)
        isolation = TenantDataIsolation(config)

        key1 = isolation.get_encryption_key("tenant_a")
        key2 = isolation.get_encryption_key("tenant_b")

        # Should be same shared key
        assert key1 == key2
        assert len(key1) == 32

    def test_encryption_key_per_tenant(self):
        """Test encryption key generation - per-tenant mode."""
        config = TenantIsolationConfig(per_tenant_keys=True)
        isolation = TenantDataIsolation(config)

        with TenantContext(tenant_id="tenant_a"):
            key1 = isolation.get_encryption_key()
        with TenantContext(tenant_id="tenant_b"):
            key2 = isolation.get_encryption_key()

        # Should be different keys
        assert key1 != key2
        assert len(key1) == 32
        assert len(key2) == 32

    def test_encryption_key_not_enabled(self):
        """Test encryption key raises when not enabled."""
        config = TenantIsolationConfig(encrypt_at_rest=False)
        isolation = TenantDataIsolation(config)

        with pytest.raises(ValueError, match="Encryption not enabled"):
            isolation.get_encryption_key("tenant_a")

    def test_set_encryption_key(self):
        """Test setting custom encryption key."""
        # Need per_tenant_keys=True for set_encryption_key to be retrievable
        config = TenantIsolationConfig(per_tenant_keys=True)
        isolation = TenantDataIsolation(config)
        custom_key = b"0" * 32

        isolation.set_encryption_key("tenant_a", custom_key)

        with TenantContext(tenant_id="tenant_a"):
            result = isolation.get_encryption_key()

        assert result == custom_key

    def test_set_encryption_key_invalid_length(self):
        """Test setting encryption key with invalid length."""
        isolation = TenantDataIsolation()

        with pytest.raises(ValueError, match="32 bytes"):
            isolation.set_encryption_key("tenant_a", b"short_key")

    def test_audit_log(self):
        """Test audit logging of access attempts."""
        isolation = TenantDataIsolation()
        resource = {"id": "123", "tenant_id": "wrong_tenant"}

        with TenantContext(tenant_id="tenant_abc"):
            try:
                isolation.validate_ownership(resource)
            except IsolationViolation:
                pass

        audit_log = isolation.get_audit_log()
        assert len(audit_log) > 0
        assert audit_log[-1].allowed is False
        assert "mismatch" in (audit_log[-1].reason or "").lower()

    def test_audit_log_filter_by_tenant(self):
        """Test filtering audit log by tenant."""
        isolation = TenantDataIsolation()

        # Create some audit entries
        with TenantContext(tenant_id="tenant_a"):
            try:
                isolation.validate_ownership({"tenant_id": "other"})
            except IsolationViolation:
                pass

        with TenantContext(tenant_id="tenant_b"):
            try:
                isolation.validate_ownership({"tenant_id": "other"})
            except IsolationViolation:
                pass

        log_a = isolation.get_audit_log(tenant_id="tenant_a")
        log_b = isolation.get_audit_log(tenant_id="tenant_b")

        assert all(e.tenant_id == "tenant_a" for e in log_a)
        assert all(e.tenant_id == "tenant_b" for e in log_b)

    def test_clear_audit_log(self):
        """Test clearing audit log."""
        isolation = TenantDataIsolation()

        # Create an audit entry
        with TenantContext(tenant_id="tenant_a"):
            try:
                isolation.validate_ownership({"tenant_id": "other"})
            except IsolationViolation:
                pass

        assert len(isolation.get_audit_log()) > 0

        isolation.clear_audit_log()

        assert len(isolation.get_audit_log()) == 0


class TestIsolatedResourceDecorator:
    """Tests for IsolatedResource decorator."""

    def test_auto_sets_tenant(self):
        """Test decorator auto-sets tenant ID on creation."""

        @IsolatedResource()
        class MyResource:
            def __init__(self):
                self.tenant_id = None
                self.data = "test"

        with TenantContext(tenant_id="auto_tenant"):
            resource = MyResource()

        assert resource.tenant_id == "auto_tenant"

    def test_validates_on_creation(self):
        """Test decorator validates tenant on creation."""

        @IsolatedResource()
        class MyResource:
            def __init__(self, tenant_id=None):
                self.tenant_id = tenant_id
                self.data = "test"

        with TenantContext(tenant_id="expected_tenant"):
            # Should raise because we're setting wrong tenant
            with pytest.raises(IsolationViolation):
                MyResource(tenant_id="wrong_tenant")

    def test_custom_tenant_field(self):
        """Test decorator with custom tenant field."""

        @IsolatedResource(tenant_field="org_id")
        class MyResource:
            def __init__(self):
                self.org_id = None
                self.data = "test"

        with TenantContext(tenant_id="my_org"):
            resource = MyResource()

        assert resource.org_id == "my_org"


class TestEnsureTenantScope:
    """Tests for ensure_tenant_scope decorator."""

    def test_sync_function_with_context(self):
        """Test decorator allows sync function when context set."""

        @ensure_tenant_scope
        def my_function():
            return "success"

        with TenantContext(tenant_id="tenant_123"):
            result = my_function()

        assert result == "success"

    def test_sync_function_without_context(self):
        """Test decorator raises for sync function without context."""

        @ensure_tenant_scope
        def my_function():
            return "success"

        with pytest.raises(TenantNotSetError):
            my_function()

    @pytest.mark.asyncio
    async def test_async_function_with_context(self):
        """Test decorator allows async function when context set."""

        @ensure_tenant_scope
        async def my_async_function():
            return "async_success"

        with TenantContext(tenant_id="tenant_123"):
            result = await my_async_function()

        assert result == "async_success"

    @pytest.mark.asyncio
    async def test_async_function_without_context(self):
        """Test decorator raises for async function without context."""

        @ensure_tenant_scope
        async def my_async_function():
            return "async_success"

        with pytest.raises(TenantNotSetError):
            await my_async_function()


class TestCrossTenantPrevention:
    """Integration tests for cross-tenant data prevention."""

    def test_cannot_access_other_tenant_data(self):
        """Test that one tenant cannot access another tenant's data."""
        isolation = TenantDataIsolation()

        tenant_a_data = {"id": "1", "tenant_id": "tenant_a", "secret": "a_secret"}
        tenant_b_data = {"id": "2", "tenant_id": "tenant_b", "secret": "b_secret"}

        # Tenant A can access their own data
        with TenantContext(tenant_id="tenant_a"):
            assert isolation.validate_ownership(tenant_a_data) is True

        # Tenant A cannot access Tenant B's data
        with TenantContext(tenant_id="tenant_a"):
            with pytest.raises(IsolationViolation):
                isolation.validate_ownership(tenant_b_data)

        # Tenant B can access their own data
        with TenantContext(tenant_id="tenant_b"):
            assert isolation.validate_ownership(tenant_b_data) is True

    def test_query_filtering_prevents_cross_tenant(self):
        """Test that query filtering isolates tenant data."""
        isolation = TenantDataIsolation()
        base_query = {"status": "active"}

        with TenantContext(tenant_id="tenant_a"):
            query_a = isolation.filter_query(base_query.copy(), "debates")

        with TenantContext(tenant_id="tenant_b"):
            query_b = isolation.filter_query(base_query.copy(), "debates")

        # Each query should have their own tenant filter
        assert query_a["tenant_id"] == "tenant_a"
        assert query_b["tenant_id"] == "tenant_b"
        assert query_a["tenant_id"] != query_b["tenant_id"]
