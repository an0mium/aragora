"""Tests for WebSocket tenant filtering."""

import pytest

from aragora.server.stream.tenant_filter import (
    TenantFilter,
    get_tenant_filter,
)


class TestTenantFilter:
    """Tests for TenantFilter class."""

    @pytest.fixture
    def tenant_filter(self):
        """Create a fresh tenant filter for each test."""
        return TenantFilter()

    def test_register_client(self, tenant_filter):
        """Should register client with tenant."""
        tenant_filter.register_client(client_id=1, tenant_id="tenant1")
        assert tenant_filter.get_client_tenant(1) == "tenant1"

    def test_register_client_without_tenant(self, tenant_filter):
        """Should register unauthenticated client."""
        tenant_filter.register_client(client_id=1, tenant_id=None)
        assert tenant_filter.get_client_tenant(1) is None

    def test_unregister_client(self, tenant_filter):
        """Should remove client registration."""
        tenant_filter.register_client(client_id=1, tenant_id="tenant1")
        tenant_filter.unregister_client(1)
        assert tenant_filter.get_client_tenant(1) is None

    def test_register_resource(self, tenant_filter):
        """Should register resource with tenant."""
        tenant_filter.register_resource(resource_id="debate1", tenant_id="tenant1")
        assert tenant_filter.get_resource_tenant("debate1") == "tenant1"

    def test_unregister_resource(self, tenant_filter):
        """Should remove resource registration."""
        tenant_filter.register_resource(resource_id="debate1", tenant_id="tenant1")
        tenant_filter.unregister_resource("debate1")
        assert tenant_filter.get_resource_tenant("debate1") is None

    def test_can_access_public_resource(self, tenant_filter):
        """Authenticated client should access public resource."""
        tenant_filter.register_client(client_id=1, tenant_id="tenant1")
        # Resource not registered = public
        assert tenant_filter.can_access_resource(client_id=1, resource_id="public1") is True

    def test_can_access_own_tenant_resource(self, tenant_filter):
        """Client should access resources in same tenant."""
        tenant_filter.register_client(client_id=1, tenant_id="tenant1")
        tenant_filter.register_resource(resource_id="debate1", tenant_id="tenant1")
        assert tenant_filter.can_access_resource(client_id=1, resource_id="debate1") is True

    def test_cannot_access_other_tenant_resource(self, tenant_filter):
        """Client should not access resources in different tenant."""
        tenant_filter.register_client(client_id=1, tenant_id="tenant1")
        tenant_filter.register_resource(resource_id="debate1", tenant_id="tenant2")
        assert tenant_filter.can_access_resource(client_id=1, resource_id="debate1") is False

    def test_unauthenticated_cannot_access_tenant_resource(self, tenant_filter):
        """Unauthenticated client should not access tenant resources."""
        tenant_filter.register_client(client_id=1, tenant_id=None)
        tenant_filter.register_resource(resource_id="debate1", tenant_id="tenant1")
        assert tenant_filter.can_access_resource(client_id=1, resource_id="debate1") is False

    def test_unauthenticated_can_access_public_resource(self, tenant_filter):
        """Unauthenticated client should access public resources."""
        tenant_filter.register_client(client_id=1, tenant_id=None)
        # No resource registration = public
        assert tenant_filter.can_access_resource(client_id=1, resource_id="public1") is True


class TestShouldReceiveEvent:
    """Tests for event filtering logic."""

    @pytest.fixture
    def tenant_filter(self):
        """Create a fresh tenant filter for each test."""
        return TenantFilter()

    def test_system_event_to_all(self, tenant_filter):
        """System events (no tenant/resource) should go to all clients."""
        tenant_filter.register_client(client_id=1, tenant_id="tenant1")
        tenant_filter.register_client(client_id=2, tenant_id=None)

        assert (
            tenant_filter.should_receive_event(
                client_id=1, event_tenant_id=None, event_resource_id=None
            )
            is True
        )
        assert (
            tenant_filter.should_receive_event(
                client_id=2, event_tenant_id=None, event_resource_id=None
            )
            is True
        )

    def test_tenant_event_to_matching_client(self, tenant_filter):
        """Tenant-scoped events should go to matching clients."""
        tenant_filter.register_client(client_id=1, tenant_id="tenant1")

        assert (
            tenant_filter.should_receive_event(
                client_id=1, event_tenant_id="tenant1", event_resource_id=None
            )
            is True
        )

    def test_tenant_event_not_to_other_tenant(self, tenant_filter):
        """Tenant-scoped events should not go to other tenants."""
        tenant_filter.register_client(client_id=1, tenant_id="tenant1")

        assert (
            tenant_filter.should_receive_event(
                client_id=1, event_tenant_id="tenant2", event_resource_id=None
            )
            is False
        )

    def test_tenant_event_not_to_unauthenticated(self, tenant_filter):
        """Tenant-scoped events should not go to unauthenticated clients."""
        tenant_filter.register_client(client_id=1, tenant_id=None)

        assert (
            tenant_filter.should_receive_event(
                client_id=1, event_tenant_id="tenant1", event_resource_id=None
            )
            is False
        )

    def test_resource_event_uses_resource_tenant(self, tenant_filter):
        """Resource events should use resource's tenant for filtering."""
        tenant_filter.register_client(client_id=1, tenant_id="tenant1")
        tenant_filter.register_resource(resource_id="debate1", tenant_id="tenant1")

        assert (
            tenant_filter.should_receive_event(
                client_id=1, event_tenant_id=None, event_resource_id="debate1"
            )
            is True
        )

    def test_resource_event_blocked_for_other_tenant(self, tenant_filter):
        """Resource events should be blocked for other tenants."""
        tenant_filter.register_client(client_id=1, tenant_id="tenant2")
        tenant_filter.register_resource(resource_id="debate1", tenant_id="tenant1")

        assert (
            tenant_filter.should_receive_event(
                client_id=1, event_tenant_id=None, event_resource_id="debate1"
            )
            is False
        )

    def test_explicit_tenant_overrides_resource(self, tenant_filter):
        """Explicit tenant_id should take precedence over resource tenant."""
        tenant_filter.register_client(client_id=1, tenant_id="tenant1")
        tenant_filter.register_resource(resource_id="debate1", tenant_id="tenant2")

        # Event has explicit tenant1, resource is tenant2
        # Should use explicit tenant1
        assert (
            tenant_filter.should_receive_event(
                client_id=1, event_tenant_id="tenant1", event_resource_id="debate1"
            )
            is True
        )


class TestFilterClientsForEvent:
    """Tests for bulk client filtering."""

    @pytest.fixture
    def tenant_filter(self):
        """Create a fresh tenant filter for each test."""
        return TenantFilter()

    def test_filter_for_system_event(self, tenant_filter):
        """System events should reach all clients."""
        tenant_filter.register_client(client_id=1, tenant_id="tenant1")
        tenant_filter.register_client(client_id=2, tenant_id="tenant2")
        tenant_filter.register_client(client_id=3, tenant_id=None)

        result = tenant_filter.filter_clients_for_event(
            client_ids={1, 2, 3},
            event_tenant_id=None,
            event_resource_id=None,
        )
        assert result == {1, 2, 3}

    def test_filter_for_tenant_event(self, tenant_filter):
        """Tenant events should only reach matching clients."""
        tenant_filter.register_client(client_id=1, tenant_id="tenant1")
        tenant_filter.register_client(client_id=2, tenant_id="tenant1")
        tenant_filter.register_client(client_id=3, tenant_id="tenant2")
        tenant_filter.register_client(client_id=4, tenant_id=None)

        result = tenant_filter.filter_clients_for_event(
            client_ids={1, 2, 3, 4},
            event_tenant_id="tenant1",
            event_resource_id=None,
        )
        assert result == {1, 2}

    def test_filter_for_resource_event(self, tenant_filter):
        """Resource events should only reach clients in resource's tenant."""
        tenant_filter.register_client(client_id=1, tenant_id="tenant1")
        tenant_filter.register_client(client_id=2, tenant_id="tenant2")
        tenant_filter.register_resource(resource_id="debate1", tenant_id="tenant1")

        result = tenant_filter.filter_clients_for_event(
            client_ids={1, 2},
            event_tenant_id=None,
            event_resource_id="debate1",
        )
        assert result == {1}


class TestValidateSubscription:
    """Tests for subscription validation."""

    @pytest.fixture
    def tenant_filter(self):
        """Create a fresh tenant filter for each test."""
        return TenantFilter()

    def test_valid_subscription_same_tenant(self, tenant_filter):
        """Should allow subscription to same-tenant resource."""
        tenant_filter.register_client(client_id=1, tenant_id="tenant1")
        tenant_filter.register_resource(resource_id="debate1", tenant_id="tenant1")

        allowed, msg = tenant_filter.validate_subscription(client_id=1, resource_id="debate1")
        assert allowed is True
        assert "allowed" in msg.lower()

    def test_invalid_subscription_different_tenant(self, tenant_filter):
        """Should deny subscription to different-tenant resource."""
        tenant_filter.register_client(client_id=1, tenant_id="tenant1")
        tenant_filter.register_resource(resource_id="debate1", tenant_id="tenant2")

        allowed, msg = tenant_filter.validate_subscription(client_id=1, resource_id="debate1")
        assert allowed is False
        assert "denied" in msg.lower()

    def test_valid_subscription_public_resource(self, tenant_filter):
        """Should allow subscription to public resource."""
        tenant_filter.register_client(client_id=1, tenant_id="tenant1")
        # Resource not registered = public

        allowed, msg = tenant_filter.validate_subscription(client_id=1, resource_id="public1")
        assert allowed is True


class TestGlobalTenantFilter:
    """Tests for global tenant filter instance."""

    def test_get_tenant_filter_returns_instance(self):
        """Should return a TenantFilter instance."""
        tf = get_tenant_filter()
        assert isinstance(tf, TenantFilter)

    def test_get_tenant_filter_returns_same_instance(self):
        """Should return same instance on repeated calls."""
        tf1 = get_tenant_filter()
        tf2 = get_tenant_filter()
        assert tf1 is tf2
