"""
Tenant filtering for WebSocket event broadcasts.

Provides tenant isolation for real-time event streams by:
1. Tracking tenant_id for each WebSocket client
2. Filtering events to only reach clients in the same tenant
3. Validating tenant ownership of subscribed resources
"""

import logging
from typing import Dict, Optional, Set

logger = logging.getLogger(__name__)


class TenantFilter:
    """
    Filters WebSocket events by tenant for secure multi-tenant broadcasts.

    Each client connection is associated with a tenant_id extracted from their
    auth token. Events are only delivered to clients within the same tenant.
    """

    def __init__(self) -> None:
        # Map client_id (websocket hash) -> tenant_id
        self._client_tenants: Dict[int, Optional[str]] = {}
        # Map resource_id (debate/loop) -> tenant_id
        self._resource_tenants: Dict[str, str] = {}
        # Map tenant_id -> set of resource_ids
        self._tenant_resources: Dict[str, Set[str]] = {}

    def register_client(self, client_id: int, tenant_id: Optional[str]) -> None:
        """Register a client connection with their tenant.

        Args:
            client_id: WebSocket connection identifier (id(websocket))
            tenant_id: Tenant ID from auth token, or None for unauthenticated
        """
        self._client_tenants[client_id] = tenant_id
        logger.debug(f"Registered client {client_id} for tenant {tenant_id}")

    def unregister_client(self, client_id: int) -> None:
        """Remove client registration on disconnect.

        Args:
            client_id: WebSocket connection identifier
        """
        self._client_tenants.pop(client_id, None)
        logger.debug(f"Unregistered client {client_id}")

    def register_resource(self, resource_id: str, tenant_id: str) -> None:
        """Register a resource (debate/loop) with its tenant.

        Args:
            resource_id: The debate/loop ID
            tenant_id: The tenant that owns this resource
        """
        self._resource_tenants[resource_id] = tenant_id
        if tenant_id not in self._tenant_resources:
            self._tenant_resources[tenant_id] = set()
        self._tenant_resources[tenant_id].add(resource_id)
        logger.debug(f"Registered resource {resource_id} for tenant {tenant_id}")

    def unregister_resource(self, resource_id: str) -> None:
        """Remove resource registration.

        Args:
            resource_id: The debate/loop ID
        """
        tenant_id = self._resource_tenants.pop(resource_id, None)
        if tenant_id and tenant_id in self._tenant_resources:
            self._tenant_resources[tenant_id].discard(resource_id)
        logger.debug(f"Unregistered resource {resource_id}")

    def get_client_tenant(self, client_id: int) -> Optional[str]:
        """Get the tenant ID for a client.

        Args:
            client_id: WebSocket connection identifier

        Returns:
            The tenant ID or None if not registered/unauthenticated
        """
        return self._client_tenants.get(client_id)

    def get_resource_tenant(self, resource_id: str) -> Optional[str]:
        """Get the tenant ID that owns a resource.

        Args:
            resource_id: The debate/loop ID

        Returns:
            The tenant ID or None if not registered
        """
        return self._resource_tenants.get(resource_id)

    def can_access_resource(self, client_id: int, resource_id: str) -> bool:
        """Check if a client can access a resource.

        Access is allowed if:
        1. Resource is not tenant-scoped (public)
        2. Client is not tenant-scoped (unauthenticated/anonymous)
        3. Client's tenant matches resource's tenant

        Args:
            client_id: WebSocket connection identifier
            resource_id: The debate/loop ID

        Returns:
            True if access is allowed
        """
        client_tenant = self._client_tenants.get(client_id)
        resource_tenant = self._resource_tenants.get(resource_id)

        # Public resources (no tenant) accessible to all
        if not resource_tenant:
            return True

        # Unauthenticated clients can only access public resources
        if not client_tenant:
            return False

        # Tenant match required
        return client_tenant == resource_tenant

    def should_receive_event(
        self,
        client_id: int,
        event_tenant_id: Optional[str],
        event_resource_id: Optional[str],
    ) -> bool:
        """Determine if a client should receive an event.

        Enforces tenant isolation:
        1. System-wide events (no tenant/resource) go to all clients
        2. Tenant-scoped events only go to clients in that tenant
        3. Resource-scoped events use resource tenant for filtering

        Args:
            client_id: WebSocket connection identifier
            event_tenant_id: Explicit tenant_id on the event
            event_resource_id: Resource ID (loop_id/debate_id) on the event

        Returns:
            True if the client should receive this event
        """
        client_tenant = self._client_tenants.get(client_id)

        # System-wide events (no scoping) - broadcast to all
        if not event_tenant_id and not event_resource_id:
            return True

        # Get effective tenant from event or resource
        effective_tenant = event_tenant_id
        if not effective_tenant and event_resource_id:
            effective_tenant = self._resource_tenants.get(event_resource_id)

        # If event has no tenant association, allow
        if not effective_tenant:
            return True

        # Require tenant match
        if not client_tenant:
            # Unauthenticated clients don't receive tenant-scoped events
            return False

        return client_tenant == effective_tenant

    def filter_clients_for_event(
        self,
        client_ids: Set[int],
        event_tenant_id: Optional[str],
        event_resource_id: Optional[str],
    ) -> Set[int]:
        """Filter a set of clients to those who should receive an event.

        Args:
            client_ids: Set of client identifiers
            event_tenant_id: Explicit tenant_id on the event
            event_resource_id: Resource ID on the event

        Returns:
            Set of client_ids that should receive the event
        """
        return {
            cid
            for cid in client_ids
            if self.should_receive_event(cid, event_tenant_id, event_resource_id)
        }

    def validate_subscription(self, client_id: int, resource_id: str) -> tuple[bool, str]:
        """Validate that a client can subscribe to a resource.

        Args:
            client_id: WebSocket connection identifier
            resource_id: The resource to subscribe to

        Returns:
            Tuple of (allowed, reason_message)
        """
        if self.can_access_resource(client_id, resource_id):
            return True, "Subscription allowed"

        client_tenant = self._client_tenants.get(client_id)
        resource_tenant = self._resource_tenants.get(resource_id)

        return False, (
            f"Access denied: client tenant '{client_tenant}' "
            f"cannot subscribe to resource owned by tenant '{resource_tenant}'"
        )


# Global tenant filter instance
_tenant_filter = TenantFilter()


def get_tenant_filter() -> TenantFilter:
    """Get the global tenant filter instance."""
    return _tenant_filter
