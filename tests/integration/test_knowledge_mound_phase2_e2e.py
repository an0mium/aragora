"""
E2E Integration Tests for Knowledge Mound Phase 2 Features.

Tests the full HTTP handler flow for:
- Global Knowledge (verified facts)
- Cross-Workspace Sharing
- Knowledge Federation
- Visibility Levels

These tests use mock HTTP handlers to verify the complete request/response flow.
"""

import pytest
import json
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
from io import BytesIO

from aragora.server.handlers.knowledge_base.mound.federation import FederationOperationsMixin
from aragora.server.handlers.knowledge_base.mound.sharing import SharingOperationsMixin
from aragora.server.handlers.knowledge_base.mound.global_knowledge import (
    GlobalKnowledgeOperationsMixin,
)
from aragora.server.handlers.utils.responses import HandlerResult


def parse_response(result: HandlerResult) -> tuple[dict, int]:
    """Parse HandlerResult into (body_dict, status_code)."""
    body = json.loads(result.body.decode("utf-8")) if result.body else {}
    return body, result.status_code


# =============================================================================
# Test Fixtures
# =============================================================================


class MockHandler:
    """Mock HTTP handler for testing."""

    def __init__(self, body: dict = None, headers: dict = None):
        self.headers = headers or {"Content-Length": "0"}
        if body:
            body_bytes = json.dumps(body).encode("utf-8")
            self.rfile = BytesIO(body_bytes)
            self.headers["Content-Length"] = str(len(body_bytes))
        else:
            self.rfile = BytesIO(b"")


class MockUser:
    """Mock user object."""

    def __init__(self, user_id: str, email: str, permissions: list = None, is_admin: bool = False):
        self.id = user_id
        self.user_id = user_id
        self.email = email
        self.permissions = permissions or []
        self.is_admin = is_admin


class MockKnowledgeMound:
    """Mock KnowledgeMound for testing handlers."""

    def __init__(self):
        self._items = {}
        self._grants = []
        self._regions = {}
        self._global_facts = []

    async def store_verified_fact(
        self, content, source, confidence, evidence_ids, verified_by, topics
    ):
        node_id = f"fact_{len(self._global_facts)}"
        self._global_facts.append(
            {
                "id": node_id,
                "content": content,
                "source": source,
                "confidence": confidence,
            }
        )
        return node_id

    async def query_global_knowledge(self, query, limit, topics=None):
        items = []
        for fact in self._global_facts[:limit]:
            item = MagicMock()
            item.to_dict = lambda f=fact: f
            items.append(item)
        return items

    async def get_system_facts(self, limit, topics=None):
        return self._global_facts[:limit]

    def get_system_workspace_id(self):
        return "__system__"

    async def promote_to_global(self, item_id, workspace_id, promoted_by, reason):
        return f"global_{item_id}"

    async def share_with_workspace(
        self, item_id, from_workspace_id, to_workspace_id, shared_by, permissions, expires_at
    ):
        grant = MagicMock()
        grant.item_id = item_id
        grant.grantee_id = to_workspace_id
        self._grants.append(grant)
        return grant

    async def share_with_user(
        self, item_id, from_workspace_id, to_user_id, shared_by, permissions, expires_at
    ):
        grant = MagicMock()
        grant.item_id = item_id
        grant.grantee_id = to_user_id
        self._grants.append(grant)
        return grant

    async def get_shared_with_me(self, workspace_id, user_id, limit, include_expired=False):
        items = []
        for i in range(min(3, limit)):
            item = MagicMock()
            item.to_dict = lambda: {"id": f"shared_{i}", "content": "shared content"}
            items.append(item)
        return items

    async def revoke_share(self, item_id, grantee_id, revoked_by):
        self._grants = [
            g for g in self._grants if g.item_id != item_id or g.grantee_id != grantee_id
        ]

    async def get_share_grants(self, shared_by, workspace_id):
        grants = []
        for g in self._grants:
            mock_grant = MagicMock()
            mock_grant.to_dict = lambda: {"item_id": g.item_id, "grantee_id": g.grantee_id}
            grants.append(mock_grant)
        return grants

    async def update_share_permissions(
        self, item_id, grantee_id, permissions, expires_at, updated_by
    ):
        grant = MagicMock()
        grant.to_dict = lambda: {
            "item_id": item_id,
            "grantee_id": grantee_id,
            "permissions": permissions,
        }
        return grant

    async def register_federated_region(self, region_id, endpoint_url, api_key, mode, sync_scope):
        region = MagicMock()
        region.region_id = region_id
        region.endpoint_url = endpoint_url
        region.mode = mode
        region.sync_scope = sync_scope
        region.enabled = True
        self._regions[region_id] = region
        return region

    async def unregister_federated_region(self, region_id):
        if region_id in self._regions:
            del self._regions[region_id]
            return True
        return False

    async def sync_to_region(self, region_id, workspace_id, since, visibility_levels):
        result = MagicMock()
        result.success = True
        result.region_id = region_id
        result.direction = "push"
        result.nodes_synced = 10
        result.nodes_skipped = 2
        result.nodes_failed = 0
        result.duration_ms = 150
        result.error = None
        return result

    async def pull_from_region(self, region_id, workspace_id, since):
        result = MagicMock()
        result.success = True
        result.region_id = region_id
        result.direction = "pull"
        result.nodes_synced = 5
        result.nodes_failed = 0
        result.duration_ms = 100
        result.error = None
        return result

    async def sync_all_regions(self, workspace_id, since):
        results = []
        for region_id in self._regions:
            result = MagicMock()
            result.region_id = region_id
            result.direction = "bidirectional"
            result.success = True
            result.nodes_synced = 5
            result.nodes_failed = 0
            result.error = None
            results.append(result)
        return results

    async def get_federation_status(self):
        return {
            region_id: {
                "enabled": r.enabled,
                "healthy": True,
                "last_sync": datetime.utcnow().isoformat(),
            }
            for region_id, r in self._regions.items()
        }


class TestableHandler(
    GlobalKnowledgeOperationsMixin,
    SharingOperationsMixin,
    FederationOperationsMixin,
):
    """Testable handler combining all mixins."""

    def __init__(self, mound: MockKnowledgeMound, user: MockUser = None):
        self._mound = mound
        self._user = user

    def _get_mound(self):
        return self._mound

    def require_auth_or_error(self, handler):
        if self._user:
            return self._user, None
        from aragora.server.handlers.base import error_response

        return None, error_response("Unauthorized", 401)

    def require_admin_or_error(self, handler):
        if self._user and self._user.is_admin:
            return self._user, None
        from aragora.server.handlers.base import error_response

        return None, error_response("Admin required", 403)


# =============================================================================
# Global Knowledge E2E Tests
# =============================================================================


class TestGlobalKnowledgeE2E:
    """E2E tests for global knowledge HTTP handlers."""

    @pytest.fixture
    def setup(self):
        mound = MockKnowledgeMound()
        admin_user = MockUser("admin1", "admin@example.com", ["admin"], is_admin=True)
        handler = TestableHandler(mound, admin_user)
        return handler, mound

    def test_store_verified_fact_success(self, setup):
        """Should store verified fact via handler."""
        handler, mound = setup
        http_handler = MockHandler(
            body={
                "content": "The Earth orbits the Sun.",
                "source": "astronomy.gov",
                "confidence": 0.99,
                "topics": ["astronomy", "physics"],
            }
        )

        with patch(
            "aragora.server.handlers.knowledge_base.mound.global_knowledge.track_global_fact"
        ):
            result = handler._handle_store_verified_fact(http_handler)

        body, status = parse_response(result)
        assert status == 201
        assert body["success"] is True
        assert "node_id" in body

    def test_store_verified_fact_requires_admin(self):
        """Should require admin for storing facts."""
        mound = MockKnowledgeMound()
        regular_user = MockUser("user1", "user@example.com", [])
        handler = TestableHandler(mound, regular_user)
        http_handler = MockHandler(
            body={
                "content": "Test fact",
                "source": "test",
            }
        )

        result = handler._handle_store_verified_fact(http_handler)
        body, status = parse_response(result)

        assert status == 403

    def test_query_global_knowledge(self, setup):
        """Should query global knowledge."""
        handler, mound = setup
        # Add some facts first
        mound._global_facts = [
            {"id": "fact1", "content": "Fact 1"},
            {"id": "fact2", "content": "Fact 2"},
        ]

        with patch(
            "aragora.server.handlers.knowledge_base.mound.global_knowledge.track_global_query"
        ):
            result = handler._handle_query_global({"query": ["test"], "limit": ["10"]})

        body, status = parse_response(result)
        assert status == 200
        assert "items" in body
        assert body["count"] >= 0

    def test_get_system_workspace_id(self, setup):
        """Should return system workspace ID."""
        handler, mound = setup

        result = handler._handle_get_system_workspace_id()

        body, status = parse_response(result)
        assert status == 200
        assert body["system_workspace_id"] == "__system__"


# =============================================================================
# Sharing E2E Tests
# =============================================================================


class TestSharingE2E:
    """E2E tests for sharing HTTP handlers."""

    @pytest.fixture
    def setup(self):
        mound = MockKnowledgeMound()
        user = MockUser("user1", "user@example.com")
        handler = TestableHandler(mound, user)
        return handler, mound

    def test_share_item_with_workspace(self, setup):
        """Should share item with workspace."""
        handler, mound = setup
        http_handler = MockHandler(
            body={
                "item_id": "item1",
                "target_type": "workspace",
                "target_id": "workspace2",
                "permissions": ["read"],
            }
        )

        with patch("aragora.server.handlers.knowledge_base.mound.sharing.track_share"):
            result = handler._handle_share_item(http_handler)

        body, status = parse_response(result)
        assert status == 201
        assert body["success"] is True
        assert body["share"]["target_type"] == "workspace"

    def test_share_item_with_user(self, setup):
        """Should share item with user."""
        handler, mound = setup
        http_handler = MockHandler(
            body={
                "item_id": "item1",
                "target_type": "user",
                "target_id": "user2",
                "permissions": ["read", "comment"],
            }
        )

        with patch("aragora.server.handlers.knowledge_base.mound.sharing.track_share"):
            result = handler._handle_share_item(http_handler)

        body, status = parse_response(result)
        assert status == 201
        assert body["share"]["target_type"] == "user"

    def test_share_requires_valid_target_type(self, setup):
        """Should reject invalid target type."""
        handler, mound = setup
        http_handler = MockHandler(
            body={
                "item_id": "item1",
                "target_type": "invalid",
                "target_id": "something",
            }
        )

        result = handler._handle_share_item(http_handler)
        body, status = parse_response(result)

        assert status == 400

    def test_get_shared_with_me(self, setup):
        """Should list items shared with user."""
        handler, mound = setup
        http_handler = MockHandler()

        result = handler._handle_shared_with_me(
            {"workspace_id": ["default"], "limit": ["50"]},
            http_handler,
        )

        body, status = parse_response(result)
        assert status == 200
        assert "items" in body
        assert "count" in body

    def test_revoke_share(self, setup):
        """Should revoke a share."""
        handler, mound = setup
        # Add a grant first
        grant = MagicMock()
        grant.item_id = "item1"
        grant.grantee_id = "user2"
        mound._grants.append(grant)

        http_handler = MockHandler(
            body={
                "item_id": "item1",
                "grantee_id": "user2",
            }
        )

        result = handler._handle_revoke_share(http_handler)
        body, status = parse_response(result)

        assert status == 200
        assert body["success"] is True


# =============================================================================
# Federation E2E Tests
# =============================================================================


class TestFederationE2E:
    """E2E tests for federation HTTP handlers."""

    @pytest.fixture
    def setup(self):
        mound = MockKnowledgeMound()
        admin_user = MockUser("admin1", "admin@example.com", ["admin"], is_admin=True)
        handler = TestableHandler(mound, admin_user)
        return handler, mound

    def test_register_region(self, setup):
        """Should register federated region."""
        handler, mound = setup
        http_handler = MockHandler(
            body={
                "region_id": "us-west",
                "endpoint_url": "https://us-west.aragora.io/api",
                "api_key": "secret_key_123",
                "mode": "bidirectional",
                "sync_scope": "summary",
            }
        )

        result = handler._handle_register_region(http_handler)
        body, status = parse_response(result)

        assert status == 201
        assert body["success"] is True
        assert body["region"]["region_id"] == "us-west"

    def test_register_region_requires_admin(self):
        """Should require admin for region registration."""
        mound = MockKnowledgeMound()
        regular_user = MockUser("user1", "user@example.com")
        handler = TestableHandler(mound, regular_user)
        http_handler = MockHandler(
            body={
                "region_id": "us-west",
                "endpoint_url": "https://us-west.aragora.io/api",
                "api_key": "key",
            }
        )

        result = handler._handle_register_region(http_handler)
        body, status = parse_response(result)

        assert status == 403

    def test_unregister_region(self, setup):
        """Should unregister federated region."""
        handler, mound = setup
        # Register first
        region = MagicMock()
        region.region_id = "us-west"
        mound._regions["us-west"] = region

        http_handler = MockHandler()

        result = handler._handle_unregister_region("us-west", http_handler)
        body, status = parse_response(result)

        assert status == 200
        assert body["success"] is True

    def test_sync_to_region(self, setup):
        """Should sync to region."""
        handler, mound = setup
        # Register region first
        region = MagicMock()
        region.region_id = "us-west"
        mound._regions["us-west"] = region

        http_handler = MockHandler(
            body={
                "region_id": "us-west",
                "workspace_id": "workspace1",
            }
        )

        with patch(
            "aragora.server.handlers.knowledge_base.mound.federation.track_federation_sync"
        ) as mock_track:
            mock_track.return_value.__enter__ = MagicMock(return_value={})
            mock_track.return_value.__exit__ = MagicMock(return_value=False)
            result = handler._handle_sync_to_region(http_handler)

        body, status = parse_response(result)
        assert status == 200
        assert body["success"] is True
        assert body["direction"] == "push"
        assert body["nodes_synced"] == 10

    def test_pull_from_region(self, setup):
        """Should pull from region."""
        handler, mound = setup
        http_handler = MockHandler(
            body={
                "region_id": "us-west",
                "workspace_id": "workspace1",
            }
        )

        with patch(
            "aragora.server.handlers.knowledge_base.mound.federation.track_federation_sync"
        ) as mock_track:
            mock_track.return_value.__enter__ = MagicMock(return_value={})
            mock_track.return_value.__exit__ = MagicMock(return_value=False)
            result = handler._handle_pull_from_region(http_handler)

        body, status = parse_response(result)
        assert status == 200
        assert body["success"] is True
        assert body["direction"] == "pull"

    def test_get_federation_status(self, setup):
        """Should get federation status."""
        handler, mound = setup
        # Register some regions
        for region_id in ["us-west", "eu-central"]:
            region = MagicMock()
            region.region_id = region_id
            region.enabled = True
            mound._regions[region_id] = region

        with patch(
            "aragora.server.handlers.knowledge_base.mound.federation.track_federation_regions"
        ):
            result = handler._handle_get_federation_status({})

        body, status = parse_response(result)
        assert status == 200
        assert "regions" in body
        assert body["total_regions"] == 2

    def test_list_regions(self, setup):
        """Should list federated regions."""
        handler, mound = setup
        # Register some regions
        for region_id in ["us-west", "eu-central"]:
            region = MagicMock()
            region.region_id = region_id
            region.enabled = True
            mound._regions[region_id] = region

        result = handler._handle_list_regions({})
        body, status = parse_response(result)

        assert status == 200
        assert "regions" in body
        assert body["count"] == 2


# =============================================================================
# Cross-Feature Integration Tests
# =============================================================================


class TestCrossFeatureIntegration:
    """Tests for interactions between Phase 2 features."""

    def test_global_fact_can_be_shared(self):
        """Global facts should be shareable across workspaces."""
        mound = MockKnowledgeMound()
        admin_user = MockUser("admin1", "admin@example.com", ["admin"], is_admin=True)
        handler = TestableHandler(mound, admin_user)

        # Store a global fact
        http_handler = MockHandler(
            body={
                "content": "Water boils at 100C at sea level",
                "source": "physics.edu",
            }
        )
        with patch(
            "aragora.server.handlers.knowledge_base.mound.global_knowledge.track_global_fact"
        ):
            result = handler._handle_store_verified_fact(http_handler)
        body, status = parse_response(result)
        assert status == 201

        # Share it with another workspace
        http_handler = MockHandler(
            body={
                "item_id": "fact_0",
                "target_type": "workspace",
                "target_id": "workspace2",
            }
        )
        with patch("aragora.server.handlers.knowledge_base.mound.sharing.track_share"):
            result = handler._handle_share_item(http_handler)
        body, status = parse_response(result)
        assert status == 201

    def test_federated_regions_respect_sharing(self):
        """Federation should respect sharing permissions."""
        mound = MockKnowledgeMound()
        admin_user = MockUser("admin1", "admin@example.com", ["admin"], is_admin=True)
        handler = TestableHandler(mound, admin_user)

        # Register a region
        http_handler = MockHandler(
            body={
                "region_id": "partner-region",
                "endpoint_url": "https://partner.aragora.io/api",
                "api_key": "partner_key",
                "sync_scope": "summary",
            }
        )
        result = handler._handle_register_region(http_handler)
        body, status = parse_response(result)
        assert status == 201

        # Sync should respect the scope
        http_handler = MockHandler(
            body={
                "region_id": "partner-region",
                "visibility_levels": ["public", "organization"],
            }
        )
        with patch(
            "aragora.server.handlers.knowledge_base.mound.federation.track_federation_sync"
        ) as mock_track:
            mock_track.return_value.__enter__ = MagicMock(return_value={})
            mock_track.return_value.__exit__ = MagicMock(return_value=False)
            result = handler._handle_sync_to_region(http_handler)
        body, status = parse_response(result)
        assert status == 200


class TestErrorHandling:
    """Tests for error handling in handlers."""

    def test_mound_unavailable_returns_503(self):
        """Should return 503 when mound is unavailable."""
        handler = TestableHandler(None, MockUser("user1", "user@example.com"))

        result = handler._handle_query_global({"query": ["test"]})
        body, status = parse_response(result)

        assert status == 503

    def test_invalid_json_returns_400(self):
        """Should return 400 for invalid JSON."""
        mound = MockKnowledgeMound()
        user = MockUser("user1", "user@example.com")
        handler = TestableHandler(mound, user)

        http_handler = MockHandler()
        http_handler.headers["Content-Length"] = "10"
        http_handler.rfile = BytesIO(b"not valid")

        result = handler._handle_share_item(http_handler)
        body, status = parse_response(result)

        assert status == 400

    def test_missing_required_fields_returns_400(self):
        """Should return 400 for missing required fields."""
        mound = MockKnowledgeMound()
        admin_user = MockUser("admin1", "admin@example.com", ["admin"], is_admin=True)
        handler = TestableHandler(mound, admin_user)

        http_handler = MockHandler(
            body={
                # Missing region_id, endpoint_url, api_key
            }
        )

        result = handler._handle_register_region(http_handler)
        body, status = parse_response(result)

        assert status == 400
