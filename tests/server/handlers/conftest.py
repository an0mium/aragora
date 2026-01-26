"""
Shared fixtures for server handler tests.

This module provides common fixtures for testing HTTP handlers,
reducing duplication and ensuring consistent test setup.

Common patterns:
    @pytest.fixture
    def handler(mock_server_context):
        from aragora.server.handlers.example import ExampleHandler
        return ExampleHandler(server_context=mock_server_context)

    def test_can_handle_route(handler):
        assert handler.can_handle("/api/v1/example") is True

    def test_endpoint(handler, mock_http_handler):
        result = handler.handle("/api/v1/example", {}, mock_http_handler, "GET")
        assert result.status_code == 200
        body = parse_handler_response(result)
        assert "data" in body
"""

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, Mock

import pytest

from aragora.rbac.models import AuthorizationContext


# ============================================================================
# RBAC Auto-Bypass Fixture
# ============================================================================


@pytest.fixture(autouse=True)
def mock_auth_for_handler_tests(request, monkeypatch):
    """Bypass RBAC authentication for handler unit tests.

    This autouse fixture patches get_auth_context to return an authenticated
    admin context by default, and patches _get_context_from_args to inject
    context into already-decorated functions.

    To test authentication/authorization behavior specifically, use the
    @pytest.mark.no_auto_auth marker to opt-out of auto-mocking:

        @pytest.mark.no_auto_auth
        def test_unauthenticated_returns_401(handler, mock_http_handler):
            # get_auth_context will NOT be mocked for this test
            result = handler.handle("/api/v1/resource", {}, mock_http_handler)
            assert result.status_code == 401
    """
    # Check if test has opted out of auto-auth
    if "no_auto_auth" in [m.name for m in request.node.iter_markers()]:
        yield
        return

    # Create a mock auth context with admin permissions
    mock_auth_ctx = AuthorizationContext(
        user_id="test-user-001",
        user_email="test@example.com",
        org_id="test-org-001",
        roles={"admin", "owner"},
        permissions={"*"},  # Wildcard grants all permissions
    )

    async def mock_get_auth_context(request, require_auth=False):
        """Mock get_auth_context that returns admin context."""
        return mock_auth_ctx

    # Patch _get_context_from_args to return mock context when no context found
    # This is critical for functions that are already decorated at import time
    try:
        from aragora.rbac import decorators

        original_get_context = decorators._get_context_from_args

        def patched_get_context_from_args(args, kwargs, context_param):
            """Return mock context if no real context found."""
            result = original_get_context(args, kwargs, context_param)
            if result is None:
                return mock_auth_ctx
            return result

        monkeypatch.setattr(decorators, "_get_context_from_args", patched_get_context_from_args)
    except (ImportError, AttributeError):
        pass

    # Patch get_auth_context at various locations
    try:
        from aragora.server.handlers.utils import auth as utils_auth

        monkeypatch.setattr(utils_auth, "get_auth_context", mock_get_auth_context)
    except (ImportError, AttributeError):
        pass

    try:
        from aragora.server.handlers import secure

        monkeypatch.setattr(secure, "get_auth_context", mock_get_auth_context)
    except (ImportError, AttributeError):
        pass

    yield mock_auth_ctx


# ============================================================================
# Cache Clearing Fixture
# ============================================================================


@pytest.fixture(autouse=True)
def clear_handler_cache():
    """Clear handler TTL cache and class-level attributes before each test.

    This prevents cached results from previous tests from interfering.
    Also clears class-level attributes like elo_system that unified_server sets.
    """
    try:
        from aragora.server.handlers.admin.cache import clear_cache

        clear_cache()  # Clear all cache entries
    except ImportError:
        pass

    # Clear class-level elo_system that might be set by unified_server
    try:
        from aragora.server.handlers.base import BaseHandler

        if hasattr(BaseHandler, "elo_system"):
            BaseHandler.elo_system = None
    except ImportError:
        pass

    yield

    # Also clear after test
    try:
        from aragora.server.handlers.admin.cache import clear_cache

        clear_cache()
    except ImportError:
        pass


# ============================================================================
# Response Helpers
# ============================================================================


def parse_handler_response(result) -> Dict[str, Any]:
    """Parse JSON body from a HandlerResult.

    Args:
        result: HandlerResult with body attribute

    Returns:
        Parsed JSON dict, or empty dict if parsing fails

    Example:
        result = handler.handle("/api/v1/test", {}, mock_http, "GET")
        body = parse_handler_response(result)
        assert body["status"] == "ok"
    """
    if result is None:
        return {}
    try:
        body = result.body
        if isinstance(body, bytes):
            body = body.decode("utf-8")
        return json.loads(body) if body else {}
    except (json.JSONDecodeError, AttributeError):
        return {}


def assert_success_response(result, expected_keys: List[str] = None):
    """Assert that a handler result is a successful JSON response.

    Args:
        result: HandlerResult to check
        expected_keys: Optional list of keys that should be in response

    Raises:
        AssertionError: If response is not successful or missing keys
    """
    assert result is not None, "Result should not be None"
    assert result.status_code == 200, f"Expected 200, got {result.status_code}"
    assert result.content_type == "application/json"

    if expected_keys:
        body = parse_handler_response(result)
        for key in expected_keys:
            assert key in body, f"Expected key '{key}' in response"


def assert_error_response(result, expected_status: int, error_substring: str = None):
    """Assert that a handler result is an error response.

    Args:
        result: HandlerResult to check
        expected_status: Expected HTTP status code
        error_substring: Optional substring to look for in error message
    """
    assert result is not None, "Result should not be None"
    assert result.status_code == expected_status, (
        f"Expected {expected_status}, got {result.status_code}"
    )

    if error_substring:
        body = parse_handler_response(result)
        error_msg = body.get("error", "") or body.get("message", "")
        assert error_substring.lower() in error_msg.lower(), (
            f"Expected '{error_substring}' in error message: {error_msg}"
        )


# ============================================================================
# HTTP Handler Mocks
# ============================================================================


@dataclass
class MockHTTPRequest:
    """Mock HTTP request for handler testing."""

    method: str = "GET"
    path: str = "/"
    headers: Dict[str, str] = None
    body: bytes = b"{}"
    query_params: Dict[str, str] = None

    def __post_init__(self):
        if self.headers is None:
            self.headers = {"Content-Length": str(len(self.body))}
        else:
            self.headers.setdefault("Content-Length", str(len(self.body)))
        if self.query_params is None:
            self.query_params = {}


@pytest.fixture
def mock_http_handler():
    """Create a mock HTTP handler for handler testing.

    Returns:
        Factory function to create mock HTTP handlers.

    Example:
        def test_post_endpoint(handler, mock_http_handler):
            http = mock_http_handler(
                method="POST",
                body={"name": "test"},
                headers={"Authorization": "Bearer token123"}
            )
            result = handler.handle("/api/v1/resource", {}, http, "POST")
    """

    def _create_handler(
        method: str = "GET",
        body: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        client_address: tuple = ("127.0.0.1", 12345),
    ) -> MagicMock:
        mock = MagicMock()
        mock.command = method

        # Set up body reading
        if body is not None:
            body_bytes = json.dumps(body).encode()
        else:
            body_bytes = b"{}"

        mock.rfile = MagicMock()
        mock.rfile.read = MagicMock(return_value=body_bytes)

        # Set up headers
        mock.headers = headers or {}
        mock.headers.setdefault("Content-Length", str(len(body_bytes)))

        # Set up client address
        mock.client_address = client_address

        return mock

    return _create_handler


@pytest.fixture
def mock_http_get(mock_http_handler):
    """Convenience fixture for GET requests."""
    return mock_http_handler(method="GET")


@pytest.fixture
def mock_http_post(mock_http_handler):
    """Convenience fixture for POST requests with empty body."""
    return mock_http_handler(method="POST", body={})


# ============================================================================
# Server Context Fixtures
# ============================================================================


@pytest.fixture
def mock_debate_storage():
    """Create a mock DebateStorage for handler testing.

    Returns:
        Mock storage with common methods pre-configured.
    """
    storage = MagicMock()

    # List debates
    storage.list_debates.return_value = [
        {
            "id": "debate-001",
            "slug": "test-debate-one",
            "task": "Test task one",
            "created_at": "2026-01-15T10:00:00Z",
            "consensus_reached": False,
        },
        {
            "id": "debate-002",
            "slug": "test-debate-two",
            "task": "Test task two",
            "created_at": "2026-01-14T10:00:00Z",
            "consensus_reached": True,
        },
    ]

    # Get single debate
    storage.get_debate.return_value = {
        "id": "debate-001",
        "slug": "test-debate-one",
        "task": "Test task one",
        "messages": [
            {"agent": "claude", "content": "Initial proposal", "round": 1},
            {"agent": "gemini", "content": "Critique of proposal", "round": 1},
        ],
        "critiques": [],
        "votes": [],
        "consensus_reached": False,
        "rounds_used": 3,
        "created_at": "2026-01-15T10:00:00Z",
    }
    storage.get_debate_by_slug.return_value = storage.get_debate.return_value

    # Search
    storage.search.return_value = storage.list_debates.return_value

    # Public/private
    storage.is_public.return_value = True

    # Save operations
    storage.save_debate.return_value = "debate-new"
    storage.update_debate.return_value = True
    storage.delete_debate.return_value = True

    return storage


@pytest.fixture
def mock_user_store():
    """Create a mock UserStore for handler testing.

    Returns:
        Mock user store with common authentication methods.
    """
    store = MagicMock()

    # User retrieval
    store.get_user.return_value = {
        "user_id": "user-001",
        "email": "test@example.com",
        "username": "testuser",
        "created_at": "2026-01-01T00:00:00Z",
        "is_active": True,
    }
    store.get_user_by_email.return_value = store.get_user.return_value
    store.get_user_by_username.return_value = store.get_user.return_value

    # Authentication
    store.verify_password.return_value = True
    store.create_session.return_value = "session-token-123"
    store.validate_session.return_value = store.get_user.return_value

    # User creation
    store.create_user.return_value = "user-new"
    store.update_user.return_value = True
    store.delete_user.return_value = True

    return store


@pytest.fixture
def mock_elo_system():
    """Create a mock ELO system for handler testing.

    Returns:
        Mock ELO system with leaderboard and rating methods.
    """
    elo = MagicMock()

    # Mock rating object
    mock_rating = MagicMock()
    mock_rating.agent_name = "claude"
    mock_rating.elo = 1650
    mock_rating.wins = 10
    mock_rating.losses = 5
    mock_rating.draws = 3
    mock_rating.games_played = 18
    mock_rating.win_rate = 0.56

    # Methods
    elo.get_rating.return_value = mock_rating
    elo.get_leaderboard.return_value = [mock_rating]
    elo.get_cached_leaderboard.return_value = [
        {
            "agent_name": "claude",
            "elo": 1650,
            "wins": 10,
            "losses": 5,
            "draws": 3,
            "games_played": 18,
            "win_rate": 0.56,
        },
        {
            "agent_name": "gemini",
            "elo": 1580,
            "wins": 8,
            "losses": 7,
            "draws": 2,
            "games_played": 17,
            "win_rate": 0.47,
        },
    ]
    elo.get_recent_matches.return_value = []
    elo.get_head_to_head.return_value = {
        "matches": 5,
        "agent_a_wins": 3,
        "agent_b_wins": 2,
        "draws": 0,
    }
    elo.record_match.return_value = None

    return elo


@pytest.fixture
def mock_knowledge_store():
    """Create a mock KnowledgeStore for handler testing.

    Returns:
        Mock knowledge store with fact/mound operations.
    """
    store = MagicMock()

    # Facts
    store.list_facts.return_value = [
        {
            "fact_id": "fact-001",
            "content": "Test fact one",
            "source": "debate-001",
            "confidence": 0.95,
            "created_at": "2026-01-15T10:00:00Z",
        },
    ]
    store.get_fact.return_value = store.list_facts.return_value[0]
    store.add_fact.return_value = "fact-new"
    store.update_fact.return_value = True
    store.delete_fact.return_value = True

    # Search
    store.search_facts.return_value = store.list_facts.return_value

    return store


@pytest.fixture
def mock_workflow_store():
    """Create a mock WorkflowStore for handler testing."""
    store = MagicMock()

    store.list_workflows.return_value = [
        {
            "workflow_id": "wf-001",
            "name": "Test Workflow",
            "status": "active",
            "created_at": "2026-01-15T10:00:00Z",
        },
    ]
    store.get_workflow.return_value = store.list_workflows.return_value[0]
    store.create_workflow.return_value = "wf-new"
    store.update_workflow.return_value = True
    store.delete_workflow.return_value = True

    return store


@pytest.fixture
def mock_workspace_store():
    """Create a mock WorkspaceStore for handler testing."""
    store = MagicMock()

    store.list_workspaces.return_value = [
        {
            "workspace_id": "ws-001",
            "name": "Test Workspace",
            "owner_id": "user-001",
            "created_at": "2026-01-15T10:00:00Z",
        },
    ]
    store.get_workspace.return_value = store.list_workspaces.return_value[0]
    store.create_workspace.return_value = "ws-new"
    store.update_workspace.return_value = True
    store.delete_workspace.return_value = True

    return store


@pytest.fixture
def mock_audit_store():
    """Create a mock AuditStore for handler testing."""
    store = MagicMock()

    store.list_events.return_value = [
        {
            "event_id": "evt-001",
            "event_type": "debate.created",
            "actor_id": "user-001",
            "resource_id": "debate-001",
            "timestamp": "2026-01-15T10:00:00Z",
        },
    ]
    store.get_event.return_value = store.list_events.return_value[0]
    store.log_event.return_value = "evt-new"

    return store


@pytest.fixture
def mock_server_context(
    mock_debate_storage,
    mock_user_store,
    mock_elo_system,
    mock_knowledge_store,
    mock_workflow_store,
    mock_workspace_store,
    mock_audit_store,
):
    """Create a complete server context with all mock dependencies.

    This provides the standard server_context dict used by handlers.

    Example:
        @pytest.fixture
        def handler(mock_server_context):
            return MyHandler(server_context=mock_server_context)
    """
    return {
        "storage": mock_debate_storage,
        "user_store": mock_user_store,
        "elo_system": mock_elo_system,
        "knowledge_store": mock_knowledge_store,
        "workflow_store": mock_workflow_store,
        "workspace_store": mock_workspace_store,
        "audit_store": mock_audit_store,
        "debate_embeddings": None,
        "critique_store": None,
        "nomic_dir": None,
    }


# ============================================================================
# RBAC Authorization Context Fixtures
# ============================================================================


@dataclass
class MockAuthorizationContext:
    """Mock authorization context for testing RBAC-protected handlers.

    This mirrors the real AuthorizationContext from aragora.rbac.models.
    """

    user_id: str = "test-user-001"
    user_email: str = "test@example.com"
    org_id: str = "test-org-001"
    workspace_id: str = "test-ws-001"
    roles: List[str] = None
    permissions: List[str] = None
    api_key_scope: Optional[str] = None
    ip_address: str = "127.0.0.1"
    user_agent: str = "test-agent"
    request_id: str = "req-test-001"
    timestamp: str = None

    def __post_init__(self):
        if self.roles is None:
            self.roles = ["admin"]
        if self.permissions is None:
            # Default comprehensive permissions for testing
            self.permissions = [
                "debates.read",
                "debates.write",
                "debates.create",
                "debates.delete",
                "agents.read",
                "agents.write",
                "memory.read",
                "memory.write",
                "knowledge.read",
                "knowledge.write",
                "workflows.read",
                "workflows.write",
                "admin.read",
                "admin.write",
                "billing.read",
                "billing.write",
                "costs.read",
                "costs.write",
                "audit.read",
            ]
        if self.timestamp is None:
            from datetime import datetime

            self.timestamp = datetime.now().isoformat()

    def has_permission(self, permission: str) -> bool:
        """Check if context has a specific permission."""
        return permission in self.permissions

    def has_role(self, role: str) -> bool:
        """Check if context has a specific role."""
        return role in self.roles


@pytest.fixture
def mock_auth_context():
    """Create a mock authorization context for RBAC-protected handlers.

    Returns:
        Factory function to create customized auth contexts.

    Example:
        def test_with_admin(mock_auth_context):
            ctx = mock_auth_context(roles=["admin"])
            assert ctx.has_role("admin")

        def test_with_limited_permissions(mock_auth_context):
            ctx = mock_auth_context(permissions=["debates.read"])
            assert not ctx.has_permission("debates.write")
    """

    def _create_context(
        user_id: str = "test-user-001",
        org_id: str = "test-org-001",
        workspace_id: str = "test-ws-001",
        roles: List[str] = None,
        permissions: List[str] = None,
        **kwargs,
    ) -> MockAuthorizationContext:
        return MockAuthorizationContext(
            user_id=user_id,
            org_id=org_id,
            workspace_id=workspace_id,
            roles=roles,
            permissions=permissions,
            **kwargs,
        )

    return _create_context


@pytest.fixture
def admin_auth_context(mock_auth_context):
    """Pre-configured admin authorization context."""
    return mock_auth_context(roles=["admin", "owner"])


@pytest.fixture
def viewer_auth_context(mock_auth_context):
    """Pre-configured viewer authorization context (read-only)."""
    return mock_auth_context(
        roles=["viewer"],
        permissions=[
            "debates.read",
            "agents.read",
            "memory.read",
            "knowledge.read",
            "workflows.read",
        ],
    )


@pytest.fixture
def editor_auth_context(mock_auth_context):
    """Pre-configured editor authorization context (read/write, no admin)."""
    return mock_auth_context(
        roles=["editor"],
        permissions=[
            "debates.read",
            "debates.write",
            "debates.create",
            "agents.read",
            "agents.write",
            "memory.read",
            "memory.write",
            "knowledge.read",
            "knowledge.write",
            "workflows.read",
            "workflows.write",
        ],
    )


@pytest.fixture
def authenticated_handler(mock_auth_context):
    """Patch a handler instance to bypass RBAC authentication.

    This fixture patches the get_auth_context and check_permission methods
    on handler instances so they don't require actual authentication.

    Returns:
        Context manager factory that patches handler auth methods.

    Example:
        def test_protected_endpoint(handler, authenticated_handler, mock_http_get):
            with authenticated_handler(handler):
                result = handler.handle("/api/v1/protected", {}, mock_http_get, "GET")
                assert result.status_code == 200

        def test_with_custom_permissions(handler, authenticated_handler, mock_http_get):
            with authenticated_handler(handler, permissions=["debates.read"]):
                result = handler.handle("/api/v1/debates", {}, mock_http_get, "GET")
    """
    from contextlib import contextmanager
    from unittest.mock import patch

    @contextmanager
    def _patch_handler(
        handler_instance,
        user_id: str = "test-user-001",
        org_id: str = "test-org-001",
        roles: List[str] = None,
        permissions: List[str] = None,
    ):
        ctx = mock_auth_context(
            user_id=user_id,
            org_id=org_id,
            roles=roles,
            permissions=permissions,
        )

        # Patch get_auth_context to return our mock context
        with patch.object(handler_instance, "get_auth_context", return_value=ctx) as mock_get_auth:
            # Patch check_permission to always return True
            with patch.object(
                handler_instance, "check_permission", return_value=True
            ) as mock_check:
                yield ctx, mock_get_auth, mock_check

    return _patch_handler


@pytest.fixture
def bypass_rbac():
    """Global patch to bypass all RBAC checks.

    Use this when you want to test handler logic without any auth.
    Patches the require_permission decorator to be a no-op.

    Example:
        def test_handler_logic(handler, bypass_rbac, mock_http_get):
            with bypass_rbac:
                result = handler.handle("/api/v1/resource", {}, mock_http_get, "GET")
    """
    from unittest.mock import patch

    def passthrough_decorator(*args, **kwargs):
        """Decorator that does nothing."""

        def decorator(func):
            return func

        return decorator

    return patch("aragora.rbac.decorators.require_permission", passthrough_decorator)


# ============================================================================
# Legacy Auth Mocking Fixtures
# ============================================================================


@pytest.fixture
def mock_auth_disabled():
    """Patch auth_config to disable authentication.

    Use as context manager or with autouse in test class.
    """
    from unittest.mock import patch

    mock_config = MagicMock()
    mock_config.enabled = False
    mock_config.api_token = None

    return patch("aragora.server.auth.auth_config", mock_config)


@pytest.fixture
def mock_auth_enabled():
    """Patch auth_config with authentication enabled.

    Returns a patch context manager and valid token.
    """
    from unittest.mock import patch

    mock_config = MagicMock()
    mock_config.enabled = True
    mock_config.api_token = "test_api_token_12345"
    mock_config.validate_token = MagicMock(return_value=True)

    return patch("aragora.server.auth.auth_config", mock_config), "test_api_token_12345"


@pytest.fixture
def authenticated_http_handler(mock_http_handler):
    """Create a mock HTTP handler with valid auth headers.

    Returns:
        Factory that creates authenticated HTTP handlers.
    """

    def _create_handler(
        method: str = "GET",
        body: Optional[Dict[str, Any]] = None,
        token: str = "test_api_token_12345",
    ) -> MagicMock:
        return mock_http_handler(
            method=method,
            body=body,
            headers={"Authorization": f"Bearer {token}"},
        )

    return _create_handler


# ============================================================================
# Rate Limiting Mocks
# ============================================================================


@pytest.fixture
def disable_rate_limiting():
    """Disable rate limiting for tests.

    Use as context manager to patch rate_limit decorator.
    """
    from unittest.mock import patch

    # Create a pass-through decorator
    def no_rate_limit(**kwargs):
        def decorator(func):
            return func

        return decorator

    return patch("aragora.server.handlers.auth.handler.rate_limit", no_rate_limit)


# ============================================================================
# Async Handler Support
# ============================================================================


@pytest.fixture
def async_mock_http_handler():
    """Create an async-compatible mock HTTP handler.

    For handlers that use async methods internally.
    """

    def _create_handler(
        method: str = "GET",
        body: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> MagicMock:
        mock = MagicMock()
        mock.command = method

        if body is not None:
            body_bytes = json.dumps(body).encode()
        else:
            body_bytes = b"{}"

        mock.rfile = MagicMock()
        mock.rfile.read = MagicMock(return_value=body_bytes)

        mock.headers = headers or {}
        mock.headers.setdefault("Content-Length", str(len(body_bytes)))

        # Async-compatible send methods
        mock.send_response = MagicMock()
        mock.send_header = MagicMock()
        mock.end_headers = MagicMock()
        mock.wfile = MagicMock()
        mock.wfile.write = MagicMock()

        return mock

    return _create_handler


# ============================================================================
# Handler Test Helpers
# ============================================================================


class HandlerTestCase:
    """Base class for handler test cases.

    Provides common assertions and utilities.

    Example:
        class TestMyHandler(HandlerTestCase):
            @pytest.fixture
            def handler(self, mock_server_context):
                from aragora.server.handlers.my import MyHandler
                return MyHandler(server_context=mock_server_context)

            def test_can_handle(self, handler):
                self.assert_can_handle(handler, "/api/v1/my/resource")

            def test_get_resource(self, handler, mock_http_get):
                result = handler.handle("/api/v1/my/resource", {}, mock_http_get, "GET")
                self.assert_success_with_keys(result, ["id", "name"])
    """

    def assert_can_handle(self, handler, path: str, expected: bool = True):
        """Assert handler can/cannot handle a path."""
        result = handler.can_handle(path)
        assert result is expected, f"Expected can_handle('{path}') to be {expected}"

    def assert_routes_include(self, handler, *paths: str):
        """Assert handler ROUTES includes all given paths."""
        for path in paths:
            assert path in handler.ROUTES, f"Expected '{path}' in handler ROUTES"

    def assert_success_with_keys(self, result, keys: List[str]):
        """Assert successful response with expected keys."""
        assert_success_response(result, expected_keys=keys)

    def assert_error(self, result, status: int, message: str = None):
        """Assert error response with status and optional message."""
        assert_error_response(result, expected_status=status, error_substring=message)

    def get_response_body(self, result) -> Dict[str, Any]:
        """Get parsed response body."""
        return parse_handler_response(result)
