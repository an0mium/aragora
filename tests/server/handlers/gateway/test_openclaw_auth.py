"""
Tests for OpenClaw handler auth context security.

Verifies that the _build_auth_context function does not accept
permissions or roles from request data (defense-in-depth).

Authorization is enforced at the endpoint level via @require_permission
decorators, not by trusting client-supplied permission claims.
"""

import pytest

from aragora.server.handlers.gateway.openclaw import (
    _build_auth_context,
    VALID_ACTOR_TYPES,
)
from aragora.gateway.openclaw.protocol import AuthorizationContext


class TestBuildAuthContextSecurity:
    """Tests for _build_auth_context security hardening."""

    def test_permissions_from_request_are_ignored(self):
        """Permissions supplied in request data should NOT be used."""
        data = {
            "permissions": ["admin:all", "gateway:execute", "system:destroy"],
            "actor_type": "user",
            "session_id": "sess-123",
        }

        ctx = _build_auth_context(user_id="user-1", data=data)

        # Permissions should be empty regardless of what was passed
        assert ctx.permissions == set()
        assert "admin:all" not in ctx.permissions
        assert "gateway:execute" not in ctx.permissions
        assert "system:destroy" not in ctx.permissions

    def test_roles_from_request_are_ignored(self):
        """Roles supplied in request data should NOT be used."""
        data = {
            "roles": ["admin", "superuser", "root"],
            "actor_type": "user",
            "session_id": "sess-123",
        }

        ctx = _build_auth_context(user_id="user-1", data=data)

        # Roles should be empty regardless of what was passed
        assert ctx.roles == []
        assert "admin" not in ctx.roles
        assert "superuser" not in ctx.roles
        assert "root" not in ctx.roles

    def test_both_permissions_and_roles_ignored(self):
        """Both permissions and roles from request should be ignored."""
        data = {
            "permissions": ["admin:all", "gateway:execute"],
            "roles": ["admin", "superuser"],
            "actor_type": "user",
            "session_id": "sess-456",
        }

        ctx = _build_auth_context(user_id="user-1", data=data)

        assert ctx.permissions == set()
        assert ctx.roles == []

    def test_actor_id_is_passed_through(self):
        """The user_id parameter should be used as actor_id."""
        data = {"actor_type": "user"}

        ctx = _build_auth_context(user_id="authenticated-user-123", data=data)

        assert ctx.actor_id == "authenticated-user-123"

    def test_session_id_is_passed_through(self):
        """Session ID from request should be used."""
        data = {"session_id": "session-abc-123"}

        ctx = _build_auth_context(user_id="user-1", data=data)

        assert ctx.session_id == "session-abc-123"


class TestActorTypeValidation:
    """Tests for actor_type validation in _build_auth_context."""

    @pytest.mark.parametrize("actor_type", ["user", "service", "agent"])
    def test_valid_actor_types_are_accepted(self, actor_type: str):
        """Valid actor types should be passed through."""
        data = {"actor_type": actor_type}

        ctx = _build_auth_context(user_id="user-1", data=data)

        assert ctx.actor_type == actor_type

    def test_valid_actor_types_constant(self):
        """VALID_ACTOR_TYPES should contain expected values."""
        assert VALID_ACTOR_TYPES == {"user", "service", "agent"}

    @pytest.mark.parametrize(
        "invalid_type",
        [
            "admin",
            "superuser",
            "root",
            "system",
            "god",
            "ADMIN",
            "User",
            "",
            "attacker",
        ],
    )
    def test_invalid_actor_types_default_to_user(self, invalid_type: str):
        """Invalid actor types should default to 'user'."""
        data = {"actor_type": invalid_type}

        ctx = _build_auth_context(user_id="user-1", data=data)

        assert ctx.actor_type == "user"

    def test_missing_actor_type_defaults_to_user(self):
        """Missing actor_type should default to 'user'."""
        data = {}

        ctx = _build_auth_context(user_id="user-1", data=data)

        assert ctx.actor_type == "user"


class TestBuildAuthContextDefaults:
    """Tests for default behavior of _build_auth_context."""

    def test_minimal_data_produces_valid_context(self):
        """Empty data dict should produce valid context with defaults."""
        ctx = _build_auth_context(user_id="user-1", data={})

        assert isinstance(ctx, AuthorizationContext)
        assert ctx.actor_id == "user-1"
        assert ctx.actor_type == "user"
        assert ctx.permissions == set()
        assert ctx.roles == []
        assert ctx.session_id is None

    def test_context_is_correct_type(self):
        """Result should be an AuthorizationContext instance."""
        ctx = _build_auth_context(user_id="user-1", data={})

        assert isinstance(ctx, AuthorizationContext)

    def test_all_valid_fields_together(self):
        """All valid fields should be correctly set."""
        data = {
            "actor_type": "service",
            "session_id": "sess-789",
            # These should be ignored:
            "permissions": ["should:be:ignored"],
            "roles": ["ignored-role"],
        }

        ctx = _build_auth_context(user_id="service-account-1", data=data)

        assert ctx.actor_id == "service-account-1"
        assert ctx.actor_type == "service"
        assert ctx.session_id == "sess-789"
        assert ctx.permissions == set()
        assert ctx.roles == []
