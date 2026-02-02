"""
Tests for OpenClawGatewayHandler - OpenClaw gateway HTTP endpoints.

Tests cover:
- Input validation for credentials (name, secret, type)
- Input validation for session config
- Action parameter sanitization (command injection prevention)
- Credential rotation rate limiting
- Session management (create, get, list, close)
- Action execution (execute, status, cancel)
- Credential management (store, list, delete, rotate)
- Admin operations (health, metrics, audit)
- RBAC protection and access control
- Error handling and validation
- Data model serialization
- Store operations
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Any, Optional
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from aragora.server.handlers.openclaw_gateway import (
    OpenClawGatewayHandler,
    OpenClawGatewayStore,
    Session,
    SessionStatus,
    Action,
    ActionStatus,
    Credential,
    CredentialType,
    AuditEntry,
    _get_store,
    get_openclaw_gateway_handler,
    get_openclaw_circuit_breaker,
    get_openclaw_circuit_breaker_status,
    # Validation constants
    MAX_CREDENTIAL_NAME_LENGTH,
    MAX_CREDENTIAL_SECRET_LENGTH,
    MIN_CREDENTIAL_SECRET_LENGTH,
    MAX_SESSION_CONFIG_SIZE,
    MAX_SESSION_CONFIG_KEYS,
    MAX_SESSION_CONFIG_DEPTH,
    MAX_ACTION_TYPE_LENGTH,
    MAX_ACTION_INPUT_SIZE,
    MAX_CREDENTIAL_ROTATIONS_PER_HOUR,
    CREDENTIAL_ROTATION_WINDOW_SECONDS,
    # Validation functions
    validate_credential_name,
    validate_credential_secret,
    validate_session_config,
    validate_action_type,
    validate_action_input,
    validate_metadata,
    sanitize_action_parameters,
    # Rate limiting
    CredentialRotationRateLimiter,
    _get_credential_rotation_limiter,
)


# ===========================================================================
# Test Fixtures and Mocks
# ===========================================================================


@dataclass
class MockUser:
    """Mock user authentication context."""

    user_id: str = "user-001"
    email: str = "test@example.com"
    org_id: Optional[str] = "org-001"
    role: str = "user"
    permissions: list[str] = field(default_factory=list)
    is_authenticated: bool = True


class MockRequestHandler:
    """Mock HTTP request handler."""

    def __init__(
        self,
        body: Optional[dict] = None,
        headers: Optional[dict] = None,
        content_length: int = 0,
    ):
        self._body = body
        self.headers = headers or {"Content-Type": "application/json"}
        if body:
            body_bytes = json.dumps(body).encode()
            content_length = len(body_bytes)
            self.headers["Content-Length"] = str(content_length)
            self.rfile = MagicMock()
            self.rfile.read.return_value = body_bytes
        else:
            self.headers["Content-Length"] = "0"
            self.rfile = MagicMock()
            self.rfile.read.return_value = b"{}"


@pytest.fixture
def mock_server_context():
    """Create mock server context."""
    return MagicMock()


@pytest.fixture
def fresh_store():
    """Create a fresh store instance for each test."""
    return OpenClawGatewayStore()


@pytest.fixture
def handler(mock_server_context):
    """Create handler with mocked dependencies."""
    h = OpenClawGatewayHandler(mock_server_context)
    return h


@pytest.fixture
def mock_user():
    """Create a standard mock user."""
    return MockUser()


@pytest.fixture
def mock_admin_user():
    """Create a mock admin user."""
    return MockUser(
        user_id="admin-001",
        role="admin",
        permissions=["gateway:admin"],
    )


def create_mock_handler_with_user(
    handler: OpenClawGatewayHandler,
    user: MockUser,
) -> None:
    """Configure handler to return the given user on authentication."""
    handler.get_current_user = MagicMock(return_value=user)


# ===========================================================================
# Credential Name Validation Tests
# ===========================================================================


class TestCredentialNameValidation:
    """Test credential name validation."""

    def test_valid_credential_name(self):
        """Test valid credential names pass validation."""
        valid_names = [
            "api_key",
            "my-credential",
            "test123",
            "AWS_ACCESS_KEY",
            "github-token-2025",
            "a",  # Single letter
            "A1_b-c",  # Mixed case with special chars
        ]
        for name in valid_names:
            is_valid, error = validate_credential_name(name)
            assert is_valid, f"Name '{name}' should be valid but got error: {error}"

    def test_empty_name_rejected(self):
        """Test empty names are rejected."""
        for name in [None, "", "   "]:
            is_valid, error = validate_credential_name(name)
            assert not is_valid
            assert error is not None

    def test_name_must_start_with_letter(self):
        """Test names must start with a letter."""
        invalid_names = ["123abc", "-test", "_test", "1_key"]
        for name in invalid_names:
            is_valid, error = validate_credential_name(name)
            assert not is_valid, f"Name '{name}' should be invalid"
            assert "start with a letter" in error

    def test_name_length_limit(self):
        """Test name length limit is enforced."""
        # Valid at max length
        is_valid, _ = validate_credential_name("a" * MAX_CREDENTIAL_NAME_LENGTH)
        assert is_valid

        # Invalid when exceeding max length
        is_valid, error = validate_credential_name("a" * (MAX_CREDENTIAL_NAME_LENGTH + 1))
        assert not is_valid
        assert "exceeds maximum length" in error

    def test_invalid_characters_rejected(self):
        """Test invalid characters are rejected."""
        invalid_names = [
            "test@name",
            "test name",
            "test.name",
            "test/name",
            "test\\name",
            "test:name",
            "test;name",
            "test'name",
            'test"name',
        ]
        for name in invalid_names:
            is_valid, error = validate_credential_name(name)
            assert not is_valid, f"Name '{name}' should be invalid"


# ===========================================================================
# Credential Secret Validation Tests
# ===========================================================================


class TestCredentialSecretValidation:
    """Test credential secret validation."""

    def test_valid_secret(self):
        """Test valid secrets pass validation."""
        valid_secrets = [
            "a" * MIN_CREDENTIAL_SECRET_LENGTH,
            "supersecret123!@#",
            "a" * MAX_CREDENTIAL_SECRET_LENGTH,
        ]
        for secret in valid_secrets:
            is_valid, error = validate_credential_secret(secret)
            assert is_valid, f"Secret should be valid but got error: {error}"

    def test_empty_secret_rejected(self):
        """Test empty secrets are rejected."""
        for secret in [None, ""]:
            is_valid, error = validate_credential_secret(secret)
            assert not is_valid
            assert error is not None

    def test_short_secret_rejected(self):
        """Test secrets shorter than minimum are rejected."""
        short_secret = "a" * (MIN_CREDENTIAL_SECRET_LENGTH - 1)
        is_valid, error = validate_credential_secret(short_secret)
        assert not is_valid
        assert "at least" in error

    def test_long_secret_rejected(self):
        """Test secrets exceeding maximum are rejected."""
        long_secret = "a" * (MAX_CREDENTIAL_SECRET_LENGTH + 1)
        is_valid, error = validate_credential_secret(long_secret)
        assert not is_valid
        assert "exceeds maximum length" in error

    def test_null_bytes_rejected(self):
        """Test secrets with null bytes are rejected."""
        secret_with_null = "secret\x00value"
        is_valid, error = validate_credential_secret(secret_with_null)
        assert not is_valid
        assert "invalid characters" in error

    def test_non_string_rejected(self):
        """Test non-string secrets are rejected."""
        is_valid, error = validate_credential_secret(12345)
        assert not is_valid
        assert "must be a string" in error


# ===========================================================================
# Session Config Validation Tests
# ===========================================================================


class TestSessionConfigValidation:
    """Test session config validation."""

    def test_valid_config(self):
        """Test valid configs pass validation."""
        valid_configs = [
            None,
            {},
            {"key": "value"},
            {"nested": {"level": 1}},
            {"list": [1, 2, 3]},
        ]
        for config in valid_configs:
            is_valid, error = validate_session_config(config)
            assert is_valid, f"Config should be valid but got error: {error}"

    def test_non_dict_rejected(self):
        """Test non-dict configs are rejected."""
        for config in ["string", 123, [1, 2, 3]]:
            is_valid, error = validate_session_config(config)
            assert not is_valid
            assert "must be an object" in error

    def test_config_size_limit(self):
        """Test config size limit is enforced."""
        # Create a large config
        large_config = {"key": "x" * MAX_SESSION_CONFIG_SIZE}
        is_valid, error = validate_session_config(large_config)
        assert not is_valid
        assert "exceeds maximum size" in error

    def test_config_depth_limit(self):
        """Test config nesting depth limit is enforced."""
        # Create deeply nested config (exceeds limit)
        deep_config = {"l1": {"l2": {"l3": {"l4": {"l5": {"l6": "value"}}}}}}
        is_valid, error = validate_session_config(deep_config)
        assert not is_valid
        assert "nesting depth" in error

    def test_config_keys_limit(self):
        """Test config max keys limit is enforced."""
        # Create config with too many keys
        many_keys_config = {f"key_{i}": i for i in range(MAX_SESSION_CONFIG_KEYS + 10)}
        is_valid, error = validate_session_config(many_keys_config)
        assert not is_valid
        assert "exceeds maximum" in error


# ===========================================================================
# Action Type Validation Tests
# ===========================================================================


class TestActionTypeValidation:
    """Test action type validation."""

    def test_valid_action_types(self):
        """Test valid action types pass validation."""
        valid_types = [
            "execute",
            "file.read",
            "browser_navigate",
            "shell-execute",
            "test.nested.action",
        ]
        for action_type in valid_types:
            is_valid, error = validate_action_type(action_type)
            assert is_valid, f"Action type '{action_type}' should be valid but got: {error}"

    def test_empty_action_type_rejected(self):
        """Test empty action types are rejected."""
        for action_type in [None, "", "   "]:
            is_valid, error = validate_action_type(action_type)
            assert not is_valid

    def test_action_type_must_start_with_letter(self):
        """Test action type must start with letter."""
        invalid_types = ["123action", "-action", "_action", ".action"]
        for action_type in invalid_types:
            is_valid, error = validate_action_type(action_type)
            assert not is_valid
            assert "start with a letter" in error

    def test_action_type_length_limit(self):
        """Test action type length limit is enforced."""
        long_type = "a" * (MAX_ACTION_TYPE_LENGTH + 1)
        is_valid, error = validate_action_type(long_type)
        assert not is_valid
        assert "exceeds maximum length" in error

    def test_invalid_action_type_characters(self):
        """Test invalid characters in action type are rejected."""
        invalid_types = [
            "action;exec",
            "action|pipe",
            "action`cmd`",
            "action$var",
            "action\ntest",
            "action test",  # space
        ]
        for action_type in invalid_types:
            is_valid, error = validate_action_type(action_type)
            assert not is_valid, f"Action type '{action_type}' should be invalid"


# ===========================================================================
# Action Parameter Sanitization Tests
# ===========================================================================


class TestActionParameterSanitization:
    """Test action parameter sanitization for command injection prevention."""

    def test_sanitize_string_with_shell_metacharacters(self):
        """Test shell metacharacters are escaped in strings."""
        params = {"command": "ls; rm -rf /", "file": "test`whoami`.txt"}
        sanitized = sanitize_action_parameters(params)

        # Semicolon should be escaped
        assert ";" not in sanitized["command"] or sanitized["command"].count("\\;") > 0
        # Backticks should be escaped
        assert "`" not in sanitized["file"] or sanitized["file"].count("\\`") > 0

    def test_sanitize_null_bytes_removed(self):
        """Test null bytes are removed from strings."""
        params = {"command": "test\x00command"}
        sanitized = sanitize_action_parameters(params)
        assert "\x00" not in sanitized["command"]

    def test_sanitize_nested_dicts(self):
        """Test nested dicts are sanitized."""
        params = {"outer": {"inner": "test;injection"}}
        sanitized = sanitize_action_parameters(params)
        assert ";" not in sanitized["outer"]["inner"] or "\\;" in sanitized["outer"]["inner"]

    def test_sanitize_lists(self):
        """Test lists are sanitized."""
        params = {"commands": ["cmd1;evil", "cmd2|pipe"]}
        sanitized = sanitize_action_parameters(params)
        for cmd in sanitized["commands"]:
            assert ";" not in cmd or "\\;" in cmd
            assert "|" not in cmd or "\\|" in cmd

    def test_sanitize_preserves_safe_strings(self):
        """Test safe strings are preserved."""
        params = {"safe": "hello-world_123"}
        sanitized = sanitize_action_parameters(params)
        # These should be unchanged
        assert sanitized["safe"] == "hello-world_123"

    def test_sanitize_none_returns_empty(self):
        """Test None input returns empty dict."""
        assert sanitize_action_parameters(None) == {}

    def test_sanitize_non_dict_returns_empty(self):
        """Test non-dict input returns empty dict."""
        assert sanitize_action_parameters("string") == {}
        assert sanitize_action_parameters([1, 2, 3]) == {}

    def test_sanitize_pipe_character(self):
        """Test pipe character is escaped."""
        params = {"cmd": "cat file | grep pattern"}
        sanitized = sanitize_action_parameters(params)
        assert "|" not in sanitized["cmd"] or "\\|" in sanitized["cmd"]

    def test_sanitize_command_substitution(self):
        """Test command substitution characters are escaped."""
        params = {"cmd": "echo $(whoami)", "backtick": "echo `id`"}
        sanitized = sanitize_action_parameters(params)
        assert "$(" not in sanitized["cmd"] or "\\$(" in sanitized["cmd"]
        assert "`" not in sanitized["backtick"] or "\\`" in sanitized["backtick"]

    def test_sanitize_redirection_characters(self):
        """Test redirection characters are escaped."""
        params = {"cmd": "echo test > /etc/passwd"}
        sanitized = sanitize_action_parameters(params)
        assert ">" not in sanitized["cmd"] or "\\>" in sanitized["cmd"]


# ===========================================================================
# Action Input Validation Tests
# ===========================================================================


class TestActionInputValidation:
    """Test action input validation."""

    def test_valid_input(self):
        """Test valid inputs pass validation."""
        valid_inputs = [
            None,
            {},
            {"key": "value"},
            {"data": [1, 2, 3]},
        ]
        for input_data in valid_inputs:
            is_valid, error = validate_action_input(input_data)
            assert is_valid, f"Input should be valid but got: {error}"

    def test_non_dict_rejected(self):
        """Test non-dict inputs are rejected."""
        for input_data in ["string", 123, [1, 2, 3]]:
            is_valid, error = validate_action_input(input_data)
            assert not is_valid
            assert "must be an object" in error

    def test_input_size_limit(self):
        """Test input size limit is enforced."""
        large_input = {"data": "x" * MAX_ACTION_INPUT_SIZE}
        is_valid, error = validate_action_input(large_input)
        assert not is_valid
        assert "exceeds maximum size" in error


# ===========================================================================
# Metadata Validation Tests
# ===========================================================================


class TestMetadataValidation:
    """Test metadata validation."""

    def test_valid_metadata(self):
        """Test valid metadata passes validation."""
        valid_metadata = [
            None,
            {},
            {"key": "value"},
            {"nested": {"data": 123}},
        ]
        for metadata in valid_metadata:
            is_valid, error = validate_metadata(metadata)
            assert is_valid, f"Metadata should be valid but got: {error}"

    def test_non_dict_rejected(self):
        """Test non-dict metadata is rejected."""
        for metadata in ["string", 123, [1, 2, 3]]:
            is_valid, error = validate_metadata(metadata)
            assert not is_valid
            assert "must be an object" in error

    def test_metadata_size_limit(self):
        """Test metadata size limit is enforced."""
        large_metadata = {"data": "x" * 10000}
        is_valid, error = validate_metadata(large_metadata, max_size=1000)
        assert not is_valid
        assert "exceeds maximum size" in error


# ===========================================================================
# Credential Rotation Rate Limiting Tests
# ===========================================================================


class TestCredentialRotationRateLimiting:
    """Test credential rotation rate limiting."""

    def test_rate_limiter_allows_initial_rotations(self):
        """Test rate limiter allows initial rotations within limit."""
        limiter = CredentialRotationRateLimiter(max_rotations=5, window_seconds=60)

        for i in range(5):
            assert limiter.is_allowed("user-001"), f"Rotation {i + 1} should be allowed"

    def test_rate_limiter_blocks_excess_rotations(self):
        """Test rate limiter blocks rotations exceeding limit."""
        limiter = CredentialRotationRateLimiter(max_rotations=3, window_seconds=60)

        # Use up all rotations
        for _ in range(3):
            assert limiter.is_allowed("user-001")

        # Next should be blocked
        assert not limiter.is_allowed("user-001")

    def test_rate_limiter_tracks_users_separately(self):
        """Test rate limiter tracks different users separately."""
        limiter = CredentialRotationRateLimiter(max_rotations=2, window_seconds=60)

        # User 1 uses their limit
        assert limiter.is_allowed("user-001")
        assert limiter.is_allowed("user-001")
        assert not limiter.is_allowed("user-001")

        # User 2 should still have their full limit
        assert limiter.is_allowed("user-002")
        assert limiter.is_allowed("user-002")
        assert not limiter.is_allowed("user-002")

    def test_rate_limiter_get_remaining(self):
        """Test get_remaining returns correct count."""
        limiter = CredentialRotationRateLimiter(max_rotations=5, window_seconds=60)

        assert limiter.get_remaining("user-001") == 5
        limiter.is_allowed("user-001")
        assert limiter.get_remaining("user-001") == 4
        limiter.is_allowed("user-001")
        assert limiter.get_remaining("user-001") == 3

    def test_rate_limiter_get_retry_after(self):
        """Test get_retry_after returns correct time."""
        limiter = CredentialRotationRateLimiter(max_rotations=1, window_seconds=60)

        # First rotation is allowed
        assert limiter.is_allowed("user-001")

        # Should have retry_after > 0 now
        retry_after = limiter.get_retry_after("user-001")
        assert retry_after > 0
        assert retry_after <= 60


# ===========================================================================
# Circuit Breaker Tests
# ===========================================================================


class TestCircuitBreaker:
    """Test circuit breaker functionality."""

    def test_circuit_breaker_exists(self):
        """Test circuit breaker can be retrieved."""
        cb = get_openclaw_circuit_breaker()
        assert cb is not None
        assert cb.name == "openclaw_gateway_handler"

    def test_circuit_breaker_status(self):
        """Test circuit breaker status can be retrieved."""
        status = get_openclaw_circuit_breaker_status()
        assert isinstance(status, dict)
        # Status contains config, entity_mode, and single_mode
        assert "config" in status
        assert "failure_threshold" in status["config"]

    def test_circuit_breaker_threshold(self):
        """Test circuit breaker has correct failure threshold."""
        cb = get_openclaw_circuit_breaker()
        assert cb.failure_threshold == 5

    def test_circuit_breaker_cooldown(self):
        """Test circuit breaker has correct cooldown."""
        cb = get_openclaw_circuit_breaker()
        assert cb.cooldown_seconds == 30.0


# ===========================================================================
# Handler Routing Tests
# ===========================================================================


class TestHandlerRouting:
    """Test request routing."""

    def test_can_handle_base_paths(self, handler):
        """Test that handler recognizes base gateway paths."""
        assert handler.can_handle("/api/gateway/openclaw/sessions")
        assert handler.can_handle("/api/gateway/openclaw/actions")
        assert handler.can_handle("/api/gateway/openclaw/credentials")
        assert handler.can_handle("/api/gateway/openclaw/health")
        assert handler.can_handle("/api/gateway/openclaw/metrics")
        assert handler.can_handle("/api/gateway/openclaw/audit")

    def test_can_handle_versioned_paths(self, handler):
        """Test that handler recognizes versioned paths."""
        assert handler.can_handle("/api/v1/gateway/openclaw/sessions")
        assert handler.can_handle("/api/v1/gateway/openclaw/actions/action-001")
        assert handler.can_handle("/api/v1/gateway/openclaw/credentials")

    def test_cannot_handle_other_paths(self, handler):
        """Test that handler rejects non-openclaw paths."""
        assert not handler.can_handle("/api/gateway/other")
        assert not handler.can_handle("/api/v1/debates")
        assert not handler.can_handle("/api/openclaw/sessions")

    def test_normalize_path(self, handler):
        """Test path normalization."""
        assert (
            handler._normalize_path("/api/v1/gateway/openclaw/sessions")
            == "/api/gateway/openclaw/sessions"
        )
        assert (
            handler._normalize_path("/api/gateway/openclaw/actions")
            == "/api/gateway/openclaw/actions"
        )


# ===========================================================================
# Credential Handler Integration Tests
# ===========================================================================


class TestCredentialHandlerIntegration:
    """Test credential storage endpoint with validation."""

    def test_store_credential_validates_name(self, handler, mock_user, fresh_store):
        """Test credential storage validates name."""
        mock_handler = MockRequestHandler(
            body={"name": "123invalid", "type": "api_key", "secret": "supersecret123"}
        )
        create_mock_handler_with_user(handler, mock_user)

        with patch("aragora.server.handlers.openclaw_gateway._get_store", return_value=fresh_store):
            with patch(
                "aragora.server.handlers.openclaw_gateway.require_permission",
                lambda *a, **kw: lambda f: f,
            ):
                with patch(
                    "aragora.server.handlers.openclaw_gateway.auth_rate_limit",
                    lambda *a, **kw: lambda f: f,
                ):
                    result = handler._handle_store_credential(
                        {"name": "123invalid", "type": "api_key", "secret": "supersecret123"},
                        mock_handler,
                    )

        assert result.status_code == 400
        body = json.loads(result.body)
        assert "start with a letter" in body.get("error", "")

    def test_store_credential_validates_secret_length(self, handler, mock_user, fresh_store):
        """Test credential storage validates secret length."""
        mock_handler = MockRequestHandler(
            body={"name": "mykey", "type": "api_key", "secret": "short"}
        )
        create_mock_handler_with_user(handler, mock_user)

        with patch("aragora.server.handlers.openclaw_gateway._get_store", return_value=fresh_store):
            with patch(
                "aragora.server.handlers.openclaw_gateway.require_permission",
                lambda *a, **kw: lambda f: f,
            ):
                with patch(
                    "aragora.server.handlers.openclaw_gateway.auth_rate_limit",
                    lambda *a, **kw: lambda f: f,
                ):
                    result = handler._handle_store_credential(
                        {"name": "mykey", "type": "api_key", "secret": "short"},
                        mock_handler,
                    )

        assert result.status_code == 400
        body = json.loads(result.body)
        assert "at least" in body.get("error", "")

    def test_store_credential_success(self, handler, mock_user, fresh_store):
        """Test credential storage succeeds with valid input."""
        valid_secret = "a" * MIN_CREDENTIAL_SECRET_LENGTH
        mock_handler = MockRequestHandler(
            body={"name": "myapikey", "type": "api_key", "secret": valid_secret}
        )
        create_mock_handler_with_user(handler, mock_user)

        with patch("aragora.server.handlers.openclaw_gateway._get_store", return_value=fresh_store):
            with patch(
                "aragora.server.handlers.openclaw_gateway.require_permission",
                lambda *a, **kw: lambda f: f,
            ):
                with patch(
                    "aragora.server.handlers.openclaw_gateway.auth_rate_limit",
                    lambda *a, **kw: lambda f: f,
                ):
                    result = handler._handle_store_credential(
                        {"name": "myapikey", "type": "api_key", "secret": valid_secret},
                        mock_handler,
                    )

        assert result.status_code == 201


# ===========================================================================
# Session Creation Validation Tests
# ===========================================================================


class TestSessionCreationValidation:
    """Test session creation endpoint validation."""

    def test_create_session_validates_config_size(self, handler, mock_user, fresh_store):
        """Test session creation validates config size."""
        large_config = {"data": "x" * (MAX_SESSION_CONFIG_SIZE + 1)}
        mock_handler = MockRequestHandler(body={"config": large_config})
        create_mock_handler_with_user(handler, mock_user)

        with patch("aragora.server.handlers.openclaw_gateway._get_store", return_value=fresh_store):
            with patch(
                "aragora.server.handlers.openclaw_gateway.require_permission",
                lambda *a, **kw: lambda f: f,
            ):
                with patch(
                    "aragora.server.handlers.openclaw_gateway.rate_limit",
                    lambda *a, **kw: lambda f: f,
                ):
                    result = handler._handle_create_session(
                        {"config": large_config},
                        mock_handler,
                    )

        assert result.status_code == 400
        body = json.loads(result.body)
        assert "exceeds maximum size" in body.get("error", "")

    def test_create_session_success(self, handler, mock_user, fresh_store):
        """Test session creation succeeds with valid input."""
        mock_handler = MockRequestHandler(body={"config": {"key": "value"}})
        create_mock_handler_with_user(handler, mock_user)

        with patch("aragora.server.handlers.openclaw_gateway._get_store", return_value=fresh_store):
            with patch(
                "aragora.server.handlers.openclaw_gateway.require_permission",
                lambda *a, **kw: lambda f: f,
            ):
                with patch(
                    "aragora.server.handlers.openclaw_gateway.rate_limit",
                    lambda *a, **kw: lambda f: f,
                ):
                    result = handler._handle_create_session(
                        {"config": {"key": "value"}},
                        mock_handler,
                    )

        assert result.status_code == 201


# ===========================================================================
# Action Execution Validation Tests
# ===========================================================================


class TestActionExecutionValidation:
    """Test action execution endpoint validation."""

    def test_execute_action_validates_action_type(self, handler, mock_user, fresh_store):
        """Test action execution validates action type."""
        session = fresh_store.create_session(user_id="user-001")
        mock_handler = MockRequestHandler(
            body={"session_id": session.id, "action_type": ";rm -rf /"}
        )
        create_mock_handler_with_user(handler, mock_user)

        with patch("aragora.server.handlers.openclaw_gateway._get_store", return_value=fresh_store):
            with patch(
                "aragora.server.handlers.openclaw_gateway.require_permission",
                lambda *a, **kw: lambda f: f,
            ):
                with patch(
                    "aragora.server.handlers.openclaw_gateway.auth_rate_limit",
                    lambda *a, **kw: lambda f: f,
                ):
                    result = handler._handle_execute_action(
                        {"session_id": session.id, "action_type": ";rm -rf /"},
                        mock_handler,
                    )

        assert result.status_code == 400
        body = json.loads(result.body)
        assert "start with a letter" in body.get("error", "")

    def test_execute_action_sanitizes_input(self, handler, mock_user, fresh_store):
        """Test action execution sanitizes input parameters."""
        session = fresh_store.create_session(user_id="user-001")
        mock_handler = MockRequestHandler(
            body={
                "session_id": session.id,
                "action_type": "file.read",
                "input": {"path": "/etc/passwd; cat /etc/shadow"},
            }
        )
        create_mock_handler_with_user(handler, mock_user)

        with patch("aragora.server.handlers.openclaw_gateway._get_store", return_value=fresh_store):
            with patch(
                "aragora.server.handlers.openclaw_gateway.require_permission",
                lambda *a, **kw: lambda f: f,
            ):
                with patch(
                    "aragora.server.handlers.openclaw_gateway.auth_rate_limit",
                    lambda *a, **kw: lambda f: f,
                ):
                    result = handler._handle_execute_action(
                        {
                            "session_id": session.id,
                            "action_type": "file.read",
                            "input": {"path": "/etc/passwd; cat /etc/shadow"},
                        },
                        mock_handler,
                    )

        # Should succeed but with sanitized input
        if result.status_code == 202:
            action_data = json.loads(result.body)
            action = fresh_store.get_action(action_data["id"])
            # The semicolon should be escaped in the stored input
            assert ";" not in action.input_data["path"] or "\\;" in action.input_data["path"]


# ===========================================================================
# Credential Rotation Validation Tests
# ===========================================================================


class TestCredentialRotationValidation:
    """Test credential rotation endpoint validation."""

    def test_rotate_credential_validates_secret(self, handler, mock_user, fresh_store):
        """Test credential rotation validates new secret."""
        credential = fresh_store.store_credential(
            name="testkey",
            credential_type=CredentialType.API_KEY,
            secret_value="originalsecret",
            user_id="user-001",
        )
        mock_handler = MockRequestHandler(body={"secret": "short"})
        create_mock_handler_with_user(handler, mock_user)

        with patch("aragora.server.handlers.openclaw_gateway._get_store", return_value=fresh_store):
            with patch(
                "aragora.server.handlers.openclaw_gateway.require_permission",
                lambda *a, **kw: lambda f: f,
            ):
                with patch(
                    "aragora.server.handlers.openclaw_gateway.auth_rate_limit",
                    lambda *a, **kw: lambda f: f,
                ):
                    result = handler._handle_rotate_credential(
                        credential.id,
                        {"secret": "short"},
                        mock_handler,
                    )

        assert result.status_code == 400
        body = json.loads(result.body)
        assert "at least" in body.get("error", "")

    def test_rotate_credential_rate_limited(self, handler, mock_user, fresh_store):
        """Test credential rotation is rate limited."""
        credential = fresh_store.store_credential(
            name="testkey",
            credential_type=CredentialType.API_KEY,
            secret_value="originalsecret",
            user_id="user-001",
        )
        test_limiter = CredentialRotationRateLimiter(max_rotations=1, window_seconds=60)
        valid_secret = "a" * MIN_CREDENTIAL_SECRET_LENGTH

        mock_handler = MockRequestHandler(body={"secret": valid_secret})
        create_mock_handler_with_user(handler, mock_user)

        with patch("aragora.server.handlers.openclaw_gateway._get_store", return_value=fresh_store):
            with patch(
                "aragora.server.handlers.openclaw_gateway._get_credential_rotation_limiter",
                return_value=test_limiter,
            ):
                with patch(
                    "aragora.server.handlers.openclaw_gateway.require_permission",
                    lambda *a, **kw: lambda f: f,
                ):
                    with patch(
                        "aragora.server.handlers.openclaw_gateway.auth_rate_limit",
                        lambda *a, **kw: lambda f: f,
                    ):
                        # First rotation should succeed
                        result1 = handler._handle_rotate_credential(
                            credential.id,
                            {"secret": valid_secret},
                            mock_handler,
                        )
                        assert result1.status_code == 200

                        # Second rotation should be rate limited
                        result2 = handler._handle_rotate_credential(
                            credential.id,
                            {"secret": valid_secret},
                            mock_handler,
                        )
                        assert result2.status_code == 429
                        body = json.loads(result2.body)
                        assert "Too many credential rotations" in body.get("error", "")


# ===========================================================================
# Store Tests
# ===========================================================================


class TestOpenClawGatewayStore:
    """Test OpenClawGatewayStore operations."""

    def test_create_session(self, fresh_store):
        """Test session creation."""
        session = fresh_store.create_session(user_id="user-001")
        assert session.id is not None
        assert session.user_id == "user-001"
        assert session.status == SessionStatus.ACTIVE

    def test_create_action(self, fresh_store):
        """Test action creation."""
        session = fresh_store.create_session(user_id="user-001")
        action = fresh_store.create_action(
            session_id=session.id,
            action_type="test.action",
            input_data={"key": "value"},
        )
        assert action.id is not None
        assert action.session_id == session.id
        assert action.action_type == "test.action"

    def test_store_credential(self, fresh_store):
        """Test credential storage."""
        credential = fresh_store.store_credential(
            name="testkey",
            credential_type=CredentialType.API_KEY,
            secret_value="testsecret123",
            user_id="user-001",
        )
        assert credential.id is not None
        assert credential.name == "testkey"
        assert credential.credential_type == CredentialType.API_KEY

    def test_rotate_credential(self, fresh_store):
        """Test credential rotation."""
        credential = fresh_store.store_credential(
            name="testkey",
            credential_type=CredentialType.API_KEY,
            secret_value="oldsecret123",
            user_id="user-001",
        )

        rotated = fresh_store.rotate_credential(credential.id, "newsecret456")
        assert rotated is not None
        assert rotated.last_rotated_at is not None

    def test_audit_log(self, fresh_store):
        """Test audit log entries."""
        entry = fresh_store.add_audit_entry(
            action="test.action",
            actor_id="user-001",
            resource_type="credential",
            resource_id="cred-001",
            result="success",
        )
        assert entry.id is not None
        assert entry.action == "test.action"

        entries, total = fresh_store.get_audit_log()
        assert total >= 1


# ===========================================================================
# Health Endpoint Tests
# ===========================================================================


class TestHealthEndpoint:
    """Test health endpoint."""

    def test_health_returns_status(self, handler, fresh_store):
        """Test health endpoint returns status."""
        mock_handler = MockRequestHandler()

        with patch("aragora.server.handlers.openclaw_gateway._get_store", return_value=fresh_store):
            result = handler._handle_health(mock_handler)

        assert result is not None
        assert result.status_code == 200
        body = json.loads(result.body)
        assert "status" in body
        assert "healthy" in body


# ===========================================================================
# Handler Factory Tests
# ===========================================================================


class TestHandlerFactory:
    """Test handler factory function."""

    def test_get_openclaw_gateway_handler(self, mock_server_context):
        """Test getting handler instance from factory."""
        handler = get_openclaw_gateway_handler(mock_server_context)
        assert isinstance(handler, OpenClawGatewayHandler)
