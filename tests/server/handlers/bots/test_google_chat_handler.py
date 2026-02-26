"""Tests for Google Chat bot handler."""

import json
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.server.handlers.bots.google_chat import (
    GoogleChatHandler,
    get_google_chat_handler,
    get_google_chat_connector,
    _verify_google_chat_token,
)

from aragora.server.handlers.bots.google_chat import (
    clear_token_cache,
    _get_cached_result,
    _set_cached_result,
    _token_cache,
    _token_cache_lock,
    _token_cache_key,
    _verify_token_via_jwt_verifier,
    _verify_token_via_google_auth,
    _verify_token_via_tokeninfo,
)


# =============================================================================
# Test Handler Initialization
# =============================================================================


class TestGoogleChatHandlerInit:
    """Tests for Google Chat handler initialization."""

    def test_handler_routes(self):
        """Should define correct routes."""
        handler = GoogleChatHandler({})
        assert "/api/v1/bots/google-chat/webhook" in handler.ROUTES
        assert "/api/v1/bots/google-chat/status" in handler.ROUTES

    def test_can_handle_webhook_route(self):
        """Should handle webhook route."""
        handler = GoogleChatHandler({})
        assert handler.can_handle("/api/v1/bots/google-chat/webhook") is True

    def test_can_handle_status_route(self):
        """Should handle status route."""
        handler = GoogleChatHandler({})
        assert handler.can_handle("/api/v1/bots/google-chat/status") is True

    def test_cannot_handle_unknown_route(self):
        """Should not handle unknown routes."""
        handler = GoogleChatHandler({})
        assert handler.can_handle("/api/v1/bots/unknown") is False


# =============================================================================
# Test Factory Functions
# =============================================================================


class TestFactoryFunctions:
    """Tests for factory functions."""

    def test_get_google_chat_handler(self):
        """Should return GoogleChatHandler instance."""
        # Reset singleton for testing
        import aragora.server.handlers.bots.google_chat as gchat_module

        gchat_module._google_chat_handler = None

        handler = get_google_chat_handler()
        assert isinstance(handler, GoogleChatHandler)

    def test_get_google_chat_connector_no_credentials(self):
        """Should return None when no credentials configured."""
        # Reset singleton for testing
        import aragora.server.handlers.bots.google_chat as gchat_module

        gchat_module._google_chat_connector = None

        with patch("aragora.server.handlers.bots.google_chat.GOOGLE_CHAT_CREDENTIALS", ""):
            connector = get_google_chat_connector()
        assert connector is None


# =============================================================================
# Test Status Endpoint
# =============================================================================


class TestGoogleChatStatus:
    """Tests for Google Chat status endpoint."""

    @pytest.mark.asyncio
    async def test_get_status(self):
        """Should return status information."""
        handler = GoogleChatHandler({})

        with patch.object(handler, "get_auth_context", new_callable=AsyncMock) as mock_auth:
            mock_auth.return_value = MagicMock(permissions=["bots.read"])
            with patch.object(handler, "check_permission"):
                mock_handler = MagicMock()
                result = await handler.handle("/api/v1/bots/google-chat/status", {}, mock_handler)

        assert result is not None
        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["platform"] == "google_chat"
        assert "enabled" in body
        assert "credentials_configured" in body
        assert "project_id_configured" in body

    @pytest.mark.asyncio
    async def test_get_status_requires_auth(self):
        """Should require authentication for status endpoint."""
        from aragora.server.handlers.secure import UnauthorizedError

        handler = GoogleChatHandler({})

        with patch.object(handler, "get_auth_context", new_callable=AsyncMock) as mock_auth:
            mock_auth.side_effect = UnauthorizedError("No auth")
            mock_handler = MagicMock()
            result = await handler.handle("/api/v1/bots/google-chat/status", {}, mock_handler)

        assert result is not None
        assert result.status_code == 401


# =============================================================================
# Test Webhook Events
# =============================================================================


def _make_mock_request(event: dict) -> MagicMock:
    """Helper to create mock request with proper headers for Google Chat webhook."""
    body = json.dumps(event).encode()
    mock_request = MagicMock()
    mock_request.headers = {
        "Content-Length": str(len(body)),
        "Authorization": "Bearer test-google-token",
    }
    mock_request.rfile.read.return_value = body
    return mock_request


def _patch_google_chat_auth():
    """Context manager to patch both credentials and token verification."""
    return patch.multiple(
        "aragora.server.handlers.bots.google_chat",
        GOOGLE_CHAT_CREDENTIALS="test-creds",
        _verify_google_chat_token=MagicMock(return_value=True),
    )


class TestGoogleChatWebhook:
    """Tests for Google Chat webhook event handling."""

    def test_handle_message_event(self):
        """Should handle MESSAGE event."""
        handler = GoogleChatHandler({})

        event = {
            "type": "MESSAGE",
            "message": {
                "text": "Hello bot",
                "sender": {"displayName": "Test User"},
            },
            "space": {"name": "spaces/123", "type": "DM"},
            "user": {"displayName": "Test User"},
        }

        mock_request = _make_mock_request(event)

        with _patch_google_chat_auth():
            result = handler.handle_post("/api/v1/bots/google-chat/webhook", {}, mock_request)

        assert result is not None
        assert result.status_code == 200

    def test_handle_added_to_space_event(self):
        """Should handle ADDED_TO_SPACE event."""
        handler = GoogleChatHandler({})

        event = {
            "type": "ADDED_TO_SPACE",
            "space": {"name": "spaces/123", "displayName": "Test Space"},
        }

        mock_request = _make_mock_request(event)

        with _patch_google_chat_auth():
            result = handler.handle_post("/api/v1/bots/google-chat/webhook", {}, mock_request)

        assert result is not None
        assert result.status_code == 200
        body = json.loads(result.body)
        # Should return welcome card
        assert "cardsV2" in body or "text" in body

    def test_handle_removed_from_space_event(self):
        """Should handle REMOVED_FROM_SPACE event."""
        handler = GoogleChatHandler({})

        event = {
            "type": "REMOVED_FROM_SPACE",
            "space": {"name": "spaces/123"},
        }

        mock_request = _make_mock_request(event)

        with _patch_google_chat_auth():
            result = handler.handle_post("/api/v1/bots/google-chat/webhook", {}, mock_request)

        assert result is not None
        assert result.status_code == 200

    def test_handle_card_clicked_event(self):
        """Should handle CARD_CLICKED event."""
        handler = GoogleChatHandler({})

        event = {
            "type": "CARD_CLICKED",
            "action": {
                "actionMethodName": "vote_agree",
                "parameters": [{"key": "debate_id", "value": "debate123"}],
            },
            "user": {"name": "users/123", "displayName": "Test User"},
            "space": {"name": "spaces/456"},
        }

        mock_request = _make_mock_request(event)

        with _patch_google_chat_auth():
            result = handler.handle_post("/api/v1/bots/google-chat/webhook", {}, mock_request)

        assert result is not None
        assert result.status_code == 200

    def test_handle_invalid_json(self):
        """Should handle invalid JSON gracefully."""
        handler = GoogleChatHandler({})

        mock_request = MagicMock()
        mock_request.headers = {
            "Content-Length": "15",
            "Authorization": "Bearer test-google-token",
        }
        mock_request.rfile.read.return_value = b"not valid json"

        with _patch_google_chat_auth():
            result = handler.handle_post("/api/v1/bots/google-chat/webhook", {}, mock_request)

        assert result is not None
        assert result.status_code == 400

    def test_handle_unknown_event_type(self):
        """Should handle unknown event types gracefully."""
        handler = GoogleChatHandler({})

        event = {"type": "UNKNOWN_EVENT"}

        mock_request = _make_mock_request(event)

        with _patch_google_chat_auth():
            result = handler.handle_post("/api/v1/bots/google-chat/webhook", {}, mock_request)

        assert result is not None
        assert result.status_code == 200

    def test_webhook_requires_credentials(self):
        """Should reject webhook when credentials not configured."""
        handler = GoogleChatHandler({})

        event = {"type": "MESSAGE", "message": {"text": "test"}}
        mock_request = _make_mock_request(event)

        with patch("aragora.server.handlers.bots.google_chat.GOOGLE_CHAT_CREDENTIALS", ""):
            result = handler.handle_post("/api/v1/bots/google-chat/webhook", {}, mock_request)

        assert result is not None
        assert result.status_code == 503

    def test_webhook_requires_bearer_token(self):
        """Should reject webhook without bearer token."""
        handler = GoogleChatHandler({})

        event = {"type": "MESSAGE", "message": {"text": "test"}}
        body = json.dumps(event).encode()
        mock_request = MagicMock()
        mock_request.headers = {"Content-Length": str(len(body))}  # No Authorization
        mock_request.rfile.read.return_value = body

        # Only patch credentials, not verification - we want to test auth failure
        with patch(
            "aragora.server.handlers.bots.google_chat.GOOGLE_CHAT_CREDENTIALS", "test-creds"
        ):
            result = handler.handle_post("/api/v1/bots/google-chat/webhook", {}, mock_request)

        assert result is not None
        assert result.status_code == 401


# =============================================================================
# Test Slash Commands
# =============================================================================


class TestGoogleChatSlashCommands:
    """Tests for Google Chat slash command handling."""

    def test_handle_help_command(self):
        """Should handle /help slash command."""
        handler = GoogleChatHandler({})

        event = {
            "type": "MESSAGE",
            "message": {
                "text": "/help",
                "slashCommand": {"commandName": "/help"},
            },
            "space": {"name": "spaces/123"},
            "user": {"displayName": "Test User", "name": "users/456"},
        }

        mock_request = _make_mock_request(event)

        with _patch_google_chat_auth():
            result = handler.handle_post("/api/v1/bots/google-chat/webhook", {}, mock_request)

        assert result is not None
        assert result.status_code == 200
        body = json.loads(result.body)
        # Should return help card
        assert "cardsV2" in body

    def test_handle_status_command(self):
        """Should handle /status slash command."""
        handler = GoogleChatHandler({})

        event = {
            "type": "MESSAGE",
            "message": {
                "text": "/status",
                "slashCommand": {"commandName": "/status"},
            },
            "space": {"name": "spaces/123"},
            "user": {"displayName": "Test User", "name": "users/456"},
        }

        mock_request = _make_mock_request(event)

        with _patch_google_chat_auth():
            result = handler.handle_post("/api/v1/bots/google-chat/webhook", {}, mock_request)

        assert result is not None
        assert result.status_code == 200

    def test_handle_debate_command_empty_topic(self):
        """Should handle /debate with empty topic."""
        handler = GoogleChatHandler({})

        event = {
            "type": "MESSAGE",
            "message": {
                "text": "/debate",
                "slashCommand": {"commandName": "/debate"},
                "argumentText": "",
            },
            "space": {"name": "spaces/123"},
            "user": {"displayName": "Test User", "name": "users/456"},
        }

        mock_request = _make_mock_request(event)

        with _patch_google_chat_auth():
            result = handler.handle_post("/api/v1/bots/google-chat/webhook", {}, mock_request)

        assert result is not None
        assert result.status_code == 200
        body = json.loads(result.body)
        # Should return error about empty topic
        assert "cardsV2" in body or "text" in body

    def test_handle_unknown_command(self):
        """Should handle unknown slash commands."""
        handler = GoogleChatHandler({})

        event = {
            "type": "MESSAGE",
            "message": {
                "text": "/unknowncmd",
                "slashCommand": {"commandName": "/unknowncmd"},
            },
            "space": {"name": "spaces/123"},
            "user": {"displayName": "Test User", "name": "users/456"},
        }

        mock_request = _make_mock_request(event)

        with _patch_google_chat_auth():
            result = handler.handle_post("/api/v1/bots/google-chat/webhook", {}, mock_request)

        assert result is not None
        assert result.status_code == 200


# =============================================================================
# Test Card Response Builder
# =============================================================================


class TestCardResponseBuilder:
    """Tests for Google Chat card response builder."""

    def test_card_response_with_title(self):
        """Should build card response with title."""
        handler = GoogleChatHandler({})

        result = handler._card_response(title="Test Title", body="Test body")

        assert result.status_code == 200
        body = json.loads(result.body)
        assert "cardsV2" in body
        card = body["cardsV2"][0]["card"]
        assert len(card["sections"]) >= 2  # header + body

    def test_card_response_with_fields(self):
        """Should build card response with fields."""
        handler = GoogleChatHandler({})

        result = handler._card_response(
            title="Test",
            fields=[("Field1", "Value1"), ("Field2", "Value2")],
        )

        assert result.status_code == 200
        body = json.loads(result.body)
        assert "cardsV2" in body

    def test_card_response_body_only(self):
        """Should build simple text response when no title."""
        handler = GoogleChatHandler({})

        result = handler._card_response(body="Simple text message")

        assert result.status_code == 200
        body = json.loads(result.body)
        # May be card or text depending on implementation
        assert "cardsV2" in body or "text" in body


# =============================================================================
# Test Input Validation
# =============================================================================


class TestInputValidation:
    """Tests for input validation."""

    def test_debate_topic_too_short(self):
        """Should reject debate topic that is too short."""
        handler = GoogleChatHandler({})

        event = {
            "type": "MESSAGE",
            "message": {
                "text": "/debate hi",
                "slashCommand": {"commandName": "/debate"},
                "argumentText": "hi",
            },
            "space": {"name": "spaces/123"},
            "user": {"displayName": "Test User", "name": "users/456"},
        }

        mock_request = _make_mock_request(event)

        with _patch_google_chat_auth():
            result = handler.handle_post("/api/v1/bots/google-chat/webhook", {}, mock_request)

        assert result is not None
        assert result.status_code == 200
        body = json.loads(result.body)
        # Should contain error message about topic being too short
        sections = body.get("cardsV2", [{}])[0].get("card", {}).get("sections", [])
        section_texts = [
            s.get("widgets", [{}])[0].get("textParagraph", {}).get("text", "")
            for s in sections
            if s.get("widgets")
        ]
        assert any("too short" in t.lower() for t in section_texts if t)


# =============================================================================
# Test Bearer Token Verification (Phase 3.1)
# =============================================================================


class TestBearerTokenHeaderParsing:
    """Tests for Bearer token header parsing (always available)."""

    def test_missing_auth_header(self):
        """Should reject when Authorization header is missing."""
        assert _verify_google_chat_token("") is False

    def test_non_bearer_auth_header(self):
        """Should reject when Authorization header is not Bearer type."""
        assert _verify_google_chat_token("Basic dXNlcjpwYXNz") is False

    def test_empty_bearer_token(self):
        """Should reject when Bearer token is empty."""
        assert _verify_google_chat_token("Bearer ") is False

    def test_bearer_with_only_whitespace(self):
        """Should reject when Bearer token is only whitespace."""
        assert _verify_google_chat_token("Bearer   ") is False

    def test_none_auth_header(self):
        """Should reject None auth header."""
        assert _verify_google_chat_token(None) is False


class TestBearerTokenLayeredVerification:
    """Tests for the layered Bearer token verification system (requires token cache)."""

    def setup_method(self):
        """Clear token cache before each test."""
        if clear_token_cache:
            clear_token_cache()

    # ---- JWT Verifier layer ----

    def test_jwt_verifier_valid_token(self):
        """Should accept token when JWT verifier returns True."""
        with patch(
            "aragora.server.handlers.bots.google_chat._verify_token_via_jwt_verifier",
            return_value=True,
        ):
            assert _verify_google_chat_token("Bearer valid-token") is True

    def test_jwt_verifier_invalid_token(self):
        """Should reject token when JWT verifier returns False."""
        with patch(
            "aragora.server.handlers.bots.google_chat._verify_token_via_jwt_verifier",
            return_value=False,
        ):
            assert _verify_google_chat_token("Bearer invalid-token") is False

    def test_jwt_verifier_unavailable_falls_through(self):
        """Should fall through to google-auth when JWT verifier returns None."""
        with (
            patch(
                "aragora.server.handlers.bots.google_chat._verify_token_via_jwt_verifier",
                return_value=None,
            ) as mock_jwt,
            patch(
                "aragora.server.handlers.bots.google_chat._verify_token_via_google_auth",
                return_value=True,
            ) as mock_gauth,
        ):
            result = _verify_google_chat_token("Bearer some-token")
            assert result is True
            mock_jwt.assert_called_once()
            mock_gauth.assert_called_once()

    # ---- Google Auth layer ----

    def test_google_auth_valid_token(self):
        """Should accept token when google-auth returns True."""
        with (
            patch(
                "aragora.server.handlers.bots.google_chat._verify_token_via_jwt_verifier",
                return_value=None,
            ),
            patch(
                "aragora.server.handlers.bots.google_chat._verify_token_via_google_auth",
                return_value=True,
            ),
        ):
            assert _verify_google_chat_token("Bearer valid-gauth") is True

    def test_google_auth_invalid_token(self):
        """Should reject token when google-auth returns False."""
        with (
            patch(
                "aragora.server.handlers.bots.google_chat._verify_token_via_jwt_verifier",
                return_value=None,
            ),
            patch(
                "aragora.server.handlers.bots.google_chat._verify_token_via_google_auth",
                return_value=False,
            ),
        ):
            assert _verify_google_chat_token("Bearer bad-gauth") is False

    def test_google_auth_unavailable_falls_through(self):
        """Should fall through to tokeninfo when google-auth returns None."""
        with (
            patch(
                "aragora.server.handlers.bots.google_chat._verify_token_via_jwt_verifier",
                return_value=None,
            ),
            patch(
                "aragora.server.handlers.bots.google_chat._verify_token_via_google_auth",
                return_value=None,
            ),
            patch(
                "aragora.server.handlers.bots.google_chat._verify_token_via_tokeninfo",
                return_value=True,
            ) as mock_tokeninfo,
        ):
            result = _verify_google_chat_token("Bearer some-token")
            assert result is True
            mock_tokeninfo.assert_called_once()

    # ---- Tokeninfo layer ----

    def test_tokeninfo_valid_token(self):
        """Should accept token when tokeninfo returns True."""
        with (
            patch(
                "aragora.server.handlers.bots.google_chat._verify_token_via_jwt_verifier",
                return_value=None,
            ),
            patch(
                "aragora.server.handlers.bots.google_chat._verify_token_via_google_auth",
                return_value=None,
            ),
            patch(
                "aragora.server.handlers.bots.google_chat._verify_token_via_tokeninfo",
                return_value=True,
            ),
        ):
            assert _verify_google_chat_token("Bearer valid-tokeninfo") is True

    def test_tokeninfo_invalid_token(self):
        """Should reject token when tokeninfo returns False."""
        with (
            patch(
                "aragora.server.handlers.bots.google_chat._verify_token_via_jwt_verifier",
                return_value=None,
            ),
            patch(
                "aragora.server.handlers.bots.google_chat._verify_token_via_google_auth",
                return_value=None,
            ),
            patch(
                "aragora.server.handlers.bots.google_chat._verify_token_via_tokeninfo",
                return_value=False,
            ),
        ):
            assert _verify_google_chat_token("Bearer bad-tokeninfo") is False

    # ---- Fail closed ----

    def test_all_verifiers_unavailable_rejects(self):
        """Should fail closed when all verification methods are unavailable."""
        with (
            patch(
                "aragora.server.handlers.bots.google_chat._verify_token_via_jwt_verifier",
                return_value=None,
            ),
            patch(
                "aragora.server.handlers.bots.google_chat._verify_token_via_google_auth",
                return_value=None,
            ),
            patch(
                "aragora.server.handlers.bots.google_chat._verify_token_via_tokeninfo",
                return_value=None,
            ),
        ):
            assert _verify_google_chat_token("Bearer any-token") is False


# =============================================================================
# Test Token Cache
# =============================================================================


class TestTokenCache:
    """Tests for the token verification cache."""

    def setup_method(self):
        """Clear token cache before each test."""
        clear_token_cache()

    def test_cache_miss_returns_none(self):
        """Should return None for uncached token."""
        result = _get_cached_result("uncached-token")
        assert result is None

    def test_cache_stores_valid_result(self):
        """Should cache and return valid token result."""
        _set_cached_result("valid-token", True)
        result = _get_cached_result("valid-token")
        assert result is True

    def test_cache_stores_invalid_result(self):
        """Should cache and return invalid token result."""
        _set_cached_result("invalid-token", False)
        result = _get_cached_result("invalid-token")
        assert result is False

    def test_cache_hit_skips_verification(self):
        """Should use cached result instead of calling verifiers."""
        # Pre-populate cache with valid result
        _set_cached_result("cached-token", True)

        with patch(
            "aragora.server.handlers.bots.google_chat._verify_token_via_jwt_verifier",
        ) as mock_jwt:
            result = _verify_google_chat_token("Bearer cached-token")
            assert result is True
            mock_jwt.assert_not_called()

    def test_cache_expired_returns_none(self):
        """Should return None for expired cache entry."""
        _set_cached_result("expiring-token", True)

        # Manually expire the entry by modifying the cache
        from aragora.server.handlers.bots.google_chat import (
            _token_cache,
            _token_cache_lock,
        )

        key = _token_cache_key("expiring-token")
        with _token_cache_lock:
            # Set expiry to the past
            _token_cache[key] = (True, time.monotonic() - 1)

        result = _get_cached_result("expiring-token")
        assert result is None

    def test_clear_token_cache(self):
        """Should clear all cached entries."""
        _set_cached_result("token-1", True)
        _set_cached_result("token-2", False)

        clear_token_cache()

        assert _get_cached_result("token-1") is None
        assert _get_cached_result("token-2") is None

    def test_cache_different_ttl_for_valid_vs_invalid(self):
        """Should use different TTLs for valid and invalid tokens."""
        from aragora.server.handlers.bots.google_chat import (
            _TOKEN_CACHE_VALID_TTL,
            _TOKEN_CACHE_INVALID_TTL,
        )

        _set_cached_result("valid-tok", True)
        _set_cached_result("invalid-tok", False)

        valid_key = _token_cache_key("valid-tok")
        invalid_key = _token_cache_key("invalid-tok")

        with _token_cache_lock:
            _, valid_expiry = _token_cache[valid_key]
            _, invalid_expiry = _token_cache[invalid_key]

        # Valid token should have a later expiry than invalid token
        # (5 min vs 1 min TTL)
        assert valid_expiry > invalid_expiry


# =============================================================================
# Test Individual Verification Layers
# =============================================================================


class TestJWTVerifierLayer:
    """Tests for _verify_token_via_jwt_verifier."""

    def test_import_error_returns_none(self):
        """Should return None when jwt_verify module is not available."""
        with patch(
            "aragora.server.handlers.bots.google_chat._verify_token_via_jwt_verifier",
            wraps=_verify_token_via_jwt_verifier,
        ):
            with patch.dict("sys.modules", {"aragora.connectors.chat.jwt_verify": None}):
                # Force import to raise ImportError
                import importlib

                result = _verify_token_via_jwt_verifier("test-token")
                # When module is None in sys.modules, import raises ImportError
                assert result is None

    def test_runtime_error_returns_none(self):
        """Should return None on RuntimeError."""
        with patch(
            "aragora.connectors.chat.jwt_verify.verify_google_chat_webhook",
            side_effect=RuntimeError("JWKS fetch failed"),
        ):
            result = _verify_token_via_jwt_verifier("test-token")
            assert result is None


class TestGoogleAuthLayer:
    """Tests for _verify_token_via_google_auth."""

    def test_import_error_returns_none(self):
        """Should return None when google-auth is not installed."""
        with patch.dict("sys.modules", {"google.oauth2": None, "google": None}):
            result = _verify_token_via_google_auth("test-token")
            assert result is None

    def test_production_rejects_without_project_id(self):
        """Should reject token in production when project ID is not set."""
        pytest.importorskip("google.oauth2.id_token")
        pytest.importorskip("google.auth.transport.requests")
        with (
            patch("aragora.server.handlers.bots.google_chat.GOOGLE_CHAT_PROJECT_ID", None),
            patch.dict("os.environ", {"ARAGORA_ENV": "production"}),
        ):
            with patch("google.oauth2.id_token.verify_oauth2_token") as mock_verify:
                # google-auth available but no project_id
                result = _verify_token_via_google_auth("test-token")
                assert result is False
                mock_verify.assert_not_called()

    def test_value_error_returns_false(self):
        """Should return False when google-auth raises ValueError (invalid token)."""
        pytest.importorskip("google.oauth2.id_token")
        pytest.importorskip("google.auth.transport.requests")
        with patch(
            "aragora.server.handlers.bots.google_chat.GOOGLE_CHAT_PROJECT_ID", "test-project"
        ):
            with (
                patch(
                    "google.oauth2.id_token.verify_oauth2_token",
                    side_effect=ValueError("Token expired"),
                ),
                patch("google.auth.transport.requests.Request"),
            ):
                result = _verify_token_via_google_auth("expired-token")
                assert result is False


class TestTokeninfoLayer:
    """Tests for _verify_token_via_tokeninfo."""

    def test_http_error_returns_false(self):
        """Should return False on HTTP errors."""
        import urllib.error

        with patch(
            "urllib.request.urlopen",
            side_effect=urllib.error.HTTPError(
                url="https://oauth2.googleapis.com/tokeninfo",
                code=400,
                msg="Bad Request",
                hdrs={},
                fp=None,
            ),
        ):
            result = _verify_token_via_tokeninfo("bad-token")
            assert result is False

    def test_network_error_returns_none(self):
        """Should return None on network errors (allows fallthrough)."""
        import urllib.error

        with patch(
            "urllib.request.urlopen",
            side_effect=urllib.error.URLError("Connection refused"),
        ):
            result = _verify_token_via_tokeninfo("any-token")
            assert result is None

    def test_audience_mismatch_returns_false(self):
        """Should reject token with wrong audience."""
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.read.return_value = json.dumps(
            {
                "aud": "wrong-project-id",
                "email": "chat@system.gserviceaccount.com",
                "exp": str(int(time.time()) + 3600),
            }
        ).encode()
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)

        with (
            patch(
                "aragora.server.handlers.bots.google_chat.GOOGLE_CHAT_PROJECT_ID",
                "correct-project-id",
            ),
            patch("urllib.request.urlopen", return_value=mock_response),
        ):
            result = _verify_token_via_tokeninfo("wrong-aud-token")
            assert result is False

    def test_wrong_email_returns_false(self):
        """Should reject token from non-Google Chat email."""
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.read.return_value = json.dumps(
            {
                "email": "attacker@evil.com",
                "exp": str(int(time.time()) + 3600),
            }
        ).encode()
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)

        with (
            patch("aragora.server.handlers.bots.google_chat.GOOGLE_CHAT_PROJECT_ID", None),
            patch("urllib.request.urlopen", return_value=mock_response),
        ):
            result = _verify_token_via_tokeninfo("wrong-email-token")
            assert result is False

    def test_expired_token_returns_false(self):
        """Should reject expired token even if tokeninfo returns it."""
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.read.return_value = json.dumps(
            {
                "email": "chat@system.gserviceaccount.com",
                "exp": str(int(time.time()) - 3600),  # 1 hour ago
            }
        ).encode()
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)

        with (
            patch("aragora.server.handlers.bots.google_chat.GOOGLE_CHAT_PROJECT_ID", None),
            patch("urllib.request.urlopen", return_value=mock_response),
        ):
            result = _verify_token_via_tokeninfo("expired-token")
            assert result is False

    def test_valid_token_returns_true(self):
        """Should accept valid token from tokeninfo."""
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.read.return_value = json.dumps(
            {
                "aud": "my-project",
                "email": "chat@system.gserviceaccount.com",
                "exp": str(int(time.time()) + 3600),
            }
        ).encode()
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)

        with (
            patch(
                "aragora.server.handlers.bots.google_chat.GOOGLE_CHAT_PROJECT_ID",
                "my-project",
            ),
            patch("urllib.request.urlopen", return_value=mock_response),
        ):
            result = _verify_token_via_tokeninfo("valid-token")
            assert result is True


# =============================================================================
# Test Webhook Auth Integration
# =============================================================================


class TestWebhookAuthIntegration:
    """Integration tests for webhook authentication flow."""

    def test_webhook_returns_401_on_invalid_token(self):
        """Should return 401 when token verification fails."""
        handler = GoogleChatHandler({})

        event = {"type": "MESSAGE", "message": {"text": "test"}}
        body = json.dumps(event).encode()
        mock_request = MagicMock()
        mock_request.headers = {
            "Content-Length": str(len(body)),
            "Authorization": "Bearer invalid-token",
        }
        mock_request.rfile.read.return_value = body

        with (
            patch("aragora.server.handlers.bots.google_chat.GOOGLE_CHAT_CREDENTIALS", "test-creds"),
            patch(
                "aragora.server.handlers.bots.google_chat._verify_google_chat_token",
                return_value=False,
            ),
        ):
            result = handler.handle_post("/api/v1/bots/google-chat/webhook", {}, mock_request)

        assert result is not None
        assert result.status_code == 401

    def test_webhook_succeeds_with_valid_token(self):
        """Should process event when token verification succeeds."""
        handler = GoogleChatHandler({})

        event = {"type": "REMOVED_FROM_SPACE", "space": {"name": "spaces/123"}}
        mock_request = _make_mock_request(event)

        with _patch_google_chat_auth():
            result = handler.handle_post("/api/v1/bots/google-chat/webhook", {}, mock_request)

        assert result is not None
        assert result.status_code == 200
