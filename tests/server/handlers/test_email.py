"""
Tests for Email Prioritization Handler.

Tests the email prioritization API endpoints including:
- Route registration and can_handle
- Email scoring and prioritization
- Inbox ranking
- User feedback and learning
- Configuration management
- Gmail OAuth flows
- Gmail status
- Cross-channel context
- Categorization endpoints
- VIP management
- Batch operations
- RBAC permission checks
- Error handling and edge cases
"""

import pytest
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

from aragora.server.handlers.email import (
    EmailHandler,
    handle_prioritize_email,
    handle_rank_inbox,
    handle_email_feedback,
    handle_get_config,
    handle_update_config,
    handle_get_context,
    handle_get_email_context_boost,
    handle_gmail_oauth_url,
    handle_gmail_oauth_callback,
    handle_gmail_status,
    handle_fetch_and_rank_inbox,
    handle_add_vip,
    handle_remove_vip,
    handle_categorize_email,
    handle_categorize_batch,
    handle_feedback_batch,
    handle_apply_category_label,
    get_gmail_connector,
    get_prioritizer,
    get_context_service,
    _check_email_permission,
    _user_configs,
    _user_configs_lock,
)

import aragora.server.handlers.email as email_module
import aragora.server.handlers.email.storage as email_storage
import aragora.server.handlers.email.categorization as email_categorization


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture(autouse=True)
def _reset_global_state():
    """Reset module-level global singletons between tests."""
    old_gmail = email_storage._gmail_connector
    old_prioritizer = email_storage._prioritizer
    old_context = email_storage._context_service
    old_categorizer = email_categorization._categorizer
    old_store = email_storage._email_store
    old_user_configs = email_storage._user_configs.copy()

    yield

    email_storage._gmail_connector = old_gmail
    email_storage._prioritizer = old_prioritizer
    email_storage._context_service = old_context
    email_categorization._categorizer = old_categorizer
    email_storage._email_store = old_store
    email_storage._user_configs.clear()
    email_storage._user_configs.update(old_user_configs)


@pytest.fixture
def sample_email_data():
    """Minimal valid email payload."""
    return {
        "id": "msg_123",
        "subject": "Urgent: Project deadline",
        "from_address": "boss@company.com",
        "body_text": "Please review the proposal.",
        "labels": ["INBOX", "IMPORTANT"],
        "is_read": False,
        "is_starred": False,
        "is_important": True,
    }


@pytest.fixture
def mock_priority_result():
    """Mock object returned by prioritizer.score_email."""
    result = MagicMock()
    result.to_dict.return_value = {
        "email_id": "msg_123",
        "score": 85,
        "confidence": 0.9,
        "tier": "high",
    }
    return result


@pytest.fixture
def handler():
    """Create an EmailHandler instance with a test context."""
    ctx = {"auth_context": None}
    h = EmailHandler(ctx)
    return h


@pytest.fixture
def allowed_auth_context():
    """Auth context that passes RBAC checks."""
    ctx = MagicMock()
    ctx.user_id = "test_user"
    return ctx


# ===========================================================================
# Route registration and can_handle
# ===========================================================================


class TestEmailHandlerRoutes:
    """Test route definitions and can_handle logic."""

    def test_all_expected_static_routes_present(self):
        expected = [
            "/api/v1/email/prioritize",
            "/api/v1/email/rank-inbox",
            "/api/v1/email/feedback",
            "/api/v1/email/feedback/batch",
            "/api/v1/email/inbox",
            "/api/v1/email/config",
            "/api/v1/email/vip",
            "/api/v1/email/categorize",
            "/api/v1/email/categorize/batch",
            "/api/v1/email/categorize/apply-label",
            "/api/v1/email/gmail/oauth/url",
            "/api/v1/email/gmail/oauth/callback",
            "/api/v1/email/gmail/status",
            "/api/v1/email/context/boost",
        ]
        for route in expected:
            assert route in EmailHandler.ROUTES, f"Missing route: {route}"

    def test_route_prefix_defined(self):
        assert "/api/v1/email/context/" in EmailHandler.ROUTE_PREFIXES

    def test_can_handle_static_route(self):
        h = EmailHandler({})
        assert h.can_handle("/api/v1/email/prioritize") is True
        assert h.can_handle("/api/v1/email/gmail/status") is True

    def test_can_handle_dynamic_context_route(self):
        h = EmailHandler({})
        assert h.can_handle("/api/v1/email/context/user@example.com") is True
        assert h.can_handle("/api/v1/email/context/someone@org.io") is True

    def test_cannot_handle_bare_prefix(self):
        """Bare prefix without an email address segment should not match."""
        h = EmailHandler({})
        # The prefix path itself (without trailing segment) should not match
        # because the check requires path != prefix.rstrip("/")
        assert h.can_handle("/api/v1/email/context") is False

    def test_cannot_handle_unknown_paths(self):
        h = EmailHandler({})
        assert h.can_handle("/api/v1/unknown") is False
        assert h.can_handle("/api/v1/debates") is False
        assert h.can_handle("/other/email/prioritize") is False

    def test_handle_returns_none(self):
        """The sync handle() method always returns None (async delegation)."""
        h = EmailHandler({})
        result = h.handle("/api/v1/email/prioritize", {}, MagicMock())
        assert result is None


# ===========================================================================
# Prioritize Email
# ===========================================================================


class TestHandlePrioritizeEmail:
    """Test handle_prioritize_email function."""

    @pytest.mark.asyncio
    async def test_success(self, sample_email_data, mock_priority_result):
        with patch("aragora.server.handlers.email.prioritization.get_prioritizer") as mock_gp:
            mock_prioritizer = MagicMock()
            mock_prioritizer.score_email = AsyncMock(return_value=mock_priority_result)
            mock_gp.return_value = mock_prioritizer

            result = await handle_prioritize_email(sample_email_data)

            assert result["success"] is True
            assert result["result"]["score"] == 85

    @pytest.mark.asyncio
    async def test_with_force_tier(self, sample_email_data):
        mock_result = MagicMock()
        mock_result.to_dict.return_value = {"score": 60}

        with patch("aragora.server.handlers.email.prioritization.get_prioritizer") as mock_gp:
            mock_prioritizer = MagicMock()
            mock_prioritizer.score_email = AsyncMock(return_value=mock_result)
            mock_gp.return_value = mock_prioritizer

            result = await handle_prioritize_email(sample_email_data, force_tier="tier_1_rules")

            assert result["success"] is True
            # Verify force_tier was passed through
            call_kwargs = mock_prioritizer.score_email.call_args
            assert call_kwargs.kwargs.get("force_tier") is not None

    @pytest.mark.asyncio
    async def test_minimal_email_data(self):
        """Emails with only an id should still succeed (defaults fill gaps)."""
        mock_result = MagicMock()
        mock_result.to_dict.return_value = {"score": 10}

        with patch("aragora.server.handlers.email.prioritization.get_prioritizer") as mock_gp:
            mock_prioritizer = MagicMock()
            mock_prioritizer.score_email = AsyncMock(return_value=mock_result)
            mock_gp.return_value = mock_prioritizer

            result = await handle_prioritize_email({"id": "bare"})
            assert result["success"] is True

    @pytest.mark.asyncio
    async def test_error_propagation(self):
        with patch("aragora.server.handlers.email.prioritization.get_prioritizer") as mock_gp:
            mock_prioritizer = MagicMock()
            mock_prioritizer.score_email = AsyncMock(side_effect=RuntimeError("scoring exploded"))
            mock_gp.return_value = mock_prioritizer

            result = await handle_prioritize_email({"id": "x"})
            assert result["success"] is False
            assert "scoring exploded" in result["error"]


# ===========================================================================
# Rank Inbox
# ===========================================================================


class TestHandleRankInbox:
    """Test handle_rank_inbox function."""

    @pytest.mark.asyncio
    async def test_success(self):
        emails = [
            {"id": "1", "subject": "A", "from_address": "a@a.com"},
            {"id": "2", "subject": "B", "from_address": "b@b.com"},
        ]
        mock_r1 = MagicMock()
        mock_r1.to_dict.return_value = {"email_id": "1", "score": 90}
        mock_r2 = MagicMock()
        mock_r2.to_dict.return_value = {"email_id": "2", "score": 40}

        with patch("aragora.server.handlers.email.prioritization.get_prioritizer") as mock_gp:
            mock_prioritizer = MagicMock()
            mock_prioritizer.rank_inbox = AsyncMock(return_value=[mock_r1, mock_r2])
            mock_gp.return_value = mock_prioritizer

            result = await handle_rank_inbox(emails)

            assert result["success"] is True
            assert result["total"] == 2
            assert len(result["results"]) == 2

    @pytest.mark.asyncio
    async def test_empty_list(self):
        with patch("aragora.server.handlers.email.prioritization.get_prioritizer") as mock_gp:
            mock_prioritizer = MagicMock()
            mock_prioritizer.rank_inbox = AsyncMock(return_value=[])
            mock_gp.return_value = mock_prioritizer

            result = await handle_rank_inbox([])
            assert result["success"] is True
            assert result["total"] == 0

    @pytest.mark.asyncio
    async def test_limit_passed_through(self):
        with patch("aragora.server.handlers.email.prioritization.get_prioritizer") as mock_gp:
            mock_prioritizer = MagicMock()
            mock_prioritizer.rank_inbox = AsyncMock(return_value=[])
            mock_gp.return_value = mock_prioritizer

            await handle_rank_inbox([{"id": "1"}], limit=5)
            mock_prioritizer.rank_inbox.assert_called_once()
            _, kwargs = mock_prioritizer.rank_inbox.call_args
            assert kwargs["limit"] == 5

    @pytest.mark.asyncio
    async def test_error(self):
        with patch("aragora.server.handlers.email.prioritization.get_prioritizer") as mock_gp:
            mock_prioritizer = MagicMock()
            mock_prioritizer.rank_inbox = AsyncMock(side_effect=Exception("rank failed"))
            mock_gp.return_value = mock_prioritizer

            result = await handle_rank_inbox([{"id": "1"}])
            assert result["success"] is False
            assert "rank failed" in result["error"]


# ===========================================================================
# Feedback
# ===========================================================================


class TestHandleEmailFeedback:
    """Test handle_email_feedback function."""

    @pytest.mark.asyncio
    async def test_success_with_auth(self, allowed_auth_context):
        with (
            patch("aragora.server.handlers.email.prioritization.get_prioritizer") as mock_gp,
            patch("aragora.server.handlers.email.storage.check_permission") as mock_cp,
        ):
            mock_cp.return_value = MagicMock(allowed=True)
            mock_prioritizer = MagicMock()
            mock_prioritizer.record_user_action = AsyncMock()
            mock_gp.return_value = mock_prioritizer

            result = await handle_email_feedback(
                "msg_1", "archived", auth_context=allowed_auth_context
            )

            assert result["success"] is True
            assert result["email_id"] == "msg_1"
            assert result["action"] == "archived"
            assert "recorded_at" in result

    @pytest.mark.asyncio
    async def test_write_denied_without_rbac(self):
        """email:write should fail closed when RBAC is unavailable and no auth."""
        result = await handle_email_feedback("msg_1", "archived", auth_context=None)
        assert result["success"] is False
        assert "Permission denied" in result["error"]

    @pytest.mark.asyncio
    async def test_with_email_data_context(self, allowed_auth_context):
        with (
            patch("aragora.server.handlers.email.prioritization.get_prioritizer") as mock_gp,
            patch("aragora.server.handlers.email.storage.check_permission") as mock_cp,
        ):
            mock_cp.return_value = MagicMock(allowed=True)
            mock_prioritizer = MagicMock()
            mock_prioritizer.record_user_action = AsyncMock()
            mock_gp.return_value = mock_prioritizer

            result = await handle_email_feedback(
                "msg_1",
                "replied",
                email_data={"id": "msg_1", "subject": "Test"},
                auth_context=allowed_auth_context,
            )
            assert result["success"] is True


# ===========================================================================
# Configuration
# ===========================================================================


class TestConfigOperations:
    """Test configuration get/update operations."""

    @pytest.mark.asyncio
    async def test_get_config_defaults(self):
        import uuid

        unique = f"cfg_test_{uuid.uuid4().hex[:8]}"
        with _user_configs_lock:
            _user_configs.pop(unique, None)

        with patch("aragora.server.handlers.email.get_email_store", return_value=None):
            result = await handle_get_config(unique)

        assert result["success"] is True
        cfg = result["config"]
        assert cfg["vip_domains"] == []
        assert cfg["vip_addresses"] == []
        assert cfg["tier_1_confidence_threshold"] == 0.7
        assert cfg["enable_slack_signals"] is True

    @pytest.mark.asyncio
    async def test_get_config_returns_cached_values(self):
        with _user_configs_lock:
            _user_configs["cached_user"] = {
                "vip_domains": ["cached.com"],
                "vip_addresses": ["me@cached.com"],
            }

        result = await handle_get_config("cached_user")
        assert "cached.com" in result["config"]["vip_domains"]
        assert "me@cached.com" in result["config"]["vip_addresses"]

        # Cleanup
        with _user_configs_lock:
            _user_configs.pop("cached_user", None)

    @pytest.mark.asyncio
    async def test_update_config(self):
        updates = {
            "vip_domains": ["important.com"],
            "vip_addresses": ["ceo@company.com"],
            "enable_slack_signals": False,
        }

        with patch("aragora.server.handlers.email.get_email_store", return_value=None):
            result = await handle_update_config("update_user", updates)

        assert result["success"] is True
        cfg = result["config"]
        assert "important.com" in cfg["vip_domains"]
        assert cfg["enable_slack_signals"] is False

        # Cleanup
        with _user_configs_lock:
            _user_configs.pop("update_user", None)

    @pytest.mark.asyncio
    async def test_update_config_resets_prioritizer(self):
        """Updating config should reset the prioritizer so it picks up new values."""
        email_storage._prioritizer = MagicMock()

        with patch("aragora.server.handlers.email.get_email_store", return_value=None):
            await handle_update_config("reset_user", {"vip_domains": ["x.com"]})

        assert email_storage._prioritizer is None

        with _user_configs_lock:
            _user_configs.pop("reset_user", None)


# ===========================================================================
# VIP Management
# ===========================================================================


class TestVIPManagement:
    """Test VIP add/remove handlers."""

    @pytest.mark.asyncio
    async def test_add_vip_email(self, allowed_auth_context):
        with (
            patch("aragora.server.handlers.email.storage.check_permission") as mock_cp,
            patch("aragora.server.handlers.email.get_email_store", return_value=None),
        ):
            mock_cp.return_value = MagicMock(allowed=True)

            result = await handle_add_vip(
                user_id="vip_user",
                email="vip@example.com",
                auth_context=allowed_auth_context,
            )

            assert result["success"] is True
            assert result["added"]["email"] == "vip@example.com"
            assert "vip@example.com" in result["vip_addresses"]

        with _user_configs_lock:
            _user_configs.pop("vip_user", None)

    @pytest.mark.asyncio
    async def test_add_vip_domain(self, allowed_auth_context):
        with (
            patch("aragora.server.handlers.email.storage.check_permission") as mock_cp,
            patch("aragora.server.handlers.email.get_email_store", return_value=None),
        ):
            mock_cp.return_value = MagicMock(allowed=True)

            result = await handle_add_vip(
                user_id="vip_user2",
                domain="bigclient.com",
                auth_context=allowed_auth_context,
            )

            assert result["success"] is True
            assert "bigclient.com" in result["vip_domains"]

        with _user_configs_lock:
            _user_configs.pop("vip_user2", None)

    @pytest.mark.asyncio
    async def test_remove_vip_email(self, allowed_auth_context):
        with _user_configs_lock:
            _user_configs["rm_user"] = {"vip_addresses": ["remove@me.com"]}

        with (
            patch("aragora.server.handlers.email.storage.check_permission") as mock_cp,
            patch("aragora.server.handlers.email.get_email_store", return_value=None),
        ):
            mock_cp.return_value = MagicMock(allowed=True)

            result = await handle_remove_vip(
                user_id="rm_user",
                email="remove@me.com",
                auth_context=allowed_auth_context,
            )

            assert result["success"] is True
            assert result["removed"]["email"] == "remove@me.com"
            assert "remove@me.com" not in result["vip_addresses"]

        with _user_configs_lock:
            _user_configs.pop("rm_user", None)

    @pytest.mark.asyncio
    async def test_add_vip_denied_without_auth(self):
        result = await handle_add_vip(user_id="no_auth", email="x@x.com", auth_context=None)
        assert result["success"] is False
        assert "Permission denied" in result["error"]


# ===========================================================================
# Gmail OAuth
# ===========================================================================


class TestGmailOAuth:
    """Test Gmail OAuth endpoint handlers."""

    @pytest.mark.asyncio
    async def test_oauth_url_denied_without_auth(self):
        """email:oauth permission should fail closed without auth."""
        result = await handle_gmail_oauth_url(
            redirect_uri="https://app.example.com/callback",
            auth_context=None,
        )
        assert result["success"] is False
        assert "Permission denied" in result["error"]

    @pytest.mark.asyncio
    async def test_oauth_url_success(self, allowed_auth_context):
        with (
            patch("aragora.server.handlers.email.storage.check_permission") as mock_cp,
            patch("aragora.server.handlers.email.oauth.get_gmail_connector") as mock_gc,
        ):
            mock_cp.return_value = MagicMock(allowed=True)
            mock_connector = MagicMock()
            mock_connector.get_oauth_url.return_value = (
                "https://accounts.google.com/o/oauth2/auth?..."
            )
            mock_gc.return_value = mock_connector

            result = await handle_gmail_oauth_url(
                redirect_uri="https://app.example.com/callback",
                scopes="readonly",
                auth_context=allowed_auth_context,
            )

            assert result["success"] is True
            assert "oauth_url" in result
            assert result["scopes"] == "readonly"

    @pytest.mark.asyncio
    async def test_oauth_callback_denied_without_auth(self):
        result = await handle_gmail_oauth_callback(
            code="auth_code", redirect_uri="https://example.com/cb", auth_context=None
        )
        assert result["success"] is False

    @pytest.mark.asyncio
    async def test_oauth_callback_success(self, allowed_auth_context):
        with (
            patch("aragora.server.handlers.email.storage.check_permission") as mock_cp,
            patch("aragora.server.handlers.email.oauth.get_gmail_connector") as mock_gc,
        ):
            mock_cp.return_value = MagicMock(allowed=True)
            mock_connector = MagicMock()
            mock_connector.authenticate = AsyncMock()
            mock_connector.get_user_info = AsyncMock(
                return_value={"emailAddress": "user@gmail.com", "messagesTotal": 1234}
            )
            mock_gc.return_value = mock_connector

            result = await handle_gmail_oauth_callback(
                code="code123",
                redirect_uri="https://app.example.com/cb",
                auth_context=allowed_auth_context,
            )

            assert result["success"] is True
            assert result["authenticated"] is True
            assert result["email"] == "user@gmail.com"

    @pytest.mark.asyncio
    async def test_oauth_callback_error(self, allowed_auth_context):
        with (
            patch("aragora.server.handlers.email.storage.check_permission") as mock_cp,
            patch("aragora.server.handlers.email.oauth.get_gmail_connector") as mock_gc,
        ):
            mock_cp.return_value = MagicMock(allowed=True)
            mock_connector = MagicMock()
            mock_connector.authenticate = AsyncMock(side_effect=Exception("bad code"))
            mock_gc.return_value = mock_connector

            result = await handle_gmail_oauth_callback(
                code="bad",
                redirect_uri="https://x.com/cb",
                auth_context=allowed_auth_context,
            )
            assert result["success"] is False
            assert "bad code" in result["error"]


# ===========================================================================
# Gmail Status
# ===========================================================================


class TestGmailStatus:
    """Test Gmail connection status handler."""

    @pytest.mark.asyncio
    async def test_not_authenticated(self):
        with patch("aragora.server.handlers.email.oauth.get_gmail_connector") as mock_gc:
            mock_connector = MagicMock()
            mock_connector._access_token = None
            mock_gc.return_value = mock_connector

            result = await handle_gmail_status()
            assert result["success"] is True
            assert result["authenticated"] is False

    @pytest.mark.asyncio
    async def test_authenticated_with_user_info(self):
        with patch("aragora.server.handlers.email.oauth.get_gmail_connector") as mock_gc:
            mock_connector = MagicMock()
            mock_connector._access_token = "valid_token"
            mock_connector.get_user_info = AsyncMock(
                return_value={"emailAddress": "user@gmail.com", "messagesTotal": 500}
            )
            mock_gc.return_value = mock_connector

            result = await handle_gmail_status()
            assert result["success"] is True
            assert result["authenticated"] is True
            assert result["email"] == "user@gmail.com"

    @pytest.mark.asyncio
    async def test_token_expired(self):
        with patch("aragora.server.handlers.email.oauth.get_gmail_connector") as mock_gc:
            mock_connector = MagicMock()
            mock_connector._access_token = "expired"
            mock_connector.get_user_info = AsyncMock(side_effect=ConnectionError("token expired"))
            mock_gc.return_value = mock_connector

            result = await handle_gmail_status()
            assert result["success"] is True
            assert result["authenticated"] is False
            assert "expired" in result.get("error", "").lower()


# ===========================================================================
# Fetch and Rank Inbox
# ===========================================================================


class TestFetchAndRankInbox:
    """Test handle_fetch_and_rank_inbox."""

    @pytest.mark.asyncio
    async def test_not_authenticated(self):
        with patch("aragora.server.handlers.email.inbox.get_gmail_connector") as mock_gc:
            mock_connector = MagicMock()
            mock_connector._access_token = None
            mock_gc.return_value = mock_connector

            result = await handle_fetch_and_rank_inbox()
            assert result["success"] is False
            assert result["needs_auth"] is True

    @pytest.mark.asyncio
    async def test_success(self):
        mock_email = MagicMock()
        mock_email.id = "msg_1"
        mock_email.thread_id = "th_1"
        mock_email.subject = "Hello"
        mock_email.from_address = "a@a.com"
        mock_email.to_addresses = ["b@b.com"]
        mock_email.date = datetime.now()
        mock_email.snippet = "Hi there"
        mock_email.labels = ["INBOX"]
        mock_email.is_read = False
        mock_email.is_starred = False
        mock_email.is_important = False
        mock_email.attachments = []

        mock_rank = MagicMock()
        mock_rank.email_id = "msg_1"
        mock_rank.to_dict.return_value = {"email_id": "msg_1", "score": 80}

        with (
            patch("aragora.server.handlers.email.inbox.get_gmail_connector") as mock_gc,
            patch("aragora.server.handlers.email.inbox.get_prioritizer") as mock_gp,
        ):
            mock_connector = MagicMock()
            mock_connector._access_token = "valid"
            mock_connector.list_messages = AsyncMock(return_value=(["msg_1"], None))
            mock_connector.get_message = AsyncMock(return_value=mock_email)
            mock_gc.return_value = mock_connector

            mock_prioritizer = MagicMock()
            mock_prioritizer.rank_inbox = AsyncMock(return_value=[mock_rank])
            mock_gp.return_value = mock_prioritizer

            result = await handle_fetch_and_rank_inbox()
            assert result["success"] is True
            assert result["total"] == 1
            assert result["inbox"][0]["email"]["id"] == "msg_1"
            assert result["inbox"][0]["priority"]["score"] == 80


# ===========================================================================
# Cross-Channel Context
# ===========================================================================


class TestCrossChannelContext:
    """Test context and context boost handlers."""

    @pytest.mark.asyncio
    async def test_get_context_success(self):
        mock_ctx = MagicMock()
        mock_ctx.to_dict.return_value = {"slack": True, "drive": False}

        with patch("aragora.server.handlers.email.context.get_context_service") as mock_gs:
            mock_service = MagicMock()
            mock_service.get_user_context = AsyncMock(return_value=mock_ctx)
            mock_gs.return_value = mock_service

            result = await handle_get_context("user@example.com")
            assert result["success"] is True
            assert result["context"]["slack"] is True

    @pytest.mark.asyncio
    async def test_get_context_error(self):
        with patch("aragora.server.handlers.email.context.get_context_service") as mock_gs:
            mock_service = MagicMock()
            mock_service.get_user_context = AsyncMock(side_effect=Exception("service down"))
            mock_gs.return_value = mock_service

            result = await handle_get_context("user@example.com")
            assert result["success"] is False

    @pytest.mark.asyncio
    async def test_context_boost_success(self):
        mock_boost = MagicMock()
        mock_boost.email_id = "msg_1"
        mock_boost.total_boost = 0.5
        mock_boost.slack_activity_boost = 0.2
        mock_boost.drive_relevance_boost = 0.1
        mock_boost.calendar_urgency_boost = 0.2
        mock_boost.slack_reason = "active channel"
        mock_boost.drive_reason = "related doc"
        mock_boost.calendar_reason = "meeting soon"
        mock_boost.related_slack_channels = ["#proj"]
        mock_boost.related_drive_files = ["doc.pdf"]
        mock_boost.related_meetings = ["standup"]

        with patch("aragora.server.handlers.email.context.get_context_service") as mock_gs:
            mock_service = MagicMock()
            mock_service.get_email_context = AsyncMock(return_value=mock_boost)
            mock_gs.return_value = mock_service

            result = await handle_get_email_context_boost(
                {"id": "msg_1", "from_address": "a@a.com"}
            )
            assert result["success"] is True
            assert result["boost"]["total_boost"] == 0.5
            assert result["boost"]["related_slack_channels"] == ["#proj"]


# ===========================================================================
# Categorization
# ===========================================================================


class TestCategorization:
    """Test email categorization handlers."""

    @pytest.mark.asyncio
    async def test_categorize_single_success(self, sample_email_data):
        mock_cat_result = MagicMock()
        mock_cat_result.to_dict.return_value = {
            "category": "work",
            "confidence": 0.92,
        }

        with patch("aragora.server.handlers.email.categorization.get_categorizer") as mock_gc:
            mock_categorizer = MagicMock()
            mock_categorizer.categorize_email = AsyncMock(return_value=mock_cat_result)
            mock_gc.return_value = mock_categorizer

            result = await handle_categorize_email(sample_email_data)
            assert result["success"] is True
            assert result["result"]["category"] == "work"

    @pytest.mark.asyncio
    async def test_categorize_batch_success(self):
        mock_r = MagicMock()
        mock_r.to_dict.return_value = {"category": "invoices"}

        with patch("aragora.server.handlers.email.categorization.get_categorizer") as mock_gc:
            mock_categorizer = MagicMock()
            mock_categorizer.categorize_batch = AsyncMock(return_value=[mock_r])
            mock_categorizer.get_category_stats.return_value = {"invoices": 1}
            mock_gc.return_value = mock_categorizer

            emails = [{"id": "1", "subject": "Invoice #100"}]
            result = await handle_categorize_batch(emails)

            assert result["success"] is True
            assert result["stats"]["invoices"] == 1

    @pytest.mark.asyncio
    async def test_apply_category_label_success(self, allowed_auth_context):
        with (
            patch("aragora.server.handlers.email.storage.check_permission") as mock_cp,
            patch("aragora.server.handlers.email.categorization.get_categorizer") as mock_gc,
        ):
            mock_cp.return_value = MagicMock(allowed=True)
            mock_categorizer = MagicMock()
            mock_categorizer.apply_gmail_label = AsyncMock(return_value=True)
            mock_gc.return_value = mock_categorizer

            result = await handle_apply_category_label(
                "msg_1", "invoices", auth_context=allowed_auth_context
            )
            assert result["success"] is True
            assert result["label_applied"] is True

    @pytest.mark.asyncio
    async def test_apply_category_label_invalid_category(self, allowed_auth_context):
        with (
            patch("aragora.server.handlers.email.storage.check_permission") as mock_cp,
            patch("aragora.server.handlers.email.categorization.get_categorizer") as mock_gc,
            patch(
                "aragora.services.email_categorizer.EmailCategory",
                side_effect=ValueError("bad category"),
            ),
        ):
            mock_cp.return_value = MagicMock(allowed=True)
            mock_categorizer = MagicMock()
            mock_gc.return_value = mock_categorizer

            result = await handle_apply_category_label(
                "msg_1", "DEFINITELY_NOT_A_CATEGORY", auth_context=allowed_auth_context
            )
            assert result["success"] is False
            assert "Invalid category" in result.get("error", "")


# ===========================================================================
# Batch Feedback
# ===========================================================================


class TestFeedbackBatch:
    """Test batch feedback handler."""

    @pytest.mark.asyncio
    async def test_success(self, allowed_auth_context):
        with (
            patch("aragora.server.handlers.email.storage.check_permission") as mock_cp,
            patch("aragora.server.handlers.email.prioritization.get_prioritizer") as mock_gp,
        ):
            mock_cp.return_value = MagicMock(allowed=True)
            mock_prioritizer = MagicMock()
            mock_prioritizer.record_user_action = AsyncMock()
            mock_gp.return_value = mock_prioritizer

            items = [
                {"email_id": "msg_1", "action": "archived"},
                {"email_id": "msg_2", "action": "replied", "response_time_minutes": 5},
            ]

            result = await handle_feedback_batch(items, auth_context=allowed_auth_context)
            assert result["success"] is True
            assert result["recorded"] == 2
            assert result["errors"] == 0

    @pytest.mark.asyncio
    async def test_missing_fields_in_items(self, allowed_auth_context):
        with (
            patch("aragora.server.handlers.email.storage.check_permission") as mock_cp,
            patch("aragora.server.handlers.email.prioritization.get_prioritizer") as mock_gp,
        ):
            mock_cp.return_value = MagicMock(allowed=True)
            mock_prioritizer = MagicMock()
            mock_prioritizer.record_user_action = AsyncMock()
            mock_gp.return_value = mock_prioritizer

            items = [
                {"email_id": "msg_1"},  # Missing action
                {"action": "archived"},  # Missing email_id
                {"email_id": "msg_3", "action": "starred"},  # Valid
            ]

            result = await handle_feedback_batch(items, auth_context=allowed_auth_context)
            assert result["success"] is True
            assert result["recorded"] == 1
            assert result["errors"] == 2

    @pytest.mark.asyncio
    async def test_denied_without_auth(self):
        result = await handle_feedback_batch(
            [{"email_id": "1", "action": "read"}], auth_context=None
        )
        assert result["success"] is False


# ===========================================================================
# RBAC Permission Check Helper
# ===========================================================================


class TestCheckEmailPermission:
    """Test the _check_email_permission helper."""

    def test_read_degrades_gracefully_without_rbac(self):
        """Read-only permissions should pass when RBAC is unavailable."""
        result = _check_email_permission(None, "email:read")
        assert result is None  # Allowed

    def test_write_fails_closed_without_rbac(self):
        result = _check_email_permission(None, "email:write")
        assert result is not None
        assert result["success"] is False

    def test_oauth_fails_closed_without_rbac(self):
        result = _check_email_permission(None, "email:oauth")
        assert result is not None
        assert result["success"] is False


# ===========================================================================
# Lazy Initialization
# ===========================================================================


class TestLazyInitialization:
    """Test lazy initialization of service singletons."""

    def test_gmail_connector_lazy_init(self):
        email_storage._gmail_connector = None

        with patch("aragora.connectors.enterprise.communication.gmail.GmailConnector") as mock_cls:
            mock_cls.return_value = MagicMock()
            connector = get_gmail_connector("user1")
            assert connector is not None

    def test_context_service_lazy_init(self):
        email_storage._context_service = None

        with patch("aragora.services.cross_channel_context.CrossChannelContextService") as mock_cls:
            mock_cls.return_value = MagicMock()
            svc = get_context_service()
            assert svc is not None

    def test_gmail_connector_reuses_singleton(self):
        sentinel = MagicMock()
        email_storage._gmail_connector = sentinel
        assert get_gmail_connector() is sentinel


# ===========================================================================
# Handler Class Methods
# ===========================================================================


class TestEmailHandlerMethods:
    """Test EmailHandler async handler methods (class-based routing)."""

    @pytest.fixture
    def handler_with_user(self):
        ctx = {"auth_context": None}
        h = EmailHandler(ctx)
        h._get_user_id = MagicMock(return_value="test_user")
        h._get_auth_context = MagicMock(return_value=None)
        return h

    @pytest.mark.asyncio
    async def test_post_prioritize_success(self, handler_with_user):
        data = {"email": {"id": "msg_1", "subject": "Hello"}}

        with patch(
            "aragora.server.handlers.email.handler.handle_prioritize_email",
            new_callable=AsyncMock,
        ) as mock_fn:
            mock_fn.return_value = {"success": True, "result": {"score": 70}}
            result = await handler_with_user.handle_post_prioritize(data)
            assert result.status_code == 200

    @pytest.mark.asyncio
    async def test_post_prioritize_failure(self, handler_with_user):
        data = {"email": {"id": "msg_1"}}

        with patch(
            "aragora.server.handlers.email.handler.handle_prioritize_email",
            new_callable=AsyncMock,
        ) as mock_fn:
            mock_fn.return_value = {"success": False, "error": "boom"}
            result = await handler_with_user.handle_post_prioritize(data)
            assert result.status_code == 400

    @pytest.mark.asyncio
    async def test_post_feedback_missing_email_id(self, handler_with_user):
        result = await handler_with_user.handle_post_feedback({"action": "read"})
        assert result.status_code == 400

    @pytest.mark.asyncio
    async def test_post_feedback_missing_action(self, handler_with_user):
        result = await handler_with_user.handle_post_feedback({"email_id": "msg_1"})
        assert result.status_code == 400

    @pytest.mark.asyncio
    async def test_post_feedback_batch_empty(self, handler_with_user):
        result = await handler_with_user.handle_post_feedback_batch({"items": []})
        assert result.status_code == 400

    @pytest.mark.asyncio
    async def test_post_gmail_oauth_url_missing_redirect(self, handler_with_user):
        result = await handler_with_user.handle_post_gmail_oauth_url({})
        assert result.status_code == 400

    @pytest.mark.asyncio
    async def test_post_gmail_oauth_callback_missing_fields(self, handler_with_user):
        result = await handler_with_user.handle_post_gmail_oauth_callback({"code": "abc"})
        assert result.status_code == 400

        result2 = await handler_with_user.handle_post_gmail_oauth_callback(
            {"redirect_uri": "https://x.com/cb"}
        )
        assert result2.status_code == 400

    @pytest.mark.asyncio
    async def test_post_categorize_apply_label_missing_fields(self, handler_with_user):
        result = await handler_with_user.handle_post_categorize_apply_label({"email_id": "msg_1"})
        assert result.status_code == 400

        result2 = await handler_with_user.handle_post_categorize_apply_label(
            {"category": "invoices"}
        )
        assert result2.status_code == 400

    @pytest.mark.asyncio
    async def test_post_categorize_batch_empty(self, handler_with_user):
        result = await handler_with_user.handle_post_categorize_batch({"emails": []})
        assert result.status_code == 400

    @pytest.mark.asyncio
    async def test_get_inbox_needs_auth_returns_401(self, handler_with_user):
        with patch(
            "aragora.server.handlers.email.handler.handle_fetch_and_rank_inbox",
            new_callable=AsyncMock,
        ) as mock_fn:
            mock_fn.return_value = {
                "success": False,
                "needs_auth": True,
                "error": "Not authenticated",
            }
            result = await handler_with_user.handle_get_inbox({})
            assert result.status_code == 401

    @pytest.mark.asyncio
    async def test_get_user_id_from_auth_context(self):
        auth_ctx = MagicMock()
        auth_ctx.user_id = "real_user_42"
        h = EmailHandler({"auth_context": auth_ctx})
        assert h._get_user_id() == "real_user_42"

    @pytest.mark.asyncio
    async def test_get_user_id_default_without_context(self):
        h = EmailHandler({})
        assert h._get_user_id() == "default"


# ===========================================================================
# Thread Safety
# ===========================================================================


class TestThreadSafety:
    """Test thread-safety of global state."""

    def test_user_configs_lock_exists(self):
        import threading

        assert isinstance(_user_configs_lock, type(threading.Lock()))

    def test_lock_can_acquire_and_release(self):
        with _user_configs_lock:
            _ = _user_configs.get("nonexistent", {})
        assert _user_configs_lock.acquire(blocking=False)
        _user_configs_lock.release()
