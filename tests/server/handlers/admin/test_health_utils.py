"""Tests for health utility functions."""
import sys
import types as _types_mod

# Pre-stub Slack modules to prevent import chain failures
_SLACK_ATTRS = [
    "SlackHandler", "get_slack_handler", "get_slack_integration",
    "get_workspace_store", "resolve_workspace", "create_tracked_task",
    "_validate_slack_url", "SLACK_SIGNING_SECRET", "SLACK_BOT_TOKEN",
    "SLACK_WEBHOOK_URL", "SLACK_ALLOWED_DOMAINS", "SignatureVerifierMixin",
    "CommandsMixin", "EventsMixin", "init_slack_handler",
]
for _mod_name in (
    "aragora.server.handlers.social.slack.handler",
    "aragora.server.handlers.social.slack",
    "aragora.server.handlers.social._slack_impl",
):
    if _mod_name not in sys.modules:
        _m = _types_mod.ModuleType(_mod_name)
        for _a in _SLACK_ATTRS:
            setattr(_m, _a, None)
        sys.modules[_mod_name] = _m

import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def clear_env():
    """Clear environment variables between tests."""
    yield


class TestCheckFilesystemHealth:
    """Tests for check_filesystem_health function."""

    def test_filesystem_healthy(self, tmp_path):
        """Test filesystem healthy when writable."""
        from aragora.server.handlers.admin.health_utils import check_filesystem_health

        result = check_filesystem_health(tmp_path)

        assert result["healthy"] is True
        assert "path" in result

    def test_filesystem_default_to_temp(self):
        """Test filesystem defaults to temp dir when none provided."""
        from aragora.server.handlers.admin.health_utils import check_filesystem_health

        result = check_filesystem_health(None)

        assert result["healthy"] is True

    def test_filesystem_permission_denied(self, tmp_path):
        """Test filesystem unhealthy on permission error."""
        from aragora.server.handlers.admin.health_utils import check_filesystem_health

        with patch("pathlib.Path.write_text", side_effect=PermissionError("Permission denied")):
            result = check_filesystem_health(tmp_path)

        assert result["healthy"] is False
        assert "Permission denied" in result["error"]


class TestCheckRedisHealth:
    """Tests for check_redis_health function."""

    def test_redis_not_configured(self):
        """Test Redis not configured returns healthy."""
        from aragora.server.handlers.admin.health_utils import check_redis_health

        with patch.dict("os.environ", {}, clear=True):
            result = check_redis_health()

        assert result["healthy"] is True
        assert result["configured"] is False

    def test_redis_configured_and_healthy(self):
        """Test Redis healthy when ping succeeds."""
        from aragora.server.handlers.admin.health_utils import check_redis_health

        mock_client = MagicMock()
        mock_client.ping.return_value = True

        with patch.dict("os.environ", {"REDIS_URL": "redis://localhost:6379"}):
            with patch("redis.from_url", return_value=mock_client):
                result = check_redis_health()

        assert result["healthy"] is True
        assert result["configured"] is True
        assert "latency_ms" in result

    def test_redis_connection_error(self):
        """Test Redis unhealthy on connection error."""
        from aragora.server.handlers.admin.health_utils import check_redis_health

        with patch.dict("os.environ", {"REDIS_URL": "redis://localhost:6379"}):
            with patch("redis.from_url", side_effect=ConnectionError("Connection refused")):
                result = check_redis_health()

        assert result["healthy"] is False
        assert result["configured"] is True


class TestCheckAiProvidersHealth:
    """Tests for check_ai_providers_health function."""

    def test_no_providers_configured(self):
        """Test no providers configured."""
        from aragora.server.handlers.admin.health_utils import check_ai_providers_health

        with patch.dict("os.environ", {}, clear=True):
            result = check_ai_providers_health()

        assert result["healthy"] is True
        assert result["any_available"] is False
        assert result["available_count"] == 0

    def test_anthropic_configured(self):
        """Test Anthropic provider configured."""
        from aragora.server.handlers.admin.health_utils import check_ai_providers_health

        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "sk-ant-1234567890"}):
            result = check_ai_providers_health()

        assert result["any_available"] is True
        assert result["providers"]["anthropic"] is True

    def test_multiple_providers_configured(self):
        """Test multiple providers configured."""
        from aragora.server.handlers.admin.health_utils import check_ai_providers_health

        with patch.dict("os.environ", {
            "ANTHROPIC_API_KEY": "sk-ant-1234567890",
            "OPENAI_API_KEY": "sk-1234567890",
        }):
            result = check_ai_providers_health()

        assert result["available_count"] == 2
        assert result["providers"]["anthropic"] is True
        assert result["providers"]["openai"] is True


class TestCheckSecurityServices:
    """Tests for check_security_services function."""

    def test_security_services_available(self):
        """Test security services available."""
        from aragora.server.handlers.admin.health_utils import check_security_services

        mock_service = MagicMock()

        with patch(
            "aragora.server.handlers.admin.health_utils.get_encryption_service",
            return_value=mock_service,
        ), patch(
            "aragora.server.handlers.admin.health_utils.get_secret",
            return_value="encryption_key_value",
        ):
            result = check_security_services()

        assert result["healthy"] is True
        assert result["encryption_available"] is True

    def test_security_encryption_not_configured_production(self):
        """Test encryption warning in production without key."""
        from aragora.server.handlers.admin.health_utils import check_security_services

        with patch(
            "aragora.server.handlers.admin.health_utils.get_encryption_service",
            return_value=MagicMock(),
        ), patch(
            "aragora.server.handlers.admin.health_utils.get_secret",
            return_value=None,
        ):
            result = check_security_services(is_production=True)

        assert result["healthy"] is False
        assert "encryption_warning" in result


class TestCheckDatabaseHealth:
    """Tests for check_database_health function."""

    def test_database_not_configured(self):
        """Test database not configured."""
        from aragora.server.handlers.admin.health_utils import check_database_health

        with patch.dict("os.environ", {}, clear=True):
            result = check_database_health()

        assert result["healthy"] is True
        assert result["configured"] is False


class TestGetUptimeInfo:
    """Tests for get_uptime_info function."""

    def test_uptime_seconds(self):
        """Test uptime in seconds."""
        from aragora.server.handlers.admin.health_utils import get_uptime_info

        start_time = time.time() - 30  # 30 seconds ago

        result = get_uptime_info(start_time)

        assert result["uptime_seconds"] >= 30
        assert "uptime_human" in result

    def test_uptime_minutes(self):
        """Test uptime in minutes."""
        from aragora.server.handlers.admin.health_utils import get_uptime_info

        start_time = time.time() - 300  # 5 minutes ago

        result = get_uptime_info(start_time)

        assert "m" in result["uptime_human"]

    def test_uptime_hours(self):
        """Test uptime in hours."""
        from aragora.server.handlers.admin.health_utils import get_uptime_info

        start_time = time.time() - 7200  # 2 hours ago

        result = get_uptime_info(start_time)

        assert "h" in result["uptime_human"]

    def test_uptime_days(self):
        """Test uptime in days."""
        from aragora.server.handlers.admin.health_utils import get_uptime_info

        start_time = time.time() - 172800  # 2 days ago

        result = get_uptime_info(start_time)

        assert "d" in result["uptime_human"]
