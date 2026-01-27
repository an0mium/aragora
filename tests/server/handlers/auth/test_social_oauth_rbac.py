"""
Tests for social OAuth handler RBAC enforcement.

Tests cover:
- Install endpoints require connector.create permission
- Callback endpoints are public (OAuth flow)
- Uninstall endpoints require connector.delete permission
- Permission format uses dots (connectors.authorize)
"""

import pytest


class TestSlackOAuthRBAC:
    """Tests for Slack OAuth RBAC."""

    def test_slack_permission_format_uses_dots(self):
        """Slack handler should use dot-separated permission format."""
        from aragora.server.handlers.social.slack_oauth import (
            CONNECTOR_READ,
            CONNECTOR_AUTHORIZE,
        )

        assert "." in CONNECTOR_READ, f"Expected dot format, got: {CONNECTOR_READ}"
        assert "." in CONNECTOR_AUTHORIZE, f"Expected dot format, got: {CONNECTOR_AUTHORIZE}"
        assert CONNECTOR_READ == "connectors.read"
        assert CONNECTOR_AUTHORIZE == "connectors.authorize"

    def test_middleware_protects_slack_install(self):
        """Middleware should protect Slack install with connector.create."""
        from aragora.rbac.middleware import DEFAULT_ROUTE_PERMISSIONS

        install_routes = [
            rp
            for rp in DEFAULT_ROUTE_PERMISSIONS
            if "slack/install" in rp.pattern.pattern and rp.permission_key
        ]
        assert len(install_routes) > 0, "Slack install should be protected"
        assert any("connector" in rp.permission_key for rp in install_routes), (
            "Should require connector permission"
        )

    def test_middleware_marks_slack_callback_as_public(self):
        """Middleware should mark Slack callback as public."""
        from aragora.rbac.middleware import DEFAULT_ROUTE_PERMISSIONS

        callback_routes = [
            rp
            for rp in DEFAULT_ROUTE_PERMISSIONS
            if "slack/callback" in rp.pattern.pattern and rp.allow_unauthenticated
        ]
        assert len(callback_routes) > 0, "Slack callback should be public"

    def test_middleware_protects_slack_uninstall(self):
        """Middleware should protect Slack uninstall with connector.delete."""
        from aragora.rbac.middleware import DEFAULT_ROUTE_PERMISSIONS

        uninstall_routes = [
            rp
            for rp in DEFAULT_ROUTE_PERMISSIONS
            if "slack/uninstall" in rp.pattern.pattern and rp.permission_key
        ]
        assert len(uninstall_routes) > 0, "Slack uninstall should be protected"


class TestTeamsOAuthRBAC:
    """Tests for Teams OAuth RBAC."""

    def test_teams_permission_format_uses_dots(self):
        """Teams handler should use dot-separated permission format."""
        from aragora.server.handlers.social.teams_oauth import CONNECTOR_AUTHORIZE

        assert "." in CONNECTOR_AUTHORIZE, f"Expected dot format, got: {CONNECTOR_AUTHORIZE}"
        assert CONNECTOR_AUTHORIZE == "connectors.authorize"

    def test_middleware_protects_teams_install(self):
        """Middleware should protect Teams install with connector.create."""
        from aragora.rbac.middleware import DEFAULT_ROUTE_PERMISSIONS

        install_routes = [
            rp
            for rp in DEFAULT_ROUTE_PERMISSIONS
            if "teams/install" in rp.pattern.pattern and rp.permission_key
        ]
        assert len(install_routes) > 0, "Teams install should be protected"

    def test_middleware_marks_teams_callback_as_public(self):
        """Middleware should mark Teams callback as public."""
        from aragora.rbac.middleware import DEFAULT_ROUTE_PERMISSIONS

        callback_routes = [
            rp
            for rp in DEFAULT_ROUTE_PERMISSIONS
            if "teams/callback" in rp.pattern.pattern and rp.allow_unauthenticated
        ]
        assert len(callback_routes) > 0, "Teams callback should be public"


class TestDiscordOAuthRBAC:
    """Tests for Discord OAuth RBAC."""

    def test_discord_permission_format_uses_dots(self):
        """Discord handler should use dot-separated permission format."""
        from aragora.server.handlers.social.discord_oauth import (
            CONNECTOR_READ,
            CONNECTOR_AUTHORIZE,
        )

        assert "." in CONNECTOR_READ, f"Expected dot format, got: {CONNECTOR_READ}"
        assert "." in CONNECTOR_AUTHORIZE, f"Expected dot format, got: {CONNECTOR_AUTHORIZE}"
        assert CONNECTOR_READ == "connectors.read"
        assert CONNECTOR_AUTHORIZE == "connectors.authorize"

    def test_middleware_protects_discord_install(self):
        """Middleware should protect Discord install with connector.create."""
        from aragora.rbac.middleware import DEFAULT_ROUTE_PERMISSIONS

        install_routes = [
            rp
            for rp in DEFAULT_ROUTE_PERMISSIONS
            if "discord/install" in rp.pattern.pattern and rp.permission_key
        ]
        assert len(install_routes) > 0, "Discord install should be protected"

    def test_middleware_marks_discord_callback_as_public(self):
        """Middleware should mark Discord callback as public."""
        from aragora.rbac.middleware import DEFAULT_ROUTE_PERMISSIONS

        callback_routes = [
            rp
            for rp in DEFAULT_ROUTE_PERMISSIONS
            if "discord/callback" in rp.pattern.pattern and rp.allow_unauthenticated
        ]
        assert len(callback_routes) > 0, "Discord callback should be public"

    def test_middleware_protects_discord_uninstall(self):
        """Middleware should protect Discord uninstall with connector.delete."""
        from aragora.rbac.middleware import DEFAULT_ROUTE_PERMISSIONS

        uninstall_routes = [
            rp
            for rp in DEFAULT_ROUTE_PERMISSIONS
            if "discord/uninstall" in rp.pattern.pattern and rp.permission_key
        ]
        assert len(uninstall_routes) > 0, "Discord uninstall should be protected"


class TestOAuthMiddlewareConsistency:
    """Tests for consistent RBAC configuration across social OAuth handlers."""

    def test_all_social_installs_protected(self):
        """All social integration installs should be protected."""
        from aragora.rbac.middleware import DEFAULT_ROUTE_PERMISSIONS

        social_installs = [
            rp
            for rp in DEFAULT_ROUTE_PERMISSIONS
            if any(
                social in rp.pattern.pattern
                for social in ["slack/install", "teams/install", "discord/install"]
            )
        ]

        for route in social_installs:
            assert route.permission_key, f"Install route {route.pattern} should be protected"
            assert "connector" in route.permission_key, (
                f"Install route should use connector permission: {route.pattern}"
            )

    def test_all_social_callbacks_public(self):
        """All social integration callbacks should be public (OAuth redirect)."""
        from aragora.rbac.middleware import DEFAULT_ROUTE_PERMISSIONS

        social_callbacks = [
            rp
            for rp in DEFAULT_ROUTE_PERMISSIONS
            if any(
                social in rp.pattern.pattern
                for social in ["slack/callback", "teams/callback", "discord/callback"]
            )
        ]

        for route in social_callbacks:
            assert route.allow_unauthenticated, f"Callback route {route.pattern} should be public"

    def test_permission_format_consistency(self):
        """All connector permissions should use dot format."""
        from aragora.rbac.middleware import DEFAULT_ROUTE_PERMISSIONS

        connector_routes = [
            rp for rp in DEFAULT_ROUTE_PERMISSIONS if "connector" in (rp.permission_key or "")
        ]

        for route in connector_routes:
            assert "." in route.permission_key, (
                f"Permission should use dots: {route.permission_key}"
            )
            assert ":" not in route.permission_key, (
                f"Permission should not use colons: {route.permission_key}"
            )
