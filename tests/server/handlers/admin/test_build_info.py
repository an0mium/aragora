"""
Tests for build info and deploy status endpoints.

Covers:
- /health/build endpoint returns SHA, build_time, version
- SHA verification via ?verify= query parameter
- Environment variable override behavior
- Git fallback when env vars not set
- Deploy status endpoint (authenticated)
- BuildInfoHandler route matching
- DeployStatusHandler RBAC enforcement
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# build_info module tests
# ---------------------------------------------------------------------------

class TestGetBuildInfo:
    """Tests for aragora.server.build_info.get_build_info."""

    def test_returns_dict_with_required_keys(self):
        from aragora.server.build_info import get_build_info

        # Clear the lru_cache so env overrides take effect
        get_build_info.cache_clear()

        info = get_build_info()
        assert "sha" in info
        assert "build_time" in info
        assert "version" in info
        assert "sha_short" in info

    def test_env_vars_take_priority(self):
        from aragora.server.build_info import get_build_info

        get_build_info.cache_clear()

        with patch.dict(
            "os.environ",
            {
                "ARAGORA_BUILD_SHA": "abc123def456",
                "ARAGORA_BUILD_TIME": "2026-01-01T00:00:00Z",
                "ARAGORA_DEPLOY_VERSION": "v1.2.3",
            },
        ):
            # Need to re-import to pick up env vars since module-level reads happen at import
            import importlib
            import aragora.server.build_info as bi

            # Patch module-level vars directly
            original_sha = bi._BUILD_SHA
            original_time = bi._BUILD_TIME
            original_ver = bi._DEPLOY_VERSION
            try:
                bi._BUILD_SHA = "abc123def456"
                bi._BUILD_TIME = "2026-01-01T00:00:00Z"
                bi._DEPLOY_VERSION = "v1.2.3"
                get_build_info.cache_clear()

                info = get_build_info()
                assert info["sha"] == "abc123def456"
                assert info["build_time"] == "2026-01-01T00:00:00Z"
                assert info["version"] == "v1.2.3"
                assert info["sha_short"] == "abc123de"
            finally:
                bi._BUILD_SHA = original_sha
                bi._BUILD_TIME = original_time
                bi._DEPLOY_VERSION = original_ver
                get_build_info.cache_clear()

    def test_sha_short_is_8_chars(self):
        from aragora.server.build_info import get_build_info

        get_build_info.cache_clear()

        import aragora.server.build_info as bi
        original = bi._BUILD_SHA
        try:
            bi._BUILD_SHA = "abcdef1234567890abcdef1234567890abcdef12"
            get_build_info.cache_clear()
            info = get_build_info()
            assert info["sha_short"] == "abcdef12"
            assert len(info["sha_short"]) == 8
        finally:
            bi._BUILD_SHA = original
            get_build_info.cache_clear()

    def test_unknown_sha_gives_unknown_short(self):
        from aragora.server.build_info import get_build_info

        get_build_info.cache_clear()

        import aragora.server.build_info as bi
        original = bi._BUILD_SHA
        try:
            bi._BUILD_SHA = ""
            get_build_info.cache_clear()

            with patch.object(bi, "_git_sha_fallback", return_value="unknown"):
                get_build_info.cache_clear()
                info = get_build_info()
                assert info["sha_short"] == "unknown"
        finally:
            bi._BUILD_SHA = original
            get_build_info.cache_clear()

    def test_git_fallback_on_empty_env(self):
        from aragora.server.build_info import get_build_info

        get_build_info.cache_clear()

        import aragora.server.build_info as bi
        original = bi._BUILD_SHA
        try:
            bi._BUILD_SHA = ""
            get_build_info.cache_clear()

            with patch.object(
                bi, "_git_sha_fallback", return_value="fedcba9876543210"
            ):
                get_build_info.cache_clear()
                info = get_build_info()
                assert info["sha"] == "fedcba9876543210"
        finally:
            bi._BUILD_SHA = original
            get_build_info.cache_clear()


class TestVerifySha:
    """Tests for aragora.server.build_info.verify_sha."""

    def test_matching_full_sha(self):
        from aragora.server.build_info import get_build_info, verify_sha

        get_build_info.cache_clear()

        import aragora.server.build_info as bi
        original = bi._BUILD_SHA
        try:
            bi._BUILD_SHA = "abc123def456"
            get_build_info.cache_clear()

            result = verify_sha("abc123def456")
            assert result["matches"] is True
            assert result["current"] == "abc123def456"
        finally:
            bi._BUILD_SHA = original
            get_build_info.cache_clear()

    def test_matching_short_sha(self):
        from aragora.server.build_info import get_build_info, verify_sha

        get_build_info.cache_clear()

        import aragora.server.build_info as bi
        original = bi._BUILD_SHA
        try:
            bi._BUILD_SHA = "abc123def456789"
            get_build_info.cache_clear()

            result = verify_sha("abc123de")
            assert result["matches"] is True
        finally:
            bi._BUILD_SHA = original
            get_build_info.cache_clear()

    def test_non_matching_sha(self):
        from aragora.server.build_info import get_build_info, verify_sha

        get_build_info.cache_clear()

        import aragora.server.build_info as bi
        original = bi._BUILD_SHA
        try:
            bi._BUILD_SHA = "abc123"
            get_build_info.cache_clear()

            result = verify_sha("zzz999")
            assert result["matches"] is False
        finally:
            bi._BUILD_SHA = original
            get_build_info.cache_clear()

    def test_unknown_sha_never_matches(self):
        from aragora.server.build_info import get_build_info, verify_sha

        get_build_info.cache_clear()

        import aragora.server.build_info as bi
        original = bi._BUILD_SHA
        try:
            bi._BUILD_SHA = ""
            get_build_info.cache_clear()

            with patch.object(bi, "_git_sha_fallback", return_value="unknown"):
                get_build_info.cache_clear()
                result = verify_sha("abc123")
                assert result["matches"] is False
        finally:
            bi._BUILD_SHA = original
            get_build_info.cache_clear()

    def test_empty_expected_sha(self):
        from aragora.server.build_info import get_build_info, verify_sha

        get_build_info.cache_clear()

        import aragora.server.build_info as bi
        original = bi._BUILD_SHA
        try:
            bi._BUILD_SHA = "abc123"
            get_build_info.cache_clear()

            result = verify_sha("")
            assert result["matches"] is False
        finally:
            bi._BUILD_SHA = original
            get_build_info.cache_clear()


# ---------------------------------------------------------------------------
# BuildInfoHandler tests
# ---------------------------------------------------------------------------

class TestBuildInfoHandler:
    """Tests for BuildInfoHandler endpoint."""

    def _make_handler(self):
        from aragora.server.handlers.admin.health.build import BuildInfoHandler

        return BuildInfoHandler(ctx={})

    def test_can_handle_routes(self):
        handler = self._make_handler()
        assert handler.can_handle("/health/build")
        assert handler.can_handle("/api/health/build")
        assert handler.can_handle("/api/v1/health/build")
        assert not handler.can_handle("/health")
        assert not handler.can_handle("/api/health")

    @pytest.mark.asyncio
    async def test_handle_returns_build_info(self):
        handler = self._make_handler()

        from aragora.server.build_info import get_build_info

        get_build_info.cache_clear()

        import aragora.server.build_info as bi
        original = bi._BUILD_SHA
        try:
            bi._BUILD_SHA = "testsha123456"
            get_build_info.cache_clear()

            result = await handler.handle("/health/build", {}, MagicMock())
            assert result is not None
            assert result.status_code == 200
            body = json.loads(result.body.decode("utf-8"))
            assert body["sha"] == "testsha123456"
            assert body["sha_short"] == "testsha1"
            assert "build_time" in body
            assert "version" in body
        finally:
            bi._BUILD_SHA = original
            get_build_info.cache_clear()

    @pytest.mark.asyncio
    async def test_handle_with_verify_param(self):
        handler = self._make_handler()

        from aragora.server.build_info import get_build_info

        get_build_info.cache_clear()

        import aragora.server.build_info as bi
        original = bi._BUILD_SHA
        try:
            bi._BUILD_SHA = "abc123def456"
            get_build_info.cache_clear()

            result = await handler.handle(
                "/health/build", {"verify": "abc123de"}, MagicMock()
            )
            assert result is not None
            body = json.loads(result.body.decode("utf-8"))
            assert "verification" in body
            assert body["verification"]["matches"] is True
        finally:
            bi._BUILD_SHA = original
            get_build_info.cache_clear()

    @pytest.mark.asyncio
    async def test_handle_unmatched_path_returns_none(self):
        handler = self._make_handler()
        result = await handler.handle("/some/other/path", {}, MagicMock())
        assert result is None

    def test_routes_are_public(self):
        handler = self._make_handler()
        for route in handler.ROUTES:
            assert route in handler.PUBLIC_ROUTES


# ---------------------------------------------------------------------------
# DeployStatusHandler tests
# ---------------------------------------------------------------------------

class TestDeployStatusHandler:
    """Tests for DeployStatusHandler endpoint."""

    def _make_handler(self):
        from aragora.server.handlers.admin.health.deploy_status import (
            DeployStatusHandler,
        )

        return DeployStatusHandler(ctx={})

    def test_can_handle_routes(self):
        handler = self._make_handler()
        assert handler.can_handle("/api/deploy/status")
        assert handler.can_handle("/api/v1/deploy/status")
        assert not handler.can_handle("/api/health")

    @pytest.mark.asyncio
    async def test_unauthenticated_returns_401(self):
        from aragora.server.handlers.utils.auth import UnauthorizedError

        handler = self._make_handler()

        with patch.object(
            handler, "get_auth_context", side_effect=UnauthorizedError("no token")
        ):
            result = await handler.handle(
                "/api/deploy/status", {}, MagicMock()
            )
            assert result is not None
            assert result.status_code == 401

    @pytest.mark.asyncio
    async def test_forbidden_returns_403(self):
        from aragora.server.handlers.utils.auth import ForbiddenError

        handler = self._make_handler()

        mock_ctx = MagicMock()
        with patch.object(
            handler, "get_auth_context", return_value=mock_ctx
        ), patch.object(
            handler, "check_permission", side_effect=ForbiddenError("denied")
        ):
            result = await handler.handle(
                "/api/v1/deploy/status", {}, MagicMock()
            )
            assert result is not None
            assert result.status_code == 403

    @pytest.mark.asyncio
    async def test_authenticated_returns_deploy_info(self):
        handler = self._make_handler()

        from aragora.server.build_info import get_build_info

        get_build_info.cache_clear()

        import aragora.server.build_info as bi
        original = bi._BUILD_SHA
        try:
            bi._BUILD_SHA = "deploy123abc"
            get_build_info.cache_clear()

            mock_ctx = MagicMock()
            with patch.object(
                handler, "get_auth_context", return_value=mock_ctx
            ), patch.object(handler, "check_permission"):
                result = await handler.handle(
                    "/api/deploy/status", {}, MagicMock()
                )
                assert result is not None
                assert result.status_code == 200
                body = json.loads(result.body.decode("utf-8"))
                assert body["deploy"]["sha"] == "deploy123abc"
                assert "health" in body
                assert "uptime" in body
                assert "timestamp" in body
        finally:
            bi._BUILD_SHA = original
            get_build_info.cache_clear()


# ---------------------------------------------------------------------------
# Git SHA fallback tests
# ---------------------------------------------------------------------------

class TestGitShaFallback:
    """Tests for _git_sha_fallback function."""

    def test_returns_string(self):
        from aragora.server.build_info import _git_sha_fallback

        result = _git_sha_fallback()
        assert isinstance(result, str)
        # Either returns a real git SHA or "unknown"
        assert len(result) > 0

    def test_returns_unknown_on_error(self):
        from aragora.server.build_info import _git_sha_fallback

        with patch("subprocess.run", side_effect=FileNotFoundError):
            result = _git_sha_fallback()
            assert result == "unknown"

    def test_returns_unknown_on_timeout(self):
        import subprocess

        from aragora.server.build_info import _git_sha_fallback

        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired("git", 5)):
            result = _git_sha_fallback()
            assert result == "unknown"

    def test_returns_unknown_on_nonzero_exit(self):
        from aragora.server.build_info import _git_sha_fallback

        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stdout = ""

        with patch("subprocess.run", return_value=mock_result):
            result = _git_sha_fallback()
            assert result == "unknown"
