"""Tests for the CritiqueHandler."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from aragora.server.handlers.critique import CritiqueHandler


class TestCritiqueHandler:
    """Tests for CritiqueHandler."""

    def _make_handler(self, ctx: dict | None = None) -> CritiqueHandler:
        return CritiqueHandler(ctx=ctx)

    def _make_http_handler(self, client_ip: str = "127.0.0.1") -> MagicMock:
        mock = MagicMock()
        mock.client_address = (client_ip, 12345)
        mock.headers = {"X-Forwarded-For": client_ip}
        return mock

    # -------------------------------------------------------------------------
    # can_handle tests
    # -------------------------------------------------------------------------

    def test_can_handle_patterns_route(self):
        handler = self._make_handler()
        assert handler.can_handle("/api/critiques/patterns") is True

    def test_can_handle_archive_route(self):
        handler = self._make_handler()
        assert handler.can_handle("/api/critiques/archive") is True

    def test_can_handle_reputation_all(self):
        handler = self._make_handler()
        assert handler.can_handle("/api/reputation/all") is True

    def test_can_handle_agent_reputation(self):
        handler = self._make_handler()
        assert handler.can_handle("/api/agent/claude/reputation") is True

    def test_can_handle_versioned_patterns(self):
        handler = self._make_handler()
        assert handler.can_handle("/api/v1/critiques/patterns") is True

    def test_cannot_handle_unrelated_route(self):
        handler = self._make_handler()
        assert handler.can_handle("/api/debates") is False

    def test_cannot_handle_partial_match(self):
        handler = self._make_handler()
        assert handler.can_handle("/api/critiques") is False

    # -------------------------------------------------------------------------
    # Route extraction tests
    # -------------------------------------------------------------------------

    def test_extract_agent_name_valid(self):
        handler = self._make_handler()
        result = handler._extract_agent_name("/api/agent/claude/reputation")
        assert result == "claude"

    def test_extract_agent_name_with_version(self):
        handler = self._make_handler()
        result = handler._extract_agent_name("/api/agent/gpt4_v2/reputation")
        assert result == "gpt4_v2"

    def test_extract_agent_name_path_traversal_blocked(self):
        handler = self._make_handler()
        result = handler._extract_agent_name("/api/agent/../secret/reputation")
        assert result is None

    def test_extract_agent_name_invalid_path(self):
        handler = self._make_handler()
        result = handler._extract_agent_name("/api/wrong/path")
        assert result is None

    # -------------------------------------------------------------------------
    # Rate limiting tests
    # -------------------------------------------------------------------------

    @patch("aragora.server.handlers.critique._critique_limiter")
    def test_rate_limit_exceeded(self, mock_limiter):
        mock_limiter.is_allowed.return_value = False
        handler = self._make_handler()
        http = self._make_http_handler()

        result = handler.handle.__wrapped__(handler, "/api/critiques/patterns", {}, http)

        assert result is not None
        assert result.status_code == 429

    @patch("aragora.server.handlers.critique._critique_limiter")
    def test_rate_limit_allowed(self, mock_limiter):
        mock_limiter.is_allowed.return_value = True
        handler = self._make_handler()
        http = self._make_http_handler()

        with patch.object(handler, "_get_critique_patterns") as mock_patterns:
            mock_patterns.return_value = MagicMock(status_code=200)
            result = handler.handle.__wrapped__(handler, "/api/critiques/patterns", {}, http)

        mock_patterns.assert_called_once()

    # -------------------------------------------------------------------------
    # Route handling tests
    # -------------------------------------------------------------------------

    @patch("aragora.server.handlers.critique._critique_limiter")
    def test_handle_patterns_route(self, mock_limiter):
        mock_limiter.is_allowed.return_value = True
        handler = self._make_handler(ctx={"nomic_dir": "/tmp/nomic"})
        http = self._make_http_handler()

        with patch.object(handler, "_get_critique_patterns") as mock_method:
            mock_method.return_value = MagicMock(status_code=200)
            handler.handle.__wrapped__(handler, "/api/critiques/patterns", {"limit": "5"}, http)

            mock_method.assert_called_once()
            call_args = mock_method.call_args
            assert call_args[0][1] == 5  # limit param

    @patch("aragora.server.handlers.critique._critique_limiter")
    def test_handle_archive_route(self, mock_limiter):
        mock_limiter.is_allowed.return_value = True
        handler = self._make_handler(ctx={"nomic_dir": "/tmp/nomic"})
        http = self._make_http_handler()

        with patch.object(handler, "_get_archive_stats") as mock_method:
            mock_method.return_value = MagicMock(status_code=200)
            handler.handle.__wrapped__(handler, "/api/critiques/archive", {}, http)

            mock_method.assert_called_once()

    @patch("aragora.server.handlers.critique._critique_limiter")
    def test_handle_reputation_all_route(self, mock_limiter):
        mock_limiter.is_allowed.return_value = True
        handler = self._make_handler(ctx={"nomic_dir": "/tmp/nomic"})
        http = self._make_http_handler()

        with patch.object(handler, "_get_all_reputations") as mock_method:
            mock_method.return_value = MagicMock(status_code=200)
            handler.handle.__wrapped__(handler, "/api/reputation/all", {}, http)

            mock_method.assert_called_once()

    @patch("aragora.server.handlers.critique._critique_limiter")
    def test_handle_agent_reputation_route(self, mock_limiter):
        mock_limiter.is_allowed.return_value = True
        handler = self._make_handler(ctx={"nomic_dir": "/tmp/nomic"})
        http = self._make_http_handler()

        with patch.object(handler, "_get_agent_reputation") as mock_method:
            mock_method.return_value = MagicMock(status_code=200)
            handler.handle.__wrapped__(handler, "/api/agent/claude/reputation", {}, http)

            mock_method.assert_called_once_with("/tmp/nomic", "claude")

    @patch("aragora.server.handlers.critique._critique_limiter")
    def test_handle_invalid_agent_name(self, mock_limiter):
        mock_limiter.is_allowed.return_value = True
        handler = self._make_handler()
        http = self._make_http_handler()

        result = handler.handle.__wrapped__(handler, "/api/agent/../bad/reputation", {}, http)

        assert result is not None
        assert result.status_code == 400

    # -------------------------------------------------------------------------
    # Query param tests
    # -------------------------------------------------------------------------

    @patch("aragora.server.handlers.critique._critique_limiter")
    def test_patterns_limit_clamped_min(self, mock_limiter):
        mock_limiter.is_allowed.return_value = True
        handler = self._make_handler()
        http = self._make_http_handler()

        with patch.object(handler, "_get_critique_patterns") as mock_method:
            mock_method.return_value = MagicMock(status_code=200)
            handler.handle.__wrapped__(handler, "/api/critiques/patterns", {"limit": "-5"}, http)

            call_args = mock_method.call_args
            assert call_args[0][1] == 1  # min_val

    @patch("aragora.server.handlers.critique._critique_limiter")
    def test_patterns_limit_clamped_max(self, mock_limiter):
        mock_limiter.is_allowed.return_value = True
        handler = self._make_handler()
        http = self._make_http_handler()

        with patch.object(handler, "_get_critique_patterns") as mock_method:
            mock_method.return_value = MagicMock(status_code=200)
            handler.handle.__wrapped__(handler, "/api/critiques/patterns", {"limit": "100"}, http)

            call_args = mock_method.call_args
            assert call_args[0][1] == 50  # max_val

    @patch("aragora.server.handlers.critique._critique_limiter")
    def test_patterns_min_success_bounded(self, mock_limiter):
        mock_limiter.is_allowed.return_value = True
        handler = self._make_handler()
        http = self._make_http_handler()

        with patch.object(handler, "_get_critique_patterns") as mock_method:
            mock_method.return_value = MagicMock(status_code=200)
            handler.handle.__wrapped__(
                handler, "/api/critiques/patterns", {"min_success": "1.5"}, http
            )

            call_args = mock_method.call_args
            assert call_args[0][2] == 1.0  # max_val

    # -------------------------------------------------------------------------
    # Critique store availability tests
    # -------------------------------------------------------------------------

    @patch("aragora.server.handlers.critique.CRITIQUE_STORE_AVAILABLE", False)
    @patch("aragora.server.handlers.critique._critique_limiter")
    def test_critique_store_unavailable(self, mock_limiter):
        mock_limiter.is_allowed.return_value = True
        handler = self._make_handler()
        http = self._make_http_handler()

        result = handler._get_critique_patterns(None, 10, 0.5)

        assert result is not None
        assert result.status_code == 503

    # -------------------------------------------------------------------------
    # ROUTES constant tests
    # -------------------------------------------------------------------------

    def test_routes_defined(self):
        handler = self._make_handler()
        assert "/api/critiques/patterns" in handler.ROUTES
        assert "/api/critiques/archive" in handler.ROUTES
        assert "/api/reputation/all" in handler.ROUTES
