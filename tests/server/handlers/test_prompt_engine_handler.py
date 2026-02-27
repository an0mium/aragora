"""Tests for prompt engine HTTP handler."""

import pytest
from unittest.mock import MagicMock


class TestPromptEngineHandler:
    @pytest.fixture
    def handler(self):
        from aragora.server.handlers.prompt_engine.handler import PromptEngineHandler

        return PromptEngineHandler({})

    def test_can_handle_sessions_path(self, handler):
        assert handler.can_handle("/api/v1/prompt-engine/sessions")

    def test_can_handle_session_id_path(self, handler):
        assert handler.can_handle("/api/v1/prompt-engine/sessions/abc-123")

    def test_cannot_handle_other_paths(self, handler):
        assert not handler.can_handle("/api/v1/debates")

    @pytest.mark.asyncio
    async def test_create_session(self, handler):
        mock_http = MagicMock()
        mock_http.command = "POST"
        mock_http.headers = {"Content-Type": "application/json"}

        body = {"prompt": "make onboarding better", "profile": "founder"}
        result = await handler.handle(
            "/api/v1/prompt-engine/sessions",
            body,
            mock_http,
        )
        assert result["status"] == 200
        assert "session_id" in result["data"]

    @pytest.mark.asyncio
    async def test_get_session(self, handler):
        mock_http = MagicMock()
        mock_http.command = "POST"
        mock_http.headers = {"Content-Type": "application/json"}

        body = {"prompt": "test", "profile": "founder"}
        create_result = await handler.handle(
            "/api/v1/prompt-engine/sessions",
            body,
            mock_http,
        )
        session_id = create_result["data"]["session_id"]

        mock_http.command = "GET"
        get_result = await handler.handle(
            f"/api/v1/prompt-engine/sessions/{session_id}",
            {},
            mock_http,
        )
        assert get_result["data"]["session_id"] == session_id

    @pytest.mark.asyncio
    async def test_delete_session(self, handler):
        mock_http = MagicMock()
        mock_http.command = "POST"
        body = {"prompt": "test", "profile": "founder"}
        create_result = await handler.handle(
            "/api/v1/prompt-engine/sessions",
            body,
            mock_http,
        )
        session_id = create_result["data"]["session_id"]

        mock_http.command = "DELETE"
        delete_result = await handler.handle(
            f"/api/v1/prompt-engine/sessions/{session_id}",
            {},
            mock_http,
        )
        assert delete_result["status"] == 200

    @pytest.mark.asyncio
    async def test_get_nonexistent_returns_404(self, handler):
        mock_http = MagicMock()
        mock_http.command = "GET"
        result = await handler.handle(
            "/api/v1/prompt-engine/sessions/nonexistent",
            {},
            mock_http,
        )
        assert result["status"] == 404
