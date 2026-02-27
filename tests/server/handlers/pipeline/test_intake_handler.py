"""Tests for the pipeline intake API handler."""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, AsyncMock, patch

from aragora.server.handlers.pipeline.intake import (
    PipelineIntakeHandler,
    _active_intakes,
)


@pytest.fixture(autouse=True)
def clear_intakes():
    """Clear active intakes between tests."""
    _active_intakes.clear()
    yield
    _active_intakes.clear()


@pytest.fixture
def handler():
    return PipelineIntakeHandler(ctx={})


@pytest.fixture
def mock_http_handler():
    h = MagicMock()
    h.request_body = {}
    h.headers = {}
    h.client_address = ("127.0.0.1", 12345)
    return h


class TestCanHandle:
    def test_matches_pipeline_start(self, handler):
        assert handler.can_handle("/api/v1/pipeline/start") is True

    def test_matches_pipeline_start_with_id(self, handler):
        assert handler.can_handle("/api/v1/pipeline/start/abc-123") is True

    def test_does_not_match_pipeline_execute(self, handler):
        assert handler.can_handle("/api/v1/pipeline/execute") is False

    def test_does_not_match_unrelated(self, handler):
        assert handler.can_handle("/api/v1/debates") is False


class TestHandleGet:
    def test_get_nonexistent_intake(self, handler, mock_http_handler):
        result = handler.handle(
            "/api/v1/pipeline/start/nonexistent",
            {},
            mock_http_handler,
        )
        assert result is not None
        assert result[1] == 404

    def test_get_existing_intake(self, handler, mock_http_handler):
        _active_intakes["test-id"] = {
            "pipeline_id": "test-id",
            "ideas": ["idea1"],
        }
        result = handler.handle(
            "/api/v1/pipeline/start/test-id",
            {},
            mock_http_handler,
        )
        assert result is not None
        body, status = result[0], result[1]
        assert status == 200

    def test_list_intakes(self, handler, mock_http_handler):
        _active_intakes["id1"] = {"pipeline_id": "id1"}
        _active_intakes["id2"] = {"pipeline_id": "id2"}

        result = handler.handle(
            "/api/v1/pipeline/start",
            {},
            mock_http_handler,
        )
        assert result is not None
        assert result[1] == 200


class TestHandlePost:
    @pytest.mark.asyncio
    async def test_post_missing_prompt(self, handler, mock_http_handler):
        mock_http_handler.request_body = {}
        result = await handler.handle_post(
            "/api/v1/pipeline/start",
            {},
            mock_http_handler,
        )
        assert result is not None
        assert result[1] == 400

    @pytest.mark.asyncio
    async def test_post_empty_prompt(self, handler, mock_http_handler):
        mock_http_handler.request_body = {"prompt": "  "}
        result = await handler.handle_post(
            "/api/v1/pipeline/start",
            {},
            mock_http_handler,
        )
        assert result is not None
        assert result[1] == 400

    @pytest.mark.asyncio
    async def test_post_invalid_autonomy_level(self, handler, mock_http_handler):
        mock_http_handler.request_body = {
            "prompt": "test prompt",
            "autonomy_level": 9,
        }
        result = await handler.handle_post(
            "/api/v1/pipeline/start",
            {},
            mock_http_handler,
        )
        assert result is not None
        assert result[1] == 400

    @pytest.mark.asyncio
    async def test_post_valid_prompt(self, handler, mock_http_handler):
        mock_http_handler.request_body = {
            "prompt": "improve error handling",
            "autonomy_level": 2,
        }

        mock_intake_result = MagicMock()
        mock_intake_result.to_dict.return_value = {
            "pipeline_id": "test-123",
            "ideas": ["improve error handling"],
            "ready_for_pipeline": True,
        }
        mock_intake_result.ready_for_pipeline = True
        mock_intake_result.pipeline_id = "test-123"

        mock_intake = AsyncMock()
        mock_intake.process.return_value = mock_intake_result

        import aragora.pipeline.intake as intake_mod

        original = intake_mod.PipelineIntake
        intake_mod.PipelineIntake = MagicMock(return_value=mock_intake)
        try:
            result = await handler.handle_post(
                "/api/v1/pipeline/start",
                {},
                mock_http_handler,
            )
        finally:
            intake_mod.PipelineIntake = original

        assert result is not None
        assert result[1] == 201

    @pytest.mark.asyncio
    async def test_post_prompt_too_long(self, handler, mock_http_handler):
        mock_http_handler.request_body = {
            "prompt": "x" * 50001,
        }
        result = await handler.handle_post(
            "/api/v1/pipeline/start",
            {},
            mock_http_handler,
        )
        assert result is not None
        assert result[1] == 400
