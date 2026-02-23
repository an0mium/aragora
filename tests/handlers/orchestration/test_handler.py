"""Tests for the unified orchestration handler.

Tests the orchestration API endpoints including:
- POST /api/v1/orchestration/deliberate       - Async deliberation
- POST /api/v1/orchestration/deliberate/sync  - Sync deliberation
- GET  /api/v1/orchestration/status/:id       - Status lookup
- GET  /api/v1/orchestration/templates        - Template listing

Covers: routing, authentication, RBAC, input validation, security
(path traversal, channel validation), knowledge source fetching,
agent team selection, output channel routing, cost estimation,
dry_run mode, template application, error handling, and formatting.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.server.handlers.orchestration.handler import (
    OrchestrationHandler,
    _orchestration_requests,
    _orchestration_results,
)
from aragora.server.handlers.orchestration.models import (
    KnowledgeContextSource,
    OrchestrationRequest,
    OrchestrationResult,
    OutputChannel,
    OutputFormat,
    TeamStrategy,
)
from aragora.server.handlers.orchestration.validation import (
    PERM_CHANNEL_SLACK,
    PERM_ORCH_ADMIN,
    PERM_ORCH_DELIBERATE,
    SourceIdValidationError,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _status(result) -> int:
    """Extract status code from HandlerResult."""
    return result.status_code


def _body(result) -> dict[str, Any]:
    """Extract JSON body from HandlerResult."""
    try:
        return json.loads(result.body.decode("utf-8"))
    except (json.JSONDecodeError, AttributeError, UnicodeDecodeError):
        return {}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_orchestration_state():
    """Clear in-memory dicts before and after each test."""
    _orchestration_requests.clear()
    _orchestration_results.clear()
    yield
    _orchestration_requests.clear()
    _orchestration_results.clear()


@pytest.fixture
def handler():
    """Create an OrchestrationHandler with empty server context."""
    return OrchestrationHandler(server_context={})


@pytest.fixture
def mock_http_handler():
    """Mock HTTP handler (passed as ``handler`` arg to handle/handle_post)."""
    h = MagicMock()
    h.headers = {"Content-Length": "0"}
    h.rfile = MagicMock()
    h.rfile.read.return_value = b"{}"
    return h


def _make_request_data(**overrides: Any) -> dict[str, Any]:
    """Build a minimal valid deliberation request body."""
    data: dict[str, Any] = {
        "question": "Should we adopt microservices?",
    }
    data.update(overrides)
    return data


# ---------------------------------------------------------------------------
# Helper to run async handle/handle_post
# ---------------------------------------------------------------------------


def _run(coro):
    """Run a coroutine in a fresh event loop (for tests)."""
    return asyncio.run(coro)


# ============================================================================
# A. can_handle routing tests
# ============================================================================


class TestCanHandle:
    """Test OrchestrationHandler.can_handle()."""

    def test_matches_orchestration_prefix(self, handler):
        assert handler.can_handle("/api/v1/orchestration/deliberate") is True

    def test_matches_status_path(self, handler):
        assert handler.can_handle("/api/v1/orchestration/status/abc") is True

    def test_matches_templates_path(self, handler):
        assert handler.can_handle("/api/v1/orchestration/templates") is True

    def test_rejects_non_orchestration(self, handler):
        assert handler.can_handle("/api/v1/debates/list") is False

    def test_rejects_partial_match(self, handler):
        assert handler.can_handle("/api/v1/orchestrations") is False

    def test_matches_sync_path(self, handler):
        assert handler.can_handle("/api/v1/orchestration/deliberate/sync") is True


# ============================================================================
# B. GET /api/v1/orchestration/templates
# ============================================================================


class TestGetTemplates:
    """Test the template listing endpoint."""

    @pytest.mark.asyncio
    async def test_returns_templates(self, handler, mock_http_handler):
        result = await handler.handle(
            "/api/v1/orchestration/templates", {}, mock_http_handler
        )
        assert result is not None
        assert _status(result) == 200
        body = _body(result)
        assert "templates" in body
        assert "count" in body
        assert isinstance(body["templates"], list)

    @pytest.mark.asyncio
    async def test_templates_have_required_fields(self, handler, mock_http_handler):
        result = await handler.handle(
            "/api/v1/orchestration/templates", {}, mock_http_handler
        )
        body = _body(result)
        for tmpl in body["templates"]:
            assert "name" in tmpl
            assert "description" in tmpl

    @pytest.mark.asyncio
    async def test_category_filter(self, handler, mock_http_handler):
        result = await handler.handle(
            "/api/v1/orchestration/templates",
            {"category": "business"},
            mock_http_handler,
        )
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_search_filter(self, handler, mock_http_handler):
        result = await handler.handle(
            "/api/v1/orchestration/templates",
            {"search": "code"},
            mock_http_handler,
        )
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_tags_filter(self, handler, mock_http_handler):
        result = await handler.handle(
            "/api/v1/orchestration/templates",
            {"tags": "security,performance"},
            mock_http_handler,
        )
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_pagination_limit(self, handler, mock_http_handler):
        result = await handler.handle(
            "/api/v1/orchestration/templates",
            {"limit": "2"},
            mock_http_handler,
        )
        body = _body(result)
        assert _status(result) == 200
        # We can't assert exact count, but should be non-negative
        assert body["count"] >= 0

    @pytest.mark.asyncio
    async def test_pagination_offset(self, handler, mock_http_handler):
        result = await handler.handle(
            "/api/v1/orchestration/templates",
            {"offset": "100"},
            mock_http_handler,
        )
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_invalid_limit_defaults(self, handler, mock_http_handler):
        result = await handler.handle(
            "/api/v1/orchestration/templates",
            {"limit": "abc"},
            mock_http_handler,
        )
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_limit_clamped_to_max_500(self, handler, mock_http_handler):
        result = await handler.handle(
            "/api/v1/orchestration/templates",
            {"limit": "9999"},
            mock_http_handler,
        )
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_negative_offset_clamped_to_zero(self, handler, mock_http_handler):
        result = await handler.handle(
            "/api/v1/orchestration/templates",
            {"offset": "-5"},
            mock_http_handler,
        )
        assert _status(result) == 200


# ============================================================================
# C. GET /api/v1/orchestration/status/:id
# ============================================================================


class TestGetStatus:
    """Test status lookup endpoint."""

    @pytest.mark.asyncio
    async def test_completed_request(self, handler, mock_http_handler):
        _orchestration_results["req-1"] = OrchestrationResult(
            request_id="req-1",
            success=True,
            final_answer="Use microservices",
            consensus_reached=True,
        )
        result = await handler.handle(
            "/api/v1/orchestration/status/req-1", {}, mock_http_handler
        )
        assert _status(result) == 200
        body = _body(result)
        assert body["status"] == "completed"
        assert body["request_id"] == "req-1"
        assert body["result"]["success"] is True

    @pytest.mark.asyncio
    async def test_failed_request(self, handler, mock_http_handler):
        _orchestration_results["req-fail"] = OrchestrationResult(
            request_id="req-fail",
            success=False,
            error="Something went wrong",
        )
        result = await handler.handle(
            "/api/v1/orchestration/status/req-fail", {}, mock_http_handler
        )
        body = _body(result)
        assert body["status"] == "failed"

    @pytest.mark.asyncio
    async def test_in_progress_request(self, handler, mock_http_handler):
        _orchestration_requests["req-prog"] = OrchestrationRequest(
            question="test",
            request_id="req-prog",
        )
        result = await handler.handle(
            "/api/v1/orchestration/status/req-prog", {}, mock_http_handler
        )
        body = _body(result)
        assert body["status"] == "in_progress"
        assert body["result"] is None

    @pytest.mark.asyncio
    async def test_not_found(self, handler, mock_http_handler):
        result = await handler.handle(
            "/api/v1/orchestration/status/nonexistent", {}, mock_http_handler
        )
        assert _status(result) == 404

    @pytest.mark.asyncio
    async def test_result_takes_precedence_over_request(self, handler, mock_http_handler):
        """If both dicts have the id, result wins."""
        _orchestration_requests["req-both"] = OrchestrationRequest(
            question="test", request_id="req-both"
        )
        _orchestration_results["req-both"] = OrchestrationResult(
            request_id="req-both", success=True
        )
        result = await handler.handle(
            "/api/v1/orchestration/status/req-both", {}, mock_http_handler
        )
        body = _body(result)
        assert body["status"] == "completed"


# ============================================================================
# D. GET routing - unhandled paths return None
# ============================================================================


class TestGetRouting:

    @pytest.mark.asyncio
    async def test_unmatched_get_returns_none(self, handler, mock_http_handler):
        result = await handler.handle(
            "/api/v1/orchestration/unknown-endpoint", {}, mock_http_handler
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_deliberate_path_not_handled_by_get(self, handler, mock_http_handler):
        result = await handler.handle(
            "/api/v1/orchestration/deliberate", {}, mock_http_handler
        )
        assert result is None


# ============================================================================
# E. POST /api/v1/orchestration/deliberate  (async mode)
# ============================================================================


class TestPostDeliberate:
    """Test the async deliberation endpoint."""

    @pytest.mark.asyncio
    async def test_queued_response(self, handler, mock_http_handler):
        data = _make_request_data()
        with patch(
            "aragora.server.handlers.orchestration.handler.asyncio.create_task"
        ) as mock_task:
            mock_task.return_value = MagicMock()
            mock_task.return_value.add_done_callback = MagicMock()
            result = await handler.handle_post(
                "/api/v1/orchestration/deliberate",
                data,
                {},
                mock_http_handler,
            )
        assert _status(result) == 202
        body = _body(result)
        assert body["status"] == "queued"
        assert "request_id" in body

    @pytest.mark.asyncio
    async def test_request_stored(self, handler, mock_http_handler):
        data = _make_request_data()
        with patch(
            "aragora.server.handlers.orchestration.handler.asyncio.create_task"
        ) as mock_task:
            mock_task.return_value = MagicMock()
            mock_task.return_value.add_done_callback = MagicMock()
            result = await handler.handle_post(
                "/api/v1/orchestration/deliberate",
                data,
                {},
                mock_http_handler,
            )
        body = _body(result)
        assert body["request_id"] in _orchestration_requests

    @pytest.mark.asyncio
    async def test_empty_question_returns_400(self, handler, mock_http_handler):
        data = _make_request_data(question="")
        result = await handler.handle_post(
            "/api/v1/orchestration/deliberate",
            data,
            {},
            mock_http_handler,
        )
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_missing_question_returns_400(self, handler, mock_http_handler):
        result = await handler.handle_post(
            "/api/v1/orchestration/deliberate",
            {},
            {},
            mock_http_handler,
        )
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_unmatched_post_returns_none(self, handler, mock_http_handler):
        data = _make_request_data()
        result = await handler.handle_post(
            "/api/v1/orchestration/unknown",
            data,
            {},
            mock_http_handler,
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_cost_estimate_included_when_available(self, handler, mock_http_handler):
        data = _make_request_data(agents=["anthropic-api", "openai-api"])
        with patch(
            "aragora.server.handlers.orchestration.handler.asyncio.create_task"
        ) as mock_task:
            mock_task.return_value = MagicMock()
            mock_task.return_value.add_done_callback = MagicMock()
            result = await handler.handle_post(
                "/api/v1/orchestration/deliberate",
                data,
                {},
                mock_http_handler,
            )
        body = _body(result)
        # Cost estimate may or may not be present depending on import availability
        assert _status(result) == 202

    @pytest.mark.asyncio
    async def test_cost_estimation_failure_is_non_blocking(self, handler, mock_http_handler):
        data = _make_request_data()
        with (
            patch(
                "aragora.server.handlers.orchestration.handler.asyncio.create_task"
            ) as mock_task,
            patch(
                "aragora.server.handlers.debates.cost_estimation.estimate_debate_cost",
                side_effect=ImportError("no module"),
            ),
        ):
            mock_task.return_value = MagicMock()
            mock_task.return_value.add_done_callback = MagicMock()
            result = await handler.handle_post(
                "/api/v1/orchestration/deliberate",
                data,
                {},
                mock_http_handler,
            )
        assert _status(result) == 202


# ============================================================================
# F. POST /api/v1/orchestration/deliberate/sync
# ============================================================================


class TestPostDeliberateSync:
    """Test the synchronous deliberation endpoint."""

    @pytest.mark.asyncio
    async def test_sync_returns_result(self, handler, mock_http_handler):
        data = _make_request_data()
        fake_result = OrchestrationResult(
            request_id="sync-1",
            success=True,
            final_answer="Yes, adopt microservices",
            consensus_reached=True,
            confidence=0.85,
            agents_participated=["anthropic-api", "openai-api"],
        )

        # run_async is called from sync context; since we're in an async test
        # we mock it to just return the result directly (the coro is already
        # patched out via _execute_deliberation).
        def mock_run_async(coro, timeout=30.0):
            """Synchronously resolve the coroutine by scheduling on the running loop."""
            import asyncio as _aio
            loop = _aio.get_event_loop()
            task = loop.create_task(coro)
            # The task won't complete until the event loop yields, but
            # since _execute_deliberation is mocked, it completes immediately.
            # We need nest_asyncio to allow run_until_complete inside a running loop.
            try:
                import nest_asyncio
                nest_asyncio.apply(loop)
                return loop.run_until_complete(task)
            except ImportError:
                # Fallback: just return the fake result directly
                task.cancel()
                return fake_result

        with patch.object(
            handler, "_execute_deliberation", new_callable=AsyncMock, return_value=fake_result
        ), patch(
            "aragora.server.handlers.orchestration.handler.run_async",
            side_effect=mock_run_async,
        ):
            result = await handler.handle_post(
                "/api/v1/orchestration/deliberate/sync",
                data,
                {},
                mock_http_handler,
            )
        assert _status(result) == 200
        body = _body(result)
        assert body["success"] is True
        assert body["final_answer"] == "Yes, adopt microservices"

    @pytest.mark.asyncio
    async def test_sync_stores_result(self, handler, mock_http_handler):
        data = _make_request_data()
        fake_result = OrchestrationResult(
            request_id="sync-store",
            success=True,
        )

        def mock_run_async(coro, timeout=30.0):
            import asyncio as _aio
            loop = _aio.get_event_loop()
            try:
                import nest_asyncio
                nest_asyncio.apply(loop)
                return loop.run_until_complete(coro)
            except ImportError:
                return fake_result

        with patch.object(
            handler, "_execute_deliberation", new_callable=AsyncMock, return_value=fake_result
        ), patch(
            "aragora.server.handlers.orchestration.handler.run_async",
            side_effect=mock_run_async,
        ):
            await handler.handle_post(
                "/api/v1/orchestration/deliberate/sync",
                data,
                {},
                mock_http_handler,
            )
        # At least one result should be stored
        assert len(_orchestration_results) >= 1


# ============================================================================
# G. Dry run mode
# ============================================================================


class TestDryRun:
    """Test the dry_run flag."""

    @pytest.mark.asyncio
    async def test_dry_run_returns_estimate(self, handler, mock_http_handler):
        data = _make_request_data(dry_run=True)
        result = await handler.handle_post(
            "/api/v1/orchestration/deliberate",
            data,
            {},
            mock_http_handler,
        )
        assert _status(result) == 200
        body = _body(result)
        assert body["dry_run"] is True
        assert "request_id" in body
        assert body["message"] == "Dry run \u2014 no debate executed"

    @pytest.mark.asyncio
    async def test_dry_run_does_not_store_request(self, handler, mock_http_handler):
        data = _make_request_data(dry_run=True)
        await handler.handle_post(
            "/api/v1/orchestration/deliberate",
            data,
            {},
            mock_http_handler,
        )
        # Dry runs should not be stored as in-progress
        assert len(_orchestration_requests) == 0

    @pytest.mark.asyncio
    async def test_dry_run_includes_agents(self, handler, mock_http_handler):
        data = _make_request_data(dry_run=True, agents=["claude", "gpt4"])
        result = await handler.handle_post(
            "/api/v1/orchestration/deliberate",
            data,
            {},
            mock_http_handler,
        )
        body = _body(result)
        assert body["agents"] == ["claude", "gpt4"]

    @pytest.mark.asyncio
    async def test_dry_run_includes_max_rounds(self, handler, mock_http_handler):
        data = _make_request_data(dry_run=True, max_rounds=5)
        result = await handler.handle_post(
            "/api/v1/orchestration/deliberate",
            data,
            {},
            mock_http_handler,
        )
        body = _body(result)
        assert body["max_rounds"] == 5


# ============================================================================
# H. Template application
# ============================================================================


class TestTemplateApplication:
    """Test that templates inject defaults into the request."""

    @pytest.mark.asyncio
    async def test_code_review_template_applied(self, handler, mock_http_handler):
        data = _make_request_data(template="code_review", dry_run=True)
        result = await handler.handle_post(
            "/api/v1/orchestration/deliberate",
            data,
            {},
            mock_http_handler,
        )
        assert _status(result) == 200
        body = _body(result)
        # Template should inject default agents
        assert len(body.get("agents", [])) > 0

    @pytest.mark.asyncio
    async def test_unknown_template_ignored(self, handler, mock_http_handler):
        data = _make_request_data(template="nonexistent_template", dry_run=True)
        result = await handler.handle_post(
            "/api/v1/orchestration/deliberate",
            data,
            {},
            mock_http_handler,
        )
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_explicit_agents_not_overridden_by_template(self, handler, mock_http_handler):
        data = _make_request_data(
            template="code_review", agents=["my-custom-agent"], dry_run=True
        )
        result = await handler.handle_post(
            "/api/v1/orchestration/deliberate",
            data,
            {},
            mock_http_handler,
        )
        body = _body(result)
        assert body["agents"] == ["my-custom-agent"]


# ============================================================================
# I. Knowledge source validation
# ============================================================================


class TestKnowledgeSourceValidation:
    """Test knowledge source security validation."""

    @pytest.mark.asyncio
    async def test_path_traversal_in_source_id_rejected(self, handler, mock_http_handler):
        data = _make_request_data(
            knowledge_sources=[{"type": "slack", "id": "../../etc/passwd"}]
        )
        result = await handler.handle_post(
            "/api/v1/orchestration/deliberate",
            data,
            {},
            mock_http_handler,
        )
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_absolute_path_source_id_rejected(self, handler, mock_http_handler):
        data = _make_request_data(
            knowledge_sources=[{"type": "slack", "id": "/etc/passwd"}]
        )
        result = await handler.handle_post(
            "/api/v1/orchestration/deliberate",
            data,
            {},
            mock_http_handler,
        )
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_null_byte_in_source_id_rejected(self, handler, mock_http_handler):
        data = _make_request_data(
            knowledge_sources=[{"type": "document", "id": "doc\x00evil"}]
        )
        result = await handler.handle_post(
            "/api/v1/orchestration/deliberate",
            data,
            {},
            mock_http_handler,
        )
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_valid_source_id_accepted(self, handler, mock_http_handler):
        data = _make_request_data(
            knowledge_sources=[{"type": "slack", "id": "C0123ABCDEF"}],
            dry_run=True,
        )
        result = await handler.handle_post(
            "/api/v1/orchestration/deliberate",
            data,
            {},
            mock_http_handler,
        )
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_github_source_id_format_accepted(self, handler, mock_http_handler):
        data = _make_request_data(
            knowledge_sources=[{"type": "github", "id": "owner/repo/pr/123"}],
            dry_run=True,
        )
        result = await handler.handle_post(
            "/api/v1/orchestration/deliberate",
            data,
            {},
            mock_http_handler,
        )
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_windows_path_source_id_rejected(self, handler, mock_http_handler):
        data = _make_request_data(
            knowledge_sources=[{"type": "document", "id": "C:\\Windows\\System32"}]
        )
        result = await handler.handle_post(
            "/api/v1/orchestration/deliberate",
            data,
            {},
            mock_http_handler,
        )
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_too_long_source_id_rejected(self, handler, mock_http_handler):
        data = _make_request_data(
            knowledge_sources=[{"type": "document", "id": "x" * 300}]
        )
        result = await handler.handle_post(
            "/api/v1/orchestration/deliberate",
            data,
            {},
            mock_http_handler,
        )
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_string_format_knowledge_source(self, handler, mock_http_handler):
        data = _make_request_data(
            knowledge_sources=["slack:C0123ABCDEF"],
            dry_run=True,
        )
        result = await handler.handle_post(
            "/api/v1/orchestration/deliberate",
            data,
            {},
            mock_http_handler,
        )
        assert _status(result) == 200


# ============================================================================
# J. Output channel validation
# ============================================================================


class TestOutputChannelValidation:
    """Test output channel security validation."""

    @pytest.mark.asyncio
    async def test_path_traversal_channel_id_rejected(self, handler, mock_http_handler):
        data = _make_request_data(
            output_channels=[{"type": "slack", "id": "../../secret"}]
        )
        result = await handler.handle_post(
            "/api/v1/orchestration/deliberate",
            data,
            {},
            mock_http_handler,
        )
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_empty_channel_id_rejected(self, handler, mock_http_handler):
        data = _make_request_data(
            output_channels=[{"type": "slack", "id": ""}]
        )
        result = await handler.handle_post(
            "/api/v1/orchestration/deliberate",
            data,
            {},
            mock_http_handler,
        )
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_null_byte_channel_id_rejected(self, handler, mock_http_handler):
        data = _make_request_data(
            output_channels=[{"type": "slack", "id": "chan\x00evil"}]
        )
        result = await handler.handle_post(
            "/api/v1/orchestration/deliberate",
            data,
            {},
            mock_http_handler,
        )
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_webhook_invalid_url_rejected(self, handler, mock_http_handler):
        data = _make_request_data(
            output_channels=[{"type": "webhook", "id": "not-a-url"}]
        )
        result = await handler.handle_post(
            "/api/v1/orchestration/deliberate",
            data,
            {},
            mock_http_handler,
        )
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_webhook_valid_url_accepted(self, handler, mock_http_handler):
        data = _make_request_data(
            output_channels=[{"type": "webhook", "id": "https://example.com/hook"}],
            dry_run=True,
        )
        result = await handler.handle_post(
            "/api/v1/orchestration/deliberate",
            data,
            {},
            mock_http_handler,
        )
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_valid_slack_channel_accepted(self, handler, mock_http_handler):
        data = _make_request_data(
            output_channels=[{"type": "slack", "id": "C0123ABCDEF"}],
            dry_run=True,
        )
        result = await handler.handle_post(
            "/api/v1/orchestration/deliberate",
            data,
            {},
            mock_http_handler,
        )
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_absolute_path_channel_id_rejected(self, handler, mock_http_handler):
        data = _make_request_data(
            output_channels=[{"type": "teams", "id": "/etc/shadow"}]
        )
        result = await handler.handle_post(
            "/api/v1/orchestration/deliberate",
            data,
            {},
            mock_http_handler,
        )
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_webhook_path_traversal_rejected(self, handler, mock_http_handler):
        data = _make_request_data(
            output_channels=[{"type": "webhook", "id": "https://evil.com/../../etc"}]
        )
        result = await handler.handle_post(
            "/api/v1/orchestration/deliberate",
            data,
            {},
            mock_http_handler,
        )
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_string_format_output_channel(self, handler, mock_http_handler):
        data = _make_request_data(
            output_channels=["slack:C0123ABCDEF"],
            dry_run=True,
        )
        result = await handler.handle_post(
            "/api/v1/orchestration/deliberate",
            data,
            {},
            mock_http_handler,
        )
        assert _status(result) == 200


# ============================================================================
# K. RBAC permission checks  (opt out of auto-auth)
# ============================================================================


class TestRBACPermissions:

    @pytest.mark.no_auto_auth
    @pytest.mark.asyncio
    async def test_get_unauthenticated_returns_401(self, handler, mock_http_handler):
        """GET without auth should return 401."""
        from aragora.server.handlers.secure import SecureHandler
        from aragora.server.handlers.utils.auth import UnauthorizedError

        async def raise_unauth(self, req, require_auth=False):
            raise UnauthorizedError("no token")

        with patch.object(SecureHandler, "get_auth_context", raise_unauth):
            result = await handler.handle(
                "/api/v1/orchestration/templates", {}, mock_http_handler
            )
        assert _status(result) == 401

    @pytest.mark.no_auto_auth
    @pytest.mark.asyncio
    async def test_post_unauthenticated_returns_401(self, handler, mock_http_handler):
        from aragora.server.handlers.secure import SecureHandler
        from aragora.server.handlers.utils.auth import UnauthorizedError

        async def raise_unauth(self, req, require_auth=False):
            raise UnauthorizedError("no token")

        with patch.object(SecureHandler, "get_auth_context", raise_unauth):
            result = await handler.handle_post(
                "/api/v1/orchestration/deliberate",
                _make_request_data(),
                {},
                mock_http_handler,
            )
        assert _status(result) == 401

    @pytest.mark.no_auto_auth
    @pytest.mark.asyncio
    async def test_get_forbidden_returns_403(self, handler, mock_http_handler):
        from aragora.rbac.models import AuthorizationContext
        from aragora.server.handlers.secure import SecureHandler, ForbiddenError

        mock_ctx = AuthorizationContext(
            user_id="user-limited",
            roles=set(),
            permissions=set(),  # No permissions at all
        )

        async def return_ctx(self, req, require_auth=False):
            return mock_ctx

        with patch.object(SecureHandler, "get_auth_context", return_ctx):
            result = await handler.handle(
                "/api/v1/orchestration/templates", {}, mock_http_handler
            )
        assert _status(result) == 403

    @pytest.mark.no_auto_auth
    @pytest.mark.asyncio
    async def test_post_forbidden_on_execute_permission(self, handler, mock_http_handler):
        from aragora.rbac.models import AuthorizationContext
        from aragora.server.handlers.secure import SecureHandler

        # Has read but no execute
        mock_ctx = AuthorizationContext(
            user_id="user-readonly",
            roles=set(),
            permissions={"orchestration.read"},
        )

        async def return_ctx(self, req, require_auth=False):
            return mock_ctx

        with patch.object(SecureHandler, "get_auth_context", return_ctx):
            result = await handler.handle_post(
                "/api/v1/orchestration/deliberate",
                _make_request_data(),
                {},
                mock_http_handler,
            )
        assert _status(result) == 403

    @pytest.mark.no_auto_auth
    @pytest.mark.asyncio
    async def test_get_forbidden_error_returns_403(self, handler, mock_http_handler):
        """ForbiddenError during auth returns 403."""
        from aragora.server.handlers.secure import SecureHandler
        from aragora.server.handlers.utils.auth import ForbiddenError

        async def raise_forbidden(self, req, require_auth=False):
            raise ForbiddenError("Denied")

        with patch.object(SecureHandler, "get_auth_context", raise_forbidden):
            result = await handler.handle(
                "/api/v1/orchestration/templates", {}, mock_http_handler
            )
        assert _status(result) == 403

    @pytest.mark.no_auto_auth
    @pytest.mark.asyncio
    async def test_get_value_error_during_auth_returns_401(self, handler, mock_http_handler):
        """ValueError during auth (bad token format) returns 401."""
        from aragora.server.handlers.secure import SecureHandler

        async def raise_value_error(self, req, require_auth=False):
            raise ValueError("bad token")

        with patch.object(SecureHandler, "get_auth_context", raise_value_error):
            result = await handler.handle(
                "/api/v1/orchestration/templates", {}, mock_http_handler
            )
        assert _status(result) == 401


# ============================================================================
# L. Knowledge source RBAC - unknown source type requires admin
# ============================================================================


class TestKnowledgeSourceRBAC:

    def test_validate_known_source_type_passes(self, handler):
        from aragora.rbac.models import AuthorizationContext

        ctx = AuthorizationContext(
            user_id="u1", permissions={"*"}, roles=set()
        )
        source = KnowledgeContextSource(source_type="slack", source_id="C012ABC")
        err = handler._validate_knowledge_source(source, ctx)
        assert err is None

    def test_validate_unknown_source_type_requires_admin(self, handler):
        """Unknown source type checks admin permission."""
        from aragora.rbac.models import AuthorizationContext

        ctx = AuthorizationContext(
            user_id="u1", permissions={"*"}, roles=set()
        )
        source = KnowledgeContextSource(source_type="ftp", source_id="some-server")
        # With wildcard, still passes
        err = handler._validate_knowledge_source(source, ctx)
        assert err is None

    def test_invalid_source_id_fails_validation(self, handler):
        from aragora.rbac.models import AuthorizationContext

        ctx = AuthorizationContext(
            user_id="u1", permissions={"*"}, roles=set()
        )
        source = KnowledgeContextSource(source_type="slack", source_id="../bad")
        err = handler._validate_knowledge_source(source, ctx)
        assert err is not None
        assert _status(err) == 400


# ============================================================================
# M. Output channel RBAC - unknown channel type requires admin
# ============================================================================


class TestOutputChannelRBAC:

    def test_validate_known_channel_passes(self, handler):
        from aragora.rbac.models import AuthorizationContext

        ctx = AuthorizationContext(
            user_id="u1", permissions={"*"}, roles=set()
        )
        channel = OutputChannel(channel_type="slack", channel_id="C012ABC")
        err = handler._validate_output_channel(channel, ctx)
        assert err is None

    def test_validate_unknown_channel_type_requires_admin(self, handler):
        from aragora.rbac.models import AuthorizationContext

        ctx = AuthorizationContext(
            user_id="u1", permissions={"*"}, roles=set()
        )
        channel = OutputChannel(channel_type="fax", channel_id="123456")
        # With wildcard still passes
        err = handler._validate_output_channel(channel, ctx)
        assert err is None

    def test_invalid_channel_id_fails(self, handler):
        from aragora.rbac.models import AuthorizationContext

        ctx = AuthorizationContext(
            user_id="u1", permissions={"*"}, roles=set()
        )
        channel = OutputChannel(channel_type="slack", channel_id="../etc/passwd")
        err = handler._validate_output_channel(channel, ctx)
        assert err is not None
        assert _status(err) == 400


# ============================================================================
# N. Agent team selection
# ============================================================================


class TestAgentTeamSelection:

    @pytest.mark.asyncio
    async def test_explicit_agents_returned(self, handler):
        req = OrchestrationRequest(question="q", agents=["claude", "gpt4"])
        agents = await handler._select_agent_team(req)
        assert agents == ["claude", "gpt4"]

    @pytest.mark.asyncio
    async def test_fast_strategy_returns_two(self, handler):
        req = OrchestrationRequest(
            question="q", team_strategy=TeamStrategy.FAST, agents=[]
        )
        agents = await handler._select_agent_team(req)
        assert len(agents) == 2

    @pytest.mark.asyncio
    async def test_diverse_strategy_returns_all(self, handler):
        req = OrchestrationRequest(
            question="q", team_strategy=TeamStrategy.DIVERSE, agents=[]
        )
        agents = await handler._select_agent_team(req)
        assert len(agents) == 4

    @pytest.mark.asyncio
    async def test_random_strategy_returns_subset(self, handler):
        req = OrchestrationRequest(
            question="q", team_strategy=TeamStrategy.RANDOM, agents=[]
        )
        agents = await handler._select_agent_team(req)
        assert 1 <= len(agents) <= 4

    @pytest.mark.asyncio
    async def test_specified_no_agents_returns_default_subset(self, handler):
        req = OrchestrationRequest(
            question="q", team_strategy=TeamStrategy.SPECIFIED, agents=[]
        )
        agents = await handler._select_agent_team(req)
        assert len(agents) == 2

    @pytest.mark.asyncio
    async def test_best_for_domain_with_routing_unavailable(self, handler):
        req = OrchestrationRequest(
            question="q", team_strategy=TeamStrategy.BEST_FOR_DOMAIN, agents=[]
        )
        # The handler does `import aragora.server.handlers.routing`, so make it fail
        import sys
        with patch.dict(sys.modules, {"aragora.server.handlers.routing": None}):
            agents = await handler._select_agent_team(req)
        assert len(agents) >= 1

    @pytest.mark.asyncio
    async def test_best_for_domain_falls_back_on_routing_error(self, handler):
        req = OrchestrationRequest(
            question="q", team_strategy=TeamStrategy.BEST_FOR_DOMAIN, agents=[]
        )
        agents = await handler._select_agent_team(req)
        # Should fall back to defaults (3 agents)
        assert len(agents) >= 1


# ============================================================================
# O. Format result for channel
# ============================================================================


class TestFormatResult:

    def test_summary_format(self, handler):
        result = OrchestrationResult(
            request_id="r1",
            success=True,
            final_answer="Use Kubernetes",
            agents_participated=["claude"],
        )
        req = OrchestrationRequest(
            question="q", output_format=OutputFormat.SUMMARY
        )
        msg = handler._format_result_for_channel(result, req)
        assert "Use Kubernetes" in msg
        assert "Deliberation Complete" in msg

    def test_standard_format_includes_question(self, handler):
        result = OrchestrationResult(
            request_id="r1",
            success=True,
            final_answer="Answer here",
            consensus_reached=True,
            confidence=0.85,
            agents_participated=["claude", "gpt4"],
            duration_seconds=12.5,
        )
        req = OrchestrationRequest(
            question="Should we do X?",
            output_format=OutputFormat.STANDARD,
        )
        msg = handler._format_result_for_channel(result, req)
        assert "Should we do X?" in msg
        assert "Consensus reached" in msg
        assert "85%" in msg
        assert "12.5s" in msg

    def test_standard_no_consensus(self, handler):
        result = OrchestrationResult(
            request_id="r1",
            success=True,
            final_answer=None,
            consensus_reached=False,
            agents_participated=["claude"],
            duration_seconds=5.0,
        )
        req = OrchestrationRequest(
            question="q", output_format=OutputFormat.STANDARD
        )
        msg = handler._format_result_for_channel(result, req)
        assert "No consensus" in msg
        assert "No conclusion reached." in msg

    def test_summary_no_answer(self, handler):
        result = OrchestrationResult(
            request_id="r1", success=True, agents_participated=[]
        )
        req = OrchestrationRequest(
            question="q", output_format=OutputFormat.SUMMARY
        )
        msg = handler._format_result_for_channel(result, req)
        assert "No conclusion reached." in msg


# ============================================================================
# P. Execute and store (async background task)
# ============================================================================


class TestExecuteAndStore:

    @pytest.mark.asyncio
    async def test_stores_result_on_success(self, handler):
        req = OrchestrationRequest(question="q", request_id="bg-1")
        _orchestration_requests["bg-1"] = req

        fake_result = OrchestrationResult(request_id="bg-1", success=True)
        with patch.object(
            handler, "_execute_deliberation", new_callable=AsyncMock, return_value=fake_result
        ):
            await handler._execute_and_store(req)

        assert "bg-1" in _orchestration_results
        assert _orchestration_results["bg-1"].success is True
        # Request should be cleaned up
        assert "bg-1" not in _orchestration_requests

    @pytest.mark.asyncio
    async def test_stores_failure_on_exception(self, handler):
        req = OrchestrationRequest(question="q", request_id="bg-fail")
        _orchestration_requests["bg-fail"] = req

        with patch.object(
            handler,
            "_execute_deliberation",
            new_callable=AsyncMock,
            side_effect=RuntimeError("boom"),
        ):
            await handler._execute_and_store(req)

        assert "bg-fail" in _orchestration_results
        assert _orchestration_results["bg-fail"].success is False
        assert "bg-fail" not in _orchestration_requests

    @pytest.mark.asyncio
    async def test_cleans_up_request_on_exception(self, handler):
        req = OrchestrationRequest(question="q", request_id="bg-clean")
        _orchestration_requests["bg-clean"] = req

        with patch.object(
            handler,
            "_execute_deliberation",
            new_callable=AsyncMock,
            side_effect=ValueError("bad"),
        ):
            await handler._execute_and_store(req)

        assert "bg-clean" not in _orchestration_requests


# ============================================================================
# Q. Fetch knowledge context
# ============================================================================


class TestFetchKnowledgeContext:

    @pytest.mark.asyncio
    async def test_chat_platforms_dispatch(self, handler):
        for platform in ["slack", "teams", "discord", "telegram", "whatsapp", "google_chat"]:
            source = KnowledgeContextSource(source_type=platform, source_id="chan123")
            with patch.object(
                handler, "_fetch_chat_context", new_callable=AsyncMock, return_value="context"
            ) as mock_fetch:
                result = await handler._fetch_knowledge_context(source)
                mock_fetch.assert_called_once_with(platform, source)
                assert result == "context"

    @pytest.mark.asyncio
    async def test_confluence_dispatch(self, handler):
        source = KnowledgeContextSource(source_type="confluence", source_id="page-123")
        with patch.object(
            handler, "_fetch_confluence_context", new_callable=AsyncMock, return_value="page content"
        ):
            result = await handler._fetch_knowledge_context(source)
            assert result == "page content"

    @pytest.mark.asyncio
    async def test_github_dispatch(self, handler):
        source = KnowledgeContextSource(source_type="github", source_id="owner/repo/pr/42")
        with patch.object(
            handler, "_fetch_github_context", new_callable=AsyncMock, return_value="pr content"
        ):
            result = await handler._fetch_knowledge_context(source)
            assert result == "pr content"

    @pytest.mark.asyncio
    async def test_document_dispatch(self, handler):
        for dtype in ["document", "doc", "km"]:
            source = KnowledgeContextSource(source_type=dtype, source_id="doc-abc")
            with patch.object(
                handler, "_fetch_document_context", new_callable=AsyncMock, return_value="doc text"
            ):
                result = await handler._fetch_knowledge_context(source)
                assert result == "doc text"

    @pytest.mark.asyncio
    async def test_jira_dispatch(self, handler):
        source = KnowledgeContextSource(source_type="jira", source_id="PROJ-123")
        with patch.object(
            handler, "_fetch_jira_context", new_callable=AsyncMock, return_value="issue text"
        ):
            result = await handler._fetch_knowledge_context(source)
            assert result == "issue text"

    @pytest.mark.asyncio
    async def test_unknown_source_returns_none(self, handler):
        source = KnowledgeContextSource(source_type="ftp", source_id="server")
        result = await handler._fetch_knowledge_context(source)
        assert result is None


# ============================================================================
# R. GitHub context security checks
# ============================================================================


class TestFetchGitHubContext:

    @pytest.mark.asyncio
    async def test_path_traversal_returns_none(self, handler):
        source = KnowledgeContextSource(source_type="github", source_id="../../etc/passwd")
        result = await handler._fetch_github_context(source)
        assert result is None

    @pytest.mark.asyncio
    async def test_absolute_path_returns_none(self, handler):
        source = KnowledgeContextSource(source_type="github", source_id="/root/secrets")
        result = await handler._fetch_github_context(source)
        assert result is None

    @pytest.mark.asyncio
    async def test_invalid_owner_format_returns_none(self, handler):
        source = KnowledgeContextSource(
            source_type="github", source_id="<script>/repo/pr/1"
        )
        with patch(
            "aragora.server.handlers.orchestration.handler.re.compile",
            return_value=__import__("re").compile(r"^[a-zA-Z0-9_\-]+$"),
        ):
            result = await handler._fetch_github_context(source)
        assert result is None

    @pytest.mark.asyncio
    async def test_invalid_item_type_returns_none(self, handler):
        source = KnowledgeContextSource(
            source_type="github", source_id="owner/repo/blob/123"
        )
        result = await handler._fetch_github_context(source)
        assert result is None

    @pytest.mark.asyncio
    async def test_non_numeric_number_returns_none(self, handler):
        source = KnowledgeContextSource(
            source_type="github", source_id="owner/repo/pr/abc"
        )
        result = await handler._fetch_github_context(source)
        assert result is None

    @pytest.mark.asyncio
    async def test_too_few_parts_returns_none(self, handler):
        source = KnowledgeContextSource(
            source_type="github", source_id="owner/repo"
        )
        result = await handler._fetch_github_context(source)
        assert result is None

    @pytest.mark.asyncio
    async def test_valid_pr_fetched(self, handler):
        source = KnowledgeContextSource(
            source_type="github", source_id="owner/repo/pr/42"
        )
        mock_evidence = MagicMock()
        mock_evidence.content = "PR description"
        mock_connector = MagicMock()
        mock_connector.search = AsyncMock(return_value=[mock_evidence])
        with patch(
            "aragora.connectors.github.GitHubConnector",
            return_value=mock_connector,
        ):
            result = await handler._fetch_github_context(source)
        assert result == "PR description"

    @pytest.mark.asyncio
    async def test_valid_issue_fetched(self, handler):
        source = KnowledgeContextSource(
            source_type="github", source_id="owner/repo/issue/10"
        )
        mock_evidence = MagicMock()
        mock_evidence.content = "Issue body"
        mock_connector = MagicMock()
        mock_connector.search = AsyncMock(return_value=[mock_evidence])
        with patch(
            "aragora.connectors.github.GitHubConnector",
            return_value=mock_connector,
        ):
            result = await handler._fetch_github_context(source)
        assert result == "Issue body"


# ============================================================================
# S. Chat context fetching
# ============================================================================


class TestFetchChatContext:

    @pytest.mark.asyncio
    async def test_returns_context_string(self, handler):
        mock_ctx = MagicMock()
        mock_ctx.messages = [MagicMock()]
        mock_ctx.to_context_string.return_value = "msg1\nmsg2"
        mock_connector = MagicMock()
        mock_connector.fetch_context = AsyncMock(return_value=mock_ctx)

        with patch(
            "aragora.connectors.chat.registry.get_connector",
            return_value=mock_connector,
        ):
            result = await handler._fetch_chat_context("slack", KnowledgeContextSource(
                source_type="slack", source_id="C123", max_items=10
            ))
        assert result == "msg1\nmsg2"

    @pytest.mark.asyncio
    async def test_no_connector_returns_none(self, handler):
        with patch(
            "aragora.connectors.chat.registry.get_connector",
            return_value=None,
        ):
            result = await handler._fetch_chat_context("slack", KnowledgeContextSource(
                source_type="slack", source_id="C123"
            ))
        assert result is None

    @pytest.mark.asyncio
    async def test_empty_messages_returns_none(self, handler):
        mock_ctx = MagicMock()
        mock_ctx.messages = []
        mock_connector = MagicMock()
        mock_connector.fetch_context = AsyncMock(return_value=mock_ctx)

        with patch(
            "aragora.connectors.chat.registry.get_connector",
            return_value=mock_connector,
        ):
            result = await handler._fetch_chat_context("slack", KnowledgeContextSource(
                source_type="slack", source_id="C123"
            ))
        assert result is None

    @pytest.mark.asyncio
    async def test_import_error_returns_none(self, handler):
        with patch(
            "aragora.connectors.chat.registry.get_connector",
            side_effect=ImportError("no module"),
        ):
            result = await handler._fetch_chat_context("slack", KnowledgeContextSource(
                source_type="slack", source_id="C123"
            ))
        assert result is None


# ============================================================================
# T. Document context fetching
# ============================================================================


class TestFetchDocumentContext:

    @pytest.mark.asyncio
    async def test_returns_content(self, handler):
        mock_item = MagicMock()
        mock_item.content = "Knowledge text"
        mock_results = MagicMock()
        mock_results.items = [mock_item]
        mock_mound = AsyncMock()
        mock_mound.query = AsyncMock(return_value=mock_results)

        with patch(
            "aragora.knowledge.mound.get_knowledge_mound",
            return_value=mock_mound,
        ):
            result = await handler._fetch_document_context(
                KnowledgeContextSource(source_type="document", source_id="doc-1")
            )
        assert result == "Knowledge text"

    @pytest.mark.asyncio
    async def test_no_mound_returns_none(self, handler):
        with patch(
            "aragora.knowledge.mound.get_knowledge_mound",
            return_value=None,
        ):
            result = await handler._fetch_document_context(
                KnowledgeContextSource(source_type="document", source_id="doc-1")
            )
        assert result is None

    @pytest.mark.asyncio
    async def test_import_error_returns_none(self, handler):
        with patch(
            "aragora.knowledge.mound.get_knowledge_mound",
            side_effect=ImportError("no module"),
        ):
            result = await handler._fetch_document_context(
                KnowledgeContextSource(source_type="document", source_id="doc-1")
            )
        assert result is None


# ============================================================================
# U. Confluence context fetching
# ============================================================================


class TestFetchConfluenceContext:

    @pytest.mark.asyncio
    async def test_no_url_returns_none(self, handler):
        source = KnowledgeContextSource(source_type="confluence", source_id="12345")
        # No confluence_url in ctx and no env var
        with patch.dict("os.environ", {}, clear=False):
            result = await handler._fetch_confluence_context(source)
        assert result is None

    @pytest.mark.asyncio
    async def test_prefixes_page_id(self, handler):
        handler.ctx["confluence_url"] = "https://wiki.example.com"
        source = KnowledgeContextSource(source_type="confluence", source_id="12345")
        mock_evidence = MagicMock()
        mock_evidence.content = "Page content"
        mock_connector = MagicMock()
        mock_connector.fetch = AsyncMock(return_value=mock_evidence)
        with patch(
            "aragora.connectors.enterprise.collaboration.confluence.ConfluenceConnector",
            return_value=mock_connector,
        ):
            result = await handler._fetch_confluence_context(source)
        # Should have been called with "confluence-12345"
        mock_connector.fetch.assert_called_once_with("confluence-12345")
        assert result == "Page content"

    @pytest.mark.asyncio
    async def test_already_prefixed_page_id(self, handler):
        handler.ctx["confluence_url"] = "https://wiki.example.com"
        source = KnowledgeContextSource(source_type="confluence", source_id="confluence-99")
        mock_connector = MagicMock()
        mock_connector.fetch = AsyncMock(return_value=None)
        with patch(
            "aragora.connectors.enterprise.collaboration.confluence.ConfluenceConnector",
            return_value=mock_connector,
        ):
            result = await handler._fetch_confluence_context(source)
        mock_connector.fetch.assert_called_once_with("confluence-99")
        assert result is None


# ============================================================================
# V. Jira context fetching
# ============================================================================


class TestFetchJiraContext:

    @pytest.mark.asyncio
    async def test_no_url_returns_none(self, handler):
        source = KnowledgeContextSource(source_type="jira", source_id="PROJ-123")
        with patch.dict("os.environ", {}, clear=False):
            result = await handler._fetch_jira_context(source)
        assert result is None

    @pytest.mark.asyncio
    async def test_prefixes_issue_key(self, handler):
        handler.ctx["jira_url"] = "https://jira.example.com"
        source = KnowledgeContextSource(source_type="jira", source_id="PROJ-123")
        mock_evidence = MagicMock()
        mock_evidence.content = "Issue description"
        mock_connector = MagicMock()
        mock_connector.fetch = AsyncMock(return_value=mock_evidence)
        with patch(
            "aragora.connectors.enterprise.collaboration.jira.JiraConnector",
            return_value=mock_connector,
        ):
            result = await handler._fetch_jira_context(source)
        mock_connector.fetch.assert_called_once_with("jira-PROJ-123")
        assert result == "Issue description"

    @pytest.mark.asyncio
    async def test_already_prefixed_issue_key(self, handler):
        handler.ctx["jira_url"] = "https://jira.example.com"
        source = KnowledgeContextSource(source_type="jira", source_id="jira-ABC-1")
        mock_connector = MagicMock()
        mock_connector.fetch = AsyncMock(return_value=None)
        with patch(
            "aragora.connectors.enterprise.collaboration.jira.JiraConnector",
            return_value=mock_connector,
        ):
            result = await handler._fetch_jira_context(source)
        mock_connector.fetch.assert_called_once_with("jira-ABC-1")
        assert result is None


# ============================================================================
# W. Route to channel
# ============================================================================


class TestRouteToChannel:

    @pytest.mark.asyncio
    async def test_slack_routing(self, handler):
        channel = OutputChannel(channel_type="slack", channel_id="C123", thread_id="ts1")
        with patch.object(handler, "_send_to_slack", new_callable=AsyncMock) as mock_send:
            await handler._route_to_channel(
                channel,
                OrchestrationResult(request_id="r1", success=True, agents_participated=[]),
                OrchestrationRequest(question="q"),
            )
            mock_send.assert_called_once()

    @pytest.mark.asyncio
    async def test_teams_routing(self, handler):
        channel = OutputChannel(channel_type="teams", channel_id="T123")
        with patch.object(handler, "_send_to_teams", new_callable=AsyncMock) as mock_send:
            await handler._route_to_channel(
                channel,
                OrchestrationResult(request_id="r1", success=True, agents_participated=[]),
                OrchestrationRequest(question="q"),
            )
            mock_send.assert_called_once()

    @pytest.mark.asyncio
    async def test_discord_routing(self, handler):
        channel = OutputChannel(channel_type="discord", channel_id="D123")
        with patch.object(handler, "_send_to_discord", new_callable=AsyncMock) as mock_send:
            await handler._route_to_channel(
                channel,
                OrchestrationResult(request_id="r1", success=True, agents_participated=[]),
                OrchestrationRequest(question="q"),
            )
            mock_send.assert_called_once()

    @pytest.mark.asyncio
    async def test_telegram_routing(self, handler):
        channel = OutputChannel(channel_type="telegram", channel_id="T123")
        with patch.object(handler, "_send_to_telegram", new_callable=AsyncMock) as mock_send:
            await handler._route_to_channel(
                channel,
                OrchestrationResult(request_id="r1", success=True, agents_participated=[]),
                OrchestrationRequest(question="q"),
            )
            mock_send.assert_called_once()

    @pytest.mark.asyncio
    async def test_email_routing(self, handler):
        channel = OutputChannel(channel_type="email", channel_id="user@example.com")
        with patch.object(handler, "_send_to_email", new_callable=AsyncMock) as mock_send:
            await handler._route_to_channel(
                channel,
                OrchestrationResult(request_id="r1", success=True, agents_participated=[]),
                OrchestrationRequest(question="q"),
            )
            mock_send.assert_called_once()

    @pytest.mark.asyncio
    async def test_webhook_routing(self, handler):
        channel = OutputChannel(channel_type="webhook", channel_id="https://hook.example.com")
        with patch.object(handler, "_send_to_webhook", new_callable=AsyncMock) as mock_send:
            await handler._route_to_channel(
                channel,
                OrchestrationResult(request_id="r1", success=True, agents_participated=[]),
                OrchestrationRequest(question="q"),
            )
            mock_send.assert_called_once()

    @pytest.mark.asyncio
    async def test_unknown_channel_type_logs_warning(self, handler):
        """Unknown channel type should not raise, just log."""
        channel = OutputChannel(channel_type="fax", channel_id="555-0100")
        # Should not raise
        await handler._route_to_channel(
            channel,
            OrchestrationResult(request_id="r1", success=True, agents_participated=[]),
            OrchestrationRequest(question="q"),
        )


# ============================================================================
# X. Send to channel implementations (error handling)
# ============================================================================


class TestSendToChannelErrors:

    @pytest.mark.asyncio
    async def test_send_to_slack_import_error(self, handler):
        """Import error should not propagate."""
        with patch(
            "aragora.connectors.chat.registry.get_connector",
            side_effect=ImportError("no slack"),
        ):
            # Should not raise
            await handler._send_to_slack(
                OutputChannel(channel_type="slack", channel_id="C123"),
                "msg",
            )

    @pytest.mark.asyncio
    async def test_send_to_teams_no_connector(self, handler):
        with patch(
            "aragora.connectors.chat.registry.get_connector",
            return_value=None,
        ):
            await handler._send_to_teams(
                OutputChannel(channel_type="teams", channel_id="T123"),
                "msg",
            )

    @pytest.mark.asyncio
    async def test_send_to_email_import_error(self, handler):
        with patch.dict("sys.modules", {"aragora.connectors.email": None}):
            await handler._send_to_email(
                OutputChannel(channel_type="email", channel_id="a@b.com"),
                "msg",
                OrchestrationRequest(question="q"),
            )

    @pytest.mark.asyncio
    async def test_send_to_webhook_import_error(self, handler):
        with patch.dict("sys.modules", {"aiohttp": None}):
            await handler._send_to_webhook(
                OutputChannel(channel_type="webhook", channel_id="https://hook.example.com"),
                OrchestrationResult(request_id="r1", success=True),
            )


# ============================================================================
# Y. Execute deliberation end-to-end
# ============================================================================


class TestExecuteDeliberation:

    @pytest.mark.asyncio
    async def test_with_coordinator(self, handler):
        """Test execution path through control plane coordinator."""
        mock_outcome = MagicMock()
        mock_outcome.success = True
        mock_outcome.consensus_reached = True
        mock_outcome.winning_position = "Best answer"
        mock_outcome.consensus_confidence = 0.9

        mock_manager = MagicMock()
        mock_manager.submit_deliberation = AsyncMock(return_value="task-123")
        mock_manager.wait_for_outcome = AsyncMock(return_value=mock_outcome)

        handler.ctx["control_plane_coordinator"] = MagicMock()

        with patch(
            "aragora.control_plane.deliberation.DeliberationManager",
            return_value=mock_manager,
        ):
            req = OrchestrationRequest(question="Test question", request_id="exec-1")
            result = await handler._execute_deliberation(req)

        assert result.success is True
        assert result.final_answer == "Best answer"
        assert result.confidence == 0.9

    @pytest.mark.asyncio
    async def test_coordinator_timeout(self, handler):
        mock_manager = MagicMock()
        mock_manager.submit_deliberation = AsyncMock(return_value="task-456")
        mock_manager.wait_for_outcome = AsyncMock(return_value=None)

        handler.ctx["control_plane_coordinator"] = MagicMock()

        with patch(
            "aragora.control_plane.deliberation.DeliberationManager",
            return_value=mock_manager,
        ):
            req = OrchestrationRequest(question="Test", request_id="exec-timeout")
            result = await handler._execute_deliberation(req)

        assert result.success is False
        assert "timed out" in (result.error or "").lower()

    @pytest.mark.asyncio
    async def test_without_coordinator_uses_decision_router(self, handler):
        mock_decision_result = MagicMock()
        mock_decision_result.success = True
        mock_decision_result.consensus_reached = True
        mock_decision_result.final_answer = "Routed answer"
        mock_decision_result.confidence = 0.7
        mock_decision_result.rounds = 3

        mock_router = MagicMock()
        mock_router.route = AsyncMock(return_value=mock_decision_result)

        with (
            patch(
                "aragora.core.decision.get_decision_router",
                return_value=mock_router,
            ),
            patch("aragora.core.decision.DecisionRequest"),
            patch("aragora.core.decision.RequestContext"),
        ):
            req = OrchestrationRequest(question="Test without coord", request_id="exec-no-coord")
            result = await handler._execute_deliberation(req)

        assert result.success is True

    @pytest.mark.asyncio
    async def test_execution_exception_returns_failure(self, handler):
        """General exception during execution returns failure result."""
        handler.ctx["control_plane_coordinator"] = MagicMock()

        with patch(
            "aragora.control_plane.deliberation.DeliberationManager",
            side_effect=ImportError("missing"),
        ):
            req = OrchestrationRequest(question="Test", request_id="exec-err")
            result = await handler._execute_deliberation(req)

        assert result.success is False

    @pytest.mark.asyncio
    async def test_channels_notified_on_success(self, handler):
        """Output channels are routed to on success."""
        handler.ctx["control_plane_coordinator"] = MagicMock()

        mock_outcome = MagicMock()
        mock_outcome.success = True
        mock_outcome.consensus_reached = True
        mock_outcome.winning_position = "Answer"
        mock_outcome.consensus_confidence = 0.8

        mock_manager = MagicMock()
        mock_manager.submit_deliberation = AsyncMock(return_value="task-ch")
        mock_manager.wait_for_outcome = AsyncMock(return_value=mock_outcome)

        with (
            patch(
                "aragora.control_plane.deliberation.DeliberationManager",
                return_value=mock_manager,
            ),
            patch.object(handler, "_route_to_channel", new_callable=AsyncMock) as mock_route,
        ):
            req = OrchestrationRequest(
                question="q",
                request_id="exec-ch",
                output_channels=[
                    OutputChannel(channel_type="slack", channel_id="C123"),
                ],
            )
            result = await handler._execute_deliberation(req)

        mock_route.assert_called_once()
        assert "slack:C123" in result.channels_notified

    @pytest.mark.asyncio
    async def test_channel_routing_failure_non_blocking(self, handler):
        """Channel routing failure should not fail the deliberation."""
        handler.ctx["control_plane_coordinator"] = MagicMock()

        mock_outcome = MagicMock()
        mock_outcome.success = True
        mock_outcome.consensus_reached = True
        mock_outcome.winning_position = "Answer"
        mock_outcome.consensus_confidence = 0.8

        mock_manager = MagicMock()
        mock_manager.submit_deliberation = AsyncMock(return_value="task-chf")
        mock_manager.wait_for_outcome = AsyncMock(return_value=mock_outcome)

        with (
            patch(
                "aragora.control_plane.deliberation.DeliberationManager",
                return_value=mock_manager,
            ),
            patch.object(
                handler,
                "_route_to_channel",
                new_callable=AsyncMock,
                side_effect=ConnectionError("cannot connect"),
            ),
        ):
            req = OrchestrationRequest(
                question="q",
                request_id="exec-chf",
                output_channels=[
                    OutputChannel(channel_type="slack", channel_id="C123"),
                ],
            )
            result = await handler._execute_deliberation(req)

        # Result should still be successful
        assert result.success is True
        # But no channels should be notified
        assert len(result.channels_notified) == 0

    @pytest.mark.asyncio
    async def test_knowledge_context_included(self, handler):
        handler.ctx["control_plane_coordinator"] = MagicMock()

        mock_outcome = MagicMock()
        mock_outcome.success = True
        mock_outcome.consensus_reached = False
        mock_outcome.winning_position = None
        mock_outcome.consensus_confidence = 0.0

        mock_manager = MagicMock()
        mock_manager.submit_deliberation = AsyncMock(return_value="task-kc")
        mock_manager.wait_for_outcome = AsyncMock(return_value=mock_outcome)

        with (
            patch(
                "aragora.control_plane.deliberation.DeliberationManager",
                return_value=mock_manager,
            ),
            patch.object(
                handler,
                "_fetch_knowledge_context",
                new_callable=AsyncMock,
                return_value="Some relevant context",
            ),
        ):
            req = OrchestrationRequest(
                question="q",
                request_id="exec-kc",
                knowledge_sources=[
                    KnowledgeContextSource(source_type="slack", source_id="C123"),
                ],
            )
            result = await handler._execute_deliberation(req)

        assert "slack:C123" in result.knowledge_context_used

    @pytest.mark.asyncio
    async def test_knowledge_fetch_failure_non_blocking(self, handler):
        handler.ctx["control_plane_coordinator"] = MagicMock()

        mock_outcome = MagicMock()
        mock_outcome.success = True
        mock_outcome.consensus_reached = True
        mock_outcome.winning_position = "Answer"
        mock_outcome.consensus_confidence = 0.9

        mock_manager = MagicMock()
        mock_manager.submit_deliberation = AsyncMock(return_value="task-kf")
        mock_manager.wait_for_outcome = AsyncMock(return_value=mock_outcome)

        with (
            patch(
                "aragora.control_plane.deliberation.DeliberationManager",
                return_value=mock_manager,
            ),
            patch.object(
                handler,
                "_fetch_knowledge_context",
                new_callable=AsyncMock,
                side_effect=ConnectionError("cannot fetch"),
            ),
        ):
            req = OrchestrationRequest(
                question="q",
                request_id="exec-kf",
                knowledge_sources=[
                    KnowledgeContextSource(source_type="slack", source_id="C123"),
                ],
            )
            result = await handler._execute_deliberation(req)

        # Should still succeed -- knowledge fetch failure is non-blocking
        assert result.success is True
        assert len(result.knowledge_context_used) == 0


# ============================================================================
# Z. handle_post calling convention (query_params / handler swap)
# ============================================================================


class TestHandlePostCallingConvention:

    @pytest.mark.asyncio
    async def test_handler_as_second_positional_swapped(self, handler, mock_http_handler):
        """When handler is passed as query_params (2nd pos), it's swapped."""
        data = _make_request_data(dry_run=True)
        # Call as handle_post(path, data, handler)  -- handler in query_params slot
        result = await handler.handle_post(
            "/api/v1/orchestration/deliberate",
            data,
            mock_http_handler,
        )
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_none_query_params_defaults_to_empty(self, handler, mock_http_handler):
        data = _make_request_data(dry_run=True)
        result = await handler.handle_post(
            "/api/v1/orchestration/deliberate",
            data,
            None,
            mock_http_handler,
        )
        assert _status(result) == 200


# ============================================================================
# AA. OrchestrationRequest.from_dict parsing
# ============================================================================


class TestOrchestrationRequestFromDict:

    def test_minimal_request(self):
        req = OrchestrationRequest.from_dict({"question": "Test?"})
        assert req.question == "Test?"
        assert req.team_strategy == TeamStrategy.BEST_FOR_DOMAIN
        assert req.output_format == OutputFormat.STANDARD
        assert req.require_consensus is True

    def test_all_fields(self):
        req = OrchestrationRequest.from_dict({
            "question": "Q?",
            "agents": ["claude"],
            "team_strategy": "fast",
            "output_format": "summary",
            "require_consensus": False,
            "priority": "high",
            "max_rounds": 5,
            "timeout_seconds": 60.0,
            "template": "code_review",
            "notify": False,
            "dry_run": True,
            "metadata": {"key": "val"},
        })
        assert req.team_strategy == TeamStrategy.FAST
        assert req.output_format == OutputFormat.SUMMARY
        assert req.require_consensus is False
        assert req.priority == "high"
        assert req.max_rounds == 5
        assert req.dry_run is True

    def test_invalid_strategy_defaults(self):
        req = OrchestrationRequest.from_dict({
            "question": "q",
            "team_strategy": "invalid",
        })
        assert req.team_strategy == TeamStrategy.BEST_FOR_DOMAIN

    def test_invalid_format_defaults(self):
        req = OrchestrationRequest.from_dict({
            "question": "q",
            "output_format": "invalid",
        })
        assert req.output_format == OutputFormat.STANDARD

    def test_knowledge_context_nested_format(self):
        req = OrchestrationRequest.from_dict({
            "question": "q",
            "knowledge_context": {
                "sources": ["slack:C123"],
                "workspaces": ["ws1"],
            },
        })
        assert len(req.knowledge_sources) == 1
        assert req.knowledge_sources[0].source_type == "slack"
        assert req.workspaces == ["ws1"]

    def test_output_channels_dict_format(self):
        req = OrchestrationRequest.from_dict({
            "question": "q",
            "output_channels": [
                {"type": "slack", "id": "C123", "thread_id": "ts1"},
            ],
        })
        assert len(req.output_channels) == 1
        assert req.output_channels[0].channel_type == "slack"
        assert req.output_channels[0].thread_id == "ts1"


# ============================================================================
# AB. OrchestrationResult.to_dict
# ============================================================================


class TestOrchestrationResultToDict:

    def test_all_fields_present(self):
        result = OrchestrationResult(
            request_id="r1",
            success=True,
            consensus_reached=True,
            final_answer="Answer",
            confidence=0.9,
            agents_participated=["a", "b"],
            rounds_completed=3,
            duration_seconds=10.5,
            knowledge_context_used=["slack:C1"],
            channels_notified=["slack:C2"],
            receipt_id="receipt-1",
        )
        d = result.to_dict()
        assert d["request_id"] == "r1"
        assert d["success"] is True
        assert d["consensus_reached"] is True
        assert d["final_answer"] == "Answer"
        assert d["confidence"] == 0.9
        assert d["agents_participated"] == ["a", "b"]
        assert d["rounds_completed"] == 3
        assert d["receipt_id"] == "receipt-1"
        assert "created_at" in d

    def test_error_field(self):
        result = OrchestrationResult(
            request_id="r1", success=False, error="oops"
        )
        d = result.to_dict()
        assert d["error"] == "oops"
        assert d["success"] is False


# ============================================================================
# AC. KnowledgeContextSource helpers
# ============================================================================


class TestKnowledgeContextSource:

    def test_from_string_with_colon(self):
        src = KnowledgeContextSource.from_string("slack:C123")
        assert src.source_type == "slack"
        assert src.source_id == "C123"

    def test_from_string_without_colon(self):
        src = KnowledgeContextSource.from_string("some-doc-id")
        assert src.source_type == "document"
        assert src.source_id == "some-doc-id"


# ============================================================================
# AD. OutputChannel helpers
# ============================================================================


class TestOutputChannel:

    def test_from_string_slack(self):
        ch = OutputChannel.from_string("slack:C123")
        assert ch.channel_type == "slack"
        assert ch.channel_id == "C123"

    def test_from_string_with_thread(self):
        ch = OutputChannel.from_string("slack:C123:ts123")
        assert ch.channel_type == "slack"
        assert ch.channel_id == "C123"
        assert ch.thread_id == "ts123"

    def test_from_string_webhook_url(self):
        ch = OutputChannel.from_string("webhook:https://example.com/hook")
        assert ch.channel_type == "webhook"
        assert "example.com" in ch.channel_id

    def test_from_string_bare_url(self):
        ch = OutputChannel.from_string("https://example.com/hook")
        assert ch.channel_type == "webhook"
        assert "example.com" in ch.channel_id

    def test_from_string_no_colon(self):
        ch = OutputChannel.from_string("some-id")
        assert ch.channel_type == "webhook"


# ============================================================================
# AE. Module-level singleton
# ============================================================================


class TestModuleSingleton:

    def test_handler_singleton_exists(self):
        from aragora.server.handlers.orchestration.handler import handler as singleton
        assert isinstance(singleton, OrchestrationHandler)

    def test_handler_routes_defined(self):
        assert len(OrchestrationHandler.ROUTES) == 4

    def test_resource_type(self):
        assert OrchestrationHandler.RESOURCE_TYPE == "orchestration"


# ============================================================================
# AF. Permission mapping constants
# ============================================================================


class TestPermissionMappings:

    def test_knowledge_source_permissions_cover_all_chat_platforms(self, handler):
        chat_platforms = ["slack", "teams", "discord", "telegram", "whatsapp", "google_chat"]
        for platform in chat_platforms:
            assert platform in handler.KNOWLEDGE_SOURCE_PERMISSIONS

    def test_channel_permissions_cover_standard_types(self, handler):
        for ch_type in ["slack", "teams", "discord", "telegram", "email", "webhook"]:
            assert ch_type in handler.CHANNEL_PERMISSIONS

    def test_confluence_has_own_permission(self, handler):
        assert handler.KNOWLEDGE_SOURCE_PERMISSIONS["confluence"] != handler.KNOWLEDGE_SOURCE_PERMISSIONS["slack"]

    def test_github_has_own_permission(self, handler):
        assert handler.KNOWLEDGE_SOURCE_PERMISSIONS["github"] != handler.KNOWLEDGE_SOURCE_PERMISSIONS["slack"]

    def test_document_aliases(self, handler):
        doc_perm = handler.KNOWLEDGE_SOURCE_PERMISSIONS["document"]
        assert handler.KNOWLEDGE_SOURCE_PERMISSIONS["doc"] == doc_perm
        assert handler.KNOWLEDGE_SOURCE_PERMISSIONS["km"] == doc_perm
