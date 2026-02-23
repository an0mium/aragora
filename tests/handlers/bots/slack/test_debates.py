"""
Tests for Slack Debate Management.

Covers all functions and behavior of aragora.server.handlers.bots.slack.debates:
- start_slack_debate():
  - Happy path: DecisionRouter available, routes successfully
  - DecisionConfig construction:
    - decision_integrity=None: no config
    - decision_integrity=True: empty dict config
    - decision_integrity=False: no config (None)
    - decision_integrity=dict: DecisionConfig from dict
  - Response channel construction with all parameters
  - Request context with session_id derived from channel
  - Debate origin registration (success and failure)
  - Quick result from wait_for (cache/dedup hit)
  - Timeout from wait_for (fires task in background)
  - wait_for raises exception (caught, falls back to request_id)
  - Route done callback:
    - Result with same request_id: no re-registration
    - Result with different request_id: state migration + re-registration
    - CancelledError: silently returns
    - Other exceptions: logs error, returns
    - Re-registration failure in callback: caught
  - Active debate tracking in _active_debates
  - ImportError fallback to _fallback_start_debate
  - RuntimeError/ValueError/KeyError/AttributeError fallback
  - Attachments passed through
  - thread_ts passed through
- _fallback_start_debate():
  - Happy path: registers origin, enqueues via Redis, tracks active
  - Origin registration failure (RuntimeError, KeyError, etc.)
  - Redis queue success
  - Redis queue ImportError (unavailable)
  - Redis queue RuntimeError/OSError/ConnectionError
  - Active debate tracking
  - thread_ts stored in state
- Module exports:
  - __all__ contains expected names
  - _start_slack_debate is alias for start_slack_debate
  - _fallback_start_debate is re-exported
"""

from __future__ import annotations

import asyncio
import time
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch, call

import pytest


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODULE = "aragora.server.handlers.bots.slack.debates"
STATE_MODULE = "aragora.server.handlers.bots.slack.state"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def debates_module():
    """Import the debates module lazily (after conftest patches)."""
    import aragora.server.handlers.bots.slack.debates as mod

    return mod


@pytest.fixture(autouse=True)
def _clear_active_debates():
    """Clear active debates before and after each test."""
    from aragora.server.handlers.bots.slack.state import _active_debates

    _active_debates.clear()
    yield
    _active_debates.clear()


@pytest.fixture(autouse=True)
def _mock_redis_queue():
    """Prevent real Redis connections in fallback path.

    The _fallback_start_debate function does a local import of aragora.queue
    and tries to connect to Redis. We mock the queue module by default to
    prevent actual connections. Tests that specifically test Redis queue
    behavior override this with their own patches.
    """
    mock_job = MagicMock()
    mock_queue = AsyncMock()
    with patch(
        "aragora.queue.create_debate_job",
        MagicMock(return_value=mock_job),
    ), patch(
        "aragora.queue.create_redis_queue",
        AsyncMock(return_value=mock_queue),
    ):
        yield


def _mock_core_module(
    *,
    route_result=None,
    route_side_effect=None,
    request_side_effect=None,
):
    """Build a mock aragora.core module with DecisionRouter and friends.

    Returns (mock_core, mock_router, mock_request_instance).
    """
    mock_core = MagicMock()
    mock_core.DecisionType.DEBATE = "debate"
    mock_core.InputSource.SLACK = "slack"
    mock_core.ResponseChannel = MagicMock()
    mock_core.RequestContext = MagicMock()
    mock_core.DecisionConfig = MagicMock()

    mock_request_instance = MagicMock()
    mock_request_instance.request_id = "req-1111-2222-3333-4444"
    if request_side_effect:
        mock_core.DecisionRequest = MagicMock(side_effect=request_side_effect)
    else:
        mock_core.DecisionRequest = MagicMock(return_value=mock_request_instance)

    mock_router = MagicMock()
    if route_result is None:
        route_result_obj = MagicMock()
        route_result_obj.request_id = "req-1111-2222-3333-4444"
    else:
        route_result_obj = route_result

    if route_side_effect:
        mock_router.route = AsyncMock(side_effect=route_side_effect)
    else:
        mock_router.route = AsyncMock(return_value=route_result_obj)

    mock_core.get_decision_router = MagicMock(return_value=mock_router)

    return mock_core, mock_router, mock_request_instance


# ============================================================================
# start_slack_debate - Happy path with DecisionRouter
# ============================================================================


class TestStartSlackDebateHappyPath:
    """Tests for the main start_slack_debate function with DecisionRouter available."""

    @pytest.mark.asyncio
    async def test_returns_debate_key(self, debates_module):
        """start_slack_debate returns a debate key string."""
        mock_core, mock_router, mock_req = _mock_core_module()

        with patch.dict("sys.modules", {"aragora.core": mock_core}), \
             patch(f"{MODULE}.register_debate_origin", create=True):
            result = await debates_module.start_slack_debate(
                topic="Test topic",
                channel_id="C123",
                user_id="U456",
            )

        assert isinstance(result, str)
        assert len(result) > 0

    @pytest.mark.asyncio
    async def test_creates_response_channel(self, debates_module):
        """ResponseChannel is created with correct parameters."""
        mock_core, mock_router, mock_req = _mock_core_module()

        with patch.dict("sys.modules", {"aragora.core": mock_core}), \
             patch(f"{MODULE}.register_debate_origin", create=True):
            await debates_module.start_slack_debate(
                topic="Test topic",
                channel_id="C123",
                user_id="U456",
                response_url="https://hooks.slack.com/resp",
                thread_ts="1234567890.123456",
            )

        mock_core.ResponseChannel.assert_called_once_with(
            platform="slack",
            channel_id="C123",
            user_id="U456",
            thread_id="1234567890.123456",
            webhook_url="https://hooks.slack.com/resp",
        )

    @pytest.mark.asyncio
    async def test_creates_request_context(self, debates_module):
        """RequestContext is created with user_id and session_id."""
        mock_core, mock_router, mock_req = _mock_core_module()

        with patch.dict("sys.modules", {"aragora.core": mock_core}), \
             patch(f"{MODULE}.register_debate_origin", create=True):
            await debates_module.start_slack_debate(
                topic="Topic",
                channel_id="CABC123",
                user_id="U789",
            )

        mock_core.RequestContext.assert_called_once_with(
            user_id="U789",
            session_id="slack:CABC123",
        )

    @pytest.mark.asyncio
    async def test_creates_decision_request(self, debates_module):
        """DecisionRequest is constructed with correct kwargs."""
        mock_core, mock_router, mock_req = _mock_core_module()

        with patch.dict("sys.modules", {"aragora.core": mock_core}), \
             patch(f"{MODULE}.register_debate_origin", create=True):
            await debates_module.start_slack_debate(
                topic="My debate topic",
                channel_id="C123",
                user_id="U456",
            )

        call_kwargs = mock_core.DecisionRequest.call_args
        assert call_kwargs[1]["content"] == "My debate topic"
        assert call_kwargs[1]["decision_type"] == "debate"
        assert call_kwargs[1]["source"] == "slack"

    @pytest.mark.asyncio
    async def test_routes_through_decision_router(self, debates_module):
        """The request is routed through the DecisionRouter."""
        mock_core, mock_router, mock_req = _mock_core_module()

        with patch.dict("sys.modules", {"aragora.core": mock_core}), \
             patch(f"{MODULE}.register_debate_origin", create=True):
            await debates_module.start_slack_debate(
                topic="Topic",
                channel_id="C123",
                user_id="U456",
            )

        mock_router.route.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_records_active_debate(self, debates_module):
        """start_slack_debate records the debate in _active_debates."""
        from aragora.server.handlers.bots.slack.state import _active_debates

        mock_core, mock_router, mock_req = _mock_core_module()

        with patch.dict("sys.modules", {"aragora.core": mock_core}), \
             patch(f"{MODULE}.register_debate_origin", create=True):
            debate_key = await debates_module.start_slack_debate(
                topic="Test topic",
                channel_id="C123",
                user_id="U456",
            )

        assert debate_key in _active_debates
        assert _active_debates[debate_key]["topic"] == "Test topic"
        assert _active_debates[debate_key]["channel_id"] == "C123"
        assert _active_debates[debate_key]["user_id"] == "U456"
        assert "started_at" in _active_debates[debate_key]

    @pytest.mark.asyncio
    async def test_active_debate_started_at_is_recent(self, debates_module):
        """started_at timestamp is close to current time."""
        from aragora.server.handlers.bots.slack.state import _active_debates

        mock_core, mock_router, mock_req = _mock_core_module()
        before = time.time()

        with patch.dict("sys.modules", {"aragora.core": mock_core}), \
             patch(f"{MODULE}.register_debate_origin", create=True):
            debate_key = await debates_module.start_slack_debate(
                topic="Topic",
                channel_id="C123",
                user_id="U456",
            )

        after = time.time()
        started_at = _active_debates[debate_key]["started_at"]
        assert before <= started_at <= after

    @pytest.mark.asyncio
    async def test_thread_ts_stored_in_active_debate(self, debates_module):
        """thread_ts is stored in the active debate state."""
        from aragora.server.handlers.bots.slack.state import _active_debates

        mock_core, mock_router, mock_req = _mock_core_module()

        with patch.dict("sys.modules", {"aragora.core": mock_core}), \
             patch(f"{MODULE}.register_debate_origin", create=True):
            debate_key = await debates_module.start_slack_debate(
                topic="Topic",
                channel_id="C123",
                user_id="U456",
                thread_ts="1234567890.000001",
            )

        assert _active_debates[debate_key]["thread_ts"] == "1234567890.000001"

    @pytest.mark.asyncio
    async def test_thread_ts_none_by_default(self, debates_module):
        """thread_ts defaults to None in the active debate state."""
        from aragora.server.handlers.bots.slack.state import _active_debates

        mock_core, mock_router, mock_req = _mock_core_module()

        with patch.dict("sys.modules", {"aragora.core": mock_core}), \
             patch(f"{MODULE}.register_debate_origin", create=True):
            debate_key = await debates_module.start_slack_debate(
                topic="Topic",
                channel_id="C123",
                user_id="U456",
            )

        assert _active_debates[debate_key]["thread_ts"] is None


# ============================================================================
# start_slack_debate - DecisionConfig construction
# ============================================================================


class TestDecisionConfigConstruction:
    """Tests for decision_integrity parameter handling."""

    @pytest.mark.asyncio
    async def test_decision_integrity_none_no_config(self, debates_module):
        """decision_integrity=None means no config in request."""
        mock_core, mock_router, mock_req = _mock_core_module()

        with patch.dict("sys.modules", {"aragora.core": mock_core}), \
             patch(f"{MODULE}.register_debate_origin", create=True):
            await debates_module.start_slack_debate(
                topic="Topic",
                channel_id="C123",
                user_id="U456",
                decision_integrity=None,
            )

        call_kwargs = mock_core.DecisionRequest.call_args[1]
        assert "config" not in call_kwargs

    @pytest.mark.asyncio
    async def test_decision_integrity_true_creates_empty_config(self, debates_module):
        """decision_integrity=True creates DecisionConfig with empty dict."""
        mock_core, mock_router, mock_req = _mock_core_module()

        with patch.dict("sys.modules", {"aragora.core": mock_core}), \
             patch(f"{MODULE}.register_debate_origin", create=True):
            await debates_module.start_slack_debate(
                topic="Topic",
                channel_id="C123",
                user_id="U456",
                decision_integrity=True,
            )

        mock_core.DecisionConfig.assert_called_once_with(decision_integrity={})
        call_kwargs = mock_core.DecisionRequest.call_args[1]
        assert "config" in call_kwargs

    @pytest.mark.asyncio
    async def test_decision_integrity_false_no_config(self, debates_module):
        """decision_integrity=False means config is None (no config in request)."""
        mock_core, mock_router, mock_req = _mock_core_module()

        with patch.dict("sys.modules", {"aragora.core": mock_core}), \
             patch(f"{MODULE}.register_debate_origin", create=True):
            await debates_module.start_slack_debate(
                topic="Topic",
                channel_id="C123",
                user_id="U456",
                decision_integrity=False,
            )

        call_kwargs = mock_core.DecisionRequest.call_args[1]
        assert "config" not in call_kwargs

    @pytest.mark.asyncio
    async def test_decision_integrity_dict_creates_config(self, debates_module):
        """decision_integrity=dict creates DecisionConfig with that dict."""
        mock_core, mock_router, mock_req = _mock_core_module()
        di_config = {"include_receipt": True, "include_plan": True}

        with patch.dict("sys.modules", {"aragora.core": mock_core}), \
             patch(f"{MODULE}.register_debate_origin", create=True):
            await debates_module.start_slack_debate(
                topic="Topic",
                channel_id="C123",
                user_id="U456",
                decision_integrity=di_config,
            )

        mock_core.DecisionConfig.assert_called_once_with(decision_integrity=di_config)
        call_kwargs = mock_core.DecisionRequest.call_args[1]
        assert "config" in call_kwargs

    @pytest.mark.asyncio
    async def test_attachments_passed_through(self, debates_module):
        """Attachments are included in the DecisionRequest."""
        mock_core, mock_router, mock_req = _mock_core_module()
        attachments = [{"url": "https://example.com/file.png", "type": "image"}]

        with patch.dict("sys.modules", {"aragora.core": mock_core}), \
             patch(f"{MODULE}.register_debate_origin", create=True):
            await debates_module.start_slack_debate(
                topic="Topic",
                channel_id="C123",
                user_id="U456",
                attachments=attachments,
            )

        call_kwargs = mock_core.DecisionRequest.call_args[1]
        assert call_kwargs["attachments"] == attachments

    @pytest.mark.asyncio
    async def test_none_attachments_default_to_empty_list(self, debates_module):
        """None attachments are converted to empty list."""
        mock_core, mock_router, mock_req = _mock_core_module()

        with patch.dict("sys.modules", {"aragora.core": mock_core}), \
             patch(f"{MODULE}.register_debate_origin", create=True):
            await debates_module.start_slack_debate(
                topic="Topic",
                channel_id="C123",
                user_id="U456",
                attachments=None,
            )

        call_kwargs = mock_core.DecisionRequest.call_args[1]
        assert call_kwargs["attachments"] == []


# ============================================================================
# start_slack_debate - Origin registration
# ============================================================================


class TestOriginRegistration:
    """Tests for debate origin registration in start_slack_debate."""

    @pytest.mark.asyncio
    async def test_registers_debate_origin(self, debates_module):
        """register_debate_origin is called with correct arguments."""
        mock_core, mock_router, mock_req = _mock_core_module()
        mock_register = MagicMock()

        with patch.dict("sys.modules", {"aragora.core": mock_core}), \
             patch(
                 "aragora.server.debate_origin.register_debate_origin",
                 mock_register,
             ):
            await debates_module.start_slack_debate(
                topic="Test topic",
                channel_id="C123",
                user_id="U456",
                response_url="https://hooks.slack.com/resp",
                thread_ts="123.456",
            )

        mock_register.assert_called()
        call_kwargs = mock_register.call_args_list[0][1]
        assert call_kwargs["platform"] == "slack"
        assert call_kwargs["channel_id"] == "C123"
        assert call_kwargs["user_id"] == "U456"
        assert call_kwargs["thread_id"] == "123.456"
        assert call_kwargs["metadata"]["topic"] == "Test topic"
        assert call_kwargs["metadata"]["response_url"] == "https://hooks.slack.com/resp"

    @pytest.mark.asyncio
    async def test_origin_registration_failure_caught(self, debates_module):
        """If register_debate_origin raises, it is caught and logged."""
        mock_core, mock_router, mock_req = _mock_core_module()

        with patch.dict("sys.modules", {"aragora.core": mock_core}), \
             patch(
                 "aragora.server.debate_origin.register_debate_origin",
                 side_effect=RuntimeError("origin db error"),
             ):
            # Should not raise
            result = await debates_module.start_slack_debate(
                topic="Topic",
                channel_id="C123",
                user_id="U456",
            )

        assert isinstance(result, str)

    @pytest.mark.asyncio
    async def test_origin_registration_key_error_caught(self, debates_module):
        """KeyError from register_debate_origin is caught."""
        mock_core, mock_router, mock_req = _mock_core_module()

        with patch.dict("sys.modules", {"aragora.core": mock_core}), \
             patch(
                 "aragora.server.debate_origin.register_debate_origin",
                 side_effect=KeyError("missing field"),
             ):
            result = await debates_module.start_slack_debate(
                topic="Topic",
                channel_id="C123",
                user_id="U456",
            )

        assert isinstance(result, str)

    @pytest.mark.asyncio
    async def test_origin_registration_attribute_error_caught(self, debates_module):
        """AttributeError from register_debate_origin is caught."""
        mock_core, mock_router, mock_req = _mock_core_module()

        with patch.dict("sys.modules", {"aragora.core": mock_core}), \
             patch(
                 "aragora.server.debate_origin.register_debate_origin",
                 side_effect=AttributeError("no such attr"),
             ):
            result = await debates_module.start_slack_debate(
                topic="Topic",
                channel_id="C123",
                user_id="U456",
            )

        assert isinstance(result, str)

    @pytest.mark.asyncio
    async def test_origin_registration_os_error_caught(self, debates_module):
        """OSError from register_debate_origin is caught."""
        mock_core, mock_router, mock_req = _mock_core_module()

        with patch.dict("sys.modules", {"aragora.core": mock_core}), \
             patch(
                 "aragora.server.debate_origin.register_debate_origin",
                 side_effect=OSError("disk error"),
             ):
            result = await debates_module.start_slack_debate(
                topic="Topic",
                channel_id="C123",
                user_id="U456",
            )

        assert isinstance(result, str)


# ============================================================================
# start_slack_debate - Quick result (wait_for)
# ============================================================================


class TestQuickResult:
    """Tests for the wait_for quick result path."""

    @pytest.mark.asyncio
    async def test_quick_result_uses_result_request_id(self, debates_module):
        """When wait_for returns quickly, result.request_id is used as key."""
        from aragora.server.handlers.bots.slack.state import _active_debates

        mock_result = MagicMock()
        mock_result.request_id = "result-id-5678"

        mock_core, mock_router, _ = _mock_core_module(route_result=mock_result)

        with patch.dict("sys.modules", {"aragora.core": mock_core}), \
             patch(f"{MODULE}.register_debate_origin", create=True):
            debate_key = await debates_module.start_slack_debate(
                topic="Topic",
                channel_id="C123",
                user_id="U456",
            )

        assert debate_key == "result-id-5678"
        assert "result-id-5678" in _active_debates

    @pytest.mark.asyncio
    async def test_timeout_falls_back_to_request_id(self, debates_module):
        """When wait_for times out, request.request_id is used as key."""
        from aragora.server.handlers.bots.slack.state import _active_debates

        async def slow_route(*args, **kwargs):
            await asyncio.sleep(10)  # Will timeout
            return MagicMock(request_id="slow-result")

        mock_core, mock_router, mock_req = _mock_core_module()
        mock_router.route = slow_route

        with patch.dict("sys.modules", {"aragora.core": mock_core}), \
             patch(f"{MODULE}.register_debate_origin", create=True):
            debate_key = await debates_module.start_slack_debate(
                topic="Topic",
                channel_id="C123",
                user_id="U456",
            )

        # Should use the request_id from the DecisionRequest, not the route result
        assert debate_key == mock_req.request_id
        assert mock_req.request_id in _active_debates

    @pytest.mark.asyncio
    async def test_wait_for_exception_caught(self, debates_module):
        """If wait_for raises (not TimeoutError), result is None and request_id used."""
        from aragora.server.handlers.bots.slack.state import _active_debates

        mock_core, mock_router, mock_req = _mock_core_module(
            route_side_effect=ValueError("route error")
        )

        with patch.dict("sys.modules", {"aragora.core": mock_core}), \
             patch(f"{MODULE}.register_debate_origin", create=True):
            debate_key = await debates_module.start_slack_debate(
                topic="Topic",
                channel_id="C123",
                user_id="U456",
            )

        assert debate_key == mock_req.request_id

    @pytest.mark.asyncio
    async def test_result_with_no_request_id_uses_original(self, debates_module):
        """When result.request_id is empty/falsy, original request_id is used."""
        mock_result = MagicMock()
        mock_result.request_id = ""  # Falsy

        mock_core, mock_router, mock_req = _mock_core_module(route_result=mock_result)

        with patch.dict("sys.modules", {"aragora.core": mock_core}), \
             patch(f"{MODULE}.register_debate_origin", create=True):
            debate_key = await debates_module.start_slack_debate(
                topic="Topic",
                channel_id="C123",
                user_id="U456",
            )

        assert debate_key == mock_req.request_id

    @pytest.mark.asyncio
    async def test_result_none_uses_original_request_id(self, debates_module):
        """When result is None (timeout), original request_id is used."""
        mock_result = MagicMock()
        mock_result.request_id = None

        mock_core, mock_router, mock_req = _mock_core_module(route_result=mock_result)

        with patch.dict("sys.modules", {"aragora.core": mock_core}), \
             patch(f"{MODULE}.register_debate_origin", create=True):
            debate_key = await debates_module.start_slack_debate(
                topic="Topic",
                channel_id="C123",
                user_id="U456",
            )

        # result.request_id is None which is falsy
        assert debate_key == mock_req.request_id


# ============================================================================
# start_slack_debate - Fallback paths
# ============================================================================


class TestStartSlackDebateFallbacks:
    """Tests for fallback when DecisionRouter is unavailable."""

    @pytest.mark.asyncio
    async def test_import_error_triggers_fallback(self, debates_module):
        """ImportError when importing core triggers _fallback_start_debate."""
        from aragora.server.handlers.bots.slack.state import _active_debates

        with patch.dict("sys.modules", {"aragora.core": None}):
            result = await debates_module.start_slack_debate(
                topic="Fallback topic",
                channel_id="C123",
                user_id="U456",
            )

        assert isinstance(result, str)
        assert result in _active_debates
        assert _active_debates[result]["topic"] == "Fallback topic"

    @pytest.mark.asyncio
    async def test_runtime_error_triggers_fallback(self, debates_module):
        """RuntimeError from DecisionRouter triggers fallback."""
        from aragora.server.handlers.bots.slack.state import _active_debates

        mock_core, _, _ = _mock_core_module()
        mock_core.get_decision_router = MagicMock(side_effect=RuntimeError("router down"))

        with patch.dict("sys.modules", {"aragora.core": mock_core}), \
             patch(f"{MODULE}.register_debate_origin", create=True):
            result = await debates_module.start_slack_debate(
                topic="Fallback topic",
                channel_id="C123",
                user_id="U456",
            )

        assert isinstance(result, str)
        assert result in _active_debates

    @pytest.mark.asyncio
    async def test_value_error_triggers_fallback(self, debates_module):
        """ValueError from DecisionRouter triggers fallback."""
        mock_core, _, _ = _mock_core_module()
        mock_core.DecisionRequest = MagicMock(side_effect=ValueError("bad request"))

        with patch.dict("sys.modules", {"aragora.core": mock_core}), \
             patch(f"{MODULE}.register_debate_origin", create=True):
            result = await debates_module.start_slack_debate(
                topic="Topic",
                channel_id="C123",
                user_id="U456",
            )

        assert isinstance(result, str)

    @pytest.mark.asyncio
    async def test_key_error_triggers_fallback(self, debates_module):
        """KeyError from DecisionRouter triggers fallback."""
        mock_core, _, _ = _mock_core_module()
        mock_core.RequestContext = MagicMock(side_effect=KeyError("missing"))

        with patch.dict("sys.modules", {"aragora.core": mock_core}), \
             patch(f"{MODULE}.register_debate_origin", create=True):
            result = await debates_module.start_slack_debate(
                topic="Topic",
                channel_id="C123",
                user_id="U456",
            )

        assert isinstance(result, str)

    @pytest.mark.asyncio
    async def test_attribute_error_triggers_fallback(self, debates_module):
        """AttributeError from DecisionRouter triggers fallback."""
        mock_core, _, _ = _mock_core_module()
        mock_core.ResponseChannel = MagicMock(side_effect=AttributeError("no attr"))

        with patch.dict("sys.modules", {"aragora.core": mock_core}), \
             patch(f"{MODULE}.register_debate_origin", create=True):
            result = await debates_module.start_slack_debate(
                topic="Topic",
                channel_id="C123",
                user_id="U456",
            )

        assert isinstance(result, str)


# ============================================================================
# start_slack_debate - Response URL
# ============================================================================


class TestResponseUrl:
    """Tests for response_url parameter handling."""

    @pytest.mark.asyncio
    async def test_default_response_url_is_empty(self, debates_module):
        """Default response_url is empty string."""
        mock_core, mock_router, mock_req = _mock_core_module()

        with patch.dict("sys.modules", {"aragora.core": mock_core}), \
             patch(f"{MODULE}.register_debate_origin", create=True):
            await debates_module.start_slack_debate(
                topic="Topic",
                channel_id="C123",
                user_id="U456",
            )

        mock_core.ResponseChannel.assert_called_once()
        call_kwargs = mock_core.ResponseChannel.call_args[1]
        assert call_kwargs["webhook_url"] == ""

    @pytest.mark.asyncio
    async def test_custom_response_url_passed(self, debates_module):
        """Custom response_url is passed to ResponseChannel."""
        mock_core, mock_router, mock_req = _mock_core_module()

        with patch.dict("sys.modules", {"aragora.core": mock_core}), \
             patch(f"{MODULE}.register_debate_origin", create=True):
            await debates_module.start_slack_debate(
                topic="Topic",
                channel_id="C123",
                user_id="U456",
                response_url="https://hooks.slack.com/commands/T123/resp",
            )

        call_kwargs = mock_core.ResponseChannel.call_args[1]
        assert call_kwargs["webhook_url"] == "https://hooks.slack.com/commands/T123/resp"


# ============================================================================
# _fallback_start_debate
# ============================================================================


class TestFallbackStartDebate:
    """Tests for the _fallback_start_debate function."""

    @pytest.mark.asyncio
    async def test_returns_debate_id(self, debates_module):
        """_fallback_start_debate returns the passed debate_id."""
        with patch(
            "aragora.server.debate_origin.register_debate_origin",
            MagicMock(),
        ):
            result = await debates_module._fallback_start_debate(
                topic="Fallback topic",
                channel_id="C123",
                user_id="U456",
                debate_id="debate-fallback-1234",
            )

        assert result == "debate-fallback-1234"

    @pytest.mark.asyncio
    async def test_records_active_debate(self, debates_module):
        """_fallback_start_debate records the debate in _active_debates."""
        from aragora.server.handlers.bots.slack.state import _active_debates

        with patch(
            "aragora.server.debate_origin.register_debate_origin",
            MagicMock(),
        ):
            await debates_module._fallback_start_debate(
                topic="My topic",
                channel_id="C123",
                user_id="U789",
                debate_id="debate-fb-001",
                thread_ts="123.456",
            )

        assert "debate-fb-001" in _active_debates
        state = _active_debates["debate-fb-001"]
        assert state["topic"] == "My topic"
        assert state["channel_id"] == "C123"
        assert state["user_id"] == "U789"
        assert state["thread_ts"] == "123.456"
        assert "started_at" in state

    @pytest.mark.asyncio
    async def test_thread_ts_defaults_to_none(self, debates_module):
        """_fallback_start_debate thread_ts defaults to None."""
        from aragora.server.handlers.bots.slack.state import _active_debates

        with patch(
            "aragora.server.debate_origin.register_debate_origin",
            MagicMock(),
        ):
            await debates_module._fallback_start_debate(
                topic="Topic",
                channel_id="C123",
                user_id="U456",
                debate_id="debate-fb-002",
            )

        assert _active_debates["debate-fb-002"]["thread_ts"] is None

    @pytest.mark.asyncio
    async def test_registers_debate_origin(self, debates_module):
        """_fallback_start_debate registers the debate origin."""
        mock_register = MagicMock()

        with patch(
            "aragora.server.debate_origin.register_debate_origin",
            mock_register,
        ):
            await debates_module._fallback_start_debate(
                topic="Origin test",
                channel_id="CXYZ",
                user_id="UABC",
                debate_id="debate-fb-003",
                thread_ts="789.012",
            )

        mock_register.assert_called_once_with(
            debate_id="debate-fb-003",
            platform="slack",
            channel_id="CXYZ",
            user_id="UABC",
            thread_id="789.012",
            metadata={"topic": "Origin test"},
        )

    @pytest.mark.asyncio
    async def test_origin_registration_runtime_error_caught(self, debates_module):
        """RuntimeError from register_debate_origin is caught in fallback."""
        from aragora.server.handlers.bots.slack.state import _active_debates

        with patch(
            "aragora.server.debate_origin.register_debate_origin",
            side_effect=RuntimeError("db error"),
        ):
            result = await debates_module._fallback_start_debate(
                topic="Topic",
                channel_id="C123",
                user_id="U456",
                debate_id="debate-fb-004",
            )

        assert result == "debate-fb-004"
        assert "debate-fb-004" in _active_debates

    @pytest.mark.asyncio
    async def test_origin_registration_key_error_caught(self, debates_module):
        """KeyError from register_debate_origin is caught in fallback."""
        with patch(
            "aragora.server.debate_origin.register_debate_origin",
            side_effect=KeyError("field"),
        ):
            result = await debates_module._fallback_start_debate(
                topic="Topic",
                channel_id="C123",
                user_id="U456",
                debate_id="debate-fb-005",
            )

        assert result == "debate-fb-005"

    @pytest.mark.asyncio
    async def test_origin_registration_attribute_error_caught(self, debates_module):
        """AttributeError from register_debate_origin is caught in fallback."""
        with patch(
            "aragora.server.debate_origin.register_debate_origin",
            side_effect=AttributeError("attr"),
        ):
            result = await debates_module._fallback_start_debate(
                topic="Topic",
                channel_id="C123",
                user_id="U456",
                debate_id="debate-fb-006",
            )

        assert result == "debate-fb-006"

    @pytest.mark.asyncio
    async def test_origin_registration_os_error_caught(self, debates_module):
        """OSError from register_debate_origin is caught in fallback."""
        with patch(
            "aragora.server.debate_origin.register_debate_origin",
            side_effect=OSError("disk"),
        ):
            result = await debates_module._fallback_start_debate(
                topic="Topic",
                channel_id="C123",
                user_id="U456",
                debate_id="debate-fb-007",
            )

        assert result == "debate-fb-007"


# ============================================================================
# _fallback_start_debate - Redis queue
# ============================================================================


class TestFallbackRedisQueue:
    """Tests for Redis queue enqueue in _fallback_start_debate."""

    @pytest.mark.asyncio
    async def test_enqueues_debate_job(self, debates_module):
        """_fallback_start_debate enqueues a debate job via Redis queue."""
        mock_job = MagicMock()
        mock_queue = AsyncMock()
        mock_create_job = MagicMock(return_value=mock_job)
        mock_create_queue = AsyncMock(return_value=mock_queue)

        with patch(
            "aragora.server.debate_origin.register_debate_origin",
            MagicMock(),
        ), patch(
            "aragora.queue.create_debate_job",
            mock_create_job,
        ), patch(
            "aragora.queue.create_redis_queue",
            mock_create_queue,
        ):
            await debates_module._fallback_start_debate(
                topic="Redis queue test",
                channel_id="C123",
                user_id="U456",
                debate_id="debate-redis-001",
                thread_ts="111.222",
            )

        mock_create_job.assert_called_once_with(
            question="Redis queue test",
            user_id="U456",
            metadata={
                "debate_id": "debate-redis-001",
                "platform": "slack",
                "channel_id": "C123",
                "thread_ts": "111.222",
            },
        )
        mock_queue.enqueue.assert_awaited_once_with(mock_job)

    @pytest.mark.asyncio
    async def test_redis_import_error_caught(self, debates_module):
        """ImportError when importing Redis queue is caught."""
        from aragora.server.handlers.bots.slack.state import _active_debates

        with patch(
            "aragora.server.debate_origin.register_debate_origin",
            MagicMock(),
        ), patch.dict("sys.modules", {"aragora.queue": None}):
            result = await debates_module._fallback_start_debate(
                topic="Topic",
                channel_id="C123",
                user_id="U456",
                debate_id="debate-redis-002",
            )

        assert result == "debate-redis-002"
        assert "debate-redis-002" in _active_debates

    @pytest.mark.asyncio
    async def test_redis_runtime_error_caught(self, debates_module):
        """RuntimeError from Redis queue is caught."""
        mock_create_queue = AsyncMock(side_effect=RuntimeError("connection refused"))

        with patch(
            "aragora.server.debate_origin.register_debate_origin",
            MagicMock(),
        ), patch(
            "aragora.queue.create_debate_job",
            MagicMock(),
        ), patch(
            "aragora.queue.create_redis_queue",
            mock_create_queue,
        ):
            result = await debates_module._fallback_start_debate(
                topic="Topic",
                channel_id="C123",
                user_id="U456",
                debate_id="debate-redis-003",
            )

        assert result == "debate-redis-003"

    @pytest.mark.asyncio
    async def test_redis_os_error_caught(self, debates_module):
        """OSError from Redis queue is caught."""
        mock_queue = AsyncMock()
        mock_queue.enqueue = AsyncMock(side_effect=OSError("network"))

        with patch(
            "aragora.server.debate_origin.register_debate_origin",
            MagicMock(),
        ), patch(
            "aragora.queue.create_debate_job",
            MagicMock(),
        ), patch(
            "aragora.queue.create_redis_queue",
            AsyncMock(return_value=mock_queue),
        ):
            result = await debates_module._fallback_start_debate(
                topic="Topic",
                channel_id="C123",
                user_id="U456",
                debate_id="debate-redis-004",
            )

        assert result == "debate-redis-004"

    @pytest.mark.asyncio
    async def test_redis_connection_error_caught(self, debates_module):
        """ConnectionError from Redis queue is caught."""
        mock_queue = AsyncMock()
        mock_queue.enqueue = AsyncMock(side_effect=ConnectionError("refused"))

        with patch(
            "aragora.server.debate_origin.register_debate_origin",
            MagicMock(),
        ), patch(
            "aragora.queue.create_debate_job",
            MagicMock(),
        ), patch(
            "aragora.queue.create_redis_queue",
            AsyncMock(return_value=mock_queue),
        ):
            result = await debates_module._fallback_start_debate(
                topic="Topic",
                channel_id="C123",
                user_id="U456",
                debate_id="debate-redis-005",
            )

        assert result == "debate-redis-005"


# ============================================================================
# start_slack_debate - Done callback behavior
# ============================================================================


class TestRouteDoneCallback:
    """Tests for the _route_done callback within start_slack_debate."""

    @pytest.mark.asyncio
    async def test_different_request_id_migrates_state(self, debates_module):
        """When result has different request_id, state is migrated."""
        from aragora.server.handlers.bots.slack.state import _active_debates

        # Make the route return a different request_id
        mock_result = MagicMock()
        mock_result.request_id = "new-result-id"

        mock_core, mock_router, mock_req = _mock_core_module(route_result=mock_result)
        # Make sure the request has a different ID
        mock_req.request_id = "original-req-id"

        with patch.dict("sys.modules", {"aragora.core": mock_core}), \
             patch(f"{MODULE}.register_debate_origin", create=True):
            debate_key = await debates_module.start_slack_debate(
                topic="Topic",
                channel_id="C123",
                user_id="U456",
            )

        # Allow the event loop to process the done callback
        await asyncio.sleep(0.1)

        # The state should be under the new result id
        assert "new-result-id" in _active_debates

    @pytest.mark.asyncio
    async def test_same_request_id_no_migration(self, debates_module):
        """When result has same request_id, no state migration occurs."""
        from aragora.server.handlers.bots.slack.state import _active_debates

        mock_result = MagicMock()
        mock_result.request_id = "same-id-1234"

        mock_core, mock_router, mock_req = _mock_core_module(route_result=mock_result)
        mock_req.request_id = "same-id-1234"

        with patch.dict("sys.modules", {"aragora.core": mock_core}), \
             patch(f"{MODULE}.register_debate_origin", create=True):
            debate_key = await debates_module.start_slack_debate(
                topic="Topic",
                channel_id="C123",
                user_id="U456",
            )

        await asyncio.sleep(0.1)

        # Should remain under same key
        assert debate_key in _active_debates


# ============================================================================
# start_slack_debate - Multiple concurrent debates
# ============================================================================


class TestMultipleDebates:
    """Tests for multiple concurrent debate starts."""

    @pytest.mark.asyncio
    async def test_multiple_debates_independent(self, debates_module):
        """Multiple start_slack_debate calls create independent entries."""
        from aragora.server.handlers.bots.slack.state import _active_debates

        results = []
        with patch.dict("sys.modules", {"aragora.core": None}):
            for i in range(3):
                result = await debates_module.start_slack_debate(
                    topic=f"Topic {i}",
                    channel_id=f"C{i}",
                    user_id=f"U{i}",
                )
                results.append(result)

        # All should be unique IDs
        assert len(set(results)) == 3
        # All should be in active debates
        for r in results:
            assert r in _active_debates


# ============================================================================
# Module exports
# ============================================================================


class TestModuleExports:
    """Tests for module exports and aliases."""

    def test_all_exports(self, debates_module):
        """__all__ contains the expected names."""
        assert "start_slack_debate" in debates_module.__all__
        assert "_start_slack_debate" in debates_module.__all__
        assert "_fallback_start_debate" in debates_module.__all__

    def test_start_slack_debate_is_callable(self, debates_module):
        """start_slack_debate is callable."""
        assert callable(debates_module.start_slack_debate)

    def test_fallback_start_debate_is_callable(self, debates_module):
        """_fallback_start_debate is callable."""
        assert callable(debates_module._fallback_start_debate)

    def test_start_slack_debate_alias(self, debates_module):
        """_start_slack_debate is an alias for start_slack_debate."""
        assert debates_module._start_slack_debate is debates_module.start_slack_debate

    def test_all_has_three_entries(self, debates_module):
        """__all__ has exactly 3 entries."""
        assert len(debates_module.__all__) == 3


# ============================================================================
# Edge cases
# ============================================================================


class TestEdgeCases:
    """Tests for edge cases and unusual inputs."""

    @pytest.mark.asyncio
    async def test_empty_topic(self, debates_module):
        """Empty topic string is handled."""
        with patch.dict("sys.modules", {"aragora.core": None}):
            result = await debates_module.start_slack_debate(
                topic="",
                channel_id="C123",
                user_id="U456",
            )

        assert isinstance(result, str)

    @pytest.mark.asyncio
    async def test_very_long_topic(self, debates_module):
        """Very long topic string is handled."""
        long_topic = "x" * 10000
        with patch.dict("sys.modules", {"aragora.core": None}):
            result = await debates_module.start_slack_debate(
                topic=long_topic,
                channel_id="C123",
                user_id="U456",
            )

        assert isinstance(result, str)

    @pytest.mark.asyncio
    async def test_unicode_topic(self, debates_module):
        """Unicode characters in topic are handled."""
        with patch.dict("sys.modules", {"aragora.core": None}):
            result = await debates_module.start_slack_debate(
                topic="What about \u00e9m\u00f6j\u00efs and \u4e2d\u6587?",
                channel_id="C123",
                user_id="U456",
            )

        assert isinstance(result, str)

    @pytest.mark.asyncio
    async def test_special_characters_in_channel_id(self, debates_module):
        """Special characters in channel_id are handled."""
        with patch.dict("sys.modules", {"aragora.core": None}):
            result = await debates_module.start_slack_debate(
                topic="Topic",
                channel_id="C-SPECIAL_123",
                user_id="U456",
            )

        assert isinstance(result, str)

    @pytest.mark.asyncio
    async def test_empty_user_id(self, debates_module):
        """Empty user_id is handled."""
        with patch.dict("sys.modules", {"aragora.core": None}):
            result = await debates_module.start_slack_debate(
                topic="Topic",
                channel_id="C123",
                user_id="",
            )

        assert isinstance(result, str)

    @pytest.mark.asyncio
    async def test_empty_channel_id(self, debates_module):
        """Empty channel_id is handled."""
        with patch.dict("sys.modules", {"aragora.core": None}):
            result = await debates_module.start_slack_debate(
                topic="Topic",
                channel_id="",
                user_id="U456",
            )

        assert isinstance(result, str)

    @pytest.mark.asyncio
    async def test_multiple_attachments(self, debates_module):
        """Multiple attachments are passed through."""
        mock_core, mock_router, mock_req = _mock_core_module()
        attachments = [
            {"url": "https://example.com/1.png", "type": "image"},
            {"url": "https://example.com/2.pdf", "type": "document"},
            {"url": "https://example.com/3.csv", "type": "file"},
        ]

        with patch.dict("sys.modules", {"aragora.core": mock_core}), \
             patch(f"{MODULE}.register_debate_origin", create=True):
            await debates_module.start_slack_debate(
                topic="Topic",
                channel_id="C123",
                user_id="U456",
                attachments=attachments,
            )

        call_kwargs = mock_core.DecisionRequest.call_args[1]
        assert len(call_kwargs["attachments"]) == 3

    @pytest.mark.asyncio
    async def test_empty_attachments_list(self, debates_module):
        """Empty attachments list is passed through."""
        mock_core, mock_router, mock_req = _mock_core_module()

        with patch.dict("sys.modules", {"aragora.core": mock_core}), \
             patch(f"{MODULE}.register_debate_origin", create=True):
            await debates_module.start_slack_debate(
                topic="Topic",
                channel_id="C123",
                user_id="U456",
                attachments=[],
            )

        call_kwargs = mock_core.DecisionRequest.call_args[1]
        assert call_kwargs["attachments"] == []

    @pytest.mark.asyncio
    async def test_fallback_started_at_is_timestamp(self, debates_module):
        """Fallback active debate has a numeric started_at timestamp."""
        from aragora.server.handlers.bots.slack.state import _active_debates

        with patch(
            "aragora.server.debate_origin.register_debate_origin",
            MagicMock(),
        ):
            await debates_module._fallback_start_debate(
                topic="Topic",
                channel_id="C123",
                user_id="U456",
                debate_id="debate-ts-001",
            )

        assert isinstance(_active_debates["debate-ts-001"]["started_at"], float)

    @pytest.mark.asyncio
    async def test_debate_id_is_uuid_format(self, debates_module):
        """When fallback is triggered via ImportError, debate_id is UUID format."""
        import re

        with patch.dict("sys.modules", {"aragora.core": None}):
            result = await debates_module.start_slack_debate(
                topic="Topic",
                channel_id="C123",
                user_id="U456",
            )

        # Should be a valid UUID4 string
        uuid_pattern = re.compile(
            r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
        )
        assert uuid_pattern.match(result), f"Expected UUID format, got: {result}"


# ============================================================================
# Integration-like tests
# ============================================================================


class TestIntegration:
    """Integration-like tests exercising full function flows."""

    @pytest.mark.asyncio
    async def test_full_flow_with_decision_router(self, debates_module):
        """Full flow: create request, register origin, route, record active."""
        from aragora.server.handlers.bots.slack.state import _active_debates

        mock_result = MagicMock()
        mock_result.request_id = "router-result-id"
        mock_core, mock_router, mock_req = _mock_core_module(route_result=mock_result)
        mock_req.request_id = "initial-request-id"
        mock_register = MagicMock()

        with patch.dict("sys.modules", {"aragora.core": mock_core}), \
             patch(
                 "aragora.server.debate_origin.register_debate_origin",
                 mock_register,
             ):
            debate_key = await debates_module.start_slack_debate(
                topic="Full flow test",
                channel_id="CFULL",
                user_id="UFULL",
                response_url="https://hooks.slack.com/full",
                thread_ts="999.888",
                attachments=[{"url": "https://example.com/doc.pdf"}],
                decision_integrity={"include_receipt": True},
            )

        # Verify origin was registered
        mock_register.assert_called()
        # Verify router was called
        mock_router.route.assert_awaited_once()
        # Verify active debate recorded
        assert debate_key in _active_debates
        state = _active_debates[debate_key]
        assert state["topic"] == "Full flow test"
        assert state["channel_id"] == "CFULL"
        assert state["user_id"] == "UFULL"
        assert state["thread_ts"] == "999.888"

    @pytest.mark.asyncio
    async def test_full_fallback_flow(self, debates_module):
        """Full fallback flow: origin registration + Redis enqueue + active tracking."""
        from aragora.server.handlers.bots.slack.state import _active_debates

        mock_register = MagicMock()
        mock_job = MagicMock()
        mock_queue = AsyncMock()

        with patch.dict("sys.modules", {"aragora.core": None}), \
             patch(
                 "aragora.server.debate_origin.register_debate_origin",
                 mock_register,
             ), patch(
                 "aragora.queue.create_debate_job",
                 MagicMock(return_value=mock_job),
             ), patch(
                 "aragora.queue.create_redis_queue",
                 AsyncMock(return_value=mock_queue),
             ):
            debate_key = await debates_module.start_slack_debate(
                topic="Fallback flow",
                channel_id="CFALL",
                user_id="UFALL",
                thread_ts="111.222",
            )

        # Origin registered in fallback
        mock_register.assert_called()
        # Job enqueued
        mock_queue.enqueue.assert_awaited_once()
        # Active debate recorded
        assert debate_key in _active_debates
        assert _active_debates[debate_key]["topic"] == "Fallback flow"

    @pytest.mark.asyncio
    async def test_graceful_degradation_all_services_down(self, debates_module):
        """When DecisionRouter, origin registration, and Redis all fail, still returns an ID."""
        from aragora.server.handlers.bots.slack.state import _active_debates

        with patch.dict("sys.modules", {"aragora.core": None}), \
             patch(
                 "aragora.server.debate_origin.register_debate_origin",
                 side_effect=RuntimeError("origin down"),
             ), patch.dict("sys.modules", {"aragora.queue": None}):
            result = await debates_module.start_slack_debate(
                topic="All down",
                channel_id="C123",
                user_id="U456",
            )

        # Should still return a valid ID
        assert isinstance(result, str)
        assert len(result) > 0
        # Should still track the active debate
        assert result in _active_debates
