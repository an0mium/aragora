"""Tests for workflow execution operations (aragora/server/handlers/workflows/execution.py).

Covers all public and internal functions in the execution module:
- _normalize_list() helper
- _extract_notification_context() metadata builder
- _should_notify_chat() event filtering
- _format_workflow_message() message formatting
- _dispatch_chat_message() chat dispatch (async)
- _schedule_chat_dispatch() async scheduling
- _build_event_callback() event bridge callback factory
- execute_workflow() main execution (async)
- get_execution() status retrieval (async)
- list_executions() listing (async)
- terminate_execution() termination (async)
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch, call

import pytest

from aragora.server.handlers.workflows.execution import (
    _normalize_list,
    _extract_notification_context,
    _should_notify_chat,
    _format_workflow_message,
    _dispatch_chat_message,
    _schedule_chat_dispatch,
    _build_event_callback,
    execute_workflow,
    get_execution,
    list_executions,
    terminate_execution,
)


# ---------------------------------------------------------------------------
# Module-level patch targets
# ---------------------------------------------------------------------------

PATCH_MOD = "aragora.server.handlers.workflows.execution"
PATCH_CORE = "aragora.server.handlers.workflows.core"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_workflow(metadata=None):
    """Create a mock workflow object."""
    wf = MagicMock()
    wf.metadata = metadata if metadata is not None else {}
    return wf


def _make_engine_result(success=True, final_output=None, steps=None, error=None, duration_ms=100):
    """Create a mock engine execution result."""
    result = MagicMock()
    result.success = success
    result.final_output = final_output or {"result": "ok"}
    result.steps = steps or []
    result.error = error
    result.total_duration_ms = duration_ms
    return result


def _make_step_result(step_id="s1", step_name="Step 1"):
    """Create a mock StepResult for _step_result_to_dict."""
    step = MagicMock()
    step.step_id = step_id
    step.step_name = step_name
    step.status = MagicMock(value="completed")
    step.started_at = datetime(2026, 1, 1, tzinfo=timezone.utc)
    step.completed_at = datetime(2026, 1, 1, 0, 0, 1, tzinfo=timezone.utc)
    step.duration_ms = 1000
    step.output = {"data": "test"}
    step.error = None
    step.metrics = {}
    step.retry_count = 0
    return step


# ---------------------------------------------------------------------------
# _normalize_list
# ---------------------------------------------------------------------------


class TestNormalizeList:
    """Test _normalize_list() helper."""

    def test_none_returns_none(self):
        assert _normalize_list(None) is None

    def test_string_single_value(self):
        assert _normalize_list("foo") == ["foo"]

    def test_string_csv(self):
        assert _normalize_list("foo, bar, baz") == ["foo", "bar", "baz"]

    def test_string_csv_with_empty_parts(self):
        assert _normalize_list("foo,,bar,") == ["foo", "bar"]

    def test_empty_string_returns_none(self):
        assert _normalize_list("") is None

    def test_string_only_commas_returns_none(self):
        assert _normalize_list(",,,") is None

    def test_list_of_strings(self):
        assert _normalize_list(["a", "b", "c"]) == ["a", "b", "c"]

    def test_tuple_of_strings(self):
        result = _normalize_list(("x", "y"))
        assert result == ["x", "y"]

    def test_set_of_strings(self):
        result = _normalize_list({"one"})
        assert result == ["one"]

    def test_list_with_empty_strings_filtered(self):
        assert _normalize_list(["a", "", "  ", "b"]) == ["a", "b"]

    def test_empty_list_returns_none(self):
        assert _normalize_list([]) is None

    def test_list_of_all_empty_returns_none(self):
        assert _normalize_list(["", " ", "  "]) is None

    def test_list_with_non_string_items(self):
        assert _normalize_list([1, 2, 3]) == ["1", "2", "3"]

    def test_integer_returns_none(self):
        assert _normalize_list(42) is None

    def test_dict_returns_none(self):
        assert _normalize_list({"key": "val"}) is None

    def test_boolean_returns_none(self):
        assert _normalize_list(True) is None


# ---------------------------------------------------------------------------
# _extract_notification_context
# ---------------------------------------------------------------------------


class TestExtractNotificationContext:
    """Test _extract_notification_context() metadata builder."""

    def test_basic_no_notification_fields(self):
        wf = _make_workflow()
        metadata, notify = _extract_notification_context(
            workflow=wf, inputs={}, tenant_id="t1", user_id="u1", org_id="o1"
        )
        assert metadata["tenant_id"] == "t1"
        assert metadata["user_id"] == "u1"
        assert metadata["org_id"] == "o1"
        assert notify["channel_targets"] == []
        assert notify["thread_id"] is None
        assert notify["thread_id_by_platform"] == {}
        assert notify["notify_steps"] is False

    def test_channel_targets_from_inputs(self):
        wf = _make_workflow()
        metadata, notify = _extract_notification_context(
            workflow=wf,
            inputs={"channel_targets": "slack:#general,telegram:chat_123"},
            tenant_id="t1",
            user_id=None,
            org_id=None,
        )
        assert notify["channel_targets"] == ["slack:#general", "telegram:chat_123"]
        assert metadata["channel_targets"] == ["slack:#general", "telegram:chat_123"]
        assert "user_id" not in metadata
        assert "org_id" not in metadata

    def test_channel_targets_from_chat_targets_input(self):
        wf = _make_workflow()
        _, notify = _extract_notification_context(
            workflow=wf,
            inputs={"chat_targets": ["slack:#dev"]},
            tenant_id="t1",
            user_id=None,
            org_id=None,
        )
        assert notify["channel_targets"] == ["slack:#dev"]

    def test_channel_targets_from_notify_channels_input(self):
        wf = _make_workflow()
        _, notify = _extract_notification_context(
            workflow=wf,
            inputs={"notify_channels": "slack:#alerts"},
            tenant_id="t1",
            user_id=None,
            org_id=None,
        )
        assert notify["channel_targets"] == ["slack:#alerts"]

    def test_channel_targets_from_workflow_metadata(self):
        wf = _make_workflow(metadata={"channel_targets": ["slack:#ops"]})
        _, notify = _extract_notification_context(
            workflow=wf, inputs={}, tenant_id="t1", user_id=None, org_id=None
        )
        assert notify["channel_targets"] == ["slack:#ops"]

    def test_approval_targets_from_inputs(self):
        wf = _make_workflow()
        metadata, _ = _extract_notification_context(
            workflow=wf,
            inputs={"approval_targets": ["user:alice"]},
            tenant_id="t1",
            user_id=None,
            org_id=None,
        )
        assert metadata.get("approval_targets") == ["user:alice"]

    def test_approval_targets_from_metadata(self):
        wf = _make_workflow(metadata={"approval_targets": "user:bob"})
        metadata, _ = _extract_notification_context(
            workflow=wf, inputs={}, tenant_id="t1", user_id=None, org_id=None
        )
        # metadata starts as dict(base_meta) so "approval_targets" already exists
        # as the raw string. setdefault won't override the existing key.
        assert metadata.get("approval_targets") == "user:bob"

    def test_notify_steps_true_from_inputs(self):
        wf = _make_workflow()
        _, notify = _extract_notification_context(
            workflow=wf,
            inputs={"notify_steps": True},
            tenant_id="t1",
            user_id=None,
            org_id=None,
        )
        assert notify["notify_steps"] is True

    def test_notify_steps_from_metadata_default_false(self):
        wf = _make_workflow(metadata={})
        _, notify = _extract_notification_context(
            workflow=wf, inputs={}, tenant_id="t1", user_id=None, org_id=None
        )
        assert notify["notify_steps"] is False

    def test_thread_id_from_inputs(self):
        wf = _make_workflow()
        metadata, notify = _extract_notification_context(
            workflow=wf,
            inputs={"thread_id": "t_abc"},
            tenant_id="t1",
            user_id=None,
            org_id=None,
        )
        assert notify["thread_id"] == "t_abc"
        assert metadata["thread_id"] == "t_abc"

    def test_thread_id_from_origin_thread_id(self):
        wf = _make_workflow()
        _, notify = _extract_notification_context(
            workflow=wf,
            inputs={"origin_thread_id": "orig_123"},
            tenant_id="t1",
            user_id=None,
            org_id=None,
        )
        assert notify["thread_id"] == "orig_123"

    def test_thread_id_by_platform_from_inputs(self):
        wf = _make_workflow()
        platforms = {"slack": "ts_1", "telegram": "msg_2"}
        metadata, notify = _extract_notification_context(
            workflow=wf,
            inputs={"thread_id_by_platform": platforms},
            tenant_id="t1",
            user_id=None,
            org_id=None,
        )
        assert notify["thread_id_by_platform"] == {"slack": "ts_1", "telegram": "msg_2"}
        assert metadata["thread_id_by_platform"] == {"slack": "ts_1", "telegram": "msg_2"}

    def test_thread_id_by_platform_ignores_none_keys(self):
        wf = _make_workflow()
        platforms = {None: "ts_1", "telegram": None, "slack": "ts_2"}
        _, notify = _extract_notification_context(
            workflow=wf,
            inputs={"thread_id_by_platform": platforms},
            tenant_id="t1",
            user_id=None,
            org_id=None,
        )
        assert notify["thread_id_by_platform"] == {"slack": "ts_2"}

    def test_thread_id_by_platform_non_dict_ignored(self):
        wf = _make_workflow()
        _, notify = _extract_notification_context(
            workflow=wf,
            inputs={"thread_id_by_platform": "not_a_dict"},
            tenant_id="t1",
            user_id=None,
            org_id=None,
        )
        assert notify["thread_id_by_platform"] == {}

    def test_fallback_to_default_chat_targets(self):
        """When notify_channels is truthy but channel_targets is empty, tries default."""
        wf = _make_workflow()
        with patch(
            f"{PATCH_MOD}.get_default_chat_targets",
            return_value=["slack:#fallback"],
            create=True,
        ):
            # Monkey-patch the import inside _extract_notification_context
            with patch.dict(
                "sys.modules",
                {
                    "aragora.approvals.chat": MagicMock(
                        get_default_chat_targets=MagicMock(return_value=["slack:#fallback"])
                    ),
                },
            ):
                metadata, notify = _extract_notification_context(
                    workflow=wf,
                    inputs={"notify_channels": True},
                    tenant_id="t1",
                    user_id=None,
                    org_id=None,
                )
        # The notify_channels input is truthy but _normalize_list(True) returns None,
        # so channel_targets starts as None. The fallback code checks if
        # inputs.get("notify_channels") is truthy, then calls get_default_chat_targets.
        assert notify["channel_targets"] == ["slack:#fallback"]

    def test_fallback_import_error_handled(self):
        """When aragora.approvals.chat is not available, gracefully returns None."""
        wf = _make_workflow()
        # Ensure the module is not importable
        with patch.dict("sys.modules", {"aragora.approvals.chat": None}):
            metadata, notify = _extract_notification_context(
                workflow=wf,
                inputs={"notify_channels": True},
                tenant_id="t1",
                user_id=None,
                org_id=None,
            )
        assert notify["channel_targets"] == []

    def test_metadata_preserves_workflow_metadata(self):
        wf = _make_workflow(metadata={"custom_key": "custom_val"})
        metadata, _ = _extract_notification_context(
            workflow=wf, inputs={}, tenant_id="t1", user_id=None, org_id=None
        )
        assert metadata["custom_key"] == "custom_val"

    def test_non_dict_workflow_metadata_ignored(self):
        wf = _make_workflow(metadata="not_a_dict")
        metadata, _ = _extract_notification_context(
            workflow=wf, inputs={}, tenant_id="t1", user_id=None, org_id=None
        )
        assert metadata["tenant_id"] == "t1"

    def test_chat_targets_setdefault_in_metadata(self):
        """channel_targets also sets chat_targets via setdefault."""
        wf = _make_workflow()
        metadata, _ = _extract_notification_context(
            workflow=wf,
            inputs={"channel_targets": ["slack:#test"]},
            tenant_id="t1",
            user_id=None,
            org_id=None,
        )
        assert metadata["chat_targets"] == ["slack:#test"]


# ---------------------------------------------------------------------------
# _should_notify_chat
# ---------------------------------------------------------------------------


class TestShouldNotifyChat:
    """Test _should_notify_chat() event filtering."""

    @pytest.mark.parametrize(
        "event_type",
        [
            "workflow_start",
            "workflow_complete",
            "workflow_failed",
            "workflow_terminated",
            "workflow_human_approval_required",
            "workflow_human_approval_received",
            "workflow_human_approval_timeout",
        ],
    )
    def test_critical_events_always_notify(self, event_type):
        assert _should_notify_chat(event_type, notify_steps=False) is True
        assert _should_notify_chat(event_type, notify_steps=True) is True

    @pytest.mark.parametrize(
        "event_type",
        [
            "workflow_step_complete",
            "workflow_step_failed",
            "workflow_step_skipped",
        ],
    )
    def test_step_events_only_when_notify_steps(self, event_type):
        assert _should_notify_chat(event_type, notify_steps=False) is False
        assert _should_notify_chat(event_type, notify_steps=True) is True

    def test_unknown_event_does_not_notify(self):
        assert _should_notify_chat("workflow_debug", notify_steps=False) is False
        assert _should_notify_chat("workflow_debug", notify_steps=True) is False

    def test_unrelated_event_does_not_notify(self):
        assert _should_notify_chat("debate_start", notify_steps=True) is False


# ---------------------------------------------------------------------------
# _format_workflow_message
# ---------------------------------------------------------------------------


class TestFormatWorkflowMessage:
    """Test _format_workflow_message() formatting."""

    def test_workflow_start(self):
        msg = _format_workflow_message(
            "workflow_start",
            {"workflow_name": "MyFlow", "execution_id": "exec_1"},
        )
        assert "started" in msg.lower()
        assert "MyFlow" in msg
        assert "exec_1" in msg

    def test_workflow_complete(self):
        msg = _format_workflow_message(
            "workflow_complete",
            {"workflow_name": "MyFlow", "workflow_id": "exec_2"},
        )
        assert "completed" in msg.lower()
        assert "MyFlow" in msg

    def test_workflow_failed_with_error(self):
        msg = _format_workflow_message(
            "workflow_failed",
            {"workflow_name": "MyFlow", "execution_id": "exec_3", "error": "timeout"},
        )
        assert "failed" in msg.lower()
        assert "timeout" in msg

    def test_workflow_failed_without_error(self):
        msg = _format_workflow_message(
            "workflow_failed",
            {"workflow_name": "MyFlow", "execution_id": "exec_3"},
        )
        assert "failed" in msg.lower()
        assert "MyFlow" in msg

    def test_workflow_terminated(self):
        msg = _format_workflow_message(
            "workflow_terminated",
            {"workflow_name": "MyFlow", "execution_id": "exec_4"},
        )
        assert "terminated" in msg.lower()

    def test_approval_required(self):
        msg = _format_workflow_message(
            "workflow_human_approval_required",
            {"workflow_name": "MyFlow", "execution_id": "exec_5", "request_id": "req_1"},
        )
        assert "approval required" in msg.lower()
        assert "req_1" in msg

    def test_approval_received(self):
        msg = _format_workflow_message(
            "workflow_human_approval_received",
            {"workflow_name": "MyFlow", "execution_id": "exec_6", "status": "approved"},
        )
        assert "approved" in msg.lower()

    def test_approval_timeout(self):
        msg = _format_workflow_message(
            "workflow_human_approval_timeout",
            {"workflow_name": "MyFlow", "execution_id": "exec_7"},
        )
        assert "timed out" in msg.lower()

    def test_step_event_with_step_name(self):
        msg = _format_workflow_message(
            "workflow_step_complete",
            {"workflow_name": "MyFlow", "step_name": "Validate", "status": "completed"},
        )
        assert "Validate" in msg
        assert "COMPLETED" in msg

    def test_step_event_uses_event_type_status_fallback(self):
        msg = _format_workflow_message(
            "workflow_step_failed",
            {"workflow_name": "MyFlow", "step_name": "Build"},
        )
        assert "FAILED" in msg

    def test_unknown_event_type(self):
        msg = _format_workflow_message(
            "workflow_custom_event",
            {"workflow_name": "MyFlow", "execution_id": "exec_8"},
        )
        assert "Workflow update" in msg
        assert "MyFlow" in msg

    def test_fallback_to_definition_id(self):
        msg = _format_workflow_message(
            "workflow_start",
            {"definition_id": "def_1", "execution_id": "exec_9"},
        )
        assert "def_1" in msg

    def test_fallback_to_workflow_as_name(self):
        msg = _format_workflow_message("workflow_start", {})
        assert "workflow" in msg.lower()

    def test_step_event_without_step_name(self):
        """Step events without step_name fall through to default."""
        msg = _format_workflow_message(
            "workflow_step_complete",
            {"workflow_name": "MyFlow", "execution_id": "exec_10"},
        )
        assert "Workflow update" in msg


# ---------------------------------------------------------------------------
# _dispatch_chat_message (async)
# ---------------------------------------------------------------------------


class TestDispatchChatMessage:
    """Test _dispatch_chat_message() async dispatch."""

    @pytest.mark.asyncio
    async def test_no_targets_returns_immediately(self):
        # Should not raise or do anything
        await _dispatch_chat_message(
            text="hello",
            channel_targets=[],
            thread_id=None,
            thread_id_by_platform={},
        )

    @pytest.mark.asyncio
    async def test_import_error_returns_gracefully(self):
        with patch.dict("sys.modules", {"aragora.approvals.chat": None}):
            await _dispatch_chat_message(
                text="hello",
                channel_targets=["slack:#general"],
                thread_id=None,
                thread_id_by_platform={},
            )

    @pytest.mark.asyncio
    async def test_sends_to_configured_connector(self):
        mock_connector = MagicMock()
        mock_connector.is_configured = True
        mock_connector.send_message = AsyncMock()

        mock_parse = MagicMock(return_value={"slack": ["#general"]})

        with patch.dict(
            "sys.modules",
            {
                "aragora.approvals.chat": MagicMock(parse_chat_targets=mock_parse),
                "aragora.connectors.chat.registry": MagicMock(
                    get_connector=MagicMock(return_value=mock_connector)
                ),
            },
        ):
            await _dispatch_chat_message(
                text="test msg",
                channel_targets=["slack:#general"],
                thread_id="t_1",
                thread_id_by_platform={},
            )
        mock_connector.send_message.assert_awaited_once_with(
            channel_id="#general",
            text="test msg",
            thread_id="t_1",
        )

    @pytest.mark.asyncio
    async def test_skips_unconfigured_connector(self):
        mock_connector = MagicMock()
        mock_connector.is_configured = False
        mock_connector.send_message = AsyncMock()

        mock_parse = MagicMock(return_value={"slack": ["#general"]})

        with patch.dict(
            "sys.modules",
            {
                "aragora.approvals.chat": MagicMock(parse_chat_targets=mock_parse),
                "aragora.connectors.chat.registry": MagicMock(
                    get_connector=MagicMock(return_value=mock_connector)
                ),
            },
        ):
            await _dispatch_chat_message(
                text="test msg",
                channel_targets=["slack:#general"],
                thread_id=None,
                thread_id_by_platform={},
            )
        mock_connector.send_message.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_skips_none_connector(self):
        mock_parse = MagicMock(return_value={"slack": ["#general"]})

        with patch.dict(
            "sys.modules",
            {
                "aragora.approvals.chat": MagicMock(parse_chat_targets=mock_parse),
                "aragora.connectors.chat.registry": MagicMock(
                    get_connector=MagicMock(return_value=None)
                ),
            },
        ):
            await _dispatch_chat_message(
                text="test msg",
                channel_targets=["slack:#general"],
                thread_id=None,
                thread_id_by_platform={},
            )

    @pytest.mark.asyncio
    async def test_uses_platform_thread_id(self):
        mock_connector = MagicMock()
        mock_connector.is_configured = True
        mock_connector.send_message = AsyncMock()

        mock_parse = MagicMock(return_value={"slack": ["#general"]})

        with patch.dict(
            "sys.modules",
            {
                "aragora.approvals.chat": MagicMock(parse_chat_targets=mock_parse),
                "aragora.connectors.chat.registry": MagicMock(
                    get_connector=MagicMock(return_value=mock_connector)
                ),
            },
        ):
            await _dispatch_chat_message(
                text="test",
                channel_targets=["slack:#general"],
                thread_id="generic_thread",
                thread_id_by_platform={"slack": "slack_specific_thread"},
            )
        mock_connector.send_message.assert_awaited_once_with(
            channel_id="#general",
            text="test",
            thread_id="slack_specific_thread",
        )

    @pytest.mark.asyncio
    async def test_send_error_logged_not_raised(self):
        mock_connector = MagicMock()
        mock_connector.is_configured = True
        mock_connector.send_message = AsyncMock(side_effect=ConnectionError("fail"))

        mock_parse = MagicMock(return_value={"slack": ["#general"]})

        with patch.dict(
            "sys.modules",
            {
                "aragora.approvals.chat": MagicMock(parse_chat_targets=mock_parse),
                "aragora.connectors.chat.registry": MagicMock(
                    get_connector=MagicMock(return_value=mock_connector)
                ),
            },
        ):
            # Should not raise
            await _dispatch_chat_message(
                text="test",
                channel_targets=["slack:#general"],
                thread_id=None,
                thread_id_by_platform={},
            )

    @pytest.mark.asyncio
    async def test_multiple_platforms_and_channels(self):
        slack_connector = MagicMock()
        slack_connector.is_configured = True
        slack_connector.send_message = AsyncMock()
        telegram_connector = MagicMock()
        telegram_connector.is_configured = True
        telegram_connector.send_message = AsyncMock()

        def _get_connector(platform):
            if platform == "slack":
                return slack_connector
            if platform == "telegram":
                return telegram_connector
            return None

        mock_parse = MagicMock(
            return_value={
                "slack": ["#general", "#alerts"],
                "telegram": ["chat_1"],
            }
        )

        with patch.dict(
            "sys.modules",
            {
                "aragora.approvals.chat": MagicMock(parse_chat_targets=mock_parse),
                "aragora.connectors.chat.registry": MagicMock(get_connector=_get_connector),
            },
        ):
            await _dispatch_chat_message(
                text="msg",
                channel_targets=["slack:#general", "slack:#alerts", "telegram:chat_1"],
                thread_id=None,
                thread_id_by_platform={},
            )
        assert slack_connector.send_message.await_count == 2
        assert telegram_connector.send_message.await_count == 1


# ---------------------------------------------------------------------------
# _schedule_chat_dispatch
# ---------------------------------------------------------------------------


class TestScheduleChatDispatch:
    """Test _schedule_chat_dispatch() scheduling."""

    @pytest.mark.asyncio
    async def test_creates_task_in_running_loop(self):
        dispatched = False

        async def fake_coro():
            nonlocal dispatched
            dispatched = True

        coro = fake_coro()
        _schedule_chat_dispatch(coro)
        # Give the event loop time to process the task
        await asyncio.sleep(0.05)
        assert dispatched

    def test_falls_back_to_asyncio_run(self):
        """When no running loop, uses asyncio.run."""
        mock_coro = AsyncMock()

        with patch(f"{PATCH_MOD}.asyncio") as mock_asyncio:
            mock_asyncio.get_running_loop.side_effect = RuntimeError("no loop")
            mock_asyncio.run = MagicMock()
            _schedule_chat_dispatch(mock_coro)
            mock_asyncio.run.assert_called_once_with(mock_coro)

    def test_asyncio_run_error_logged(self):
        """When asyncio.run raises, the error is caught."""
        mock_coro = MagicMock()

        with patch(f"{PATCH_MOD}.asyncio") as mock_asyncio:
            mock_asyncio.get_running_loop.side_effect = RuntimeError("no loop")
            mock_asyncio.run.side_effect = RuntimeError("nested")
            # Should not raise
            _schedule_chat_dispatch(mock_coro)


# ---------------------------------------------------------------------------
# _build_event_callback
# ---------------------------------------------------------------------------


class TestBuildEventCallback:
    """Test _build_event_callback() event bridge factory."""

    def test_callback_sets_default_payload_fields(self):
        emitter = MagicMock()
        # Make the StreamEvent import fail so we can isolate payload building
        with patch.dict(
            "sys.modules",
            {
                "aragora.events.types": MagicMock(
                    StreamEventType=MagicMock(side_effect=ValueError("unknown")),
                    StreamEvent=MagicMock(),
                ),
                "aragora.events.dispatcher": MagicMock(dispatch_event=MagicMock()),
            },
        ):
            cb = _build_event_callback(
                event_emitter=emitter,
                tenant_id="t1",
                user_id="u1",
                org_id="o1",
                workflow_definition_id="wf_1",
                execution_id="exec_1",
                notify_config={
                    "channel_targets": [],
                    "thread_id": None,
                    "thread_id_by_platform": {},
                    "notify_steps": False,
                },
            )
            cb("workflow_start", {"custom": "data"})

    def test_callback_emits_stream_event(self):
        emitter = MagicMock()
        mock_stream_event_type = MagicMock()
        mock_stream_event = MagicMock()

        with patch.dict(
            "sys.modules",
            {
                "aragora.events.types": MagicMock(
                    StreamEventType=mock_stream_event_type,
                    StreamEvent=mock_stream_event,
                ),
                "aragora.events.dispatcher": MagicMock(dispatch_event=MagicMock()),
            },
        ):
            cb = _build_event_callback(
                event_emitter=emitter,
                tenant_id="t1",
                user_id=None,
                org_id=None,
                workflow_definition_id="wf_1",
                execution_id="exec_1",
                notify_config={
                    "channel_targets": [],
                    "thread_id": None,
                    "thread_id_by_platform": {},
                    "notify_steps": False,
                },
            )
            cb("workflow_start", {})
            emitter.emit.assert_called_once()

    def test_callback_dispatches_event(self):
        mock_dispatch = MagicMock()

        with patch.dict(
            "sys.modules",
            {
                "aragora.events.types": MagicMock(
                    StreamEventType=MagicMock(side_effect=ValueError("x")),
                ),
                "aragora.events.dispatcher": MagicMock(dispatch_event=mock_dispatch),
            },
        ):
            cb = _build_event_callback(
                event_emitter=None,
                tenant_id="t1",
                user_id=None,
                org_id=None,
                workflow_definition_id="wf_1",
                execution_id="exec_1",
                notify_config={
                    "channel_targets": [],
                    "thread_id": None,
                    "thread_id_by_platform": {},
                    "notify_steps": False,
                },
            )
            cb("workflow_complete", {"key": "val"})
            mock_dispatch.assert_called_once()
            call_args = mock_dispatch.call_args
            assert call_args[0][0] == "workflow_complete"
            payload = call_args[0][1]
            assert payload["tenant_id"] == "t1"
            assert payload["workflow_definition_id"] == "wf_1"
            assert payload["execution_id"] == "exec_1"

    def test_callback_triggers_chat_for_critical_events(self):
        with (
            patch(f"{PATCH_MOD}._schedule_chat_dispatch") as mock_schedule,
            patch.dict(
                "sys.modules",
                {
                    "aragora.events.types": MagicMock(
                        StreamEventType=MagicMock(side_effect=ValueError)
                    ),
                    "aragora.events.dispatcher": MagicMock(dispatch_event=MagicMock()),
                },
            ),
        ):
            cb = _build_event_callback(
                event_emitter=None,
                tenant_id="t1",
                user_id=None,
                org_id=None,
                workflow_definition_id="wf_1",
                execution_id="exec_1",
                notify_config={
                    "channel_targets": ["slack:#general"],
                    "thread_id": None,
                    "thread_id_by_platform": {},
                    "notify_steps": False,
                },
            )
            cb("workflow_start", {"workflow_name": "TestFlow"})
            mock_schedule.assert_called_once()

    def test_callback_skips_chat_for_non_critical_without_notify_steps(self):
        with (
            patch(f"{PATCH_MOD}._schedule_chat_dispatch") as mock_schedule,
            patch.dict(
                "sys.modules",
                {
                    "aragora.events.types": MagicMock(
                        StreamEventType=MagicMock(side_effect=ValueError)
                    ),
                    "aragora.events.dispatcher": MagicMock(dispatch_event=MagicMock()),
                },
            ),
        ):
            cb = _build_event_callback(
                event_emitter=None,
                tenant_id="t1",
                user_id=None,
                org_id=None,
                workflow_definition_id="wf_1",
                execution_id="exec_1",
                notify_config={
                    "channel_targets": ["slack:#general"],
                    "thread_id": None,
                    "thread_id_by_platform": {},
                    "notify_steps": False,
                },
            )
            cb("workflow_step_complete", {"step_name": "Build"})
            mock_schedule.assert_not_called()

    def test_callback_triggers_chat_for_step_events_when_notify_steps(self):
        with (
            patch(f"{PATCH_MOD}._schedule_chat_dispatch") as mock_schedule,
            patch.dict(
                "sys.modules",
                {
                    "aragora.events.types": MagicMock(
                        StreamEventType=MagicMock(side_effect=ValueError)
                    ),
                    "aragora.events.dispatcher": MagicMock(dispatch_event=MagicMock()),
                },
            ),
        ):
            cb = _build_event_callback(
                event_emitter=None,
                tenant_id="t1",
                user_id=None,
                org_id=None,
                workflow_definition_id="wf_1",
                execution_id="exec_1",
                notify_config={
                    "channel_targets": ["slack:#general"],
                    "thread_id": "t_1",
                    "thread_id_by_platform": {},
                    "notify_steps": True,
                },
            )
            cb("workflow_step_complete", {"step_name": "Build"})
            mock_schedule.assert_called_once()

    def test_callback_no_chat_when_no_channel_targets(self):
        with (
            patch(f"{PATCH_MOD}._schedule_chat_dispatch") as mock_schedule,
            patch.dict(
                "sys.modules",
                {
                    "aragora.events.types": MagicMock(
                        StreamEventType=MagicMock(side_effect=ValueError)
                    ),
                    "aragora.events.dispatcher": MagicMock(dispatch_event=MagicMock()),
                },
            ),
        ):
            cb = _build_event_callback(
                event_emitter=None,
                tenant_id="t1",
                user_id=None,
                org_id=None,
                workflow_definition_id="wf_1",
                execution_id="exec_1",
                notify_config={
                    "channel_targets": [],
                    "thread_id": None,
                    "thread_id_by_platform": {},
                    "notify_steps": True,
                },
            )
            cb("workflow_start", {})
            mock_schedule.assert_not_called()

    def test_callback_handles_emitter_error(self):
        """Stream event emitter error is caught and logged."""
        emitter = MagicMock()
        emitter.emit.side_effect = TypeError("bad emit")

        with patch.dict(
            "sys.modules",
            {
                "aragora.events.types": MagicMock(
                    StreamEventType=MagicMock(return_value="workflow_start"),
                    StreamEvent=MagicMock(return_value=MagicMock()),
                ),
                "aragora.events.dispatcher": MagicMock(dispatch_event=MagicMock()),
            },
        ):
            cb = _build_event_callback(
                event_emitter=emitter,
                tenant_id="t1",
                user_id=None,
                org_id=None,
                workflow_definition_id="wf_1",
                execution_id="exec_1",
                notify_config={
                    "channel_targets": [],
                    "thread_id": None,
                    "thread_id_by_platform": {},
                    "notify_steps": False,
                },
            )
            # Should not raise
            cb("workflow_start", {})

    def test_callback_handles_dispatch_import_error(self):
        """Event dispatcher import error is caught."""
        with patch.dict(
            "sys.modules",
            {
                "aragora.events.types": MagicMock(
                    StreamEventType=MagicMock(side_effect=ValueError)
                ),
                "aragora.events.dispatcher": None,  # Simulate ImportError
            },
        ):
            cb = _build_event_callback(
                event_emitter=None,
                tenant_id="t1",
                user_id=None,
                org_id=None,
                workflow_definition_id="wf_1",
                execution_id="exec_1",
                notify_config={
                    "channel_targets": [],
                    "thread_id": None,
                    "thread_id_by_platform": {},
                    "notify_steps": False,
                },
            )
            # Should not raise even though dispatcher import fails
            cb("workflow_start", {})


# ---------------------------------------------------------------------------
# execute_workflow (async)
# ---------------------------------------------------------------------------


class TestExecuteWorkflow:
    """Test execute_workflow() async function."""

    @pytest.mark.asyncio
    async def test_successful_execution(self):
        mock_store = MagicMock()
        mock_wf = _make_workflow()
        mock_store.get_workflow.return_value = mock_wf

        mock_engine = MagicMock()
        engine_result = _make_engine_result(success=True)
        mock_engine.execute = AsyncMock(return_value=engine_result)

        mock_audit = MagicMock()

        with (
            patch(f"{PATCH_MOD}._get_store", return_value=mock_store),
            patch(f"{PATCH_MOD}._get_engine", return_value=mock_engine),
            patch("aragora.server.handlers.workflows.audit_data", mock_audit),
        ):
            result = await execute_workflow(
                "wf_1", inputs={"key": "val"}, tenant_id="t1", user_id="u1", org_id="o1"
            )

        assert result["status"] == "completed"
        assert result["workflow_id"] == "wf_1"
        assert result["tenant_id"] == "t1"
        assert result["outputs"] == {"result": "ok"}
        assert "id" in result
        assert result["id"].startswith("exec_")
        # Verify store interactions
        assert mock_store.save_execution.call_count == 2  # initial + completed
        mock_engine.execute.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_workflow_not_found_raises(self):
        mock_store = MagicMock()
        mock_store.get_workflow.return_value = None

        with patch(f"{PATCH_MOD}._get_store", return_value=mock_store):
            with pytest.raises(ValueError, match="Workflow not found"):
                await execute_workflow("wf_missing")

    @pytest.mark.asyncio
    async def test_failed_execution(self):
        mock_store = MagicMock()
        mock_wf = _make_workflow()
        mock_store.get_workflow.return_value = mock_wf

        mock_engine = MagicMock()
        engine_result = _make_engine_result(success=False, error="step failed", final_output=None)
        mock_engine.execute = AsyncMock(return_value=engine_result)

        with (
            patch(f"{PATCH_MOD}._get_store", return_value=mock_store),
            patch(f"{PATCH_MOD}._get_engine", return_value=mock_engine),
            patch("aragora.server.handlers.workflows.audit_data", MagicMock()),
        ):
            result = await execute_workflow("wf_1")

        assert result["status"] == "failed"
        assert result["error"] == "step failed"

    @pytest.mark.asyncio
    async def test_value_error_during_execution(self):
        mock_store = MagicMock()
        mock_wf = _make_workflow()
        mock_store.get_workflow.return_value = mock_wf

        mock_engine = MagicMock()
        mock_engine.execute = AsyncMock(side_effect=ValueError("bad config"))

        with (
            patch(f"{PATCH_MOD}._get_store", return_value=mock_store),
            patch(f"{PATCH_MOD}._get_engine", return_value=mock_engine),
        ):
            with pytest.raises(ValueError, match="bad config"):
                await execute_workflow("wf_1")

        # Verify execution was saved as failed
        save_calls = mock_store.save_execution.call_args_list
        last_saved = save_calls[-1][0][0]
        assert last_saved["status"] == "failed"
        assert last_saved["error"] == "Invalid workflow configuration or inputs"

    @pytest.mark.asyncio
    async def test_os_error_during_execution(self):
        mock_store = MagicMock()
        mock_wf = _make_workflow()
        mock_store.get_workflow.return_value = mock_wf

        mock_engine = MagicMock()
        mock_engine.execute = AsyncMock(side_effect=OSError("disk full"))

        with (
            patch(f"{PATCH_MOD}._get_store", return_value=mock_store),
            patch(f"{PATCH_MOD}._get_engine", return_value=mock_engine),
        ):
            with pytest.raises(OSError, match="disk full"):
                await execute_workflow("wf_1")

        save_calls = mock_store.save_execution.call_args_list
        last_saved = save_calls[-1][0][0]
        assert last_saved["status"] == "failed"
        assert last_saved["error"] == "Storage error during workflow execution"

    @pytest.mark.asyncio
    async def test_connection_error_during_execution(self):
        """ConnectionError is a subclass of OSError, so the OSError handler catches it."""
        mock_store = MagicMock()
        mock_wf = _make_workflow()
        mock_store.get_workflow.return_value = mock_wf

        mock_engine = MagicMock()
        mock_engine.execute = AsyncMock(side_effect=ConnectionError("refused"))

        with (
            patch(f"{PATCH_MOD}._get_store", return_value=mock_store),
            patch(f"{PATCH_MOD}._get_engine", return_value=mock_engine),
        ):
            with pytest.raises(ConnectionError, match="refused"):
                await execute_workflow("wf_1")

        save_calls = mock_store.save_execution.call_args_list
        last_saved = save_calls[-1][0][0]
        assert last_saved["status"] == "failed"
        # ConnectionError is a subclass of OSError, so the OSError handler catches it
        assert last_saved["error"] == "Storage error during workflow execution"

    @pytest.mark.asyncio
    async def test_timeout_error_during_execution(self):
        mock_store = MagicMock()
        mock_wf = _make_workflow()
        mock_store.get_workflow.return_value = mock_wf

        mock_engine = MagicMock()
        mock_engine.execute = AsyncMock(side_effect=TimeoutError("timed out"))

        with (
            patch(f"{PATCH_MOD}._get_store", return_value=mock_store),
            patch(f"{PATCH_MOD}._get_engine", return_value=mock_engine),
        ):
            with pytest.raises(TimeoutError, match="timed out"):
                await execute_workflow("wf_1")

        save_calls = mock_store.save_execution.call_args_list
        last_saved = save_calls[-1][0][0]
        assert last_saved["status"] == "failed"

    @pytest.mark.asyncio
    async def test_key_error_during_execution(self):
        mock_store = MagicMock()
        mock_wf = _make_workflow()
        mock_store.get_workflow.return_value = mock_wf

        mock_engine = MagicMock()
        mock_engine.execute = AsyncMock(side_effect=KeyError("missing key"))

        with (
            patch(f"{PATCH_MOD}._get_store", return_value=mock_store),
            patch(f"{PATCH_MOD}._get_engine", return_value=mock_engine),
        ):
            with pytest.raises(KeyError):
                await execute_workflow("wf_1")

        save_calls = mock_store.save_execution.call_args_list
        last_saved = save_calls[-1][0][0]
        assert last_saved["status"] == "failed"

    @pytest.mark.asyncio
    async def test_none_inputs_defaults_to_empty_dict(self):
        mock_store = MagicMock()
        mock_wf = _make_workflow()
        mock_store.get_workflow.return_value = mock_wf

        mock_engine = MagicMock()
        engine_result = _make_engine_result(success=True)
        mock_engine.execute = AsyncMock(return_value=engine_result)

        with (
            patch(f"{PATCH_MOD}._get_store", return_value=mock_store),
            patch(f"{PATCH_MOD}._get_engine", return_value=mock_engine),
            patch("aragora.server.handlers.workflows.audit_data", MagicMock()),
        ):
            result = await execute_workflow("wf_1", inputs=None)

        assert result["inputs"] == {}

    @pytest.mark.asyncio
    async def test_execution_with_steps_converted_to_dict(self):
        mock_store = MagicMock()
        mock_wf = _make_workflow()
        mock_store.get_workflow.return_value = mock_wf

        step = _make_step_result()
        mock_engine = MagicMock()
        engine_result = _make_engine_result(success=True, steps=[step])
        mock_engine.execute = AsyncMock(return_value=engine_result)

        with (
            patch(f"{PATCH_MOD}._get_store", return_value=mock_store),
            patch(f"{PATCH_MOD}._get_engine", return_value=mock_engine),
            patch("aragora.server.handlers.workflows.audit_data", MagicMock()),
        ):
            result = await execute_workflow("wf_1")

        assert len(result["steps"]) == 1
        assert result["steps"][0]["step_id"] == "s1"

    @pytest.mark.asyncio
    async def test_audit_data_called_on_success(self):
        mock_store = MagicMock()
        mock_wf = _make_workflow()
        mock_store.get_workflow.return_value = mock_wf

        mock_engine = MagicMock()
        engine_result = _make_engine_result(success=True)
        mock_engine.execute = AsyncMock(return_value=engine_result)

        mock_audit = MagicMock()

        with (
            patch(f"{PATCH_MOD}._get_store", return_value=mock_store),
            patch(f"{PATCH_MOD}._get_engine", return_value=mock_engine),
            patch("aragora.server.handlers.workflows.audit_data", mock_audit),
        ):
            await execute_workflow("wf_1", tenant_id="t1")

        mock_audit.assert_called_once()
        call_kwargs = mock_audit.call_args[1]
        assert call_kwargs["resource_type"] == "workflow_execution"
        assert call_kwargs["action"] == "execute"
        assert call_kwargs["workflow_id"] == "wf_1"
        assert call_kwargs["status"] == "completed"
        assert call_kwargs["tenant_id"] == "t1"

    @pytest.mark.asyncio
    async def test_audit_data_none_skipped(self):
        mock_store = MagicMock()
        mock_wf = _make_workflow()
        mock_store.get_workflow.return_value = mock_wf

        mock_engine = MagicMock()
        engine_result = _make_engine_result(success=True)
        mock_engine.execute = AsyncMock(return_value=engine_result)

        with (
            patch(f"{PATCH_MOD}._get_store", return_value=mock_store),
            patch(f"{PATCH_MOD}._get_engine", return_value=mock_engine),
            patch("aragora.server.handlers.workflows.audit_data", None),
        ):
            # Should not raise
            result = await execute_workflow("wf_1")
            assert result["status"] == "completed"

    @pytest.mark.asyncio
    async def test_event_callback_passed_to_engine(self):
        mock_store = MagicMock()
        mock_wf = _make_workflow()
        mock_store.get_workflow.return_value = mock_wf

        mock_engine = MagicMock()
        engine_result = _make_engine_result(success=True)
        mock_engine.execute = AsyncMock(return_value=engine_result)

        emitter = MagicMock()

        with (
            patch(f"{PATCH_MOD}._get_store", return_value=mock_store),
            patch(f"{PATCH_MOD}._get_engine", return_value=mock_engine),
            patch("aragora.server.handlers.workflows.audit_data", MagicMock()),
        ):
            await execute_workflow("wf_1", event_emitter=emitter)

        execute_call = mock_engine.execute.call_args
        assert "event_callback" in execute_call.kwargs
        assert execute_call.kwargs["event_callback"] is not None

    @pytest.mark.asyncio
    async def test_execution_id_format(self):
        mock_store = MagicMock()
        mock_wf = _make_workflow()
        mock_store.get_workflow.return_value = mock_wf

        mock_engine = MagicMock()
        engine_result = _make_engine_result(success=True)
        mock_engine.execute = AsyncMock(return_value=engine_result)

        with (
            patch(f"{PATCH_MOD}._get_store", return_value=mock_store),
            patch(f"{PATCH_MOD}._get_engine", return_value=mock_engine),
            patch("aragora.server.handlers.workflows.audit_data", MagicMock()),
        ):
            result = await execute_workflow("wf_1")

        assert result["id"].startswith("exec_")
        # exec_ prefix + 12 hex chars
        assert len(result["id"]) == 17

    @pytest.mark.asyncio
    async def test_started_at_and_completed_at_set(self):
        mock_store = MagicMock()
        mock_wf = _make_workflow()
        mock_store.get_workflow.return_value = mock_wf

        mock_engine = MagicMock()
        engine_result = _make_engine_result(success=True)
        mock_engine.execute = AsyncMock(return_value=engine_result)

        with (
            patch(f"{PATCH_MOD}._get_store", return_value=mock_store),
            patch(f"{PATCH_MOD}._get_engine", return_value=mock_engine),
            patch("aragora.server.handlers.workflows.audit_data", MagicMock()),
        ):
            result = await execute_workflow("wf_1")

        assert "started_at" in result
        assert "completed_at" in result

    @pytest.mark.asyncio
    async def test_duration_ms_set(self):
        mock_store = MagicMock()
        mock_wf = _make_workflow()
        mock_store.get_workflow.return_value = mock_wf

        mock_engine = MagicMock()
        engine_result = _make_engine_result(success=True, duration_ms=500)
        mock_engine.execute = AsyncMock(return_value=engine_result)

        with (
            patch(f"{PATCH_MOD}._get_store", return_value=mock_store),
            patch(f"{PATCH_MOD}._get_engine", return_value=mock_engine),
            patch("aragora.server.handlers.workflows.audit_data", MagicMock()),
        ):
            result = await execute_workflow("wf_1")

        assert result["duration_ms"] == 500

    @pytest.mark.asyncio
    async def test_default_tenant_id(self):
        mock_store = MagicMock()
        mock_wf = _make_workflow()
        mock_store.get_workflow.return_value = mock_wf

        mock_engine = MagicMock()
        engine_result = _make_engine_result(success=True)
        mock_engine.execute = AsyncMock(return_value=engine_result)

        with (
            patch(f"{PATCH_MOD}._get_store", return_value=mock_store),
            patch(f"{PATCH_MOD}._get_engine", return_value=mock_engine),
            patch("aragora.server.handlers.workflows.audit_data", MagicMock()),
        ):
            result = await execute_workflow("wf_1")

        assert result["tenant_id"] == "default"


# ---------------------------------------------------------------------------
# get_execution (async)
# ---------------------------------------------------------------------------


class TestGetExecution:
    """Test get_execution() async function."""

    @pytest.mark.asyncio
    async def test_returns_execution(self):
        mock_store = MagicMock()
        mock_exec = {"id": "exec_1", "status": "completed"}
        mock_store.get_execution.return_value = mock_exec

        with patch(f"{PATCH_MOD}._get_store", return_value=mock_store):
            result = await get_execution("exec_1")

        assert result == mock_exec
        mock_store.get_execution.assert_called_once_with("exec_1")

    @pytest.mark.asyncio
    async def test_returns_none_when_not_found(self):
        mock_store = MagicMock()
        mock_store.get_execution.return_value = None

        with patch(f"{PATCH_MOD}._get_store", return_value=mock_store):
            result = await get_execution("exec_missing")

        assert result is None


# ---------------------------------------------------------------------------
# list_executions (async)
# ---------------------------------------------------------------------------


class TestListExecutions:
    """Test list_executions() async function."""

    @pytest.mark.asyncio
    async def test_returns_executions(self):
        mock_store = MagicMock()
        mock_execs = [{"id": "exec_1"}, {"id": "exec_2"}]
        mock_store.list_executions.return_value = (mock_execs, 2)

        with patch(f"{PATCH_MOD}._get_store", return_value=mock_store):
            result = await list_executions()

        assert len(result) == 2
        assert result[0]["id"] == "exec_1"

    @pytest.mark.asyncio
    async def test_passes_filters(self):
        mock_store = MagicMock()
        mock_store.list_executions.return_value = ([], 0)

        with patch(f"{PATCH_MOD}._get_store", return_value=mock_store):
            await list_executions(workflow_id="wf_1", tenant_id="t1", limit=10)

        mock_store.list_executions.assert_called_once_with(
            workflow_id="wf_1", tenant_id="t1", limit=10
        )

    @pytest.mark.asyncio
    async def test_default_params(self):
        mock_store = MagicMock()
        mock_store.list_executions.return_value = ([], 0)

        with patch(f"{PATCH_MOD}._get_store", return_value=mock_store):
            await list_executions()

        mock_store.list_executions.assert_called_once_with(
            workflow_id=None, tenant_id="default", limit=20
        )


# ---------------------------------------------------------------------------
# terminate_execution (async)
# ---------------------------------------------------------------------------


class TestTerminateExecution:
    """Test terminate_execution() async function."""

    @pytest.mark.asyncio
    async def test_terminates_running_execution(self):
        mock_store = MagicMock()
        mock_exec = {"id": "exec_1", "status": "running"}
        mock_store.get_execution.return_value = mock_exec

        mock_engine = MagicMock()

        with (
            patch(f"{PATCH_MOD}._get_store", return_value=mock_store),
            patch(f"{PATCH_MOD}._get_engine", return_value=mock_engine),
        ):
            result = await terminate_execution("exec_1")

        assert result is True
        mock_engine.request_termination.assert_called_once_with("User requested")
        assert mock_exec["status"] == "terminated"
        assert "completed_at" in mock_exec
        mock_store.save_execution.assert_called_once_with(mock_exec)

    @pytest.mark.asyncio
    async def test_returns_false_when_not_found(self):
        mock_store = MagicMock()
        mock_store.get_execution.return_value = None

        with patch(f"{PATCH_MOD}._get_store", return_value=mock_store):
            result = await terminate_execution("exec_missing")

        assert result is False

    @pytest.mark.asyncio
    async def test_returns_false_when_not_running(self):
        mock_store = MagicMock()
        mock_exec = {"id": "exec_1", "status": "completed"}
        mock_store.get_execution.return_value = mock_exec

        with patch(f"{PATCH_MOD}._get_store", return_value=mock_store):
            result = await terminate_execution("exec_1")

        assert result is False

    @pytest.mark.asyncio
    async def test_returns_false_for_failed_execution(self):
        mock_store = MagicMock()
        mock_exec = {"id": "exec_1", "status": "failed"}
        mock_store.get_execution.return_value = mock_exec

        with patch(f"{PATCH_MOD}._get_store", return_value=mock_store):
            result = await terminate_execution("exec_1")

        assert result is False

    @pytest.mark.asyncio
    async def test_returns_false_for_terminated_execution(self):
        mock_store = MagicMock()
        mock_exec = {"id": "exec_1", "status": "terminated"}
        mock_store.get_execution.return_value = mock_exec

        with patch(f"{PATCH_MOD}._get_store", return_value=mock_store):
            result = await terminate_execution("exec_1")

        assert result is False

    @pytest.mark.asyncio
    async def test_completed_at_timestamp_set(self):
        mock_store = MagicMock()
        mock_exec = {"id": "exec_1", "status": "running"}
        mock_store.get_execution.return_value = mock_exec

        mock_engine = MagicMock()

        with (
            patch(f"{PATCH_MOD}._get_store", return_value=mock_store),
            patch(f"{PATCH_MOD}._get_engine", return_value=mock_engine),
        ):
            await terminate_execution("exec_1")

        assert mock_exec["completed_at"] is not None
        # Verify it's a valid ISO format timestamp
        datetime.fromisoformat(mock_exec["completed_at"])
