"""Focused tests for execution notifier routing, idempotency, and observability."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.pipeline.execution_notifier import ExecutionNotifier


class TestExecutionNotifierRouting:
    @pytest.mark.asyncio
    async def test_dispatch_uses_platform_specific_thread_mapping(self) -> None:
        notifier = ExecutionNotifier(
            debate_id="debate-1",
            plan_id="plan-1",
            channel_targets=["slack:#eng", "teams:ops"],
            thread_id="default-thread",
            thread_id_by_platform={"slack": "slack-thread"},
            notify_channel=False,
            notify_websocket=False,
        )

        slack = MagicMock()
        slack.is_configured = True
        slack.send_message = AsyncMock()

        teams = MagicMock()
        teams.is_configured = True
        teams.send_message = AsyncMock()

        def _connector_for(platform: str):
            return {"slack": slack, "teams": teams}.get(platform)

        with (
            patch(
                "aragora.approvals.chat.parse_chat_targets",
                return_value={"slack": ["#eng"], "teams": ["ops"]},
            ),
            patch(
                "aragora.connectors.chat.registry.get_connector",
                side_effect=_connector_for,
            ),
        ):
            await notifier._dispatch_to_targets(
                {
                    "completed_tasks": 1,
                    "failed_tasks": 0,
                    "total_tasks": 2,
                    "progress_pct": 50.0,
                },
                is_complete=False,
            )

        slack.send_message.assert_awaited_once_with(
            channel_id="#eng",
            text="Execution update (50%)\n- Tasks: 1/2 completed",
            thread_id="slack-thread",
        )
        teams.send_message.assert_awaited_once_with(
            channel_id="ops",
            text="Execution update (50%)\n- Tasks: 1/2 completed",
            thread_id="default-thread",
        )


class TestExecutionNotifierIdempotency:
    def test_duplicate_task_callback_is_ignored(self) -> None:
        notifier = ExecutionNotifier(
            debate_id="debate-2",
            total_tasks=2,
            notify_channel=False,
            notify_websocket=False,
        )
        result = MagicMock(success=True, model_used="claude", duration_seconds=0.2, error=None)

        with patch.object(notifier, "_dispatch_progress") as mock_dispatch:
            notifier.on_task_complete("task-1", result)
            notifier.on_task_complete("task-1", result)

        assert notifier.progress.completed_tasks == 1
        assert notifier.progress.failed_tasks == 0
        assert len(notifier.progress.task_results) == 1
        assert mock_dispatch.call_count == 1

    @pytest.mark.asyncio
    async def test_completion_summary_is_idempotent(self) -> None:
        notifier = ExecutionNotifier(
            debate_id="debate-3",
            channel_targets=["slack:#eng"],
            notify_channel=False,
            notify_websocket=False,
        )

        slack = MagicMock()
        slack.is_configured = True
        slack.send_message = AsyncMock()

        with (
            patch(
                "aragora.approvals.chat.parse_chat_targets",
                return_value={"slack": ["#eng"]},
            ),
            patch(
                "aragora.connectors.chat.registry.get_connector",
                return_value=slack,
            ),
        ):
            await notifier.send_completion_summary()
            await notifier.send_completion_summary()

        slack.send_message.assert_awaited_once()


class TestExecutionNotifierObservability:
    @pytest.mark.asyncio
    async def test_target_dispatch_retries_and_records_error(self) -> None:
        notifier = ExecutionNotifier(
            debate_id="debate-4",
            channel_targets=["slack:#eng"],
            notify_channel=False,
            notify_websocket=False,
        )

        slack = MagicMock()
        slack.is_configured = True
        slack.send_message = AsyncMock(side_effect=[RuntimeError("temporary"), None])

        with (
            patch(
                "aragora.approvals.chat.parse_chat_targets",
                return_value={"slack": ["#eng"]},
            ),
            patch(
                "aragora.connectors.chat.registry.get_connector",
                return_value=slack,
            ),
        ):
            await notifier._dispatch_to_targets(
                {
                    "completed_tasks": 1,
                    "failed_tasks": 0,
                    "total_tasks": 1,
                    "progress_pct": 100.0,
                },
                is_complete=True,
            )

        assert slack.send_message.await_count == 2
        assert len(notifier.delivery_errors) == 1
        err = notifier.delivery_errors[0]
        assert err["stage"] == "target_dispatch"
        assert err["platform"] == "slack"
        assert err["channel_id"] == "#eng"
        assert err["retryable"] is True
        assert err["error_type"] == "RuntimeError"
        assert err["error_message"] == "temporary"
