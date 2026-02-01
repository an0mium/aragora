"""
Comprehensive tests for Built-in Hook Handlers.

Tests cover:
- log_event handler with interpolation
- log_metric handler with tags
- send_webhook handler
- send_slack_notification handler
- save_checkpoint handler
- store_fact handler
- set_context_var handler
- delay_execution handler
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.hooks.builtin import (
    delay_execution,
    log_event,
    log_metric,
    save_checkpoint,
    send_slack_notification,
    send_webhook,
    set_context_var,
    store_fact,
)


# =============================================================================
# log_event Tests
# =============================================================================


class TestLogEvent:
    """Tests for log_event handler."""

    @pytest.mark.asyncio
    async def test_log_event_basic(self, caplog):
        """Test basic log_event call."""
        import logging

        with caplog.at_level(logging.INFO):
            await log_event(message="Test message", level="info")

        assert "Test message" in caplog.text

    @pytest.mark.asyncio
    async def test_log_event_interpolation(self, caplog):
        """Test message interpolation."""
        import logging

        with caplog.at_level(logging.INFO):
            await log_event(
                message="Debate {debate_id} completed with {status}",
                level="info",
                debate_id="123",
                status="success",
            )

        assert "Debate 123 completed with success" in caplog.text

    @pytest.mark.asyncio
    async def test_log_event_failed_interpolation(self, caplog):
        """Test graceful handling of failed interpolation."""
        import logging

        with caplog.at_level(logging.INFO):
            # Missing context variable
            await log_event(
                message="Missing {missing_var}",
                level="info",
                other_var="present",
            )

        # Should log the original message without crashing
        assert "Missing {missing_var}" in caplog.text

    @pytest.mark.asyncio
    async def test_log_event_different_levels(self, caplog):
        """Test different log levels."""
        import logging

        with caplog.at_level(logging.DEBUG):
            await log_event(message="Debug message", level="debug")
            await log_event(message="Info message", level="info")
            await log_event(message="Warning message", level="warning")
            await log_event(message="Error message", level="error")

        assert "Debug message" in caplog.text
        assert "Info message" in caplog.text
        assert "Warning message" in caplog.text
        assert "Error message" in caplog.text

    @pytest.mark.asyncio
    async def test_log_event_include_context(self, caplog):
        """Test including context in log."""
        import logging

        with caplog.at_level(logging.INFO):
            await log_event(
                message="Test",
                level="info",
                include_context=True,
                debate_id="123",
                confidence=0.95,
            )

        # Context should be included
        assert "debate_id" in caplog.text
        assert "123" in caplog.text

    @pytest.mark.asyncio
    async def test_log_event_invalid_level_defaults_to_info(self, caplog):
        """Test that invalid level defaults to info."""
        import logging

        with caplog.at_level(logging.INFO):
            await log_event(message="Test", level="invalid_level")

        assert "Test" in caplog.text

    @pytest.mark.asyncio
    async def test_log_event_empty_message(self, caplog):
        """Test empty message."""
        import logging

        with caplog.at_level(logging.INFO):
            await log_event(message="", level="info")

        # Should not raise


# =============================================================================
# log_metric Tests
# =============================================================================


class TestLogMetric:
    """Tests for log_metric handler."""

    @pytest.mark.asyncio
    async def test_log_metric_basic(self, caplog):
        """Test basic metric logging."""
        import logging

        with caplog.at_level(logging.INFO):
            await log_metric(
                metric_name="debate_duration",
                value=42.5,
            )

        assert "METRIC" in caplog.text
        assert "debate_duration=42.5" in caplog.text

    @pytest.mark.asyncio
    async def test_log_metric_with_tags(self, caplog):
        """Test metric with explicit tags."""
        import logging

        with caplog.at_level(logging.INFO):
            await log_metric(
                metric_name="consensus_score",
                value=0.95,
                tags={"env": "prod", "region": "us-west"},
            )

        assert "env=prod" in caplog.text
        assert "region=us-west" in caplog.text

    @pytest.mark.asyncio
    async def test_log_metric_extracts_context_tags(self, caplog):
        """Test that common context values are extracted as tags."""
        import logging

        with caplog.at_level(logging.INFO):
            await log_metric(
                metric_name="round_count",
                value=5,
                debate_id="debate-123",
                platform="api",
            )

        assert "debate_id=debate-123" in caplog.text
        assert "platform=api" in caplog.text

    @pytest.mark.asyncio
    async def test_log_metric_integer_value(self, caplog):
        """Test metric with integer value."""
        import logging

        with caplog.at_level(logging.INFO):
            await log_metric(metric_name="agent_count", value=5)

        assert "agent_count=5" in caplog.text

    @pytest.mark.asyncio
    async def test_log_metric_string_value(self, caplog):
        """Test metric with string value."""
        import logging

        with caplog.at_level(logging.INFO):
            await log_metric(metric_name="status", value="completed")

        assert "status=completed" in caplog.text


# =============================================================================
# send_webhook Tests
# =============================================================================


class TestSendWebhook:
    """Tests for send_webhook handler."""

    @pytest.mark.asyncio
    async def test_send_webhook_post(self):
        """Test POST webhook."""
        import httpx

        with patch.object(httpx, "AsyncClient") as mock_client:
            mock_response = MagicMock()
            mock_response.is_success = True
            mock_response.status_code = 200

            mock_instance = AsyncMock()
            mock_instance.post = AsyncMock(return_value=mock_response)
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=None)
            mock_client.return_value = mock_instance

            result = await send_webhook(
                url="https://example.com/webhook",
                method="POST",
            )

            assert result is True
            mock_instance.post.assert_called_once()

    @pytest.mark.asyncio
    async def test_send_webhook_put(self):
        """Test PUT webhook."""
        import httpx

        with patch.object(httpx, "AsyncClient") as mock_client:
            mock_response = MagicMock()
            mock_response.is_success = True

            mock_instance = AsyncMock()
            mock_instance.put = AsyncMock(return_value=mock_response)
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=None)
            mock_client.return_value = mock_instance

            result = await send_webhook(
                url="https://example.com/webhook",
                method="PUT",
            )

            assert result is True
            mock_instance.put.assert_called_once()

    @pytest.mark.asyncio
    async def test_send_webhook_get(self):
        """Test GET webhook."""
        import httpx

        with patch.object(httpx, "AsyncClient") as mock_client:
            mock_response = MagicMock()
            mock_response.is_success = True

            mock_instance = AsyncMock()
            mock_instance.get = AsyncMock(return_value=mock_response)
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=None)
            mock_client.return_value = mock_instance

            result = await send_webhook(
                url="https://example.com/webhook",
                method="GET",
            )

            assert result is True
            mock_instance.get.assert_called_once()

    @pytest.mark.asyncio
    async def test_send_webhook_with_payload_template(self):
        """Test webhook with payload template interpolation."""
        import httpx

        with patch.object(httpx, "AsyncClient") as mock_client:
            mock_response = MagicMock()
            mock_response.is_success = True

            mock_instance = AsyncMock()
            mock_instance.post = AsyncMock(return_value=mock_response)
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=None)
            mock_client.return_value = mock_instance

            result = await send_webhook(
                url="https://example.com/webhook",
                payload_template={
                    "debate_id": "{debate_id}",
                    "result": "{result}",
                    "static_field": "static_value",
                },
                debate_id="123",
                result="consensus",
            )

            assert result is True
            call_args = mock_instance.post.call_args
            payload = call_args.kwargs.get("json", {})
            assert payload["debate_id"] == "123"
            assert payload["result"] == "consensus"
            assert payload["static_field"] == "static_value"

    @pytest.mark.asyncio
    async def test_send_webhook_default_payload(self):
        """Test webhook with default payload."""
        import httpx

        with patch.object(httpx, "AsyncClient") as mock_client:
            mock_response = MagicMock()
            mock_response.is_success = True

            mock_instance = AsyncMock()
            mock_instance.post = AsyncMock(return_value=mock_response)
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=None)
            mock_client.return_value = mock_instance

            result = await send_webhook(
                url="https://example.com/webhook",
                trigger="post_debate",
                debate_id="123",
            )

            assert result is True
            call_args = mock_instance.post.call_args
            payload = call_args.kwargs.get("json", {})
            assert payload["event"] == "post_debate"
            assert "timestamp" in payload
            assert "data" in payload

    @pytest.mark.asyncio
    async def test_send_webhook_with_headers(self):
        """Test webhook with custom headers."""
        import httpx

        with patch.object(httpx, "AsyncClient") as mock_client:
            mock_response = MagicMock()
            mock_response.is_success = True

            mock_instance = AsyncMock()
            mock_instance.post = AsyncMock(return_value=mock_response)
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=None)
            mock_client.return_value = mock_instance

            result = await send_webhook(
                url="https://example.com/webhook",
                headers={"Authorization": "Bearer token123"},
            )

            assert result is True
            call_args = mock_instance.post.call_args
            headers = call_args.kwargs.get("headers", {})
            assert headers["Authorization"] == "Bearer token123"

    @pytest.mark.asyncio
    async def test_send_webhook_failure(self):
        """Test webhook failure response."""
        import httpx

        with patch.object(httpx, "AsyncClient") as mock_client:
            mock_response = MagicMock()
            mock_response.is_success = False
            mock_response.status_code = 500

            mock_instance = AsyncMock()
            mock_instance.post = AsyncMock(return_value=mock_response)
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=None)
            mock_client.return_value = mock_instance

            result = await send_webhook(url="https://example.com/webhook")

            assert result is False

    @pytest.mark.asyncio
    async def test_send_webhook_exception(self):
        """Test webhook with network exception."""
        import httpx

        with patch.object(httpx, "AsyncClient") as mock_client:
            mock_instance = AsyncMock()
            mock_instance.post = AsyncMock(side_effect=Exception("Network error"))
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=None)
            mock_client.return_value = mock_instance

            result = await send_webhook(url="https://example.com/webhook")

            assert result is False


# =============================================================================
# send_slack_notification Tests
# =============================================================================


class TestSendSlackNotification:
    """Tests for send_slack_notification handler."""

    @pytest.mark.asyncio
    async def test_slack_notification_no_webhook_url(self):
        """Test Slack notification without webhook URL."""
        with patch.dict(os.environ, {}, clear=True):
            # Remove SLACK_WEBHOOK_URL if present
            os.environ.pop("SLACK_WEBHOOK_URL", None)

            result = await send_slack_notification(
                channel="#general",
                message_template="Test message",
            )

            assert result is False

    @pytest.mark.asyncio
    async def test_slack_notification_with_webhook(self):
        """Test Slack notification with webhook URL."""
        with patch.dict(os.environ, {"SLACK_WEBHOOK_URL": "https://hooks.slack.com/test"}):
            with patch(
                "aragora.hooks.builtin.send_webhook", new_callable=AsyncMock
            ) as mock_webhook:
                mock_webhook.return_value = True

                result = await send_slack_notification(
                    channel="#general",
                    message_template="Test message",
                )

                assert result is True
                mock_webhook.assert_called_once()
                call_args = mock_webhook.call_args
                assert call_args.kwargs["url"] == "https://hooks.slack.com/test"
                payload = call_args.kwargs["payload_template"]
                assert payload["channel"] == "#general"
                assert payload["text"] == "Test message"

    @pytest.mark.asyncio
    async def test_slack_notification_message_interpolation(self):
        """Test Slack notification with message interpolation."""
        with patch.dict(os.environ, {"SLACK_WEBHOOK_URL": "https://hooks.slack.com/test"}):
            with patch(
                "aragora.hooks.builtin.send_webhook", new_callable=AsyncMock
            ) as mock_webhook:
                mock_webhook.return_value = True

                result = await send_slack_notification(
                    channel="#alerts",
                    message_template="Debate {debate_id} completed: {status}",
                    debate_id="123",
                    status="consensus",
                )

                assert result is True
                call_args = mock_webhook.call_args
                payload = call_args.kwargs["payload_template"]
                assert payload["text"] == "Debate 123 completed: consensus"

    @pytest.mark.asyncio
    async def test_slack_notification_custom_username(self):
        """Test Slack notification with custom username."""
        with patch.dict(os.environ, {"SLACK_WEBHOOK_URL": "https://hooks.slack.com/test"}):
            with patch(
                "aragora.hooks.builtin.send_webhook", new_callable=AsyncMock
            ) as mock_webhook:
                mock_webhook.return_value = True

                await send_slack_notification(
                    channel="#general",
                    message_template="Test",
                    username="CustomBot",
                    icon_emoji=":wave:",
                )

                call_args = mock_webhook.call_args
                payload = call_args.kwargs["payload_template"]
                assert payload["username"] == "CustomBot"
                assert payload["icon_emoji"] == ":wave:"

    @pytest.mark.asyncio
    async def test_slack_notification_failed_interpolation(self):
        """Test Slack notification with failed interpolation."""
        with patch.dict(os.environ, {"SLACK_WEBHOOK_URL": "https://hooks.slack.com/test"}):
            with patch(
                "aragora.hooks.builtin.send_webhook", new_callable=AsyncMock
            ) as mock_webhook:
                mock_webhook.return_value = True

                await send_slack_notification(
                    channel="#general",
                    message_template="Missing {missing_var}",
                )

                # Should use original message
                call_args = mock_webhook.call_args
                payload = call_args.kwargs["payload_template"]
                assert payload["text"] == "Missing {missing_var}"


# =============================================================================
# save_checkpoint Tests
# =============================================================================


class TestSaveCheckpoint:
    """Tests for save_checkpoint handler."""

    @pytest.mark.asyncio
    async def test_save_checkpoint_basic(self, tmp_path):
        """Test basic checkpoint saving."""
        result = await save_checkpoint(
            path=str(tmp_path / "checkpoints"),
            filename_template="test_{debate_id}.json",
            debate_id="123",
        )

        assert result is not None
        assert Path(result).exists()

        # Verify content
        with open(result) as f:
            data = json.load(f)
            assert data["debate_id"] == "123"
            assert "_checkpoint_time" in data

    @pytest.mark.asyncio
    async def test_save_checkpoint_creates_directory(self, tmp_path):
        """Test that checkpoint creates directory if needed."""
        checkpoint_dir = tmp_path / "nested" / "checkpoints"

        result = await save_checkpoint(
            path=str(checkpoint_dir),
            debate_id="123",
        )

        assert result is not None
        assert checkpoint_dir.exists()

    @pytest.mark.asyncio
    async def test_save_checkpoint_include_fields(self, tmp_path):
        """Test checkpoint with specific fields."""
        result = await save_checkpoint(
            path=str(tmp_path),
            include_fields=["debate_id", "confidence"],
            debate_id="123",
            confidence=0.95,
            extra_field="should_be_excluded",
        )

        assert result is not None

        with open(result) as f:
            data = json.load(f)
            assert "debate_id" in data
            assert "confidence" in data
            assert "extra_field" not in data

    @pytest.mark.asyncio
    async def test_save_checkpoint_filename_interpolation(self, tmp_path):
        """Test checkpoint filename interpolation."""
        result = await save_checkpoint(
            path=str(tmp_path),
            filename_template="debate_{debate_id}_round_{round_num}.json",
            debate_id="abc",
            round_num=3,
        )

        assert result is not None
        assert "debate_abc_round_3.json" in result

    @pytest.mark.asyncio
    async def test_save_checkpoint_failed_interpolation(self, tmp_path):
        """Test checkpoint with failed filename interpolation."""
        result = await save_checkpoint(
            path=str(tmp_path),
            filename_template="debate_{missing_var}.json",
            debate_id="123",
        )

        # Should fall back to default filename
        assert result is not None
        assert "checkpoint_" in result

    @pytest.mark.asyncio
    async def test_save_checkpoint_complex_data(self, tmp_path):
        """Test checkpoint with complex data."""
        result = await save_checkpoint(
            path=str(tmp_path),
            debate_id="123",
            nested={"key": "value", "list": [1, 2, 3]},
            result={"consensus": True},
        )

        assert result is not None

        with open(result) as f:
            data = json.load(f)
            assert data["nested"]["key"] == "value"
            assert data["nested"]["list"] == [1, 2, 3]

    @pytest.mark.asyncio
    async def test_save_checkpoint_invalid_path(self):
        """Test checkpoint with invalid path."""
        # Use a path that will fail (e.g., device file on Unix)
        result = await save_checkpoint(
            path="/dev/null/cannot_create_here",
            debate_id="123",
        )

        assert result is None


# =============================================================================
# store_fact Tests
# =============================================================================


class TestStoreFact:
    """Tests for store_fact handler."""

    @pytest.mark.asyncio
    async def test_store_fact_import_failure(self):
        """Test store_fact when knowledge mound is not available."""
        # The import happens inside the function, so we patch the import mechanism
        import sys

        original_modules = sys.modules.copy()

        # Remove the module if it's cached
        if "aragora.knowledge.mound" in sys.modules:
            del sys.modules["aragora.knowledge.mound"]

        with patch.dict(sys.modules, {"aragora.knowledge.mound": None}):
            result = await store_fact(
                fact_type="consensus",
                content_field="answer",
                answer="Test answer",
            )

            # When import fails, should return False
            assert result is False

    @pytest.mark.asyncio
    async def test_store_fact_success(self):
        """Test successful fact storage."""
        mock_mound = AsyncMock()
        mock_mound.store_verified_fact = AsyncMock()

        # Create mock module
        mock_km_module = MagicMock()
        mock_km_module.get_knowledge_mound = MagicMock(return_value=mock_mound)

        with patch.dict("sys.modules", {"aragora.knowledge.mound": mock_km_module}):
            result = await store_fact(
                fact_type="consensus",
                content_field="final_answer",
                confidence_field="confidence",
                source="debate",
                final_answer="The answer is 42",
                confidence=0.95,
                debate_id="123",
            )

            assert result is True
            mock_mound.store_verified_fact.assert_called_once()

    @pytest.mark.asyncio
    async def test_store_fact_default_fields(self):
        """Test store_fact with default field names."""
        mock_mound = AsyncMock()
        mock_mound.store_verified_fact = AsyncMock()

        mock_km_module = MagicMock()
        mock_km_module.get_knowledge_mound = MagicMock(return_value=mock_mound)

        with patch.dict("sys.modules", {"aragora.knowledge.mound": mock_km_module}):
            result = await store_fact(
                fact_type="finding",
                final_answer="Important finding",
                confidence=0.8,
            )

            assert result is True

    @pytest.mark.asyncio
    async def test_store_fact_exception(self):
        """Test store_fact with exception."""
        mock_mound = AsyncMock()
        mock_mound.store_verified_fact = AsyncMock(side_effect=Exception("Storage error"))

        mock_km_module = MagicMock()
        mock_km_module.get_knowledge_mound = MagicMock(return_value=mock_mound)

        with patch.dict("sys.modules", {"aragora.knowledge.mound": mock_km_module}):
            result = await store_fact(
                fact_type="consensus",
                final_answer="Test",
            )

            assert result is False


# =============================================================================
# set_context_var Tests
# =============================================================================


class TestSetContextVar:
    """Tests for set_context_var handler.

    Note: set_context_var modifies the **kwargs dict locally, which doesn't
    affect the original context passed by the caller. The function is designed
    to be used within a hook wrapper that can observe these changes.
    """

    @pytest.mark.asyncio
    async def test_set_context_var_runs_without_error(self):
        """Test set_context_var completes without raising."""
        # The function modifies kwargs locally, not the original context
        await set_context_var(
            var_name="new_var",
            value="new_value",
            existing="value",
        )
        # Should not raise

    @pytest.mark.asyncio
    async def test_set_context_var_with_various_values(self):
        """Test set_context_var with various value types."""
        # String value
        await set_context_var(var_name="str_var", value="string")

        # Integer value
        await set_context_var(var_name="int_var", value=42)

        # List value
        await set_context_var(var_name="list_var", value=[1, 2, 3])

        # Dict value
        await set_context_var(var_name="dict_var", value={"key": "value"})

        # None value
        await set_context_var(var_name="none_var", value=None)

    @pytest.mark.asyncio
    async def test_set_context_var_complex_value(self):
        """Test setting complex nested value."""
        await set_context_var(
            var_name="data",
            value={"nested": {"key": "value"}, "list": [1, 2, 3]},
        )
        # Should complete without error


# =============================================================================
# delay_execution Tests
# =============================================================================


class TestDelayExecution:
    """Tests for delay_execution handler."""

    @pytest.mark.asyncio
    async def test_delay_execution_basic(self):
        """Test basic delay."""
        import time

        start = time.time()
        await delay_execution(seconds=0.1)
        elapsed = time.time() - start

        assert elapsed >= 0.1
        assert elapsed < 0.2  # Should not take too long

    @pytest.mark.asyncio
    async def test_delay_execution_zero(self):
        """Test zero delay."""
        import time

        start = time.time()
        await delay_execution(seconds=0)
        elapsed = time.time() - start

        assert elapsed < 0.1

    @pytest.mark.asyncio
    async def test_delay_execution_default(self):
        """Test default delay value."""
        import time

        start = time.time()
        await delay_execution()
        elapsed = time.time() - start

        # Default is 1.0 seconds
        assert elapsed >= 1.0
        assert elapsed < 1.2

    @pytest.mark.asyncio
    async def test_delay_execution_ignores_context(self):
        """Test that delay ignores context kwargs."""
        import time

        start = time.time()
        await delay_execution(
            seconds=0.1,
            debate_id="123",
            other="value",
        )
        elapsed = time.time() - start

        assert elapsed >= 0.1


# =============================================================================
# Handler Error Handling Tests
# =============================================================================


class TestHandlerErrorHandling:
    """Tests for handler error handling."""

    @pytest.mark.asyncio
    async def test_log_event_no_crash_on_empty_context(self):
        """Test log_event doesn't crash with empty context."""
        await log_event()  # Should not raise

    @pytest.mark.asyncio
    async def test_log_metric_no_crash_on_none_tags(self):
        """Test log_metric handles None tags."""
        await log_metric(metric_name="test", value=1, tags=None)

    @pytest.mark.asyncio
    async def test_save_checkpoint_handles_non_serializable(self, tmp_path):
        """Test checkpoint handles non-JSON-serializable data."""

        class NonSerializable:
            pass

        result = await save_checkpoint(
            path=str(tmp_path),
            debate_id="123",
            obj=NonSerializable(),
        )

        # Should succeed with default=str serialization
        assert result is not None


# =============================================================================
# Integration Tests
# =============================================================================


class TestBuiltinIntegration:
    """Integration tests for built-in handlers."""

    @pytest.mark.asyncio
    async def test_handler_chain(self, tmp_path, caplog):
        """Test chaining multiple handlers."""
        import logging

        with caplog.at_level(logging.INFO):
            # Log event
            await log_event(
                message="Starting checkpoint",
                level="info",
            )

            # Save checkpoint
            result = await save_checkpoint(
                path=str(tmp_path),
                debate_id="123",
            )

            # Log metric
            await log_metric(
                metric_name="checkpoint_saved",
                value=1 if result else 0,
            )

        assert "Starting checkpoint" in caplog.text
        assert result is not None
        assert "checkpoint_saved" in caplog.text

    @pytest.mark.asyncio
    async def test_all_handlers_importable(self):
        """Test that all exported handlers are importable."""
        from aragora.hooks.builtin import (
            delay_execution,
            log_event,
            log_metric,
            save_checkpoint,
            send_slack_notification,
            send_webhook,
            set_context_var,
            store_fact,
        )

        # All should be callable
        assert callable(log_event)
        assert callable(log_metric)
        assert callable(send_webhook)
        assert callable(send_slack_notification)
        assert callable(save_checkpoint)
        assert callable(store_fact)
        assert callable(set_context_var)
        assert callable(delay_execution)

    @pytest.mark.asyncio
    async def test_all_handlers_are_async(self):
        """Test that all handlers are async functions."""
        import inspect

        from aragora.hooks.builtin import (
            delay_execution,
            log_event,
            log_metric,
            save_checkpoint,
            send_slack_notification,
            send_webhook,
            set_context_var,
            store_fact,
        )

        handlers = [
            log_event,
            log_metric,
            send_webhook,
            send_slack_notification,
            save_checkpoint,
            store_fact,
            set_context_var,
            delay_execution,
        ]

        for handler in handlers:
            assert inspect.iscoroutinefunction(handler), f"{handler.__name__} should be async"
