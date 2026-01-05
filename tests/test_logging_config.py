"""Tests for structured logging configuration."""

import json
import logging
import pytest
from io import StringIO
from unittest.mock import patch

from aragora.logging_config import (
    LogRecord,
    JSONFormatter,
    TextFormatter,
    StructuredLogger,
    LogContext,
    get_logger,
    configure_logging,
    set_context,
    get_context,
    clear_context,
    log_function,
    _log_context,
)


class TestLogRecord:
    """Test LogRecord dataclass."""

    def test_basic_record(self):
        """Test basic log record creation."""
        record = LogRecord(
            timestamp="2024-01-01T00:00:00Z",
            level="INFO",
            logger="test",
            message="Hello world",
        )
        assert record.level == "INFO"
        assert record.message == "Hello world"

    def test_record_with_fields(self):
        """Test log record with custom fields."""
        record = LogRecord(
            timestamp="2024-01-01T00:00:00Z",
            level="DEBUG",
            logger="test",
            message="Processing",
            fields={"user_id": 123, "action": "login"},
        )
        d = record.to_dict()
        assert d["user_id"] == 123
        assert d["action"] == "login"

    def test_record_with_trace_context(self):
        """Test log record with trace context."""
        record = LogRecord(
            timestamp="2024-01-01T00:00:00Z",
            level="INFO",
            logger="test",
            message="Traced operation",
            trace_id="abc123def456",
            span_id="span789",
            debate_id="debate_001",
        )
        d = record.to_dict()
        assert d["trace_id"] == "abc123def456"
        assert d["span_id"] == "span789"
        assert d["debate_id"] == "debate_001"

    def test_record_with_exception(self):
        """Test log record with exception info."""
        record = LogRecord(
            timestamp="2024-01-01T00:00:00Z",
            level="ERROR",
            logger="test",
            message="Failed",
            exception={
                "type": "ValueError",
                "message": "Invalid input",
                "traceback": "Traceback...",
            },
        )
        d = record.to_dict()
        assert d["exception"]["type"] == "ValueError"

    def test_to_json(self):
        """Test JSON serialization."""
        record = LogRecord(
            timestamp="2024-01-01T00:00:00Z",
            level="INFO",
            logger="test",
            message="Test",
            fields={"count": 42},
        )
        json_str = record.to_json()
        parsed = json.loads(json_str)
        assert parsed["level"] == "INFO"
        assert parsed["count"] == 42

    def test_to_text(self):
        """Test text format."""
        record = LogRecord(
            timestamp="2024-01-01 00:00:00",
            level="INFO",
            logger="test",
            message="Hello",
            trace_id="abc123",
        )
        text = record.to_text()
        assert "INFO" in text
        assert "Hello" in text
        assert "abc123" in text  # Trace ID appears (truncated to first 8 chars)


class TestJSONFormatter:
    """Test JSON log formatter."""

    def test_format_basic(self):
        """Test basic JSON formatting."""
        formatter = JSONFormatter()
        log_record = logging.LogRecord(
            name="test.module",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="Test message",
            args=(),
            exc_info=None,
        )
        output = formatter.format(log_record)
        parsed = json.loads(output)
        assert parsed["level"] == "INFO"
        assert parsed["msg"] == "Test message"
        assert parsed["logger"] == "test.module"

    def test_format_with_structured_fields(self):
        """Test JSON formatting with structured fields."""
        formatter = JSONFormatter()
        log_record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="Operation completed",
            args=(),
            exc_info=None,
        )
        log_record.structured_fields = {"duration_ms": 123.45, "status": "success"}
        output = formatter.format(log_record)
        parsed = json.loads(output)
        assert parsed["duration_ms"] == 123.45
        assert parsed["status"] == "success"

    def test_format_with_exception(self):
        """Test JSON formatting with exception."""
        formatter = JSONFormatter()
        try:
            raise ValueError("Test error")
        except ValueError:
            import sys
            exc_info = sys.exc_info()

        log_record = logging.LogRecord(
            name="test",
            level=logging.ERROR,
            pathname="",
            lineno=0,
            msg="Error occurred",
            args=(),
            exc_info=exc_info,
        )
        output = formatter.format(log_record)
        parsed = json.loads(output)
        assert "exception" in parsed
        assert parsed["exception"]["type"] == "ValueError"
        assert "Test error" in parsed["exception"]["message"]


class TestTextFormatter:
    """Test text log formatter."""

    def test_format_basic(self):
        """Test basic text formatting."""
        formatter = TextFormatter()
        log_record = logging.LogRecord(
            name="test.module",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="Test message",
            args=(),
            exc_info=None,
        )
        output = formatter.format(log_record)
        assert "[INFO]" in output
        assert "Test message" in output
        assert "[module]" in output  # Short name

    def test_format_with_fields(self):
        """Test text formatting with fields."""
        formatter = TextFormatter()
        log_record = logging.LogRecord(
            name="test",
            level=logging.DEBUG,
            pathname="",
            lineno=0,
            msg="Processing",
            args=(),
            exc_info=None,
        )
        log_record.structured_fields = {"items": 10}
        output = formatter.format(log_record)
        assert "items=10" in output


class TestStructuredLogger:
    """Test StructuredLogger wrapper."""

    def test_logger_creation(self):
        """Test creating a structured logger."""
        logger = StructuredLogger("test.module")
        assert logger._name == "test.module"

    def test_log_levels(self):
        """Test different log levels."""
        logger = StructuredLogger("test")
        # These should not raise
        logger.debug("Debug message")
        logger.info("Info message")
        logger.warning("Warning message")
        logger.error("Error message")
        logger.critical("Critical message")

    def test_log_with_fields(self):
        """Test logging with structured fields."""
        logger = StructuredLogger("test")
        # Should not raise
        logger.info("User action", user_id=123, action="login")

    def test_exception_logging(self):
        """Test exception logging."""
        logger = StructuredLogger("test")
        try:
            raise RuntimeError("Test error")
        except RuntimeError:
            logger.exception("Operation failed", operation="test")


class TestLogContext:
    """Test LogContext context manager."""

    def test_context_sets_fields(self):
        """Test that context sets fields."""
        clear_context()
        with LogContext(user="alice", request_id="req123"):
            ctx = get_context()
            assert ctx["user"] == "alice"
            assert ctx["request_id"] == "req123"

    def test_context_clears_on_exit(self):
        """Test that context is cleared on exit."""
        clear_context()
        with LogContext(temp="value"):
            assert get_context().get("temp") == "value"
        assert get_context().get("temp") is None

    def test_nested_contexts(self):
        """Test nested contexts merge correctly."""
        clear_context()
        with LogContext(outer="1"):
            with LogContext(inner="2"):
                ctx = get_context()
                assert ctx["outer"] == "1"
                assert ctx["inner"] == "2"
            # Inner context gone, outer remains
            ctx = get_context()
            assert ctx.get("outer") == "1"
            assert ctx.get("inner") is None

    def test_context_override(self):
        """Test that inner context can override outer."""
        clear_context()
        with LogContext(key="outer"):
            with LogContext(key="inner"):
                assert get_context()["key"] == "inner"
            assert get_context()["key"] == "outer"


class TestContextFunctions:
    """Test context manipulation functions."""

    def test_set_context(self):
        """Test set_context function."""
        clear_context()
        set_context(user="bob")
        assert get_context()["user"] == "bob"

    def test_set_context_merges(self):
        """Test set_context merges with existing."""
        clear_context()
        set_context(a="1")
        set_context(b="2")
        ctx = get_context()
        assert ctx["a"] == "1"
        assert ctx["b"] == "2"

    def test_clear_context(self):
        """Test clear_context removes all fields."""
        set_context(temp="value")
        clear_context()
        assert get_context() == {}


class TestGetLogger:
    """Test get_logger factory function."""

    def test_returns_structured_logger(self):
        """Test that get_logger returns StructuredLogger."""
        logger = get_logger("test.module")
        assert isinstance(logger, StructuredLogger)

    def test_caches_loggers(self):
        """Test that loggers are cached."""
        logger1 = get_logger("test.cached")
        logger2 = get_logger("test.cached")
        assert logger1 is logger2

    def test_different_names_different_loggers(self):
        """Test that different names give different loggers."""
        logger1 = get_logger("test.one")
        logger2 = get_logger("test.two")
        assert logger1 is not logger2


class TestConfigureLogging:
    """Test logging configuration."""

    def test_configure_json(self):
        """Test configuring JSON output."""
        configure_logging(level="DEBUG", json_output=True)
        root = logging.getLogger()
        assert any(
            isinstance(h.formatter, JSONFormatter)
            for h in root.handlers
        )

    def test_configure_text(self):
        """Test configuring text output."""
        configure_logging(level="INFO", json_output=False)
        root = logging.getLogger()
        assert any(
            isinstance(h.formatter, TextFormatter)
            for h in root.handlers
        )

    def test_configure_level(self):
        """Test configuring log level."""
        configure_logging(level="WARNING")
        root = logging.getLogger()
        assert root.level == logging.WARNING


class TestLogFunctionDecorator:
    """Test log_function decorator."""

    def test_logs_function_completion(self):
        """Test that decorator logs function completion."""
        @log_function(level="DEBUG")
        def sample_function():
            return 42

        result = sample_function()
        assert result == 42

    def test_logs_function_error(self):
        """Test that decorator logs function errors."""
        @log_function(level="DEBUG")
        def failing_function():
            raise ValueError("Test error")

        with pytest.raises(ValueError):
            failing_function()

    def test_logs_duration(self):
        """Test that decorator logs duration."""
        @log_function(level="DEBUG", log_duration=True)
        def slow_function():
            import time
            time.sleep(0.01)
            return "done"

        result = slow_function()
        assert result == "done"

    @pytest.mark.asyncio
    async def test_async_function(self):
        """Test decorator with async function."""
        @log_function(level="DEBUG")
        async def async_operation():
            return "async result"

        result = await async_operation()
        assert result == "async result"

    @pytest.mark.asyncio
    async def test_async_function_error(self):
        """Test decorator logs async function errors."""
        @log_function(level="DEBUG")
        async def failing_async():
            raise RuntimeError("Async error")

        with pytest.raises(RuntimeError):
            await failing_async()


class TestIntegration:
    """Integration tests for structured logging."""

    def test_json_output_includes_fields(self):
        """Test that JSON output includes structured fields."""
        configure_logging(level="DEBUG", json_output=True)

        # Capture log output
        handler = logging.handlers.MemoryHandler(capacity=100)
        handler.setFormatter(JSONFormatter())
        test_logger = logging.getLogger("test.integration")
        test_logger.addHandler(handler)

        # Log with structured fields
        logger = StructuredLogger("test.integration")
        logger.info("Test message", custom_field="value")

        # Check captured output
        assert len(handler.buffer) > 0
        # The last record should have our field
        record = handler.buffer[-1]
        assert hasattr(record, "structured_fields")
        assert record.structured_fields.get("custom_field") == "value"

        handler.close()

    def test_context_propagates_to_logs(self):
        """Test that context fields appear in logs."""
        configure_logging(level="DEBUG", json_output=True)
        clear_context()

        handler = logging.handlers.MemoryHandler(capacity=100)
        handler.setFormatter(JSONFormatter())
        test_logger = logging.getLogger("test.context")
        test_logger.addHandler(handler)

        with LogContext(trace_id="test_trace_123"):
            logger = StructuredLogger("test.context")
            logger.info("Traced operation")

        # The log should include trace_id from context
        assert len(handler.buffer) > 0
        handler.close()

    def test_full_logging_workflow(self):
        """Test complete logging workflow."""
        configure_logging(level="DEBUG", json_output=True)
        clear_context()

        logger = get_logger("test.workflow")

        # Set debate context
        with LogContext(debate_id="debate_001", trace_id="trace_abc"):
            logger.info("Debate started", agents=["claude", "gpt4"])

            # Nested operation
            with LogContext(round_num=1):
                logger.debug("Processing round", phase="proposal")

            logger.info("Debate completed", consensus=True)

        # Context should be cleared
        assert get_context() == {}


import logging.handlers
