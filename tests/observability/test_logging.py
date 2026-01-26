"""
Tests for structured logging utilities.

Tests cover:
- Correlation ID functions (set, get, generate, context)
- Sensitive data redaction
- LogConfig dataclass
- JSONFormatter
- HumanFormatter
- StructuredLogger
- configure_logging
- get_logger
"""

import pytest
import json
import logging
import os
import threading
from unittest.mock import patch, MagicMock
from io import StringIO

from aragora.observability.logging import (
    set_correlation_id,
    get_correlation_id,
    generate_correlation_id,
    correlation_context,
    SENSITIVE_FIELDS,
    _is_sensitive_value,
    _redact_sensitive,
    LogConfig,
    JSONFormatter,
    HumanFormatter,
    StructuredLogger,
    configure_logging,
    get_logger,
)


@pytest.fixture(autouse=True)
def reset_correlation_id():
    """Reset correlation ID between tests."""
    from aragora.observability import logging as log_module

    log_module._correlation_id.value = None
    yield
    log_module._correlation_id.value = None


@pytest.fixture
def reset_logging():
    """Reset global logging state."""
    from aragora.observability import logging as log_module

    old_config = log_module._log_config
    old_loggers = log_module._loggers.copy()
    yield
    log_module._log_config = old_config
    log_module._loggers = old_loggers


class TestCorrelationId:
    """Tests for correlation ID functions."""

    def test_set_and_get_correlation_id(self):
        """Sets and gets correlation ID."""
        set_correlation_id("test-123")
        assert get_correlation_id() == "test-123"

    def test_get_returns_none_when_not_set(self):
        """Returns None when correlation ID not set."""
        assert get_correlation_id() is None

    def test_generate_correlation_id(self):
        """Generates unique correlation IDs."""
        id1 = generate_correlation_id()
        id2 = generate_correlation_id()

        assert isinstance(id1, str)
        assert len(id1) == 8
        assert id1 != id2

    def test_correlation_context_sets_id(self):
        """Context manager sets correlation ID."""
        with correlation_context("ctx-123") as cid:
            assert cid == "ctx-123"
            assert get_correlation_id() == "ctx-123"

    def test_correlation_context_generates_id(self):
        """Context manager generates ID if not provided."""
        with correlation_context() as cid:
            assert isinstance(cid, str)
            assert get_correlation_id() == cid

    def test_correlation_context_restores_previous(self):
        """Context manager restores previous ID."""
        set_correlation_id("outer-123")

        with correlation_context("inner-456"):
            assert get_correlation_id() == "inner-456"

        assert get_correlation_id() == "outer-123"

    def test_correlation_context_clears_when_no_previous(self):
        """Context manager clears ID when no previous ID."""
        with correlation_context("temp-123"):
            pass

        assert get_correlation_id() is None

    def test_correlation_id_thread_local(self):
        """Correlation IDs are thread-local."""
        results = {}

        def thread_func(name, cid):
            set_correlation_id(cid)
            results[name] = get_correlation_id()

        t1 = threading.Thread(target=thread_func, args=("t1", "id-1"))
        t2 = threading.Thread(target=thread_func, args=("t2", "id-2"))

        t1.start()
        t2.start()
        t1.join()
        t2.join()

        assert results["t1"] == "id-1"
        assert results["t2"] == "id-2"


class TestSensitiveRedaction:
    """Tests for sensitive data redaction."""

    def test_sensitive_fields_exist(self):
        """SENSITIVE_FIELDS contains expected fields."""
        assert "password" in SENSITIVE_FIELDS
        assert "token" in SENSITIVE_FIELDS
        assert "api_key" in SENSITIVE_FIELDS
        assert "authorization" in SENSITIVE_FIELDS

    def test_is_sensitive_value_base64(self):
        """Detects Base64 encoded values."""
        base64_value = "dGhpcyBpcyBhIHRlc3QgdmFsdWUgdGhhdCBpcyBsb25nIGVub3VnaA=="
        assert _is_sensitive_value(base64_value) is True

    def test_is_sensitive_value_openai_key(self):
        """Detects OpenAI API key pattern."""
        assert _is_sensitive_value("sk-1234567890123456789012345678901234") is True

    def test_is_sensitive_value_short_string(self):
        """Short strings are not considered sensitive."""
        assert _is_sensitive_value("short") is False

    def test_is_sensitive_value_non_string(self):
        """Non-strings are not considered sensitive."""
        assert _is_sensitive_value(12345678901234567890) is False
        assert _is_sensitive_value(None) is False

    def test_redact_sensitive_password(self):
        """Redacts password field."""
        data = {"username": "alice", "password": "secret123"}
        result = _redact_sensitive(data)

        assert result["username"] == "alice"
        assert result["password"] == "[REDACTED]"

    def test_redact_sensitive_nested(self):
        """Redacts nested sensitive fields."""
        data = {"user": {"name": "alice", "api_key": "secret"}}
        result = _redact_sensitive(data)

        assert result["user"]["name"] == "alice"
        assert result["user"]["api_key"] == "[REDACTED]"

    def test_redact_sensitive_value_pattern(self):
        """Redacts values matching secret patterns."""
        data = {"key": "sk-1234567890123456789012345678901234"}
        result = _redact_sensitive(data)

        assert "[REDACTED:" in result["key"]

    def test_redact_preserves_non_sensitive(self):
        """Preserves non-sensitive fields."""
        data = {"name": "alice", "count": 42, "active": True}
        result = _redact_sensitive(data)

        assert result == data


class TestLogConfig:
    """Tests for LogConfig dataclass."""

    def test_default_values(self):
        """LogConfig has sensible defaults."""
        config = LogConfig()

        assert config.environment == "development"
        assert config.level == "INFO"
        assert config.format == "human"
        assert config.service == "aragora"
        assert config.redact_sensitive is True

    def test_custom_values(self):
        """LogConfig accepts custom values."""
        config = LogConfig(
            environment="production",
            level="WARNING",
            format="json",
        )

        assert config.environment == "production"
        assert config.level == "WARNING"
        assert config.format == "json"

    def test_from_env_development(self):
        """from_env creates development config."""
        with patch.dict(os.environ, {"ARAGORA_ENV": "development"}, clear=False):
            config = LogConfig.from_env()

            assert config.environment == "development"
            assert config.format == "human"

    def test_from_env_production(self):
        """from_env creates production config."""
        with patch.dict(os.environ, {"ARAGORA_ENV": "production"}, clear=False):
            config = LogConfig.from_env()

            assert config.environment == "production"
            assert config.level == "INFO"
            assert config.format == "json"

    def test_from_env_overrides(self):
        """from_env respects environment variable overrides."""
        env = {
            "ARAGORA_ENV": "staging",
            "ARAGORA_LOG_LEVEL": "DEBUG",
            "ARAGORA_LOG_FORMAT": "json",
        }
        with patch.dict(os.environ, env, clear=False):
            config = LogConfig.from_env()

            assert config.environment == "staging"
            assert config.level == "DEBUG"
            assert config.format == "json"


class TestJSONFormatter:
    """Tests for JSONFormatter."""

    @pytest.fixture
    def formatter(self):
        """JSON formatter for tests."""
        config = LogConfig(environment="production", format="json")
        return JSONFormatter(config)

    def test_format_produces_valid_json(self, formatter):
        """Produces valid JSON output."""
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test message",
            args=(),
            exc_info=None,
        )

        output = formatter.format(record)
        data = json.loads(output)

        assert data["message"] == "Test message"
        assert data["level"] == "INFO"
        assert data["logger"] == "test"

    def test_format_includes_correlation_id(self, formatter):
        """Includes correlation ID when set."""
        set_correlation_id("json-123")

        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test",
            args=(),
            exc_info=None,
        )

        output = formatter.format(record)
        data = json.loads(output)

        assert data["correlation_id"] == "json-123"

    def test_format_includes_extra_fields(self, formatter):
        """Includes extra fields from record."""
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test",
            args=(),
            exc_info=None,
        )
        record.extra_fields = {"user_id": "123", "action": "login"}

        output = formatter.format(record)
        data = json.loads(output)

        assert data["user_id"] == "123"
        assert data["action"] == "login"

    def test_format_redacts_sensitive(self, formatter):
        """Redacts sensitive fields."""
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test",
            args=(),
            exc_info=None,
        )
        record.extra_fields = {"password": "secret", "username": "alice"}

        output = formatter.format(record)
        data = json.loads(output)

        assert data["password"] == "[REDACTED]"
        assert data["username"] == "alice"

    def test_format_error_includes_source(self, formatter):
        """Error logs include source location."""
        record = logging.LogRecord(
            name="test",
            level=logging.ERROR,
            pathname="/path/to/test.py",
            lineno=42,
            msg="Error occurred",
            args=(),
            exc_info=None,
        )
        record.funcName = "test_func"

        output = formatter.format(record)
        data = json.loads(output)

        assert "source" in data
        assert data["source"]["line"] == 42


class TestHumanFormatter:
    """Tests for HumanFormatter."""

    @pytest.fixture
    def formatter(self):
        """Human formatter for tests (no colors)."""
        config = LogConfig(environment="development", format="human")
        return HumanFormatter(config, use_colors=False)

    def test_format_includes_message(self, formatter):
        """Includes log message."""
        record = logging.LogRecord(
            name="test.module",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Hello world",
            args=(),
            exc_info=None,
        )

        output = formatter.format(record)

        assert "Hello world" in output
        assert "test.module" in output

    def test_format_includes_correlation_id(self, formatter):
        """Includes correlation ID when set."""
        set_correlation_id("human-123")

        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test",
            args=(),
            exc_info=None,
        )

        output = formatter.format(record)

        assert "[human-123]" in output

    def test_format_includes_extra_fields(self, formatter):
        """Includes extra fields."""
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test",
            args=(),
            exc_info=None,
        )
        record.extra_fields = {"user": "alice"}

        output = formatter.format(record)

        assert "user=" in output
        assert "alice" in output


class TestStructuredLogger:
    """Tests for StructuredLogger."""

    @pytest.fixture
    def logger(self):
        """Structured logger for tests."""
        config = LogConfig(level="DEBUG")
        base_logger = logging.getLogger("test.structured")
        base_logger.setLevel(logging.DEBUG)
        return StructuredLogger(base_logger, config)

    def test_debug(self, logger):
        """debug method logs at DEBUG level."""
        with patch.object(logger._logger, "handle") as mock_handle:
            logger.debug("Debug message", key="value")

            mock_handle.assert_called_once()
            record = mock_handle.call_args[0][0]
            assert record.levelno == logging.DEBUG
            assert record.extra_fields == {"key": "value"}

    def test_info(self, logger):
        """info method logs at INFO level."""
        with patch.object(logger._logger, "handle") as mock_handle:
            logger.info("Info message", count=42)

            mock_handle.assert_called_once()
            record = mock_handle.call_args[0][0]
            assert record.levelno == logging.INFO

    def test_warning(self, logger):
        """warning method logs at WARNING level."""
        with patch.object(logger._logger, "handle") as mock_handle:
            logger.warning("Warning message")

            record = mock_handle.call_args[0][0]
            assert record.levelno == logging.WARNING

    def test_error(self, logger):
        """error method logs at ERROR level."""
        with patch.object(logger._logger, "handle") as mock_handle:
            logger.error("Error message", error_code=500)

            record = mock_handle.call_args[0][0]
            assert record.levelno == logging.ERROR

    def test_critical(self, logger):
        """critical method logs at CRITICAL level."""
        with patch.object(logger._logger, "handle") as mock_handle:
            logger.critical("Critical message")

            record = mock_handle.call_args[0][0]
            assert record.levelno == logging.CRITICAL

    def test_exception(self, logger):
        """exception method includes traceback."""
        with patch.object(logger._logger, "handle") as mock_handle:
            try:
                raise ValueError("Test error")
            except ValueError:
                logger.exception("Caught error")

            record = mock_handle.call_args[0][0]
            assert record.exc_info is not None


class TestConfigureLogging:
    """Tests for configure_logging function."""

    def test_configure_logging_returns_config(self, reset_logging):
        """Returns LogConfig."""
        config = configure_logging(environment="test")

        assert isinstance(config, LogConfig)
        assert config.environment == "test"

    def test_configure_logging_sets_level(self, reset_logging):
        """Sets log level on root logger."""
        configure_logging(level="WARNING")

        root = logging.getLogger()
        assert root.level == logging.WARNING

    def test_configure_logging_json_format(self, reset_logging):
        """Configures JSON formatter."""
        configure_logging(format="json")

        root = logging.getLogger()
        handler = root.handlers[0]
        assert isinstance(handler.formatter, JSONFormatter)

    def test_configure_logging_human_format(self, reset_logging):
        """Configures human formatter."""
        configure_logging(format="human")

        root = logging.getLogger()
        handler = root.handlers[0]
        assert isinstance(handler.formatter, HumanFormatter)


class TestGetLogger:
    """Tests for get_logger function."""

    def test_get_logger_returns_structured_logger(self, reset_logging):
        """Returns StructuredLogger."""
        logger = get_logger("test.module")

        assert isinstance(logger, StructuredLogger)

    def test_get_logger_caches_loggers(self, reset_logging):
        """Caches loggers by name."""
        logger1 = get_logger("test.cached")
        logger2 = get_logger("test.cached")

        assert logger1 is logger2

    def test_get_logger_different_names(self, reset_logging):
        """Different names return different loggers."""
        logger1 = get_logger("module.a")
        logger2 = get_logger("module.b")

        assert logger1 is not logger2
