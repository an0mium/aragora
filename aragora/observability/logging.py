"""
Structured logging for Aragora.

Provides JSON-formatted logging with:
- Correlation IDs for request tracing
- Structured key-value logging
- Sensitive data redaction
- Environment-aware configuration

Features:
- JSON output for production (machine-parseable)
- Human-readable output for development
- Automatic correlation ID propagation
- Sensitive field redaction (passwords, tokens)

Usage:
    from aragora.observability.logging import configure_logging, get_logger

    # Configure at application startup
    configure_logging(environment="production")

    # Get a logger for a module
    logger = get_logger(__name__)

    # Structured logging
    logger.info("debate_started", debate_id="123", agents=["claude", "gpt"])
    logger.error("agent_failed", agent="claude", error="timeout")

    # With correlation ID context
    with correlation_context("request-123"):
        logger.info("processing_request")
        # All logs in this context will include correlation_id
"""

import json
import logging
import os
import re
import sys
import threading
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional, Generator

# Thread-local storage for correlation IDs
_correlation_id = threading.local()


def set_correlation_id(correlation_id: str) -> None:
    """Set the correlation ID for the current thread."""
    _correlation_id.value = correlation_id


def get_correlation_id() -> Optional[str]:
    """Get the correlation ID for the current thread."""
    return getattr(_correlation_id, "value", None)


def generate_correlation_id() -> str:
    """Generate a new correlation ID."""
    return str(uuid.uuid4())[:8]


@contextmanager
def correlation_context(correlation_id: Optional[str] = None) -> Generator[str, None, None]:
    """Context manager for correlation ID scope.

    Args:
        correlation_id: ID to use (generates new if None)

    Yields:
        The correlation ID being used
    """
    cid = correlation_id or generate_correlation_id()
    old_id = get_correlation_id()
    set_correlation_id(cid)
    try:
        yield cid
    finally:
        if old_id:
            set_correlation_id(old_id)
        else:
            _correlation_id.value = None


# Fields to redact in logs
SENSITIVE_FIELDS = frozenset({
    "password",
    "secret",
    "token",
    "api_key",
    "apikey",
    "api-key",
    "authorization",
    "auth",
    "credential",
    "private_key",
    "privatekey",
    "access_token",
    "refresh_token",
})

# Pattern to detect potential secrets in values
SECRET_PATTERNS = [
    re.compile(r"^[A-Za-z0-9+/]{32,}={0,2}$"),  # Base64
    re.compile(r"^sk-[A-Za-z0-9]{32,}$"),  # OpenAI key pattern
    re.compile(r"^[A-Za-z0-9]{32,}$"),  # Generic API key
]


def _is_sensitive_value(value: Any) -> bool:
    """Check if a value looks like a secret."""
    if not isinstance(value, str):
        return False
    if len(value) < 20:
        return False
    for pattern in SECRET_PATTERNS:
        if pattern.match(value):
            return True
    return False


def _redact_sensitive(data: dict[str, Any]) -> dict[str, Any]:
    """Redact sensitive fields from a dict."""
    result: dict[str, Any] = {}
    for key, value in data.items():
        key_lower = key.lower().replace("-", "_")
        if key_lower in SENSITIVE_FIELDS:
            result[key] = "[REDACTED]"
        elif isinstance(value, dict):
            result[key] = _redact_sensitive(value)
        elif isinstance(value, str) and _is_sensitive_value(value):
            result[key] = f"[REDACTED:{len(value)}chars]"
        else:
            result[key] = value
    return result


@dataclass
class LogConfig:
    """Configuration for structured logging."""

    # Environment (development, staging, production)
    environment: str = "development"

    # Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    level: str = "INFO"

    # Output format (json, human)
    format: str = "human"

    # Service name for log identification
    service: str = "aragora"

    # Enable redaction of sensitive fields
    redact_sensitive: bool = True

    # Include stack traces in error logs
    include_stacktrace: bool = True

    # Additional static fields to include in all logs
    static_fields: dict = field(default_factory=dict)

    @classmethod
    def from_env(cls) -> "LogConfig":
        """Create config from environment variables."""
        env = os.getenv("ARAGORA_ENV", "development")
        return cls(
            environment=env,
            level=os.getenv("ARAGORA_LOG_LEVEL", "INFO" if env == "production" else "DEBUG"),
            format=os.getenv("ARAGORA_LOG_FORMAT", "json" if env == "production" else "human"),
            service=os.getenv("ARAGORA_SERVICE_NAME", "aragora"),
            redact_sensitive=os.getenv("ARAGORA_LOG_REDACT", "true").lower() == "true",
            include_stacktrace=os.getenv("ARAGORA_LOG_STACKTRACE", "true").lower() == "true",
        )


class JSONFormatter(logging.Formatter):
    """JSON log formatter for production."""

    def __init__(self, config: LogConfig):
        super().__init__()
        self.config = config

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_data: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "service": self.config.service,
            "environment": self.config.environment,
        }

        # Add correlation ID if present
        correlation_id = get_correlation_id()
        if correlation_id:
            log_data["correlation_id"] = correlation_id

        # Add extra fields from the record
        if hasattr(record, "extra_fields"):
            extra = record.extra_fields
            if self.config.redact_sensitive:
                extra = _redact_sensitive(extra)
            log_data.update(extra)

        # Add static fields
        if self.config.static_fields:
            log_data.update(self.config.static_fields)

        # Add exception info if present
        if record.exc_info and self.config.include_stacktrace:
            log_data["exception"] = self.formatException(record.exc_info)

        # Add source location for errors
        if record.levelno >= logging.ERROR:
            log_data["source"] = {
                "file": record.pathname,
                "line": record.lineno,
                "function": record.funcName,
            }

        return json.dumps(log_data, default=str)


class HumanFormatter(logging.Formatter):
    """Human-readable log formatter for development."""

    COLORS = {
        "DEBUG": "\033[36m",     # Cyan
        "INFO": "\033[32m",      # Green
        "WARNING": "\033[33m",   # Yellow
        "ERROR": "\033[31m",     # Red
        "CRITICAL": "\033[35m",  # Magenta
    }
    RESET = "\033[0m"

    def __init__(self, config: LogConfig, use_colors: bool = True):
        super().__init__()
        self.config = config
        self.use_colors = use_colors and sys.stderr.isatty()

    def format(self, record: logging.LogRecord) -> str:
        """Format log record for human readability."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        level = record.levelname

        if self.use_colors:
            color = self.COLORS.get(level, "")
            level_str = f"{color}{level:8}{self.RESET}"
        else:
            level_str = f"{level:8}"

        message = record.getMessage()

        # Add correlation ID
        correlation_id = get_correlation_id()
        cid_str = f"[{correlation_id}] " if correlation_id else ""

        # Format extra fields
        extra_str = ""
        if hasattr(record, "extra_fields") and record.extra_fields:
            extra = record.extra_fields
            if self.config.redact_sensitive:
                extra = _redact_sensitive(extra)
            extra_parts = [f"{k}={v!r}" for k, v in extra.items()]
            extra_str = " | " + ", ".join(extra_parts)

        # Format exception
        exc_str = ""
        if record.exc_info and self.config.include_stacktrace:
            exc_str = "\n" + self.formatException(record.exc_info)

        return f"{timestamp} {level_str} {cid_str}{record.name}: {message}{extra_str}{exc_str}"


class StructuredLogger:
    """Logger wrapper with structured logging support."""

    def __init__(self, logger: logging.Logger, config: LogConfig):
        self._logger = logger
        self._config = config

    def _log(self, level: int, message: str, **kwargs: Any) -> None:
        """Internal log method with extra fields."""
        record = self._logger.makeRecord(
            self._logger.name,
            level,
            "(unknown file)",
            0,
            message,
            (),
            None,
        )
        record.extra_fields = kwargs
        self._logger.handle(record)

    def debug(self, message: str, **kwargs: Any) -> None:
        """Log at DEBUG level with structured data."""
        if self._logger.isEnabledFor(logging.DEBUG):
            self._log(logging.DEBUG, message, **kwargs)

    def info(self, message: str, **kwargs: Any) -> None:
        """Log at INFO level with structured data."""
        if self._logger.isEnabledFor(logging.INFO):
            self._log(logging.INFO, message, **kwargs)

    def warning(self, message: str, **kwargs: Any) -> None:
        """Log at WARNING level with structured data."""
        if self._logger.isEnabledFor(logging.WARNING):
            self._log(logging.WARNING, message, **kwargs)

    def error(self, message: str, exc_info: bool = False, **kwargs: Any) -> None:
        """Log at ERROR level with structured data."""
        if self._logger.isEnabledFor(logging.ERROR):
            record = self._logger.makeRecord(
                self._logger.name,
                logging.ERROR,
                "(unknown file)",
                0,
                message,
                (),
                sys.exc_info() if exc_info else None,
            )
            record.extra_fields = kwargs
            self._logger.handle(record)

    def critical(self, message: str, exc_info: bool = False, **kwargs: Any) -> None:
        """Log at CRITICAL level with structured data."""
        if self._logger.isEnabledFor(logging.CRITICAL):
            record = self._logger.makeRecord(
                self._logger.name,
                logging.CRITICAL,
                "(unknown file)",
                0,
                message,
                (),
                sys.exc_info() if exc_info else None,
            )
            record.extra_fields = kwargs
            self._logger.handle(record)

    def exception(self, message: str, **kwargs: Any) -> None:
        """Log an exception with traceback."""
        self.error(message, exc_info=True, **kwargs)


# Global configuration
_log_config: Optional[LogConfig] = None
_loggers: dict[str, StructuredLogger] = {}


def configure_logging(
    environment: Optional[str] = None,
    level: Optional[str] = None,
    format: Optional[str] = None,
    **kwargs: Any,
) -> LogConfig:
    """Configure structured logging.

    Should be called once at application startup.

    Args:
        environment: Environment name (development, production)
        level: Log level (DEBUG, INFO, WARNING, ERROR)
        format: Output format (json, human)
        **kwargs: Additional LogConfig fields

    Returns:
        The configured LogConfig
    """
    global _log_config

    # Start with environment-based config
    config = LogConfig.from_env()

    # Override with explicit parameters
    if environment:
        config.environment = environment
    if level:
        config.level = level
    if format:
        config.format = format
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)

    _log_config = config

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, config.level.upper()))

    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Add new handler with appropriate formatter
    handler = logging.StreamHandler(sys.stderr)
    if config.format == "json":
        handler.setFormatter(JSONFormatter(config))
    else:
        handler.setFormatter(HumanFormatter(config))

    root_logger.addHandler(handler)

    return config


def get_logger(name: str) -> StructuredLogger:
    """Get a structured logger for a module.

    Args:
        name: Logger name (typically __name__)

    Returns:
        StructuredLogger instance
    """
    global _log_config, _loggers

    if name in _loggers:
        return _loggers[name]

    if _log_config is None:
        _log_config = configure_logging()

    logger = logging.getLogger(name)
    structured = StructuredLogger(logger, _log_config)
    _loggers[name] = structured
    return structured
