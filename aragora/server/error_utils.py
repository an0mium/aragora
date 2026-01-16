"""
DEPRECATED: This module has been consolidated into aragora.server.errors.

All functionality from this module is now available in aragora.server.errors.
Please update your imports:

    # Old (deprecated)
    from aragora.server.error_utils import safe_error_message, ErrorCode

    # New (preferred)
    from aragora.server.errors import safe_error_message, ErrorCode

This module will be removed in a future version.
"""

import warnings

# Emit deprecation warning on import
warnings.warn(
    "aragora.server.error_utils is deprecated. " "Import from aragora.server.errors instead.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export everything from errors.py for backward compatibility
from aragora.server.errors import (
    # Error codes
    ErrorCode,
    # Error classes
    AragoraAPIError,
    safe_error_message,
    ERROR_SUGGESTIONS,
    get_error_suggestion,
    format_cli_error,
    ErrorFormatter,
    _STATUS_TO_CODE,
    # Utilities
    format_error_response,
    get_status_code,
    wrap_exception,
    log_error,
)

# Re-export sanitization utilities (originally imported from utils.error_sanitizer)
from aragora.utils.error_sanitizer import sanitize_error, sanitize_error_text

# Legacy aliases for backward compatibility
APIError = AragoraAPIError

# ErrorContext from errors.py (different from the class in old error_utils.py)
from aragora.server.errors import ErrorContext

__all__ = [
    # Error codes
    "ErrorCode",
    # Error class (legacy alias)
    "APIError",
    # Context
    "ErrorContext",
    # Functions
    "safe_error_message",
    "sanitize_error",
    "sanitize_error_text",
    "ERROR_SUGGESTIONS",
    "get_error_suggestion",
    "format_cli_error",
    "ErrorFormatter",
    "_STATUS_TO_CODE",
    "format_error_response",
    "get_status_code",
    "wrap_exception",
    "log_error",
    # Kept for interface compatibility (deprecated)
    "log_and_suppress",
    "with_error_context",
]


# Provide legacy context manager functions with deprecation
def log_and_suppress(operation: str, default_value=None, **context):
    """DEPRECATED: Use try/except with logging instead.

    This function is kept for backward compatibility but will be removed.
    """
    warnings.warn(
        "log_and_suppress is deprecated. Use try/except with logging instead.",
        DeprecationWarning,
        stacklevel=2,
    )

    class _SuppressingContext:
        def __init__(self):
            self.result = default_value

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            if exc_val is not None:
                import logging

                logger = logging.getLogger(__name__)
                logger.warning(f"Error during {operation}: {exc_val}")
                return True
            return False

    return _SuppressingContext()


def with_error_context(operation: str, **default_context):
    """DEPRECATED: Use try/except with logging instead.

    This function is kept for backward compatibility but will be removed.
    """
    warnings.warn(
        "with_error_context is deprecated. Use try/except with logging instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    from functools import wraps

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                import logging

                logger = logging.getLogger(__name__)
                logger.error(f"Error in {operation}: {e}", exc_info=True)
                raise

        return wrapper

    return decorator
