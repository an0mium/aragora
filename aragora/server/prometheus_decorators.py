"""
Prometheus instrumentation decorators for Aragora server.

Extracted from prometheus.py for maintainability.
Provides timing decorators for HTTP handlers, agent generation, and DB queries.
"""

import logging
import time
from functools import wraps
from typing import Callable

logger = logging.getLogger(__name__)


def timed_http_request(endpoint: str) -> Callable[[Callable], Callable]:
    """Decorator to time HTTP request handlers.

    Args:
        endpoint: The HTTP endpoint being timed (e.g., "/api/debates")

    Returns:
        Decorator function that wraps handlers with timing instrumentation.
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            from aragora.server.prometheus_recording import record_http_request

            start = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                status = getattr(result, "status_code", 200) if result else 200
                return result
            except (ValueError, TypeError, KeyError, AttributeError, RuntimeError, OSError) as e:
                logger.warning("HTTP request to %s failed: %s", endpoint, e)
                status = 500
                raise
            finally:
                duration = time.perf_counter() - start
                record_http_request("GET", endpoint, status, duration)

        return wrapper

    return decorator


def timed_agent_generation(agent_type: str, model: str) -> Callable[[Callable], Callable]:
    """Decorator to time agent generation.

    Args:
        agent_type: Type of agent being timed (e.g., "anthropic-api")
        model: Model name being used (e.g., "claude-3-sonnet")

    Returns:
        Async decorator function that wraps generators with timing instrumentation.
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            from aragora.server.prometheus_recording import (
                record_agent_failure,
                record_agent_generation,
            )

            start = time.perf_counter()
            try:
                result = await func(*args, **kwargs)
                return result
            except (ValueError, TypeError, KeyError, RuntimeError, TimeoutError, OSError) as e:
                logger.warning("Agent %s generation failed: %s", agent_type, e)
                record_agent_failure(agent_type, type(e).__name__)
                raise
            finally:
                duration = time.perf_counter() - start
                record_agent_generation(agent_type, model, duration)

        return wrapper

    return decorator


def timed_db_query(operation: str, table: str) -> Callable[[Callable], Callable]:
    """Decorator to time database query execution.

    Args:
        operation: Query operation type (select, insert, update, delete)
        table: Table name being queried

    Returns:
        Decorator function that wraps queries with timing instrumentation.

    Usage:
        @timed_db_query("select", "debates")
        def list_debates(self, limit: int):
            ...
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            from aragora.server.prometheus_recording import record_db_error, record_db_query

            start = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                return result
            except (ValueError, TypeError, KeyError, RuntimeError, TimeoutError, OSError) as e:
                logger.warning("DB %s on %s failed: %s", operation, table, e)
                record_db_error(type(e).__name__, operation)
                raise
            finally:
                duration = time.perf_counter() - start
                record_db_query(operation, table, duration)

        return wrapper

    return decorator


def timed_db_query_async(operation: str, table: str) -> Callable[[Callable], Callable]:
    """Async decorator to time database query execution.

    Args:
        operation: Query operation type (select, insert, update, delete)
        table: Table name being queried

    Returns:
        Async decorator function that wraps queries with timing instrumentation.

    Usage:
        @timed_db_query_async("select", "debates")
        async def list_debates(self, limit: int):
            ...
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            from aragora.server.prometheus_recording import record_db_error, record_db_query

            start = time.perf_counter()
            try:
                result = await func(*args, **kwargs)
                return result
            except (ValueError, TypeError, KeyError, RuntimeError, TimeoutError, OSError) as e:
                logger.warning("Async DB %s on %s failed: %s", operation, table, e)
                record_db_error(type(e).__name__, operation)
                raise
            finally:
                duration = time.perf_counter() - start
                record_db_query(operation, table, duration)

        return wrapper

    return decorator
