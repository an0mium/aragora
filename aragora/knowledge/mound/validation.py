"""
Input validation and production hardening for Knowledge Mound.

Provides:
- Input validation with configurable limits
- Standardized error responses
- Resource limit enforcement
- Thread-safe operation utilities
"""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TypeVar

logger = logging.getLogger(__name__)

# Type variable for generic validation
T = TypeVar("T")


# =============================================================================
# Validation Limits Configuration
# =============================================================================


@dataclass
class ValidationLimits:
    """Configurable limits for Knowledge Mound operations."""

    # Content limits
    max_content_size: int = 100_000  # 100KB max content
    max_topics: int = 100  # Maximum topics per node
    max_metadata_size: int = 10_000  # 10KB max metadata
    max_query_length: int = 10_000  # 10KB max query string

    # ID/name limits
    max_id_length: int = 128
    max_workspace_id_length: int = 64
    workspace_id_pattern: str = r"^[a-zA-Z0-9_-]+$"

    # Graph limits
    max_graph_depth: int = 5
    max_graph_nodes: int = 1000
    max_relationships_per_node: int = 100

    # Batch limits
    max_batch_size: int = 100
    max_concurrent_operations: int = 10

    # Query limits
    max_query_results: int = 1000
    default_query_limit: int = 20

    # Event/log limits
    max_event_log_size: int = 1000


# Global default limits
DEFAULT_LIMITS = ValidationLimits()


# =============================================================================
# Validation Errors
# =============================================================================


class ValidationError(Exception):
    """Base class for validation errors."""

    def __init__(self, message: str, field: Optional[str] = None, code: str = "VALIDATION_ERROR"):
        self.message = message
        self.field = field
        self.code = code
        super().__init__(message)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to error response dict."""
        return {
            "error": self.code,
            "message": self.message,
            "field": self.field,
        }


class ContentTooLargeError(ValidationError):
    """Content exceeds maximum allowed size."""

    def __init__(self, size: int, max_size: int, field: str = "content"):
        super().__init__(
            f"Content size {size} bytes exceeds maximum {max_size} bytes",
            field=field,
            code="CONTENT_TOO_LARGE",
        )


class InvalidIdError(ValidationError):
    """Invalid ID format."""

    def __init__(self, value: str, field: str = "id"):
        super().__init__(
            f"Invalid ID format: {value[:50]}{'...' if len(value) > 50 else ''}",
            field=field,
            code="INVALID_ID",
        )


class ResourceLimitExceededError(ValidationError):
    """Resource limit exceeded."""

    def __init__(self, resource: str, limit: int, actual: int):
        super().__init__(
            f"{resource} limit exceeded: {actual} > {limit}",
            field=resource,
            code="RESOURCE_LIMIT_EXCEEDED",
        )


class NotFoundError(ValidationError):
    """Resource not found."""

    def __init__(self, resource_type: str, resource_id: str):
        super().__init__(
            f"{resource_type} not found: {resource_id}",
            field="id",
            code="NOT_FOUND",
        )


class AccessDeniedError(ValidationError):
    """Access denied to resource."""

    def __init__(self, resource_id: str, reason: str = "insufficient permissions"):
        super().__init__(
            f"Access denied to {resource_id}: {reason}",
            field="id",
            code="ACCESS_DENIED",
        )


# =============================================================================
# Input Validators
# =============================================================================


def validate_content(
    content: str,
    limits: ValidationLimits = DEFAULT_LIMITS,
) -> str:
    """Validate content string.

    Args:
        content: Content to validate
        limits: Validation limits configuration

    Returns:
        Validated content string

    Raises:
        ValidationError: If content is invalid or too large
    """
    if not content:
        raise ValidationError("Content cannot be empty", field="content", code="EMPTY_CONTENT")

    content_size = len(content.encode("utf-8"))
    if content_size > limits.max_content_size:
        raise ContentTooLargeError(content_size, limits.max_content_size)

    return content


def validate_id(
    value: str,
    field_name: str = "id",
    limits: ValidationLimits = DEFAULT_LIMITS,
) -> str:
    """Validate an ID string.

    Args:
        value: ID value to validate
        field_name: Field name for error messages
        limits: Validation limits configuration

    Returns:
        Validated ID string

    Raises:
        ValidationError: If ID is invalid
    """
    if not value:
        raise ValidationError(f"{field_name} cannot be empty", field=field_name, code="EMPTY_ID")

    if len(value) > limits.max_id_length:
        raise InvalidIdError(value, field=field_name)

    # Basic format validation (alphanumeric, underscores, hyphens)
    if not re.match(r"^[a-zA-Z0-9_-]+$", value):
        raise InvalidIdError(value, field=field_name)

    return value


def validate_workspace_id(
    workspace_id: str,
    limits: ValidationLimits = DEFAULT_LIMITS,
) -> str:
    """Validate workspace ID.

    Args:
        workspace_id: Workspace ID to validate
        limits: Validation limits configuration

    Returns:
        Validated workspace ID

    Raises:
        ValidationError: If workspace ID is invalid
    """
    if not workspace_id:
        raise ValidationError(
            "workspace_id cannot be empty",
            field="workspace_id",
            code="EMPTY_WORKSPACE_ID",
        )

    if len(workspace_id) > limits.max_workspace_id_length:
        raise InvalidIdError(workspace_id, field="workspace_id")

    if not re.match(limits.workspace_id_pattern, workspace_id):
        raise ValidationError(
            f"Invalid workspace_id format: must match {limits.workspace_id_pattern}",
            field="workspace_id",
            code="INVALID_WORKSPACE_ID",
        )

    return workspace_id


def validate_topics(
    topics: Optional[List[str]],
    limits: ValidationLimits = DEFAULT_LIMITS,
) -> List[str]:
    """Validate topics list.

    Args:
        topics: List of topics to validate
        limits: Validation limits configuration

    Returns:
        Validated topics list

    Raises:
        ValidationError: If topics list is invalid
    """
    if not topics:
        return []

    if len(topics) > limits.max_topics:
        raise ResourceLimitExceededError("topics", limits.max_topics, len(topics))

    # Validate each topic
    validated = []
    for topic in topics:
        if topic and isinstance(topic, str):
            # Truncate very long topics
            validated.append(topic[:256])

    return validated


def validate_metadata(
    metadata: Optional[Dict[str, Any]],
    limits: ValidationLimits = DEFAULT_LIMITS,
) -> Dict[str, Any]:
    """Validate metadata dict.

    Args:
        metadata: Metadata dict to validate
        limits: Validation limits configuration

    Returns:
        Validated metadata dict

    Raises:
        ValidationError: If metadata is too large
    """
    if not metadata:
        return {}

    # Check serialized size
    import json

    try:
        serialized = json.dumps(metadata)
        if len(serialized) > limits.max_metadata_size:
            raise ResourceLimitExceededError(
                "metadata",
                limits.max_metadata_size,
                len(serialized),
            )
    except (TypeError, ValueError) as e:
        raise ValidationError(
            f"Metadata is not JSON serializable: {e}",
            field="metadata",
            code="INVALID_METADATA",
        )

    return metadata


def validate_query(
    query: str,
    limits: ValidationLimits = DEFAULT_LIMITS,
) -> str:
    """Validate query string.

    Args:
        query: Query string to validate
        limits: Validation limits configuration

    Returns:
        Validated query string

    Raises:
        ValidationError: If query is invalid
    """
    if not query:
        raise ValidationError("Query cannot be empty", field="query", code="EMPTY_QUERY")

    if len(query) > limits.max_query_length:
        raise ResourceLimitExceededError("query", limits.max_query_length, len(query))

    return query


def validate_graph_params(
    depth: int,
    max_nodes: int,
    limits: ValidationLimits = DEFAULT_LIMITS,
) -> tuple[int, int]:
    """Validate graph traversal parameters.

    Args:
        depth: Graph traversal depth
        max_nodes: Maximum nodes to return
        limits: Validation limits configuration

    Returns:
        Tuple of (validated_depth, validated_max_nodes)

    Raises:
        ValidationError: If parameters exceed limits
    """
    if depth < 0:
        raise ValidationError("Depth cannot be negative", field="depth", code="INVALID_DEPTH")

    if depth > limits.max_graph_depth:
        raise ResourceLimitExceededError("depth", limits.max_graph_depth, depth)

    if max_nodes < 0:
        raise ValidationError(
            "max_nodes cannot be negative",
            field="max_nodes",
            code="INVALID_MAX_NODES",
        )

    if max_nodes > limits.max_graph_nodes:
        raise ResourceLimitExceededError("max_nodes", limits.max_graph_nodes, max_nodes)

    return depth, max_nodes


def validate_pagination(
    limit: Optional[int],
    offset: Optional[int],
    limits: ValidationLimits = DEFAULT_LIMITS,
) -> tuple[int, int]:
    """Validate pagination parameters.

    Args:
        limit: Query limit
        offset: Query offset
        limits: Validation limits configuration

    Returns:
        Tuple of (validated_limit, validated_offset)
    """
    # Default limit if not specified
    validated_limit = limit if limit is not None else limits.default_query_limit

    # Enforce maximum
    if validated_limit > limits.max_query_results:
        validated_limit = limits.max_query_results

    # Ensure non-negative
    if validated_limit < 0:
        validated_limit = limits.default_query_limit

    # Validate offset
    validated_offset = max(0, offset if offset is not None else 0)

    return validated_limit, validated_offset


# =============================================================================
# Thread-Safe Utilities
# =============================================================================


@dataclass
class BoundedList:
    """Thread-safe bounded list that implements FIFO eviction."""

    max_size: int
    _items: List[Any] = field(default_factory=list)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    async def append(self, item: Any) -> None:
        """Append item, evicting oldest if at capacity."""
        async with self._lock:
            self._items.append(item)
            if len(self._items) > self.max_size:
                self._items.pop(0)

    async def get_all(self) -> List[Any]:
        """Get copy of all items."""
        async with self._lock:
            return list(self._items)

    async def clear(self) -> None:
        """Clear all items."""
        async with self._lock:
            self._items.clear()

    def __len__(self) -> int:
        """Get current size (not thread-safe, for monitoring only)."""
        return len(self._items)


@dataclass
class ConcurrencyLimiter:
    """Limit concurrent operations."""

    max_concurrent: int
    _semaphore: Optional[asyncio.Semaphore] = None

    def __post_init__(self):
        self._semaphore = asyncio.Semaphore(self.max_concurrent)

    async def __aenter__(self):
        await self._semaphore.acquire()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self._semaphore.release()


# =============================================================================
# Validation Decorators
# =============================================================================


def validate_input(validation_func):
    """Decorator to validate input parameters.

    Example:
        @validate_input
        async def store(self, request):
            # request is already validated
            ...
    """

    def decorator(func):
        async def wrapper(*args, **kwargs):
            # Run validation
            validation_func(*args, **kwargs)
            # Call original function
            return await func(*args, **kwargs)

        return wrapper

    return decorator


# =============================================================================
# HTTP Status Code Mapping
# =============================================================================

ERROR_STATUS_CODES: Dict[str, int] = {
    "VALIDATION_ERROR": 400,
    "CONTENT_TOO_LARGE": 413,
    "INVALID_ID": 400,
    "EMPTY_ID": 400,
    "EMPTY_CONTENT": 400,
    "EMPTY_QUERY": 400,
    "EMPTY_WORKSPACE_ID": 400,
    "INVALID_WORKSPACE_ID": 400,
    "INVALID_METADATA": 400,
    "INVALID_DEPTH": 400,
    "INVALID_MAX_NODES": 400,
    "RESOURCE_LIMIT_EXCEEDED": 413,
    "NOT_FOUND": 404,
    "ACCESS_DENIED": 403,
}


def get_http_status(error: ValidationError) -> int:
    """Get HTTP status code for a validation error."""
    return ERROR_STATUS_CODES.get(error.code, 400)
