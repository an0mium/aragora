"""
Shared type definitions for HTTP handlers.

This module provides:
1. TypedDict definitions for API request bodies and response payloads
2. Protocol classes for handler interfaces
3. Type aliases for handler functions and middleware
4. Common parameter types (pagination, filtering, sorting)

These types improve type safety across handler files and reduce duplication.

Usage:
    from aragora.server.handlers.types import (
        CreateDebateRequest,
        HandlerFunction,
        PaginationParams,
        FilterParams,
    )

    def handle_create(self, body: CreateDebateRequest) -> DebateResponse:
        task = body.get("task", "")
        agents = body.get("agents", [])
        ...
"""

from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
    Awaitable,
    Callable,
    Protocol,
    TypedDict,
    TypeVar,
    Union,
    runtime_checkable,
)

from typing_extensions import NotRequired

if TYPE_CHECKING:
    from aragora.server.handlers.utils.responses import HandlerResult


# =============================================================================
# Type Variables
# =============================================================================

T = TypeVar("T")
T_co = TypeVar("T_co", covariant=True)


# =============================================================================
# Handler Protocol
# =============================================================================


@runtime_checkable
class HandlerProtocol(Protocol):
    """Protocol defining the interface for HTTP request handlers.

    This protocol specifies the contract that all endpoint handlers must implement.
    It allows handlers to be type-checked without importing the full BaseHandler
    implementation, reducing circular dependencies.

    Example:
        def register_handler(handler: HandlerProtocol) -> None:
            result = handler.handle("/api/v1/test", {}, mock_handler)

    See Also:
        aragora.server.handlers.interface.HandlerInterface for the full interface.
    """

    def handle(
        self, path: str, query_params: dict[str, Any], handler: Any
    ) -> "HandlerResult | Awaitable[HandlerResult | None] | None":
        """Handle a GET request."""
        ...

    def handle_post(
        self, path: str, query_params: dict[str, Any], handler: Any
    ) -> "HandlerResult | Awaitable[HandlerResult | None] | None":
        """Handle a POST request."""
        ...

    def handle_delete(
        self, path: str, query_params: dict[str, Any], handler: Any
    ) -> "HandlerResult | Awaitable[HandlerResult | None] | None":
        """Handle a DELETE request."""
        ...

    def handle_patch(
        self, path: str, query_params: dict[str, Any], handler: Any
    ) -> "HandlerResult | Awaitable[HandlerResult | None] | None":
        """Handle a PATCH request."""
        ...

    def handle_put(
        self, path: str, query_params: dict[str, Any], handler: Any
    ) -> "HandlerResult | Awaitable[HandlerResult | None] | None":
        """Handle a PUT request."""
        ...


# =============================================================================
# Request Context Types
# =============================================================================


class RequestContext(TypedDict, total=False):
    """Type for request context objects passed through middleware and handlers.

    This captures the common fields that may be attached to a request as it
    flows through the handler pipeline. Not all fields are always present.

    Attributes:
        user_id: Authenticated user ID (if authenticated)
        org_id: Organization/tenant ID for multi-tenancy
        workspace_id: Workspace ID within the organization
        trace_id: Distributed tracing ID for request correlation
        request_id: Unique request identifier
        client_ip: Client IP address
        user_agent: User-Agent header value
        auth_method: How the request was authenticated (e.g., "bearer", "api_key")
        permissions: List of permissions the user has
        roles: List of roles assigned to the user
        metadata: Additional request-specific metadata
    """

    user_id: str
    org_id: str
    workspace_id: str
    trace_id: str
    request_id: str
    client_ip: str
    user_agent: str
    auth_method: str
    permissions: list[str]
    roles: list[str]
    metadata: dict[str, Any]


# =============================================================================
# Response Type Aliases
# =============================================================================

# Union type for all valid handler response types
# Handlers may return:
# - HandlerResult: A complete response object
# - None: Indicates the handler did not handle the request
# - Awaitable[HandlerResult | None]: For async handlers
ResponseType = Union["HandlerResult", None, Awaitable[Union["HandlerResult", None]]]


# =============================================================================
# Handler Function Type Aliases
# =============================================================================

# Type for synchronous handler functions
# Takes path, query params, and HTTP handler; returns response or None
HandlerFunction = Callable[
    [str, dict[str, Any], Any],
    Union["HandlerResult", None],
]

# Type for async handler functions
AsyncHandlerFunction = Callable[
    [str, dict[str, Any], Any],
    Awaitable[Union["HandlerResult", None]],
]

# Type for handler functions that may be sync or async
MaybeAsyncHandlerFunction = Callable[
    [str, dict[str, Any], Any],
    Union["HandlerResult", None, Awaitable[Union["HandlerResult", None]]],
]


# =============================================================================
# Middleware Function Type Aliases
# =============================================================================

# Type for middleware that wraps a handler function
# Takes a handler function and returns a wrapped handler function
MiddlewareFunction = Callable[[HandlerFunction], HandlerFunction]

# Type for async middleware
AsyncMiddlewareFunction = Callable[[AsyncHandlerFunction], AsyncHandlerFunction]

# Type for middleware that can wrap both sync and async handlers
MaybeAsyncMiddlewareFunction = Callable[[MaybeAsyncHandlerFunction], MaybeAsyncHandlerFunction]

# Type for decorator factories (e.g., @rate_limit(requests_per_minute=30))
MiddlewareFactory = Callable[..., MiddlewareFunction]


# =============================================================================
# Common Parameter Types
# =============================================================================


class PaginationParams(TypedDict, total=False):
    """Standard pagination parameters used across list endpoints.

    These parameters control how results are paginated. All fields are optional
    with sensible defaults applied by handlers.

    Attributes:
        limit: Maximum number of items to return (default: 20, max: 100)
        offset: Number of items to skip (default: 0)
        page: Page number (1-indexed, alternative to offset)
        cursor: Opaque cursor for cursor-based pagination
    """

    limit: int
    offset: int
    page: int
    cursor: str


class FilterParams(TypedDict, total=False):
    """Common filtering parameters used across list endpoints.

    These parameters allow filtering results by various criteria.
    All fields are optional and interpreted by each handler.

    Attributes:
        status: Filter by status (e.g., "active", "completed", "archived")
        agent: Filter by agent name
        user_id: Filter by user ID
        org_id: Filter by organization ID
        created_after: Filter items created after this ISO timestamp
        created_before: Filter items created before this ISO timestamp
        updated_after: Filter items updated after this ISO timestamp
        updated_before: Filter items updated before this ISO timestamp
        tags: Filter by tags (comma-separated or list)
        search: Full-text search query
        ids: Filter by specific IDs (comma-separated or list)
    """

    status: str
    agent: str
    user_id: str
    org_id: str
    created_after: str
    created_before: str
    updated_after: str
    updated_before: str
    tags: str | list[str]
    search: str
    ids: str | list[str]


class SortParams(TypedDict, total=False):
    """Sorting parameters for list endpoints.

    Attributes:
        sort_by: Field name to sort by
        sort_order: Sort direction ("asc" or "desc")
        order_by: Alternative field name format
    """

    sort_by: str
    sort_order: str  # "asc" | "desc"
    order_by: str


class QueryParams(TypedDict, total=False):
    """Combined query parameters including pagination, filtering, and sorting.

    This is a convenience type that combines all common query parameter types
    for handlers that support pagination, filtering, and sorting together.
    """

    # Pagination
    limit: int
    offset: int
    page: int
    cursor: str
    # Filtering
    status: str
    agent: str
    user_id: str
    org_id: str
    created_after: str
    created_before: str
    updated_after: str
    updated_before: str
    tags: str | list[str]
    search: str
    ids: str | list[str]
    # Sorting
    sort_by: str
    sort_order: str
    order_by: str


# =============================================================================
# Debate Request/Response Types
# =============================================================================


class CreateDebateRequest(TypedDict, total=False):
    """Request body for POST /api/debate or /api/debates."""

    task: str
    question: str  # Alternative to task
    agents: list[str]
    mode: str
    rounds: int
    consensus: str


class DebateUpdateRequest(TypedDict, total=False):
    """Request body for PATCH /api/debates/{id}."""

    title: str
    status: str  # "active", "paused", "concluded", "archived"
    tags: list[str]


class DebateSummaryResponse(TypedDict):
    """Response for debate summary."""

    id: str
    task: str
    status: str
    created_at: str
    agents: list[str]
    round_count: int


class DebateDetailResponse(TypedDict):
    """Detailed debate response with messages."""

    id: str
    task: str
    status: str
    created_at: str
    agents: list[str]
    round_count: int
    messages: list[dict[str, Any]]
    consensus: NotRequired[dict[str, Any]]


class DebateListResponse(TypedDict):
    """Response for GET /api/debates."""

    debates: list[DebateSummaryResponse]
    total: int
    offset: int
    limit: int


# =============================================================================
# Fork Request Types
# =============================================================================


class ForkRequest(TypedDict):
    """Request body for POST /api/debates/{id}/fork."""

    branch_point: int
    modified_context: NotRequired[str]


class ForkResponse(TypedDict):
    """Response for debate fork."""

    fork_id: str
    parent_id: str
    branch_point: int
    created_at: str


# =============================================================================
# Batch Operation Types
# =============================================================================


class BatchDebateItem(TypedDict, total=False):
    """Single item in batch debate submission."""

    task: str
    question: str
    agents: list[str]
    rounds: int
    mode: str


class BatchSubmitRequest(TypedDict):
    """Request body for POST /api/debates/batch."""

    items: list[BatchDebateItem]
    webhook_url: NotRequired[str]
    max_parallel: NotRequired[int]


class BatchSubmitResponse(TypedDict):
    """Response for batch debate submission."""

    batch_id: str
    total_items: int
    status: str


class BatchStatusResponse(TypedDict):
    """Response for GET /api/debates/batch/{id}/status."""

    batch_id: str
    status: str
    total: int
    completed: int
    failed: int
    results: NotRequired[list[dict[str, Any]]]


# =============================================================================
# Authentication Types
# =============================================================================


class UserRegisterRequest(TypedDict):
    """Request body for POST /api/auth/register."""

    email: str
    password: str
    name: NotRequired[str]


class UserLoginRequest(TypedDict):
    """Request body for POST /api/auth/login."""

    email: str
    password: str


class AuthResponse(TypedDict):
    """Response for authentication endpoints."""

    token: str
    user: dict[str, Any]
    expires_at: str


class UserResponse(TypedDict):
    """User data in responses."""

    id: str
    email: str
    name: NotRequired[str]
    role: str
    org_id: NotRequired[str]
    created_at: str


# =============================================================================
# Organization Types
# =============================================================================


class OrgCreateRequest(TypedDict):
    """Request body for POST /api/organizations."""

    name: str
    slug: NotRequired[str]


class OrgInviteRequest(TypedDict):
    """Request body for POST /api/organizations/{id}/invite."""

    email: str
    role: NotRequired[str]  # "member" or "admin"


class OrgResponse(TypedDict):
    """Organization data in responses."""

    id: str
    name: str
    slug: str
    owner_id: str
    created_at: str
    member_count: NotRequired[int]


# =============================================================================
# Gauntlet Types
# =============================================================================


class GauntletRunRequest(TypedDict, total=False):
    """Request body for POST /api/gauntlet/run."""

    input_content: str  # Required
    input_type: str  # "spec", "code", "text", "url", "file"
    agents: list[str]
    persona: str
    profile: str


class GauntletResponse(TypedDict):
    """Response for gauntlet run."""

    run_id: str
    status: str
    result: NotRequired[dict[str, Any]]


# =============================================================================
# Verification Types
# =============================================================================


class VerificationRequest(TypedDict):
    """Request body for POST /api/verify."""

    claim: str
    context: NotRequired[str]


class VerificationResponse(TypedDict):
    """Response for verification endpoint."""

    verified: bool
    confidence: float
    evidence: list[dict[str, Any]]
    reasoning: str


# =============================================================================
# Memory Types
# =============================================================================


class MemoryCleanupRequest(TypedDict, total=False):
    """Request body for POST /api/memory/cleanup."""

    tier: str  # "fast", "medium", "slow", "glacial"
    archive: str  # "true" or "false"
    max_age_hours: float


class MemoryEntry(TypedDict):
    """Memory entry in responses."""

    id: str
    tier: str
    content: str
    metadata: dict[str, Any]
    created_at: str
    expires_at: NotRequired[str]


# =============================================================================
# Agent Types
# =============================================================================


class AgentConfigRequest(TypedDict, total=False):
    """Request body for agent configuration."""

    name: str  # Required
    model: str
    temperature: float
    max_tokens: int
    system_prompt: str


class AgentStatusResponse(TypedDict):
    """Response for agent status."""

    name: str
    status: str  # "available", "busy", "unavailable"
    model: str
    elo_rating: NotRequired[int]
    total_debates: NotRequired[int]


# =============================================================================
# Probe Types
# =============================================================================


class ProbeRunRequest(TypedDict, total=False):
    """Request body for POST /api/probes/run."""

    agent_name: str  # Required
    probe_types: list[str]
    probes_per_type: int
    model_type: str


class ProbeResultResponse(TypedDict):
    """Response for probe run."""

    probe_id: str
    agent_name: str
    results: list[dict[str, Any]]
    summary: dict[str, Any]


# =============================================================================
# Social/Sharing Types
# =============================================================================


class SocialPublishRequest(TypedDict, total=False):
    """Request body for POST /api/debates/{id}/publish."""

    include_audio_link: str  # "true" or "false"
    thread_mode: str
    title: str
    description: str
    tags: list[str]


class ShareUpdateRequest(TypedDict, total=False):
    """Request body for PATCH /api/share/{id}."""

    visibility: str  # "private", "team", "public"
    expires_in_hours: int
    allow_comments: str
    allow_forking: str


class ShareResponse(TypedDict):
    """Response for share endpoints."""

    share_id: str
    debate_id: str
    visibility: str
    url: str
    expires_at: NotRequired[str]


# =============================================================================
# Billing Types
# =============================================================================


class CheckoutSessionRequest(TypedDict):
    """Request body for POST /api/billing/checkout."""

    tier: str  # "starter", "professional", "enterprise"
    success_url: str
    cancel_url: str


class CheckoutSessionResponse(TypedDict):
    """Response for checkout session creation."""

    session_id: str
    checkout_url: str


class UsageResponse(TypedDict):
    """Response for GET /api/billing/usage."""

    debates_used: int
    debates_limit: int
    tokens_used: int
    tokens_limit: int
    period_start: str
    period_end: str


# =============================================================================
# Plugin Types
# =============================================================================


class PluginRunRequest(TypedDict, total=False):
    """Request body for POST /api/plugins/{name}/run."""

    input: str
    config: str
    working_dir: str


class PluginInstallRequest(TypedDict, total=False):
    """Request body for POST /api/plugins/{name}/install."""

    config: str
    enabled: str  # "true" or "false"


class PluginResponse(TypedDict):
    """Response for plugin operations."""

    name: str
    version: str
    enabled: bool
    status: str


# =============================================================================
# Internal Metrics Types
# =============================================================================


class ConvergenceMetrics(TypedDict):
    """Metrics from convergence detection during debate."""

    status: str  # "converged", "refining", "diverging"
    similarity: float
    per_agent: dict[str, float]
    rounds_to_converge: NotRequired[int]


class PhaseMetrics(TypedDict):
    """Metrics from phase execution."""

    phase_name: str
    duration_ms: float
    status: str  # "completed", "failed", "skipped"
    error: NotRequired[str]


class DebateMetrics(TypedDict):
    """Comprehensive debate metrics."""

    debate_id: str
    total_duration_ms: float
    rounds_used: int
    messages_count: int
    consensus_reached: bool
    confidence: float
    convergence: NotRequired[ConvergenceMetrics]
    phases: NotRequired[list["PhaseMetrics"]]


class AgentPerformanceMetrics(TypedDict):
    """Per-agent performance metrics."""

    agent_name: str
    elo_rating: float
    win_rate: float
    debates_participated: int
    avg_confidence: float
    calibration_score: NotRequired[float]


class DashboardSummary(TypedDict):
    """Summary data for dashboard."""

    total_debates: int
    active_debates: int
    total_agents: int
    avg_confidence: float
    consensus_rate: float


class DashboardResponse(TypedDict):
    """Full dashboard response."""

    summary: DashboardSummary
    recent_activity: list[dict[str, Any]]
    agent_performance: list[AgentPerformanceMetrics]
    top_domains: list[dict[str, Any]]
    generated_at: float


# =============================================================================
# Error Response Types
# =============================================================================


class ErrorDetail(TypedDict, total=False):
    """Error detail object in structured error responses."""

    code: str
    message: str
    details: dict[str, Any]
    trace_id: str
    suggestion: str


class ErrorResponse(TypedDict, total=False):
    """Structured error response body."""

    error: str | ErrorDetail


# =============================================================================
# Common Response Wrappers
# =============================================================================


class PaginatedResponse(TypedDict):
    """Base type for paginated responses."""

    items: list[Any]
    total: int
    offset: int
    limit: int
    has_more: bool


class StatusResponse(TypedDict):
    """Simple status response."""

    status: str
    message: NotRequired[str]


__all__ = [
    # Handler protocol
    "HandlerProtocol",
    # Request context
    "RequestContext",
    # Response type aliases
    "ResponseType",
    # Handler function type aliases
    "HandlerFunction",
    "AsyncHandlerFunction",
    "MaybeAsyncHandlerFunction",
    # Middleware function type aliases
    "MiddlewareFunction",
    "AsyncMiddlewareFunction",
    "MaybeAsyncMiddlewareFunction",
    "MiddlewareFactory",
    # Common parameter types
    "PaginationParams",
    "FilterParams",
    "SortParams",
    "QueryParams",
    # Debate types
    "CreateDebateRequest",
    "DebateUpdateRequest",
    "DebateSummaryResponse",
    "DebateDetailResponse",
    "DebateListResponse",
    # Fork types
    "ForkRequest",
    "ForkResponse",
    # Batch types
    "BatchDebateItem",
    "BatchSubmitRequest",
    "BatchSubmitResponse",
    "BatchStatusResponse",
    # Auth types
    "UserRegisterRequest",
    "UserLoginRequest",
    "AuthResponse",
    "UserResponse",
    # Org types
    "OrgCreateRequest",
    "OrgInviteRequest",
    "OrgResponse",
    # Gauntlet types
    "GauntletRunRequest",
    "GauntletResponse",
    # Verification types
    "VerificationRequest",
    "VerificationResponse",
    # Memory types
    "MemoryCleanupRequest",
    "MemoryEntry",
    # Agent types
    "AgentConfigRequest",
    "AgentStatusResponse",
    # Probe types
    "ProbeRunRequest",
    "ProbeResultResponse",
    # Social types
    "SocialPublishRequest",
    "ShareUpdateRequest",
    "ShareResponse",
    # Billing types
    "CheckoutSessionRequest",
    "CheckoutSessionResponse",
    "UsageResponse",
    # Plugin types
    "PluginRunRequest",
    "PluginInstallRequest",
    "PluginResponse",
    # Metrics types
    "ConvergenceMetrics",
    "PhaseMetrics",
    "DebateMetrics",
    "AgentPerformanceMetrics",
    "DashboardSummary",
    "DashboardResponse",
    # Error types
    "ErrorDetail",
    "ErrorResponse",
    # Common types
    "PaginatedResponse",
    "StatusResponse",
]
