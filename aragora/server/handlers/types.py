"""
TypedDict definitions for handler request/response types.

These TypedDicts provide static type checking for API request bodies
and response payloads. They mirror the validation schemas in
aragora/server/validation/schema.py.

Usage:
    from aragora.server.handlers.types import CreateDebateRequest

    def handle_create(self, body: CreateDebateRequest) -> DebateResponse:
        task = body.get("task", "")
        agents = body.get("agents", [])
        ...
"""

from __future__ import annotations

from typing import Any, Dict, List, TypedDict

from typing_extensions import NotRequired

# =============================================================================
# Debate Request/Response Types
# =============================================================================


class CreateDebateRequest(TypedDict, total=False):
    """Request body for POST /api/debate or /api/debates."""

    task: str
    question: str  # Alternative to task
    agents: List[str]
    mode: str
    rounds: int
    consensus: str


class DebateUpdateRequest(TypedDict, total=False):
    """Request body for PATCH /api/debates/{id}."""

    title: str
    status: str  # "active", "paused", "concluded", "archived"
    tags: List[str]


class DebateSummaryResponse(TypedDict):
    """Response for debate summary."""

    id: str
    task: str
    status: str
    created_at: str
    agents: List[str]
    round_count: int


class DebateDetailResponse(TypedDict):
    """Detailed debate response with messages."""

    id: str
    task: str
    status: str
    created_at: str
    agents: List[str]
    round_count: int
    messages: List[Dict[str, Any]]
    consensus: NotRequired[Dict[str, Any]]


class DebateListResponse(TypedDict):
    """Response for GET /api/debates."""

    debates: List[DebateSummaryResponse]
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
    agents: List[str]
    rounds: int
    mode: str


class BatchSubmitRequest(TypedDict):
    """Request body for POST /api/debates/batch."""

    items: List[BatchDebateItem]
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
    results: NotRequired[List[Dict[str, Any]]]


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
    user: Dict[str, Any]
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
    agents: List[str]
    persona: str
    profile: str


class GauntletResponse(TypedDict):
    """Response for gauntlet run."""

    run_id: str
    status: str
    result: NotRequired[Dict[str, Any]]


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
    evidence: List[Dict[str, Any]]
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
    metadata: Dict[str, Any]
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
    probe_types: List[str]
    probes_per_type: int
    model_type: str


class ProbeResultResponse(TypedDict):
    """Response for probe run."""

    probe_id: str
    agent_name: str
    results: List[Dict[str, Any]]
    summary: Dict[str, Any]


# =============================================================================
# Social/Sharing Types
# =============================================================================


class SocialPublishRequest(TypedDict, total=False):
    """Request body for POST /api/debates/{id}/publish."""

    include_audio_link: str  # "true" or "false"
    thread_mode: str
    title: str
    description: str
    tags: List[str]


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
# Error Response Types
# =============================================================================


class ErrorDetail(TypedDict, total=False):
    """Error detail object in structured error responses."""

    code: str
    message: str
    details: Dict[str, Any]
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

    items: List[Any]
    total: int
    offset: int
    limit: int
    has_more: bool


class StatusResponse(TypedDict):
    """Simple status response."""

    status: str
    message: NotRequired[str]


__all__ = [
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
    # Error types
    "ErrorDetail",
    "ErrorResponse",
    # Common types
    "PaginatedResponse",
    "StatusResponse",
]
