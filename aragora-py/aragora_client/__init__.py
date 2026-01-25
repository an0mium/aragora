"""
Aragora Python SDK.

A Python client for the Aragora control plane for multi-agent vetted decisionmaking.

Example:
    >>> from aragora_client import AragoraClient
    >>> client = AragoraClient("http://localhost:8080")
    >>> debate = await client.debates.run(task="Should we use microservices?")
    >>> print(debate.consensus.conclusion)
"""

from aragora_client.audit import (
    AuditAPI,
    AuditEvent,
    AuditExportResponse,
    AuditIntegrityResult,
    AuditRetentionPolicy,
    AuditStats,
)
from aragora_client.auth import (
    APIKey,
    AuthAPI,
    AuthToken,
    MFASetupResponse,
    MFAVerifyResponse,
    OAuthUrl,
    Session,
    User,
)
from aragora_client.client import AragoraClient, CodebaseAPI, GmailAPI
from aragora_client.control_plane import (
    AgentHealth,
    ControlPlaneAPI,
    ControlPlaneStatus,
    RegisteredAgent,
    ResourceUtilization,
    Task,
)
from aragora_client.exceptions import (
    AragoraAuthenticationError,
    AragoraConnectionError,
    AragoraError,
    AragoraNotFoundError,
    AragoraTimeoutError,
    AragoraValidationError,
)
from aragora_client.onboarding import (
    OnboardingAPI,
    OnboardingFlow,
    OnboardingInvitation,
    OnboardingStep,
    OnboardingTemplate,
)
from aragora_client.rbac import (
    RBACAPI,
    Permission,
    PermissionCheck,
    Role,
    RoleAssignment,
)
from aragora_client.tenancy import (
    QuotaStatus,
    TenancyAPI,
    Tenant,
    TenantMember,
)
from aragora_client.tournaments import (
    Tournament,
    TournamentBracket,
    TournamentMatch,
    TournamentsAPI,
    TournamentStanding,
    TournamentStandings,
)
from aragora_client.types import (
    AgentProfile,
    AgentScore,
    ConsensusResult,
    Debate,
    DebateEvent,
    DebateStatus,
    GauntletReceipt,
    GraphBranch,
    GraphDebate,
    HealthStatus,
    MatrixConclusion,
    MatrixDebate,
    MemoryAnalytics,
    SelectionPlugins,
    TeamSelection,
    VerificationResult,
    VerificationStatus,
)
from aragora_client.websocket import DebateStream, stream_debate

__version__ = "2.2.0"
__all__ = [
    # Client
    "AragoraClient",
    "CodebaseAPI",
    "GmailAPI",
    # Exceptions
    "AragoraError",
    "AragoraConnectionError",
    "AragoraAuthenticationError",
    "AragoraNotFoundError",
    "AragoraValidationError",
    "AragoraTimeoutError",
    # Types - Core
    "Debate",
    "DebateStatus",
    "ConsensusResult",
    "AgentProfile",
    "GraphDebate",
    "GraphBranch",
    "MatrixDebate",
    "MatrixConclusion",
    "VerificationResult",
    "VerificationStatus",
    "GauntletReceipt",
    "MemoryAnalytics",
    "HealthStatus",
    "DebateEvent",
    "SelectionPlugins",
    "TeamSelection",
    "AgentScore",
    # WebSocket
    "DebateStream",
    "stream_debate",
    # Control Plane
    "ControlPlaneAPI",
    "RegisteredAgent",
    "AgentHealth",
    "Task",
    "ControlPlaneStatus",
    "ResourceUtilization",
    # Auth
    "AuthAPI",
    "AuthToken",
    "User",
    "Session",
    "APIKey",
    "MFASetupResponse",
    "MFAVerifyResponse",
    "OAuthUrl",
    # Tenancy
    "TenancyAPI",
    "Tenant",
    "TenantMember",
    "QuotaStatus",
    # RBAC
    "RBACAPI",
    "Role",
    "Permission",
    "RoleAssignment",
    "PermissionCheck",
    # Tournaments
    "TournamentsAPI",
    "Tournament",
    "TournamentMatch",
    "TournamentStanding",
    "TournamentStandings",
    "TournamentBracket",
    # Audit
    "AuditAPI",
    "AuditEvent",
    "AuditStats",
    "AuditExportResponse",
    "AuditIntegrityResult",
    "AuditRetentionPolicy",
    # Onboarding
    "OnboardingAPI",
    "OnboardingFlow",
    "OnboardingStep",
    "OnboardingTemplate",
    "OnboardingInvitation",
]
