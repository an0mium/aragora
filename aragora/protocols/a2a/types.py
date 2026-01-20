"""
A2A Protocol Types.

Based on the A2A protocol specification for agent-to-agent communication.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


class TaskStatus(str, Enum):
    """Status of an A2A task."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    WAITING_INPUT = "waiting_input"  # Agent needs more info


class TaskPriority(str, Enum):
    """Priority level for A2A tasks."""

    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"


class AgentCapability(str, Enum):
    """Standard capabilities an agent can advertise."""

    DEBATE = "debate"
    CONSENSUS = "consensus"
    CRITIQUE = "critique"
    SYNTHESIS = "synthesis"
    AUDIT = "audit"
    VERIFICATION = "verification"
    CODE_REVIEW = "code_review"
    DOCUMENT_ANALYSIS = "document_analysis"
    RESEARCH = "research"
    REASONING = "reasoning"


@dataclass
class SecurityCard:
    """
    Security credentials for A2A agent authentication.

    Used to verify agent identity and permissions.
    """

    issuer: str
    subject: str
    public_key: Optional[str] = None
    signature: Optional[str] = None
    issued_at: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None
    permissions: List[str] = field(default_factory=list)

    def is_valid(self) -> bool:
        """Check if the security card is valid."""
        if self.expires_at and datetime.now() > self.expires_at:
            return False
        return True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "issuer": self.issuer,
            "subject": self.subject,
            "public_key": self.public_key,
            "issued_at": self.issued_at.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "permissions": self.permissions,
        }


@dataclass
class AgentCard:
    """
    Agent capability advertisement.

    Describes what an agent can do and how to invoke it.
    Based on A2A agent card specification.
    """

    # Identity
    name: str
    description: str
    version: str = "1.0.0"

    # Capabilities
    capabilities: List[AgentCapability] = field(default_factory=list)
    input_modes: List[str] = field(default_factory=lambda: ["text"])
    output_modes: List[str] = field(default_factory=lambda: ["text"])

    # Endpoint
    endpoint: Optional[str] = None
    protocol: str = "a2a"

    # Metadata
    tags: List[str] = field(default_factory=list)
    organization: Optional[str] = None
    documentation_url: Optional[str] = None

    # Security
    security_card: Optional[SecurityCard] = None
    requires_auth: bool = False

    # Rate limiting
    max_concurrent_tasks: int = 10
    estimated_response_time_ms: int = 5000

    def supports_capability(self, capability: AgentCapability) -> bool:
        """Check if agent supports a capability."""
        return capability in self.capabilities

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "capabilities": [c.value for c in self.capabilities],
            "input_modes": self.input_modes,
            "output_modes": self.output_modes,
            "endpoint": self.endpoint,
            "protocol": self.protocol,
            "tags": self.tags,
            "organization": self.organization,
            "documentation_url": self.documentation_url,
            "requires_auth": self.requires_auth,
            "max_concurrent_tasks": self.max_concurrent_tasks,
            "estimated_response_time_ms": self.estimated_response_time_ms,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentCard":
        """Create from dictionary."""
        return cls(
            name=data["name"],
            description=data["description"],
            version=data.get("version", "1.0.0"),
            capabilities=[AgentCapability(c) for c in data.get("capabilities", [])],
            input_modes=data.get("input_modes", ["text"]),
            output_modes=data.get("output_modes", ["text"]),
            endpoint=data.get("endpoint"),
            protocol=data.get("protocol", "a2a"),
            tags=data.get("tags", []),
            organization=data.get("organization"),
            documentation_url=data.get("documentation_url"),
            requires_auth=data.get("requires_auth", False),
            max_concurrent_tasks=data.get("max_concurrent_tasks", 10),
            estimated_response_time_ms=data.get("estimated_response_time_ms", 5000),
        )


@dataclass
class ContextItem:
    """
    A context item to pass to an agent.

    Can be text, a file reference, or structured data.
    """

    type: str  # "text", "file", "structured"
    content: str
    mime_type: str = "text/plain"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "type": self.type,
            "content": self.content,
            "mime_type": self.mime_type,
            "metadata": self.metadata,
        }


@dataclass
class TaskRequest:
    """
    Request for an agent to perform a task.

    Based on A2A task lifecycle specification.
    """

    # Task identity
    task_id: str

    # Task description
    instruction: str

    # Optional fields (must come after required fields)
    parent_task_id: Optional[str] = None  # For subtasks
    context: List[ContextItem] = field(default_factory=list)

    # Execution parameters
    capability: Optional[AgentCapability] = None
    priority: TaskPriority = TaskPriority.NORMAL
    timeout_ms: int = 300000  # 5 minutes default

    # Requester info
    requester_agent: Optional[str] = None
    requester_card: Optional[AgentCard] = None

    # Options
    stream_output: bool = False
    return_intermediate: bool = False

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "task_id": self.task_id,
            "parent_task_id": self.parent_task_id,
            "instruction": self.instruction,
            "context": [c.to_dict() for c in self.context],
            "capability": self.capability.value if self.capability else None,
            "priority": self.priority.value,
            "timeout_ms": self.timeout_ms,
            "requester_agent": self.requester_agent,
            "stream_output": self.stream_output,
            "return_intermediate": self.return_intermediate,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TaskRequest":
        """Create from dictionary."""
        return cls(
            task_id=data["task_id"],
            parent_task_id=data.get("parent_task_id"),
            instruction=data["instruction"],
            context=[
                ContextItem(
                    type=c["type"],
                    content=c["content"],
                    mime_type=c.get("mime_type", "text/plain"),
                    metadata=c.get("metadata", {}),
                )
                for c in data.get("context", [])
            ],
            capability=AgentCapability(data["capability"]) if data.get("capability") else None,
            priority=TaskPriority(data.get("priority", "normal")),
            timeout_ms=data.get("timeout_ms", 300000),
            requester_agent=data.get("requester_agent"),
            stream_output=data.get("stream_output", False),
            return_intermediate=data.get("return_intermediate", False),
            metadata=data.get("metadata", {}),
        )


@dataclass
class TaskResult:
    """
    Result of an A2A task execution.

    Based on A2A task result specification.
    """

    # Identity
    task_id: str
    agent_name: str

    # Status
    status: TaskStatus
    error_message: Optional[str] = None

    # Result
    output: Optional[str] = None
    output_type: str = "text"
    structured_output: Optional[Dict[str, Any]] = None

    # Metrics
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    tokens_used: int = 0

    # Provenance
    intermediate_results: List[Dict[str, Any]] = field(default_factory=list)
    subtasks: List[str] = field(default_factory=list)

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def duration_ms(self) -> Optional[int]:
        """Get task duration in milliseconds."""
        if self.started_at and self.completed_at:
            delta = self.completed_at - self.started_at
            return int(delta.total_seconds() * 1000)
        return None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "task_id": self.task_id,
            "agent_name": self.agent_name,
            "status": self.status.value,
            "error_message": self.error_message,
            "output": self.output,
            "output_type": self.output_type,
            "structured_output": self.structured_output,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "duration_ms": self.duration_ms,
            "tokens_used": self.tokens_used,
            "intermediate_results": self.intermediate_results,
            "subtasks": self.subtasks,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TaskResult":
        """Create from dictionary."""
        return cls(
            task_id=data["task_id"],
            agent_name=data["agent_name"],
            status=TaskStatus(data["status"]),
            error_message=data.get("error_message"),
            output=data.get("output"),
            output_type=data.get("output_type", "text"),
            structured_output=data.get("structured_output"),
            started_at=(
                datetime.fromisoformat(data["started_at"]) if data.get("started_at") else None
            ),
            completed_at=(
                datetime.fromisoformat(data["completed_at"]) if data.get("completed_at") else None
            ),
            tokens_used=data.get("tokens_used", 0),
            intermediate_results=data.get("intermediate_results", []),
            subtasks=data.get("subtasks", []),
            metadata=data.get("metadata", {}),
        )


# Pre-defined Aragora agent cards
ARAGORA_AGENT_CARDS = {
    "debate-orchestrator": AgentCard(
        name="aragora-debate-orchestrator",
        description="Multi-agent debate orchestration with consensus detection",
        capabilities=[
            AgentCapability.DEBATE,
            AgentCapability.CONSENSUS,
            AgentCapability.SYNTHESIS,
        ],
        input_modes=["text"],
        output_modes=["text", "structured"],
        tags=["debate", "multi-agent", "consensus"],
        organization="aragora",
        estimated_response_time_ms=30000,
    ),
    "audit-engine": AgentCard(
        name="aragora-audit-engine",
        description="Multi-vertical document and code auditing",
        capabilities=[
            AgentCapability.AUDIT,
            AgentCapability.DOCUMENT_ANALYSIS,
            AgentCapability.CODE_REVIEW,
        ],
        input_modes=["text", "file"],
        output_modes=["structured"],
        tags=["audit", "compliance", "security"],
        organization="aragora",
        estimated_response_time_ms=60000,
    ),
    "gauntlet": AgentCard(
        name="aragora-gauntlet",
        description="Adversarial stress-testing for content and code",
        capabilities=[
            AgentCapability.CRITIQUE,
            AgentCapability.VERIFICATION,
            AgentCapability.CODE_REVIEW,
        ],
        input_modes=["text"],
        output_modes=["structured"],
        tags=["security", "testing", "adversarial"],
        organization="aragora",
        estimated_response_time_ms=45000,
    ),
    "research": AgentCard(
        name="aragora-research",
        description="Multi-agent research and synthesis",
        capabilities=[
            AgentCapability.RESEARCH,
            AgentCapability.REASONING,
            AgentCapability.SYNTHESIS,
        ],
        input_modes=["text"],
        output_modes=["text", "structured"],
        tags=["research", "knowledge", "synthesis"],
        organization="aragora",
        estimated_response_time_ms=60000,
    ),
}


__all__ = [
    "TaskStatus",
    "TaskPriority",
    "AgentCapability",
    "SecurityCard",
    "AgentCard",
    "ContextItem",
    "TaskRequest",
    "TaskResult",
    "ARAGORA_AGENT_CARDS",
]
