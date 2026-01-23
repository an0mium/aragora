"""
GraphQL Schema for Aragora API.

Provides a GraphQL interface that wraps existing REST functionality,
enabling efficient data fetching and reducing over-fetching issues.

The schema is defined using SDL (Schema Definition Language) and uses
a lightweight internal implementation that doesn't require external
GraphQL libraries. For production use with advanced features, consider
using strawberry-graphql or graphene.

Usage:
    from aragora.server.graphql.schema import SCHEMA, parse_and_validate_query

    # Parse a GraphQL query
    parsed = parse_and_validate_query(query_string)
    if parsed.errors:
        # Handle validation errors
        pass
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set


# =============================================================================
# Schema Definition Language (SDL)
# =============================================================================

SCHEMA_SDL = '''
"""GraphQL Schema for Aragora Debate Platform"""

# =============================================================================
# Enums
# =============================================================================

"""Status of a debate"""
enum DebateStatus {
    PENDING
    RUNNING
    COMPLETED
    FAILED
    CANCELLED
}

"""Status of an agent"""
enum AgentStatus {
    AVAILABLE
    BUSY
    OFFLINE
    DEGRADED
}

"""Status of a task"""
enum TaskStatus {
    PENDING
    RUNNING
    COMPLETED
    FAILED
    CANCELLED
}

"""Task priority levels"""
enum Priority {
    LOW
    NORMAL
    HIGH
    URGENT
}

"""Health status levels"""
enum HealthStatus {
    HEALTHY
    DEGRADED
    UNHEALTHY
}

# =============================================================================
# Input Types
# =============================================================================

"""Input for starting a new debate"""
input StartDebateInput {
    """The question or topic to debate (required)"""
    question: String!

    """Comma-separated list of agent names, or 'auto' for auto-selection"""
    agents: String

    """Number of debate rounds (default: 3, max: 10)"""
    rounds: Int

    """Consensus method: 'majority', 'unanimous', 'weighted'"""
    consensus: String

    """Whether to auto-select agents based on ELO and topic"""
    autoSelect: Boolean

    """Custom tags for the debate"""
    tags: [String!]
}

"""Input for submitting a vote"""
input VoteInput {
    """The agent to vote for"""
    agentId: String!

    """Reason for the vote"""
    reason: String

    """Confidence in the vote (0.0 to 1.0)"""
    confidence: Float
}

"""Input for submitting a task"""
input SubmitTaskInput {
    """Type of task (e.g., 'debate', 'analysis', 'document_processing')"""
    taskType: String!

    """Task payload data"""
    payload: JSON

    """Required agent capabilities"""
    requiredCapabilities: [String!]

    """Task priority"""
    priority: Priority

    """Timeout in seconds"""
    timeoutSeconds: Int

    """Additional metadata"""
    metadata: JSON
}

"""Input for registering an agent"""
input RegisterAgentInput {
    """Unique agent identifier"""
    agentId: String!

    """List of agent capabilities"""
    capabilities: [String!]!

    """Model name (e.g., 'gpt-4', 'claude-3')"""
    model: String!

    """Provider name (e.g., 'openai', 'anthropic')"""
    provider: String!

    """Additional metadata"""
    metadata: JSON
}

# =============================================================================
# Object Types
# =============================================================================

"""Represents a debate session"""
type Debate {
    """Unique debate identifier"""
    id: ID!

    """The debate topic/question"""
    topic: String!

    """Alternative name for topic"""
    task: String

    """Current status"""
    status: DebateStatus!

    """List of debate rounds"""
    rounds: [Round!]!

    """Participating agents"""
    participants: [Agent!]!

    """Consensus information if reached"""
    consensus: Consensus

    """When the debate was created"""
    createdAt: DateTime!

    """When the debate completed (if completed)"""
    completedAt: DateTime

    """Number of rounds configured"""
    roundCount: Int!

    """Tags associated with the debate"""
    tags: [String!]

    """Whether consensus was reached"""
    consensusReached: Boolean

    """Confidence score of the final answer"""
    confidence: Float

    """The winning agent (if determined)"""
    winner: String
}

"""Paginated connection for debates"""
type DebateConnection {
    """List of debates"""
    debates: [Debate!]!

    """Total count of debates matching filter"""
    total: Int!

    """Whether there are more results"""
    hasMore: Boolean!

    """Cursor for pagination"""
    cursor: String
}

"""A single round in a debate"""
type Round {
    """Round number (1-indexed)"""
    number: Int!

    """Messages in this round"""
    messages: [Message!]!

    """Critiques made during this round"""
    critiques: [Critique!]

    """Whether this round is complete"""
    completed: Boolean!
}

"""A message in a debate"""
type Message {
    """Message index in debate"""
    index: Int!

    """Role of the sender (agent, user, system)"""
    role: String!

    """Message content"""
    content: String!

    """Name of the agent (if from agent)"""
    agent: String

    """Round number this message belongs to"""
    round: Int!

    """When the message was sent"""
    timestamp: DateTime
}

"""A critique of a response"""
type Critique {
    """Critique identifier"""
    id: ID!

    """Agent providing the critique"""
    critic: String!

    """Agent being critiqued"""
    target: String!

    """Critique content"""
    content: String!

    """Severity score (0.0 to 1.0)"""
    severity: Float!

    """Whether the critique was accepted"""
    accepted: Boolean
}

"""Consensus information for a debate"""
type Consensus {
    """Whether consensus was reached"""
    reached: Boolean!

    """Final answer text"""
    answer: String

    """Agents who agreed"""
    agreeingAgents: [String!]!

    """Agents who disagreed"""
    dissentingAgents: [String!]

    """Confidence score"""
    confidence: Float

    """Method used for consensus"""
    method: String
}

"""Represents an AI agent"""
type Agent {
    """Agent identifier"""
    id: ID!

    """Agent name"""
    name: String!

    """Current status"""
    status: AgentStatus!

    """List of capabilities"""
    capabilities: [String!]!

    """Deployment region"""
    region: String

    """Currently assigned task"""
    currentTask: Task

    """Agent statistics"""
    stats: AgentStats!

    """ELO rating"""
    elo: Float

    """Model identifier"""
    model: String

    """Provider name"""
    provider: String
}

"""Statistics for an agent"""
type AgentStats {
    """Total games played"""
    totalGames: Int!

    """Number of wins"""
    wins: Int!

    """Number of losses"""
    losses: Int!

    """Number of draws"""
    draws: Int!

    """Win rate percentage"""
    winRate: Float!

    """Current ELO rating"""
    elo: Float!

    """Calibration accuracy"""
    calibrationAccuracy: Float

    """Consistency score"""
    consistencyScore: Float
}

"""Represents a task in the control plane"""
type Task {
    """Task identifier"""
    id: ID!

    """Task type"""
    type: String!

    """Current status"""
    status: TaskStatus!

    """Task priority"""
    priority: Priority!

    """Assigned agent"""
    assignedAgent: Agent

    """Task result (if completed)"""
    result: JSON

    """When the task was created"""
    createdAt: DateTime!

    """When the task completed"""
    completedAt: DateTime

    """Task payload"""
    payload: JSON

    """Task metadata"""
    metadata: JSON
}

"""Paginated connection for tasks"""
type TaskConnection {
    """List of tasks"""
    tasks: [Task!]!

    """Total count of tasks"""
    total: Int!

    """Whether there are more results"""
    hasMore: Boolean!
}

"""Vote cast by a user"""
type Vote {
    """Vote identifier"""
    id: ID!

    """Debate the vote is for"""
    debateId: ID!

    """Agent voted for"""
    agentId: String!

    """Reason for the vote"""
    reason: String

    """Confidence in the vote"""
    confidence: Float

    """When the vote was cast"""
    createdAt: DateTime!
}

"""System health information"""
type SystemHealth {
    """Overall system status"""
    status: HealthStatus!

    """System uptime in seconds"""
    uptimeSeconds: Int!

    """System version"""
    version: String!

    """Individual component health"""
    components: [ComponentHealth!]!
}

"""Health status of a system component"""
type ComponentHealth {
    """Component name"""
    name: String!

    """Component status"""
    status: HealthStatus!

    """Latency in milliseconds"""
    latencyMs: Int

    """Error message if unhealthy"""
    error: String
}

"""System statistics"""
type SystemStats {
    """Number of active jobs"""
    activeJobs: Int!

    """Number of queued jobs"""
    queuedJobs: Int!

    """Number of completed jobs today"""
    completedJobsToday: Int!

    """Number of available agents"""
    availableAgents: Int!

    """Number of busy agents"""
    busyAgents: Int!

    """Total number of agents"""
    totalAgents: Int!

    """Documents processed today"""
    documentsProcessedToday: Int!
}

"""Event for debate updates (subscriptions)"""
type DebateEvent {
    """Event type"""
    type: String!

    """Debate ID"""
    debateId: ID!

    """Event payload"""
    data: JSON!

    """Event timestamp"""
    timestamp: DateTime!
}

"""Event for task updates (subscriptions)"""
type TaskEvent {
    """Event type"""
    type: String!

    """Task ID"""
    taskId: ID!

    """Event payload"""
    data: JSON!

    """Event timestamp"""
    timestamp: DateTime!
}

# =============================================================================
# Scalar Types
# =============================================================================

"""ISO 8601 DateTime string"""
scalar DateTime

"""Arbitrary JSON data"""
scalar JSON

# =============================================================================
# Root Types
# =============================================================================

type Query {
    # === Debates ===

    """Get a single debate by ID"""
    debate(id: ID!): Debate

    """List debates with optional filtering"""
    debates(
        """Filter by status"""
        status: DebateStatus
        """Maximum number of results (default: 20)"""
        limit: Int
        """Offset for pagination (default: 0)"""
        offset: Int
    ): DebateConnection!

    """Search debates by query string"""
    searchDebates(
        """Search query"""
        query: String!
        """Maximum number of results"""
        limit: Int
    ): DebateConnection!

    # === Agents ===

    """Get a single agent by ID"""
    agent(id: ID!): Agent

    """List agents with optional filtering"""
    agents(
        """Filter by status"""
        status: AgentStatus
        """Filter by capability"""
        capability: String
        """Filter by region"""
        region: String
    ): [Agent!]!

    """Get agent leaderboard"""
    leaderboard(
        """Maximum number of results"""
        limit: Int
        """Filter by domain"""
        domain: String
    ): [Agent!]!

    # === Tasks ===

    """Get a single task by ID"""
    task(id: ID!): Task

    """List tasks with optional filtering"""
    tasks(
        """Filter by status"""
        status: TaskStatus
        """Filter by type"""
        type: String
        """Maximum number of results"""
        limit: Int
    ): TaskConnection!

    # === Control Plane ===

    """Get system health status"""
    systemHealth: SystemHealth!

    """Get system statistics"""
    stats: SystemStats!
}

type Mutation {
    # === Debates ===

    """Start a new debate"""
    startDebate(input: StartDebateInput!): Debate!

    """Submit a vote for an agent in a debate"""
    submitVote(debateId: ID!, vote: VoteInput!): Vote!

    """Cancel a running debate"""
    cancelDebate(id: ID!): Debate!

    # === Tasks ===

    """Submit a new task"""
    submitTask(input: SubmitTaskInput!): Task!

    """Cancel a pending or running task"""
    cancelTask(id: ID!): Task!

    # === Agents ===

    """Register a new agent"""
    registerAgent(input: RegisterAgentInput!): Agent!

    """Unregister an agent"""
    unregisterAgent(id: ID!): Boolean!
}

type Subscription {
    """Subscribe to debate updates"""
    debateUpdates(debateId: ID!): DebateEvent!

    """Subscribe to task updates"""
    taskUpdates(taskId: ID): TaskEvent!
}
'''


# =============================================================================
# Schema Parsing and Validation
# =============================================================================


class OperationType(Enum):
    """GraphQL operation types."""

    QUERY = "query"
    MUTATION = "mutation"
    SUBSCRIPTION = "subscription"


@dataclass
class Field:
    """Represents a field selection in a GraphQL query."""

    name: str
    alias: Optional[str] = None
    arguments: Dict[str, Any] = field(default_factory=dict)
    selections: List["Field"] = field(default_factory=list)
    directives: List[str] = field(default_factory=list)


@dataclass
class Operation:
    """Represents a parsed GraphQL operation."""

    type: OperationType
    name: Optional[str]
    variables: Dict[str, Any]
    selections: List[Field]


@dataclass
class ParsedQuery:
    """Result of parsing a GraphQL query."""

    operations: List[Operation]
    fragments: Dict[str, List[Field]]
    errors: List[str] = field(default_factory=list)


@dataclass
class ValidationResult:
    """Result of validating a GraphQL query."""

    valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


# Type definitions extracted from SDL
QUERY_FIELDS: Set[str] = {
    "debate",
    "debates",
    "searchDebates",
    "agent",
    "agents",
    "leaderboard",
    "task",
    "tasks",
    "systemHealth",
    "stats",
}

MUTATION_FIELDS: Set[str] = {
    "startDebate",
    "submitVote",
    "cancelDebate",
    "submitTask",
    "cancelTask",
    "registerAgent",
    "unregisterAgent",
}

SUBSCRIPTION_FIELDS: Set[str] = {
    "debateUpdates",
    "taskUpdates",
}

# Object types and their fields
OBJECT_TYPES: Dict[str, Set[str]] = {
    "Debate": {
        "id",
        "topic",
        "task",
        "status",
        "rounds",
        "participants",
        "consensus",
        "createdAt",
        "completedAt",
        "roundCount",
        "tags",
        "consensusReached",
        "confidence",
        "winner",
    },
    "DebateConnection": {"debates", "total", "hasMore", "cursor"},
    "Round": {"number", "messages", "critiques", "completed"},
    "Message": {"index", "role", "content", "agent", "round", "timestamp"},
    "Critique": {"id", "critic", "target", "content", "severity", "accepted"},
    "Consensus": {
        "reached",
        "answer",
        "agreeingAgents",
        "dissentingAgents",
        "confidence",
        "method",
    },
    "Agent": {
        "id",
        "name",
        "status",
        "capabilities",
        "region",
        "currentTask",
        "stats",
        "elo",
        "model",
        "provider",
    },
    "AgentStats": {
        "totalGames",
        "wins",
        "losses",
        "draws",
        "winRate",
        "elo",
        "calibrationAccuracy",
        "consistencyScore",
    },
    "Task": {
        "id",
        "type",
        "status",
        "priority",
        "assignedAgent",
        "result",
        "createdAt",
        "completedAt",
        "payload",
        "metadata",
    },
    "TaskConnection": {"tasks", "total", "hasMore"},
    "Vote": {"id", "debateId", "agentId", "reason", "confidence", "createdAt"},
    "SystemHealth": {"status", "uptimeSeconds", "version", "components"},
    "ComponentHealth": {"name", "status", "latencyMs", "error"},
    "SystemStats": {
        "activeJobs",
        "queuedJobs",
        "completedJobsToday",
        "availableAgents",
        "busyAgents",
        "totalAgents",
        "documentsProcessedToday",
    },
    "DebateEvent": {"type", "debateId", "data", "timestamp"},
    "TaskEvent": {"type", "taskId", "data", "timestamp"},
}

# Input types and their fields
INPUT_TYPES: Dict[str, Set[str]] = {
    "StartDebateInput": {"question", "agents", "rounds", "consensus", "autoSelect", "tags"},
    "VoteInput": {"agentId", "reason", "confidence"},
    "SubmitTaskInput": {
        "taskType",
        "payload",
        "requiredCapabilities",
        "priority",
        "timeoutSeconds",
        "metadata",
    },
    "RegisterAgentInput": {"agentId", "capabilities", "model", "provider", "metadata"},
}

# Enum values
ENUM_VALUES: Dict[str, Set[str]] = {
    "DebateStatus": {"PENDING", "RUNNING", "COMPLETED", "FAILED", "CANCELLED"},
    "AgentStatus": {"AVAILABLE", "BUSY", "OFFLINE", "DEGRADED"},
    "TaskStatus": {"PENDING", "RUNNING", "COMPLETED", "FAILED", "CANCELLED"},
    "Priority": {"LOW", "NORMAL", "HIGH", "URGENT"},
    "HealthStatus": {"HEALTHY", "DEGRADED", "UNHEALTHY"},
}


class GraphQLParser:
    """Lightweight GraphQL query parser.

    This parser handles common GraphQL query patterns without requiring
    external dependencies. For full GraphQL specification compliance,
    use a library like graphql-core.
    """

    def __init__(self) -> None:
        self._errors: List[str] = []

    def parse(self, query: str) -> ParsedQuery:
        """Parse a GraphQL query string.

        Args:
            query: GraphQL query string

        Returns:
            ParsedQuery with operations, fragments, and any errors
        """
        self._errors = []
        operations: List[Operation] = []
        fragments: Dict[str, List[Field]] = {}

        # Remove comments
        query = re.sub(r"#[^\n]*", "", query)

        # Find all operations
        op_pattern = r"(query|mutation|subscription)\s*(\w+)?\s*(\([^)]*\))?\s*\{"
        matches = list(re.finditer(op_pattern, query, re.IGNORECASE))

        if not matches:
            # Handle shorthand query syntax (just { ... })
            if "{" in query:
                body = self._extract_body(query, query.index("{"))
                selections = self._parse_selections(body)
                operations.append(
                    Operation(
                        type=OperationType.QUERY,
                        name=None,
                        variables={},
                        selections=selections,
                    )
                )
            else:
                self._errors.append("No valid GraphQL operation found")
        else:
            for match in matches:
                op_type_str = match.group(1).lower()
                op_name = match.group(2)
                variables_str = match.group(3)

                op_type = OperationType(op_type_str)
                variables = self._parse_variables(variables_str) if variables_str else {}

                # Extract body
                body_start = match.end() - 1  # Include the {
                body = self._extract_body(query, body_start)
                selections = self._parse_selections(body)

                operations.append(
                    Operation(
                        type=op_type,
                        name=op_name,
                        variables=variables,
                        selections=selections,
                    )
                )

        # Find fragments
        frag_pattern = r"fragment\s+(\w+)\s+on\s+(\w+)\s*\{"
        frag_matches = list(re.finditer(frag_pattern, query, re.IGNORECASE))

        for match in frag_matches:
            frag_name = match.group(1)
            body_start = match.end() - 1
            body = self._extract_body(query, body_start)
            fragments[frag_name] = self._parse_selections(body)

        return ParsedQuery(
            operations=operations,
            fragments=fragments,
            errors=self._errors,
        )

    def _extract_body(self, query: str, start: int) -> str:
        """Extract a balanced braces block from query."""
        depth = 0
        end = start

        for i, char in enumerate(query[start:], start):
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    end = i
                    break

        return query[start + 1 : end].strip()

    def _parse_variables(self, variables_str: str) -> Dict[str, Any]:
        """Parse variable definitions."""
        variables: Dict[str, Any] = {}

        if not variables_str:
            return variables

        # Remove parentheses
        variables_str = variables_str.strip("()")

        # Simple variable extraction
        var_pattern = r"\$(\w+)\s*:\s*(\w+!?(?:\s*=\s*[^,]+)?)"
        for match in re.finditer(var_pattern, variables_str):
            var_name = match.group(1)
            var_type = match.group(2)
            variables[var_name] = {"type": var_type}

        return variables

    def _parse_selections(self, body: str) -> List[Field]:
        """Parse field selections from a query body."""
        selections: List[Field] = []

        # Tokenize the body
        i = 0
        while i < len(body):
            # Skip whitespace
            while i < len(body) and body[i].isspace():
                i += 1

            if i >= len(body):
                break

            # Parse field
            field_match = re.match(r"(\w+)(?:\s*:\s*(\w+))?\s*(\([^)]*\))?\s*(@\w+)?", body[i:])
            if not field_match:
                i += 1
                continue

            alias = None
            name = field_match.group(1)

            # Check for alias
            if field_match.group(2):
                alias = name
                name = field_match.group(2)

            # Parse arguments
            arguments: Dict[str, Any] = {}
            if field_match.group(3):
                arguments = self._parse_arguments(field_match.group(3))

            # Parse directive
            directives: List[str] = []
            if field_match.group(4):
                directives.append(field_match.group(4))

            i += field_match.end()

            # Check for nested selections
            sub_selections: List[Field] = []
            while i < len(body) and body[i].isspace():
                i += 1

            if i < len(body) and body[i] == "{":
                sub_body = self._extract_body(body, i)
                sub_selections = self._parse_selections(sub_body)
                # Move past the closing brace
                depth = 1
                i += 1
                while i < len(body) and depth > 0:
                    if body[i] == "{":
                        depth += 1
                    elif body[i] == "}":
                        depth -= 1
                    i += 1

            selections.append(
                Field(
                    name=name,
                    alias=alias,
                    arguments=arguments,
                    selections=sub_selections,
                    directives=directives,
                )
            )

        return selections

    def _parse_arguments(self, args_str: str) -> Dict[str, Any]:
        """Parse field arguments."""
        arguments: Dict[str, Any] = {}

        if not args_str:
            return arguments

        # Remove parentheses
        args_str = args_str.strip("()")

        # Simple argument extraction
        arg_pattern = (
            r'(\w+)\s*:\s*(?:(\$\w+)|"([^"]*)"|(\d+(?:\.\d+)?)|(\w+)|\{([^}]*)\}|\[([^\]]*)\])'
        )
        for match in re.finditer(arg_pattern, args_str):
            arg_name = match.group(1)

            if match.group(2):  # Variable reference
                arguments[arg_name] = {"$var": match.group(2)[1:]}
            elif match.group(3) is not None:  # String
                arguments[arg_name] = match.group(3)
            elif match.group(4):  # Number
                num_str = match.group(4)
                arguments[arg_name] = float(num_str) if "." in num_str else int(num_str)
            elif match.group(5):  # Enum or boolean
                val = match.group(5)
                if val.lower() == "true":
                    arguments[arg_name] = True
                elif val.lower() == "false":
                    arguments[arg_name] = False
                elif val.lower() == "null":
                    arguments[arg_name] = None
                else:
                    arguments[arg_name] = val  # Enum value
            elif match.group(6):  # Object
                arguments[arg_name] = self._parse_arguments(match.group(6))
            elif match.group(7):  # Array
                arguments[arg_name] = self._parse_array(match.group(7))

        return arguments

    def _parse_array(self, array_str: str) -> List[Any]:
        """Parse an array value."""
        values: List[Any] = []

        # Simple array parsing
        val_pattern = r'"([^"]*)"|(\d+(?:\.\d+)?)|(\w+)'
        for match in re.finditer(val_pattern, array_str):
            if match.group(1) is not None:
                values.append(match.group(1))
            elif match.group(2):
                num_str = match.group(2)
                values.append(float(num_str) if "." in num_str else int(num_str))
            elif match.group(3):
                values.append(match.group(3))

        return values


class GraphQLValidator:
    """Validates GraphQL queries against the schema."""

    def validate(self, parsed: ParsedQuery) -> ValidationResult:
        """Validate a parsed GraphQL query.

        Args:
            parsed: Parsed query to validate

        Returns:
            ValidationResult with validation status and any errors
        """
        errors: List[str] = list(parsed.errors)
        warnings: List[str] = []

        for operation in parsed.operations:
            # Validate operation type fields
            root_fields = self._get_root_fields(operation.type)

            for selection in operation.selections:
                if selection.name not in root_fields:
                    errors.append(
                        f"Field '{selection.name}' does not exist on type "
                        f"'{operation.type.value.capitalize()}'"
                    )
                else:
                    # Recursively validate nested selections
                    self._validate_selections(
                        selection.selections,
                        self._get_return_type(operation.type, selection.name),
                        errors,
                        warnings,
                    )

        return ValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
        )

    def _get_root_fields(self, op_type: OperationType) -> Set[str]:
        """Get valid root fields for an operation type."""
        if op_type == OperationType.QUERY:
            return QUERY_FIELDS
        elif op_type == OperationType.MUTATION:
            return MUTATION_FIELDS
        elif op_type == OperationType.SUBSCRIPTION:
            return SUBSCRIPTION_FIELDS
        return set()

    def _get_return_type(self, op_type: OperationType, field_name: str) -> Optional[str]:
        """Get the return type for a root field."""
        # Map of field -> return type
        type_map = {
            # Query fields
            "debate": "Debate",
            "debates": "DebateConnection",
            "searchDebates": "DebateConnection",
            "agent": "Agent",
            "agents": "Agent",
            "leaderboard": "Agent",
            "task": "Task",
            "tasks": "TaskConnection",
            "systemHealth": "SystemHealth",
            "stats": "SystemStats",
            # Mutation fields
            "startDebate": "Debate",
            "submitVote": "Vote",
            "cancelDebate": "Debate",
            "submitTask": "Task",
            "cancelTask": "Task",
            "registerAgent": "Agent",
            "unregisterAgent": None,  # Returns Boolean
            # Subscription fields
            "debateUpdates": "DebateEvent",
            "taskUpdates": "TaskEvent",
        }
        return type_map.get(field_name)

    def _validate_selections(
        self,
        selections: List[Field],
        type_name: Optional[str],
        errors: List[str],
        warnings: List[str],
    ) -> None:
        """Validate field selections against a type."""
        if not type_name or type_name not in OBJECT_TYPES:
            return

        valid_fields = OBJECT_TYPES[type_name]

        for selection in selections:
            if selection.name == "__typename":
                continue  # Introspection field, always valid

            if selection.name not in valid_fields:
                errors.append(f"Field '{selection.name}' does not exist on type '{type_name}'")
            else:
                # Get nested type and continue validation
                nested_type = self._get_nested_type(type_name, selection.name)
                if nested_type and selection.selections:
                    self._validate_selections(
                        selection.selections,
                        nested_type,
                        errors,
                        warnings,
                    )

    def _get_nested_type(self, parent_type: str, field_name: str) -> Optional[str]:
        """Get the type of a nested field."""
        # Map of parent_type.field -> nested_type
        nested_types = {
            "Debate.rounds": "Round",
            "Debate.participants": "Agent",
            "Debate.consensus": "Consensus",
            "DebateConnection.debates": "Debate",
            "Round.messages": "Message",
            "Round.critiques": "Critique",
            "Agent.currentTask": "Task",
            "Agent.stats": "AgentStats",
            "Task.assignedAgent": "Agent",
            "TaskConnection.tasks": "Task",
            "SystemHealth.components": "ComponentHealth",
        }
        return nested_types.get(f"{parent_type}.{field_name}")


def parse_and_validate_query(query: str) -> ParsedQuery:
    """Parse and validate a GraphQL query.

    Args:
        query: GraphQL query string

    Returns:
        ParsedQuery with operations, fragments, and validation errors
    """
    parser = GraphQLParser()
    parsed = parser.parse(query)

    validator = GraphQLValidator()
    validation = validator.validate(parsed)

    # Merge validation errors
    parsed.errors.extend(validation.errors)

    return parsed


# Export the schema for documentation
SCHEMA = SCHEMA_SDL
