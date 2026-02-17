"""Step Type Catalog for Visual Workflow Builder.

Provides metadata, JSON schemas, and display information for all
registered step types. Used by the visual builder frontend to render
step palettes and configuration forms.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class StepTypeInfo:
    """Metadata for a step type in the visual builder."""

    type_name: str
    display_name: str
    description: str
    category: str  # agent, control, memory, human, debate, integration, extraction
    color: str  # Hex color for visual builder
    icon: str  # Icon identifier
    config_schema: dict[str, Any] = field(default_factory=dict)  # JSON Schema
    default_config: dict[str, Any] = field(default_factory=dict)
    inputs: list[str] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "type_name": self.type_name,
            "display_name": self.display_name,
            "description": self.description,
            "category": self.category,
            "color": self.color,
            "icon": self.icon,
            "config_schema": self.config_schema,
            "default_config": self.default_config,
            "inputs": self.inputs,
            "outputs": self.outputs,
        }


# Category color constants
_AGENT_COLOR = "#3B82F6"
_CONTROL_COLOR = "#8B5CF6"
_MEMORY_COLOR = "#10B981"
_HUMAN_COLOR = "#F59E0B"
_DEBATE_COLOR = "#EF4444"
_INTEGRATION_COLOR = "#6366F1"
_EXTRACTION_COLOR = "#EC4899"


_CATALOG: dict[str, StepTypeInfo] = {}


def _register(info: StepTypeInfo) -> None:
    _CATALOG[info.type_name] = info


# ---------------------------------------------------------------------------
# Agent category
# ---------------------------------------------------------------------------

_register(StepTypeInfo(
    type_name="agent",
    display_name="Agent",
    description="Execute an AI agent with a prompt and optional tools.",
    category="agent",
    color=_AGENT_COLOR,
    icon="bot",
    config_schema={
        "type": "object",
        "properties": {
            "agent_type": {"type": "string", "description": "Agent model identifier (e.g. claude, gpt4)"},
            "prompt": {"type": "string", "description": "Prompt to send to the agent"},
            "temperature": {"type": "number", "minimum": 0, "maximum": 2, "default": 0.7},
            "max_tokens": {"type": "integer", "minimum": 1, "default": 4096},
        },
        "required": ["agent_type", "prompt"],
    },
    default_config={"agent_type": "claude", "temperature": 0.7, "max_tokens": 4096},
    inputs=["prompt", "context"],
    outputs=["response", "tokens_used"],
))

_register(StepTypeInfo(
    type_name="task",
    display_name="Task",
    description="A generic task step with configurable action.",
    category="agent",
    color=_AGENT_COLOR,
    icon="clipboard",
    config_schema={
        "type": "object",
        "properties": {
            "action": {"type": "string", "description": "Task action to perform"},
            "params": {"type": "object", "description": "Action parameters"},
        },
        "required": ["action"],
    },
    default_config={"action": "process"},
    inputs=["data"],
    outputs=["result"],
))

_register(StepTypeInfo(
    type_name="nomic",
    display_name="Nomic Step",
    description="Run a single Nomic Loop phase (debate, design, implement, verify).",
    category="agent",
    color=_AGENT_COLOR,
    icon="refresh-cw",
    config_schema={
        "type": "object",
        "properties": {
            "phase": {"type": "string", "enum": ["debate", "design", "implement", "verify"]},
            "goal": {"type": "string", "description": "Goal for the Nomic phase"},
        },
        "required": ["phase"],
    },
    default_config={"phase": "debate"},
    inputs=["goal", "context"],
    outputs=["result", "artifacts"],
))

_register(StepTypeInfo(
    type_name="nomic_loop",
    display_name="Nomic Loop",
    description="Run the full Nomic self-improvement loop with multiple cycles.",
    category="agent",
    color=_AGENT_COLOR,
    icon="repeat",
    config_schema={
        "type": "object",
        "properties": {
            "goal": {"type": "string"},
            "cycles": {"type": "integer", "minimum": 1, "default": 1},
            "require_approval": {"type": "boolean", "default": True},
        },
        "required": ["goal"],
    },
    default_config={"cycles": 1, "require_approval": True},
    inputs=["goal"],
    outputs=["improvements", "cycle_count"],
))

_register(StepTypeInfo(
    type_name="implementation",
    display_name="Implementation",
    description="Generate code implementation from a design specification.",
    category="agent",
    color=_AGENT_COLOR,
    icon="code",
    config_schema={
        "type": "object",
        "properties": {
            "spec": {"type": "string", "description": "Design specification"},
            "language": {"type": "string", "default": "python"},
            "agent_type": {"type": "string", "default": "claude"},
        },
        "required": ["spec"],
    },
    default_config={"language": "python", "agent_type": "claude"},
    inputs=["spec", "context"],
    outputs=["code", "files_modified"],
))

_register(StepTypeInfo(
    type_name="verification",
    display_name="Verification",
    description="Verify implementation with tests, linting, and type checks.",
    category="agent",
    color=_AGENT_COLOR,
    icon="check-circle",
    config_schema={
        "type": "object",
        "properties": {
            "test_command": {"type": "string", "default": "pytest"},
            "lint": {"type": "boolean", "default": True},
            "type_check": {"type": "boolean", "default": False},
        },
    },
    default_config={"test_command": "pytest", "lint": True},
    inputs=["code", "files"],
    outputs=["passed", "test_results", "lint_results"],
))

# ---------------------------------------------------------------------------
# Control flow category
# ---------------------------------------------------------------------------

_register(StepTypeInfo(
    type_name="parallel",
    display_name="Parallel",
    description="Execute multiple steps concurrently and merge results.",
    category="control",
    color=_CONTROL_COLOR,
    icon="git-branch",
    config_schema={
        "type": "object",
        "properties": {
            "branches": {"type": "array", "items": {"type": "string"}, "description": "Step IDs to run in parallel"},
            "merge_strategy": {"type": "string", "enum": ["all", "first", "majority"], "default": "all"},
            "max_concurrency": {"type": "integer", "minimum": 1, "default": 5},
        },
        "required": ["branches"],
    },
    default_config={"merge_strategy": "all", "max_concurrency": 5},
    inputs=["data"],
    outputs=["merged_results"],
))

_register(StepTypeInfo(
    type_name="conditional",
    display_name="Conditional",
    description="Branch execution based on a condition expression.",
    category="control",
    color=_CONTROL_COLOR,
    icon="git-merge",
    config_schema={
        "type": "object",
        "properties": {
            "condition": {"type": "string", "description": "Python expression evaluated against context"},
            "true_step": {"type": "string", "description": "Step ID if condition is true"},
            "false_step": {"type": "string", "description": "Step ID if condition is false"},
        },
        "required": ["condition"],
    },
    default_config={},
    inputs=["data"],
    outputs=["branch_taken"],
))

_register(StepTypeInfo(
    type_name="loop",
    display_name="Loop",
    description="Repeat a step or group of steps until a condition is met.",
    category="control",
    color=_CONTROL_COLOR,
    icon="rotate-cw",
    config_schema={
        "type": "object",
        "properties": {
            "condition": {"type": "string", "description": "Continue while this condition is true"},
            "max_iterations": {"type": "integer", "minimum": 1, "default": 10},
            "body_steps": {"type": "array", "items": {"type": "string"}, "description": "Step IDs in loop body"},
        },
        "required": ["condition"],
    },
    default_config={"max_iterations": 10},
    inputs=["data"],
    outputs=["iterations", "final_result"],
))

_register(StepTypeInfo(
    type_name="switch",
    display_name="Switch",
    description="Multi-way branch based on a value expression.",
    category="control",
    color=_CONTROL_COLOR,
    icon="layers",
    config_schema={
        "type": "object",
        "properties": {
            "expression": {"type": "string", "description": "Expression to evaluate"},
            "cases": {"type": "object", "description": "Map of value → step ID"},
            "default_step": {"type": "string", "description": "Step ID for unmatched values"},
        },
        "required": ["expression", "cases"],
    },
    default_config={},
    inputs=["data"],
    outputs=["matched_case"],
))

_register(StepTypeInfo(
    type_name="decision",
    display_name="Decision",
    description="Binary or multi-option decision point with criteria evaluation.",
    category="control",
    color=_CONTROL_COLOR,
    icon="help-circle",
    config_schema={
        "type": "object",
        "properties": {
            "question": {"type": "string", "description": "Decision question to evaluate"},
            "options": {"type": "array", "items": {"type": "string"}, "description": "Available options"},
            "criteria": {"type": "array", "items": {"type": "string"}, "description": "Evaluation criteria"},
        },
        "required": ["question"],
    },
    default_config={"options": ["yes", "no"]},
    inputs=["context"],
    outputs=["decision", "reasoning"],
))

# ---------------------------------------------------------------------------
# Memory category
# ---------------------------------------------------------------------------

_register(StepTypeInfo(
    type_name="memory_read",
    display_name="Memory Read",
    description="Read from Knowledge Mound or memory systems.",
    category="memory",
    color=_MEMORY_COLOR,
    icon="database",
    config_schema={
        "type": "object",
        "properties": {
            "source": {"type": "string", "enum": ["knowledge_mound", "continuum", "supermemory"], "default": "knowledge_mound"},
            "query": {"type": "string", "description": "Search query"},
            "limit": {"type": "integer", "minimum": 1, "default": 10},
        },
        "required": ["query"],
    },
    default_config={"source": "knowledge_mound", "limit": 10},
    inputs=["query"],
    outputs=["results", "count"],
))

_register(StepTypeInfo(
    type_name="memory_write",
    display_name="Memory Write",
    description="Write data to Knowledge Mound or memory systems.",
    category="memory",
    color=_MEMORY_COLOR,
    icon="save",
    config_schema={
        "type": "object",
        "properties": {
            "target": {"type": "string", "enum": ["knowledge_mound", "continuum", "supermemory"], "default": "knowledge_mound"},
            "key": {"type": "string", "description": "Storage key"},
            "ttl_seconds": {"type": "integer", "description": "Time-to-live in seconds"},
        },
        "required": ["key"],
    },
    default_config={"target": "knowledge_mound"},
    inputs=["data", "key"],
    outputs=["stored", "entry_id"],
))

# ---------------------------------------------------------------------------
# Human category
# ---------------------------------------------------------------------------

_register(StepTypeInfo(
    type_name="human_checkpoint",
    display_name="Human Checkpoint",
    description="Pause workflow for human review and approval.",
    category="human",
    color=_HUMAN_COLOR,
    icon="user-check",
    config_schema={
        "type": "object",
        "properties": {
            "prompt": {"type": "string", "description": "Instructions for the reviewer"},
            "timeout_hours": {"type": "number", "default": 24},
            "required_approvers": {"type": "integer", "minimum": 1, "default": 1},
            "escalation_after_hours": {"type": "number", "description": "Hours before escalation"},
        },
        "required": ["prompt"],
    },
    default_config={"timeout_hours": 24, "required_approvers": 1},
    inputs=["data_for_review"],
    outputs=["approved", "reviewer_notes"],
))

# ---------------------------------------------------------------------------
# Debate category
# ---------------------------------------------------------------------------

_register(StepTypeInfo(
    type_name="debate",
    display_name="Debate",
    description="Run a full multi-agent debate with configurable rounds and consensus.",
    category="debate",
    color=_DEBATE_COLOR,
    icon="message-square",
    config_schema={
        "type": "object",
        "properties": {
            "task": {"type": "string", "description": "Debate topic or question"},
            "agents": {"type": "array", "items": {"type": "string"}, "description": "Agent types to include"},
            "rounds": {"type": "integer", "minimum": 1, "default": 3},
            "consensus_mode": {"type": "string", "enum": ["majority", "supermajority", "unanimous"], "default": "majority"},
        },
        "required": ["task"],
    },
    default_config={"rounds": 3, "consensus_mode": "majority"},
    inputs=["task", "context"],
    outputs=["consensus", "positions", "receipt"],
))

_register(StepTypeInfo(
    type_name="quick_debate",
    display_name="Quick Debate",
    description="Lightweight single-round debate for fast decisions.",
    category="debate",
    color=_DEBATE_COLOR,
    icon="zap",
    config_schema={
        "type": "object",
        "properties": {
            "task": {"type": "string", "description": "Question to decide"},
            "agents": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["task"],
    },
    default_config={},
    inputs=["task"],
    outputs=["decision", "confidence"],
))

# ---------------------------------------------------------------------------
# Integration category
# ---------------------------------------------------------------------------

_register(StepTypeInfo(
    type_name="connector",
    display_name="Connector",
    description="Connect to an external service (Slack, GitHub, email, etc.).",
    category="integration",
    color=_INTEGRATION_COLOR,
    icon="plug",
    config_schema={
        "type": "object",
        "properties": {
            "connector_type": {"type": "string", "description": "Connector identifier"},
            "action": {"type": "string", "description": "Action to perform"},
            "params": {"type": "object", "description": "Action-specific parameters"},
        },
        "required": ["connector_type", "action"],
    },
    default_config={},
    inputs=["data"],
    outputs=["response", "status"],
))

_register(StepTypeInfo(
    type_name="openclaw_action",
    display_name="OpenClaw Action",
    description="Execute a single OpenClaw agent action with reputation tracking.",
    category="integration",
    color=_INTEGRATION_COLOR,
    icon="shield",
    config_schema={
        "type": "object",
        "properties": {
            "agent_id": {"type": "string", "description": "OpenClaw agent identifier"},
            "action": {"type": "string", "description": "Action to execute"},
            "require_verified": {"type": "boolean", "default": True},
        },
        "required": ["agent_id", "action"],
    },
    default_config={"require_verified": True},
    inputs=["action", "context"],
    outputs=["result", "reputation_delta"],
))

_register(StepTypeInfo(
    type_name="openclaw_session",
    display_name="OpenClaw Session",
    description="Manage a multi-step OpenClaw agent session with audit trail.",
    category="integration",
    color=_INTEGRATION_COLOR,
    icon="terminal",
    config_schema={
        "type": "object",
        "properties": {
            "agent_id": {"type": "string"},
            "session_type": {"type": "string", "enum": ["interactive", "batch"], "default": "batch"},
            "max_steps": {"type": "integer", "minimum": 1, "default": 10},
        },
        "required": ["agent_id"],
    },
    default_config={"session_type": "batch", "max_steps": 10},
    inputs=["instructions"],
    outputs=["session_log", "artifacts"],
))

_register(StepTypeInfo(
    type_name="computer_use_task",
    display_name="Computer Use",
    description="Execute a computer-use task via browser or desktop automation.",
    category="integration",
    color=_INTEGRATION_COLOR,
    icon="monitor",
    config_schema={
        "type": "object",
        "properties": {
            "task": {"type": "string", "description": "Task description for the computer-use agent"},
            "url": {"type": "string", "description": "Starting URL for browser tasks"},
            "timeout_seconds": {"type": "integer", "default": 120},
        },
        "required": ["task"],
    },
    default_config={"timeout_seconds": 120},
    inputs=["task", "url"],
    outputs=["screenshots", "result"],
))

# ---------------------------------------------------------------------------
# Extraction category
# ---------------------------------------------------------------------------

_register(StepTypeInfo(
    type_name="content_extraction",
    display_name="Content Extraction",
    description="Extract structured data from documents, web pages, or other content.",
    category="extraction",
    color=_EXTRACTION_COLOR,
    icon="file-text",
    config_schema={
        "type": "object",
        "properties": {
            "source_type": {"type": "string", "enum": ["url", "file", "text"], "default": "text"},
            "schema": {"type": "object", "description": "JSON schema for extracted data structure"},
            "agent_type": {"type": "string", "default": "claude"},
        },
        "required": ["source_type"],
    },
    default_config={"source_type": "text", "agent_type": "claude"},
    inputs=["content"],
    outputs=["extracted", "confidence"],
))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_step_catalog() -> dict[str, StepTypeInfo]:
    """Return the full step catalog."""
    return dict(_CATALOG)


def get_step_type_info(type_name: str) -> StepTypeInfo | None:
    """Return info for a specific step type."""
    return _CATALOG.get(type_name)


def list_step_categories() -> list[str]:
    """Return unique categories."""
    return sorted(set(info.category for info in _CATALOG.values()))


def get_known_step_types() -> set[str]:
    """Return the set of all known step type names."""
    return set(_CATALOG.keys())
