"""
Idea-to-Execution Pipeline Stage Types.

Extends the canvas data model with stage-aware node types for the
four-stage pipeline: Ideas → Goals → Actions → Orchestration.

Each stage is a DAG view with a consistent visual language.
Nodes carry provenance linking every output back to originating ideas.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class PipelineStage(str, Enum):
    """The four stages of the idea-to-execution pipeline."""

    IDEAS = "ideas"  # Stage 1: Organizing ideas into relationships
    GOALS = "goals"  # Stage 2: Deriving goals and principles
    ACTIONS = "actions"  # Stage 3: Project management action sequences
    ORCHESTRATION = "orchestration"  # Stage 4: Multi-agent AI execution


class IdeaNodeType(str, Enum):
    """Node types for Stage 1 (Idea Organization)."""

    CONCEPT = "concept"
    CLUSTER = "cluster"
    QUESTION = "question"
    INSIGHT = "insight"
    EVIDENCE = "evidence"
    ASSUMPTION = "assumption"
    CONSTRAINT = "constraint"


class GoalNodeType(str, Enum):
    """Node types for Stage 2 (Goals & Principles)."""

    GOAL = "goal"
    PRINCIPLE = "principle"
    STRATEGY = "strategy"
    MILESTONE = "milestone"
    METRIC = "metric"
    RISK = "risk"


class ActionNodeType(str, Enum):
    """Node types for Stage 3 (Project Management)."""

    TASK = "task"
    EPIC = "epic"
    CHECKPOINT = "checkpoint"
    DELIVERABLE = "deliverable"
    DEPENDENCY = "dependency"


class OrchestrationNodeType(str, Enum):
    """Node types for Stage 4 (Multi-Agent Orchestration)."""

    AGENT_TASK = "agent_task"
    DEBATE = "debate"
    HUMAN_GATE = "human_gate"
    PARALLEL_FAN = "parallel_fan"
    MERGE = "merge"
    VERIFICATION = "verification"


class StageEdgeType(str, Enum):
    """Edge types across all stages."""

    # Intra-stage
    SUPPORTS = "supports"
    REFUTES = "refutes"
    REQUIRES = "requires"
    CONFLICTS = "conflicts"
    RELATES_TO = "relates_to"
    DECOMPOSES_INTO = "decomposes_into"
    BLOCKS = "blocks"
    FOLLOWS = "follows"
    # Cross-stage provenance
    DERIVED_FROM = "derived_from"  # Goal derived from idea cluster
    IMPLEMENTS = "implements"  # Action implements a goal
    EXECUTES = "executes"  # Orchestration step executes an action


@dataclass
class ProvenanceLink:
    """Blockchain-like provenance linking a node to its origins.

    Each node records which upstream nodes it was derived from,
    with a SHA-256 hash of the source content for integrity.
    """

    source_node_id: str
    source_stage: PipelineStage
    target_node_id: str
    target_stage: PipelineStage
    content_hash: str  # SHA-256 of source content at derivation time
    timestamp: float = field(default_factory=time.time)
    method: str = ""  # How derivation happened (e.g., "ai_synthesis", "manual")

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_node_id": self.source_node_id,
            "source_stage": self.source_stage.value,
            "target_node_id": self.target_node_id,
            "target_stage": self.target_stage.value,
            "content_hash": self.content_hash,
            "timestamp": self.timestamp,
            "method": self.method,
        }


@dataclass
class StageTransition:
    """Record of a transition between pipeline stages.

    Captures the AI-generated mapping from one stage's DAG to the next,
    along with the human review status.
    """

    id: str
    from_stage: PipelineStage
    to_stage: PipelineStage
    provenance: list[ProvenanceLink] = field(default_factory=list)
    status: str = "pending"  # pending, approved, rejected, revised
    confidence: float = 0.0
    ai_rationale: str = ""
    human_notes: str = ""
    created_at: float = field(default_factory=time.time)
    reviewed_at: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "from_stage": self.from_stage.value,
            "to_stage": self.to_stage.value,
            "provenance": [p.to_dict() for p in self.provenance],
            "status": self.status,
            "confidence": self.confidence,
            "ai_rationale": self.ai_rationale,
            "human_notes": self.human_notes,
            "created_at": self.created_at,
            "reviewed_at": self.reviewed_at,
        }


def content_hash(content: str) -> str:
    """Compute SHA-256 hash for provenance integrity."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]


# Stage-specific color palettes for visual consistency
STAGE_COLORS: dict[PipelineStage, dict[str, str]] = {
    PipelineStage.IDEAS: {
        "primary": "#818cf8",  # Indigo
        "secondary": "#c7d2fe",
        "accent": "#4f46e5",
    },
    PipelineStage.GOALS: {
        "primary": "#34d399",  # Emerald
        "secondary": "#a7f3d0",
        "accent": "#059669",
    },
    PipelineStage.ACTIONS: {
        "primary": "#fbbf24",  # Amber
        "secondary": "#fde68a",
        "accent": "#d97706",
    },
    PipelineStage.ORCHESTRATION: {
        "primary": "#f472b6",  # Pink
        "secondary": "#fbcfe8",
        "accent": "#db2777",
    },
}

# Node type colors within each stage
NODE_TYPE_COLORS: dict[str, str] = {
    # Ideas
    "concept": "#818cf8",
    "cluster": "#6366f1",
    "question": "#a78bfa",
    "insight": "#8b5cf6",
    "evidence": "#7c3aed",
    "assumption": "#c4b5fd",
    "constraint": "#ddd6fe",
    # Goals
    "goal": "#34d399",
    "principle": "#10b981",
    "strategy": "#059669",
    "milestone": "#6ee7b7",
    "metric": "#a7f3d0",
    "risk": "#ef4444",
    # Actions
    "task": "#fbbf24",
    "epic": "#f59e0b",
    "checkpoint": "#d97706",
    "deliverable": "#fde68a",
    "dependency": "#fcd34d",
    # Orchestration
    "agent_task": "#f472b6",
    "debate": "#ec4899",
    "human_gate": "#f43f5e",
    "parallel_fan": "#fb7185",
    "merge": "#fda4af",
    "verification": "#e879f9",
}
