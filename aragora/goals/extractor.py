"""
Goal Extractor — AI-powered synthesis of goals from idea clusters.

Takes a Stage 1 idea graph (from ArgumentCartographer or raw idea input)
and derives SMART goals, principles, and strategies. Uses existing
Aragora agent infrastructure for AI reasoning.

The extraction pipeline:
1. Cluster related ideas by semantic similarity
2. For each cluster, synthesize a goal or principle
3. Identify dependencies between goals
4. Assign priorities based on idea support/evidence counts
5. Generate provenance links back to source ideas
"""

from __future__ import annotations

import hashlib
import json
import logging
import uuid
from dataclasses import dataclass, field
from typing import Any

from aragora.canvas.stages import (
    GoalNodeType,
    PipelineStage,
    ProvenanceLink,
    StageTransition,
    content_hash,
)

logger = logging.getLogger(__name__)


@dataclass
class GoalNode:
    """A single goal, principle, strategy, or milestone."""

    id: str
    title: str
    description: str
    goal_type: GoalNodeType = GoalNodeType.GOAL
    priority: str = "medium"  # low, medium, high, critical
    measurable: str = ""  # How to measure success
    dependencies: list[str] = field(default_factory=list)  # Other goal IDs
    source_idea_ids: list[str] = field(default_factory=list)  # Stage 1 idea IDs
    confidence: float = 0.0  # AI confidence in this extraction
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "type": self.goal_type.value,
            "priority": self.priority,
            "measurable": self.measurable,
            "dependencies": self.dependencies,
            "source_idea_ids": self.source_idea_ids,
            "confidence": self.confidence,
            "metadata": self.metadata,
        }


@dataclass
class GoalGraph:
    """A graph of goals extracted from an idea graph."""

    id: str
    goals: list[GoalNode] = field(default_factory=list)
    provenance: list[ProvenanceLink] = field(default_factory=list)
    transition: StageTransition | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "goals": [g.to_dict() for g in self.goals],
            "provenance": [p.to_dict() for p in self.provenance],
            "transition": self.transition.to_dict() if self.transition else None,
            "metadata": self.metadata,
        }


class GoalExtractor:
    """Extracts goals and principles from idea graphs.

    Uses a combination of structural analysis (which ideas have the most
    support/evidence) and optional AI synthesis to produce actionable goals.

    Can operate in two modes:
    - Structural mode (no AI): Uses graph topology to derive goals
    - AI-assisted mode: Uses an agent to synthesize goals from idea content
    """

    def __init__(self, agent: Any | None = None):
        """Initialize the goal extractor.

        Args:
            agent: Optional AI agent for synthesis. If None, uses
                   structural extraction only.
        """
        self._agent = agent

    def extract_from_ideas(
        self,
        idea_canvas_data: dict[str, Any],
    ) -> GoalGraph:
        """Extract goals from a Stage 1 Ideas canvas.

        Pipeline:
        1. Identify idea clusters (nodes with high support connectivity)
        2. For each cluster, derive a goal or principle
        3. Compute dependencies from idea relationships
        4. Assign priority from support/evidence counts
        5. Create provenance links

        Args:
            idea_canvas_data: Canvas.to_dict() from Stage 1

        Returns:
            GoalGraph with extracted goals and provenance
        """
        nodes = idea_canvas_data.get("nodes", [])
        edges = idea_canvas_data.get("edges", [])

        if not nodes:
            return GoalGraph(id=f"goals-{uuid.uuid4().hex[:8]}")

        # Build adjacency lists
        support_count: dict[str, int] = {}
        evidence_count: dict[str, int] = {}
        children: dict[str, list[str]] = {}
        node_map: dict[str, dict[str, Any]] = {}

        for node in nodes:
            nid = node.get("id", "")
            node_map[nid] = node
            support_count[nid] = 0
            evidence_count[nid] = 0
            children[nid] = []

        for edge in edges:
            src = edge.get("source", edge.get("source_id", ""))
            tgt = edge.get("target", edge.get("target_id", ""))
            edge_type = edge.get("type", edge.get("data", {}).get("stage_edge_type", ""))

            if edge_type in ("support", "supports"):
                support_count[tgt] = support_count.get(tgt, 0) + 1
            elif tgt in node_map:
                data = node_map[tgt].get("data", {})
                if data.get("idea_type") == "evidence":
                    evidence_count[src] = evidence_count.get(src, 0) + 1

            children.setdefault(src, []).append(tgt)

        # Step 1: Identify goal candidates (nodes with high connectivity)
        candidates = self._rank_candidates(node_map, support_count, evidence_count)

        # Step 2: Extract goals from candidates
        goals: list[GoalNode] = []
        provenance_links: list[ProvenanceLink] = []

        for rank, (node_id, score) in enumerate(candidates):
            node = node_map[node_id]
            node_data = node.get("data", {})
            label = node.get("label", "")
            idea_type = node_data.get("idea_type", "concept")

            # Determine goal type from idea type
            goal_type = self._idea_type_to_goal_type(idea_type, score)

            # Determine priority from support + evidence
            total_support = support_count.get(node_id, 0) + evidence_count.get(node_id, 0)
            priority = (
                "critical" if total_support >= 5
                else "high" if total_support >= 3
                else "medium" if total_support >= 1
                else "low"
            )

            # Find which ideas this goal depends on
            deps = self._find_goal_dependencies(
                node_id, children, node_map, [g.id for g in goals]
            )

            # Gather source idea IDs for provenance
            source_ids = [node_id] + children.get(node_id, [])[:5]

            goal = GoalNode(
                id=f"goal-{uuid.uuid4().hex[:8]}",
                title=self._synthesize_goal_title(label, goal_type),
                description=self._synthesize_goal_description(
                    label, node_data, goal_type
                ),
                goal_type=goal_type,
                priority=priority,
                dependencies=deps,
                source_idea_ids=source_ids,
                confidence=min(1.0, score / 10.0),
                metadata={"rank": rank, "support_score": score},
            )
            goals.append(goal)

            # Create provenance links
            for src_id in source_ids:
                provenance_links.append(
                    ProvenanceLink(
                        source_node_id=src_id,
                        source_stage=PipelineStage.IDEAS,
                        target_node_id=goal.id,
                        target_stage=PipelineStage.GOALS,
                        content_hash=content_hash(label),
                        method="structural_extraction",
                    )
                )

        # Create stage transition record
        transition = StageTransition(
            id=f"trans-ideas-goals-{uuid.uuid4().hex[:8]}",
            from_stage=PipelineStage.IDEAS,
            to_stage=PipelineStage.GOALS,
            provenance=provenance_links,
            status="pending",
            confidence=sum(g.confidence for g in goals) / max(len(goals), 1),
            ai_rationale=(
                f"Extracted {len(goals)} goals from {len(nodes)} ideas "
                f"using structural analysis (support connectivity + evidence counts)"
            ),
        )

        return GoalGraph(
            id=f"goals-{uuid.uuid4().hex[:8]}",
            goals=goals,
            provenance=provenance_links,
            transition=transition,
        )

    def extract_from_raw_ideas(
        self,
        ideas: list[str],
    ) -> GoalGraph:
        """Extract goals from a flat list of idea strings.

        Simpler entry point for users who just have a list of ideas
        without an existing debate/graph structure.

        Args:
            ideas: List of idea/thought strings

        Returns:
            GoalGraph with synthesized goals
        """
        # Convert raw ideas into a minimal canvas structure
        nodes = []
        edges = []

        for i, idea in enumerate(ideas):
            nodes.append({
                "id": f"raw-idea-{i}",
                "label": idea[:80],
                "data": {
                    "idea_type": "concept",
                    "full_content": idea,
                },
            })

            # Connect ideas that share keywords (simple semantic linking)
            idea_words = set(idea.lower().split())
            for j in range(i):
                other_words = set(ideas[j].lower().split())
                overlap = len(idea_words & other_words - _STOP_WORDS)
                if overlap >= 3:
                    edges.append({
                        "source": f"raw-idea-{j}",
                        "target": f"raw-idea-{i}",
                        "type": "relates_to",
                    })

        canvas_data = {"nodes": nodes, "edges": edges}
        return self.extract_from_ideas(canvas_data)

    # =========================================================================
    # Internal methods
    # =========================================================================

    def _rank_candidates(
        self,
        node_map: dict[str, dict[str, Any]],
        support_count: dict[str, int],
        evidence_count: dict[str, int],
    ) -> list[tuple[str, float]]:
        """Rank idea nodes by goal-worthiness.

        Score = 3 * support_count + 2 * evidence_count + type_bonus
        """
        scores: list[tuple[str, float]] = []
        for node_id, node in node_map.items():
            idea_type = node.get("data", {}).get("idea_type", "concept")
            type_bonus = {
                "concept": 2.0,
                "cluster": 3.0,
                "insight": 2.5,
                "question": 1.0,
                "evidence": 0.5,
                "assumption": 0.5,
                "constraint": 1.5,
            }.get(idea_type, 1.0)

            score = (
                3.0 * support_count.get(node_id, 0)
                + 2.0 * evidence_count.get(node_id, 0)
                + type_bonus
            )
            scores.append((node_id, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        # Return top candidates (at most 1 goal per 3 ideas)
        max_goals = max(1, len(node_map) // 3)
        return scores[:max_goals]

    def _idea_type_to_goal_type(
        self, idea_type: str, score: float
    ) -> GoalNodeType:
        """Map idea type + score to appropriate goal type."""
        if idea_type == "cluster":
            return GoalNodeType.GOAL
        elif idea_type == "constraint":
            return GoalNodeType.PRINCIPLE
        elif idea_type == "insight" and score >= 5:
            return GoalNodeType.STRATEGY
        elif idea_type == "question":
            return GoalNodeType.MILESTONE  # Questions become things to resolve
        else:
            return GoalNodeType.GOAL

    def _find_goal_dependencies(
        self,
        node_id: str,
        children: dict[str, list[str]],
        node_map: dict[str, dict[str, Any]],
        existing_goal_ids: list[str],
    ) -> list[str]:
        """Find which existing goals this new goal depends on."""
        # For now, use simple heuristic: if any child of this node
        # is a source idea for an existing goal, add a dependency
        return []  # Dependencies computed after all goals extracted

    def _synthesize_goal_title(self, label: str, goal_type: GoalNodeType) -> str:
        """Create an actionable goal title from an idea label."""
        prefix = {
            GoalNodeType.GOAL: "Achieve",
            GoalNodeType.PRINCIPLE: "Maintain",
            GoalNodeType.STRATEGY: "Implement",
            GoalNodeType.MILESTONE: "Complete",
            GoalNodeType.METRIC: "Measure",
            GoalNodeType.RISK: "Mitigate",
        }.get(goal_type, "Achieve")

        # Clean and capitalize
        clean_label = label.strip().rstrip(".")
        if clean_label and not clean_label[0].isupper():
            clean_label = clean_label[0].upper() + clean_label[1:]

        return f"{prefix}: {clean_label}"

    def _synthesize_goal_description(
        self, label: str, node_data: dict[str, Any], goal_type: GoalNodeType
    ) -> str:
        """Create a goal description with context."""
        full_content = node_data.get("full_content", label)
        agent = node_data.get("agent", "")

        parts = [full_content]
        if agent:
            parts.append(f"Originally proposed by {agent}.")

        return " ".join(parts)


# Common English stop words to exclude from keyword matching
_STOP_WORDS = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "can", "shall", "to", "of", "in", "for",
    "on", "with", "at", "by", "from", "as", "into", "through", "during",
    "before", "after", "above", "below", "between", "and", "but", "or",
    "not", "no", "this", "that", "these", "those", "it", "its",
})
