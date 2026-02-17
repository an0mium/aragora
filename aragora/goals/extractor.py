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
import re
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from aragora.canvas.stages import (
    GoalNodeType,
    PipelineStage,
    ProvenanceLink,
    StageTransition,
    content_hash,
)

if TYPE_CHECKING:
    from aragora.reasoning.belief import PropagationResult

logger = logging.getLogger(__name__)


@dataclass
class GoalExtractionConfig:
    """Configuration for debate-analysis-based goal extraction."""

    confidence_threshold: float = 0.6
    max_goals: int = 10
    require_consensus: bool = True
    smart_scoring: bool = True
    min_centrality: float = 0.0


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

    def to_prioritized_goal(self, track: str = "core") -> dict[str, Any]:
        """Convert to MetaPlanner-compatible PrioritizedGoal dict."""
        priority_score = {
            "critical": 1.0,
            "high": 0.75,
            "medium": 0.5,
            "low": 0.25,
        }.get(self.priority, 0.5)

        return {
            "goal": self.title,
            "description": self.description,
            "priority": priority_score,
            "track": track,
            "confidence": self.confidence,
            "source_goal_id": self.id,
            "metadata": {
                "goal_type": self.goal_type.value,
                "measurable": self.measurable,
                "source_idea_ids": self.source_idea_ids,
            },
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

        If an AI agent is available, uses AI synthesis for richer goals.
        Falls back to structural extraction if agent is None or fails.

        Args:
            idea_canvas_data: Canvas.to_dict() from Stage 1

        Returns:
            GoalGraph with extracted goals and provenance
        """
        # Try AI-assisted extraction first if agent is available
        if self._agent is not None:
            try:
                result = self._extract_with_ai(idea_canvas_data)
                if result and result.goals:
                    return result
            except Exception:
                logger.warning("AI goal synthesis failed, falling back to structural")

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

    def extract_from_debate_analysis(
        self,
        cartographer_output: dict[str, Any],
        belief_result: PropagationResult | None = None,
        config: GoalExtractionConfig | None = None,
    ) -> GoalGraph:
        """Extract goals from ArgumentCartographer + BeliefNetwork output.

        Bridges debate analysis (argument mapping + belief propagation) into
        actionable goals by finding consensus/vote nodes and ranking by
        centrality from the belief network.

        Args:
            cartographer_output: ArgumentCartographer.to_dict() output
            belief_result: Optional PropagationResult from BeliefNetwork
            config: Extraction configuration

        Returns:
            GoalGraph with debate-derived goals
        """
        cfg = config or GoalExtractionConfig()
        nodes = cartographer_output.get("nodes", [])

        if not nodes:
            return GoalGraph(id=f"goals-{uuid.uuid4().hex[:8]}")

        # Build centrality lookup from belief result
        centralities: dict[str, float] = {}
        if belief_result is not None:
            centralities = getattr(belief_result, "centralities", None) or {}

        # Step 1: Find consensus/vote nodes (high-signal debate outcomes)
        candidates: list[tuple[dict[str, Any], float]] = []
        for node in nodes:
            node_type = node.get("node_type", node.get("type", ""))
            node_id = node.get("id", "")

            # Accept consensus, vote, and claim nodes
            if cfg.require_consensus and node_type not in (
                "consensus", "vote", "claim", "synthesis",
            ):
                continue

            # Base score from node attributes
            score = node.get("weight", 0.5)
            if isinstance(node.get("data"), dict):
                score = max(score, node["data"].get("confidence", 0.0))

            # Cross-reference with belief centralities
            centrality = centralities.get(node_id, 0.0)
            if centrality < cfg.min_centrality:
                continue
            score = score * 0.6 + centrality * 0.4 if centralities else score

            # Filter by confidence threshold
            if score < cfg.confidence_threshold:
                continue

            candidates.append((node, score))

        # If require_consensus yielded nothing, fall back to all nodes
        if not candidates and cfg.require_consensus:
            return self.extract_from_debate_analysis(
                cartographer_output,
                belief_result,
                GoalExtractionConfig(
                    confidence_threshold=cfg.confidence_threshold,
                    max_goals=cfg.max_goals,
                    require_consensus=False,
                    smart_scoring=cfg.smart_scoring,
                    min_centrality=cfg.min_centrality,
                ),
            )

        # Sort by score descending and limit
        candidates.sort(key=lambda x: x[1], reverse=True)
        candidates = candidates[: cfg.max_goals]

        # Step 2: Create goals from candidates
        goals: list[GoalNode] = []
        provenance_links: list[ProvenanceLink] = []

        for rank, (node, score) in enumerate(candidates):
            node_id = node.get("id", "")
            label = node.get("label", node.get("content", ""))
            node_type = node.get("node_type", node.get("type", ""))

            # Map debate node types to goal types
            goal_type = {
                "consensus": GoalNodeType.GOAL,
                "vote": GoalNodeType.MILESTONE,
                "claim": GoalNodeType.STRATEGY,
                "synthesis": GoalNodeType.GOAL,
            }.get(node_type, GoalNodeType.GOAL)

            # Priority from score
            priority = (
                "critical" if score >= 0.9
                else "high" if score >= 0.7
                else "medium" if score >= 0.4
                else "low"
            )

            # SMART scoring
            smart_meta: dict[str, Any] = {"rank": rank, "debate_score": score}
            measurable = ""
            if cfg.smart_scoring:
                specificity = _score_specificity(label)
                measurability = _score_measurability(label)
                smart_meta["specificity"] = specificity
                smart_meta["measurability"] = measurability
                if measurability > 0.5:
                    measurable = f"Score: {measurability:.1f} (auto-detected metrics)"

            goal = GoalNode(
                id=f"goal-{uuid.uuid4().hex[:8]}",
                title=self._synthesize_goal_title(label, goal_type),
                description=label,
                goal_type=goal_type,
                priority=priority,
                measurable=measurable,
                source_idea_ids=[node_id],
                confidence=min(1.0, score),
                metadata=smart_meta,
            )
            goals.append(goal)

            provenance_links.append(
                ProvenanceLink(
                    source_node_id=node_id,
                    source_stage=PipelineStage.IDEAS,
                    target_node_id=goal.id,
                    target_stage=PipelineStage.GOALS,
                    content_hash=content_hash(label),
                    method="debate_analysis",
                )
            )

        transition = StageTransition(
            id=f"trans-debate-goals-{uuid.uuid4().hex[:8]}",
            from_stage=PipelineStage.IDEAS,
            to_stage=PipelineStage.GOALS,
            provenance=provenance_links,
            status="pending",
            confidence=sum(g.confidence for g in goals) / max(len(goals), 1),
            ai_rationale=(
                f"Extracted {len(goals)} goals from debate analysis "
                f"({len(nodes)} argument nodes, "
                f"{'with' if belief_result else 'without'} belief propagation)"
            ),
        )

        return GoalGraph(
            id=f"goals-{uuid.uuid4().hex[:8]}",
            goals=goals,
            provenance=provenance_links,
            transition=transition,
        )

    # =========================================================================
    # AI-assisted extraction
    # =========================================================================

    def _extract_with_ai(
        self,
        idea_canvas_data: dict[str, Any],
    ) -> GoalGraph | None:
        """Use an AI agent to synthesize goals from idea clusters.

        Prompts the agent with idea summaries and asks it to derive
        SMART goals with priorities, dependencies, and measurability.

        Returns None if the agent response cannot be parsed.
        """
        nodes = idea_canvas_data.get("nodes", [])
        if not nodes:
            return None

        # Build a summary of ideas for the prompt
        idea_summaries = []
        for node in nodes:
            label = node.get("label", "")
            data = node.get("data", {})
            idea_type = data.get("idea_type", "concept")
            full_content = data.get("full_content", label)
            idea_summaries.append(
                f"- [{idea_type}] {full_content}"
            )

        prompt = (
            "Given these ideas from a structured brainstorming session, "
            "synthesize SMART goals (Specific, Measurable, Achievable, "
            "Relevant, Time-bound).\n\n"
            "Ideas:\n" + "\n".join(idea_summaries) + "\n\n"
            "For each goal, provide:\n"
            "1. title: An actionable goal title starting with a verb\n"
            "2. description: One-sentence explanation\n"
            "3. type: One of goal/principle/strategy/milestone/metric/risk\n"
            "4. priority: One of critical/high/medium/low\n"
            "5. measurable: How to measure success\n"
            "6. source_ideas: Which idea numbers (0-indexed) this derives from\n\n"
            "Return as JSON array of objects. Return ONLY the JSON array."
        )

        try:
            # Use agent's generate method (standard Aragora agent interface)
            if hasattr(self._agent, "generate"):
                response = self._agent.generate(prompt)
            elif hasattr(self._agent, "complete"):
                response = self._agent.complete(prompt)
            elif callable(self._agent):
                response = self._agent(prompt)
            else:
                logger.warning("Agent has no generate/complete/callable interface")
                return None

            # Extract text from response
            text = response if isinstance(response, str) else str(response)

            # Parse JSON from response
            parsed = self._parse_ai_goals(text, nodes)
            return parsed
        except Exception as e:
            logger.warning("AI goal extraction failed: %s", e)
            return None

    def _parse_ai_goals(
        self,
        response_text: str,
        source_nodes: list[dict[str, Any]],
    ) -> GoalGraph | None:
        """Parse AI response into a GoalGraph."""
        # Try to extract JSON array from response
        text = response_text.strip()
        # Find JSON array boundaries
        start = text.find("[")
        end = text.rfind("]")
        if start == -1 or end == -1:
            return None

        try:
            goal_dicts = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return None

        if not isinstance(goal_dicts, list) or not goal_dicts:
            return None

        goals: list[GoalNode] = []
        provenance_links: list[ProvenanceLink] = []

        for i, gd in enumerate(goal_dicts):
            if not isinstance(gd, dict):
                continue

            goal_type_str = gd.get("type", "goal")
            try:
                goal_type = GoalNodeType(goal_type_str)
            except ValueError:
                goal_type = GoalNodeType.GOAL

            # Map source_ideas indices to node IDs
            source_indices = gd.get("source_ideas", [])
            source_ids = []
            for idx in source_indices:
                if isinstance(idx, int) and 0 <= idx < len(source_nodes):
                    source_ids.append(source_nodes[idx].get("id", f"idea-{idx}"))

            goal = GoalNode(
                id=f"goal-ai-{uuid.uuid4().hex[:8]}",
                title=gd.get("title", f"Goal {i + 1}"),
                description=gd.get("description", ""),
                goal_type=goal_type,
                priority=gd.get("priority", "medium"),
                measurable=gd.get("measurable", ""),
                source_idea_ids=source_ids,
                confidence=0.8,  # AI-generated goals get 0.8 baseline
                metadata={"extraction_method": "ai_synthesis"},
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
                        content_hash=content_hash(goal.title),
                        method="ai_synthesis",
                    )
                )

        if not goals:
            return None

        transition = StageTransition(
            id=f"trans-ideas-goals-ai-{uuid.uuid4().hex[:8]}",
            from_stage=PipelineStage.IDEAS,
            to_stage=PipelineStage.GOALS,
            provenance=provenance_links,
            status="pending",
            confidence=0.8,
            ai_rationale=(
                f"AI synthesized {len(goals)} SMART goals from "
                f"{len(source_nodes)} ideas"
            ),
        )

        return GoalGraph(
            id=f"goals-ai-{uuid.uuid4().hex[:8]}",
            goals=goals,
            provenance=provenance_links,
            transition=transition,
            metadata={"extraction_method": "ai_synthesis"},
        )

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


# =========================================================================
# SMART scoring helpers
# =========================================================================

_SPECIFICITY_PATTERNS = [
    re.compile(r"\b\d+%?\b"),  # Numbers/percentages
    re.compile(r"\b(implement|build|create|deploy|add|remove|fix|update)\b", re.I),
    re.compile(r"\b(api|database|server|client|module|service|endpoint)\b", re.I),
    re.compile(r"\b(by|within|before|after|during)\s+\w+", re.I),
]

_MEASURABILITY_PATTERNS = [
    re.compile(r"\b\d+\s*%", re.I),  # Percentages
    re.compile(r"\b(reduce|increase|improve|decrease|achieve)\s+\w+\s+by\b", re.I),
    re.compile(r"\b(metric|measure|kpi|target|benchmark|score|rate)\b", re.I),
    re.compile(r"\b(latency|throughput|uptime|coverage|accuracy)\b", re.I),
    re.compile(r"\b\d+\s*(ms|seconds?|minutes?|hours?|days?)\b", re.I),
]


def _score_specificity(text: str) -> float:
    """Score how specific/actionable a text is (0.0 to 1.0)."""
    if not text:
        return 0.0
    matches = sum(1 for p in _SPECIFICITY_PATTERNS if p.search(text))
    return min(1.0, matches / len(_SPECIFICITY_PATTERNS))


def _score_measurability(text: str) -> float:
    """Score how measurable a text is (0.0 to 1.0)."""
    if not text:
        return 0.0
    matches = sum(1 for p in _MEASURABILITY_PATTERNS if p.search(text))
    return min(1.0, matches / len(_MEASURABILITY_PATTERNS))


# Common English stop words to exclude from keyword matching
_STOP_WORDS = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "can", "shall", "to", "of", "in", "for",
    "on", "with", "at", "by", "from", "as", "into", "through", "during",
    "before", "after", "above", "below", "between", "and", "but", "or",
    "not", "no", "this", "that", "these", "those", "it", "its",
})
