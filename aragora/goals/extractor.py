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

    Supports domain-specific extraction via the ``domain`` parameter, which
    injects vertical-specific goals (e.g., HIPAA compliance for healthcare).
    """

    def __init__(self, agent: Any | None = None, domain: str | None = None):
        """Initialize the goal extractor.

        Args:
            agent: Optional AI agent for synthesis. If None, uses
                   structural extraction only.
            domain: Optional vertical domain for domain-specific goals.
                    Supported: "healthcare", "financial", "legal"
        """
        self._agent = agent
        self._domain = domain

    def extract_from_principles(
        self,
        principles_canvas_data: dict[str, Any],
        *,
        ideas_canvas_data: dict[str, Any] | None = None,
    ) -> GoalGraph:
        """Extract goals informed by principles/values extracted from ideas.

        Goals are derived from principles rather than raw ideas, ensuring
        they align with identified values and priorities.

        Args:
            principles_canvas_data: Canvas data from the Principles stage
            ideas_canvas_data: Optional original ideas canvas for provenance

        Returns:
            GoalGraph with principle-derived goals
        """
        nodes = principles_canvas_data.get("nodes", [])
        edges = principles_canvas_data.get("edges", [])

        if not nodes:
            return GoalGraph(id=f"goals-{uuid.uuid4().hex[:8]}")

        # Separate principle types
        goals: list[GoalNode] = []
        provenance_links: list[ProvenanceLink] = []

        # Build constraint map for dependency tracking
        constraint_targets: dict[str, list[str]] = {}
        for edge in edges:
            edge_data = edge.get("data", {})
            etype = edge_data.get("stage_edge_type", "")
            if etype == "constrains":
                src = edge.get("source", edge.get("source_id", ""))
                tgt = edge.get("target", edge.get("target_id", ""))
                constraint_targets.setdefault(tgt, []).append(src)

        # Group nodes by theme (via edges)
        theme_members: dict[str, list[dict[str, Any]]] = {}
        node_theme: dict[str, str] = {}
        for edge in edges:
            edge_data = edge.get("data", {})
            etype = edge_data.get("stage_edge_type", "")
            if etype == "relates_to":
                src = edge.get("source", edge.get("source_id", ""))
                tgt = edge.get("target", edge.get("target_id", ""))
                node_theme[src] = tgt
                theme_members.setdefault(tgt, [])

        # Process each principle/value/priority node
        existing_goal_ids: list[str] = []
        for node in nodes:
            data = node.get("data", {})
            p_type = data.get("principle_type", "")
            if p_type not in ("principle", "value", "priority", "constraint"):
                continue

            label = node.get("label", "")
            node_id = node.get("id", "")
            source_idea_id = data.get("source_idea_id", "")

            # Map principle type to goal type
            goal_type = {
                "priority": GoalNodeType.GOAL,
                "value": GoalNodeType.PRINCIPLE,
                "principle": GoalNodeType.STRATEGY,
                "constraint": GoalNodeType.RISK,
            }.get(p_type, GoalNodeType.GOAL)

            # Priority from principle type
            priority = {
                "priority": "high",
                "constraint": "high",
                "value": "medium",
                "principle": "medium",
            }.get(p_type, "medium")

            # Dependencies from constraints
            deps = [
                gid for gid in existing_goal_ids
                if node_id in constraint_targets.get(gid, [])
            ]

            goal = GoalNode(
                id=f"goal-{uuid.uuid4().hex[:8]}",
                title=self._synthesize_goal_title(label, goal_type),
                description=self._synthesize_goal_description(
                    label, data, goal_type
                ),
                goal_type=goal_type,
                priority=priority,
                dependencies=deps,
                source_idea_ids=[source_idea_id] if source_idea_id else [node_id],
                confidence=0.75,
                metadata={
                    "source_principle_type": p_type,
                    "source_principle_id": node_id,
                    "theme": node_theme.get(node_id, ""),
                },
            )
            goals.append(goal)
            existing_goal_ids.append(goal.id)

            # Provenance: principle → goal
            provenance_links.append(
                ProvenanceLink(
                    source_node_id=node_id,
                    source_stage=PipelineStage.PRINCIPLES,
                    target_node_id=goal.id,
                    target_stage=PipelineStage.GOALS,
                    content_hash=content_hash(label),
                    method="principle_extraction",
                )
            )

            # Provenance: original idea → goal (if available)
            if source_idea_id:
                provenance_links.append(
                    ProvenanceLink(
                        source_node_id=source_idea_id,
                        source_stage=PipelineStage.IDEAS,
                        target_node_id=goal.id,
                        target_stage=PipelineStage.GOALS,
                        content_hash=content_hash(label),
                        method="principle_extraction",
                    )
                )

        transition = StageTransition(
            id=f"trans-principles-goals-{uuid.uuid4().hex[:8]}",
            from_stage=PipelineStage.PRINCIPLES,
            to_stage=PipelineStage.GOALS,
            provenance=provenance_links,
            status="pending",
            confidence=sum(g.confidence for g in goals) / max(len(goals), 1),
            ai_rationale=(
                f"Extracted {len(goals)} goals from {len(nodes)} principle nodes"
            ),
        )

        return GoalGraph(
            id=f"goals-{uuid.uuid4().hex[:8]}",
            goals=goals,
            provenance=provenance_links,
            transition=transition,
            metadata={"extraction_method": "from_principles"},
        )

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
            except (RuntimeError, ValueError, TypeError, AttributeError) as exc:
                logger.warning(
                    "AI goal synthesis failed, falling back to structural: %s",
                    exc,
                    exc_info=True,
                )

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
                "critical"
                if total_support >= 5
                else "high"
                if total_support >= 3
                else "medium"
                if total_support >= 1
                else "low"
            )

            # Find which ideas this goal depends on
            deps = self._find_goal_dependencies(node_id, children, node_map, [g.id for g in goals])

            # Gather source idea IDs for provenance
            source_ids = [node_id] + children.get(node_id, [])[:5]

            goal = GoalNode(
                id=f"goal-{uuid.uuid4().hex[:8]}",
                title=self._synthesize_goal_title(label, goal_type),
                description=self._synthesize_goal_description(label, node_data, goal_type),
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

        # Inject domain-specific goals if a vertical domain is set
        if self._domain:
            domain_goals = _get_domain_goals(self._domain)
            for dg in domain_goals:
                dg_node = GoalNode(
                    id=f"goal-domain-{uuid.uuid4().hex[:8]}",
                    title=dg["title"],
                    description=dg["description"],
                    goal_type=GoalNodeType(dg.get("type", "goal")),
                    priority=dg.get("priority", "high"),
                    measurable=dg.get("measurable", ""),
                    confidence=0.9,
                    metadata={"domain": self._domain, "domain_injected": True},
                )
                goals.append(dg_node)

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
            nodes.append(
                {
                    "id": f"raw-idea-{i}",
                    "label": idea[:80],
                    "data": {
                        "idea_type": "concept",
                        "full_content": idea,
                    },
                }
            )

            # Connect ideas that share keywords (simple semantic linking)
            idea_words = set(idea.lower().split())
            for j in range(i):
                other_words = set(ideas[j].lower().split())
                overlap = len(idea_words & other_words - _STOP_WORDS)
                if overlap >= 3:
                    edges.append(
                        {
                            "source": f"raw-idea-{j}",
                            "target": f"raw-idea-{i}",
                            "type": "relates_to",
                        }
                    )

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
        edges = cartographer_output.get("edges", [])

        if not nodes:
            return GoalGraph(id=f"goals-{uuid.uuid4().hex[:8]}")

        # Build centrality lookup from belief result
        centralities: dict[str, float] = {}
        if belief_result is not None:
            centralities = getattr(belief_result, "centralities", None) or {}

        # Build edge topology scores: support/refute tallies per node
        in_support: dict[str, float] = {}
        in_refute: dict[str, float] = {}
        node_supporters: dict[str, list[str]] = {}
        node_refuters: dict[str, list[str]] = {}
        for edge in edges:
            src = edge.get("source", edge.get("source_id", ""))
            tgt = edge.get("target", edge.get("target_id", ""))
            rel = edge.get("type", edge.get("relation", "")).lower()
            weight = float(edge.get("weight", edge.get("strength", 1.0)))
            if rel in ("supports", "support"):
                in_support[tgt] = in_support.get(tgt, 0.0) + weight
                node_supporters.setdefault(tgt, []).append(src)
            elif rel in ("contradicts", "refutes", "refute"):
                in_refute[tgt] = in_refute.get(tgt, 0.0) + weight
                node_refuters.setdefault(tgt, []).append(src)

        # Step 1: Find consensus/vote nodes (high-signal debate outcomes)
        candidates: list[tuple[dict[str, Any], float]] = []
        for node in nodes:
            node_type = node.get("node_type", node.get("type", ""))
            node_id = node.get("id", "")

            # Accept consensus, vote, and claim nodes
            if cfg.require_consensus and node_type not in (
                "consensus",
                "vote",
                "claim",
                "synthesis",
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

            # Integrate edge topology: support boosts, refutation dampens
            support_total = in_support.get(node_id, 0.0)
            refute_total = in_refute.get(node_id, 0.0)
            edge_score = (support_total - refute_total * 0.5) * 0.1
            score += edge_score

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
                "critical"
                if score >= 0.9
                else "high"
                if score >= 0.7
                else "medium"
                if score >= 0.4
                else "low"
            )

            # SMART scoring
            smart_meta: dict[str, Any] = {
                "rank": rank,
                "debate_score": score,
                "support_edges": len(node_supporters.get(node_id, [])),
                "refute_edges": len(node_refuters.get(node_id, [])),
                "has_refutation": node_id in node_refuters,
            }
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
            idea_summaries.append(f"- [{idea_type}] {full_content}")

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
        except (RuntimeError, ValueError, TypeError, OSError, AttributeError, KeyError, ConnectionError, TimeoutError) as e:
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
                f"AI synthesized {len(goals)} SMART goals from {len(source_nodes)} ideas"
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

    def _idea_type_to_goal_type(self, idea_type: str, score: float) -> GoalNodeType:
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
    # Semantic clustering (2A)
    # =========================================================================

    def cluster_ideas_semantically(
        self,
        idea_canvas_data: dict[str, Any],
        similarity_threshold: float = 0.2,
        min_cluster_size: int = 2,
    ) -> dict[str, Any]:
        """Cluster ideas by semantic similarity using word overlap.

        Uses Jaccard similarity with stopword removal to find related ideas,
        then applies agglomerative clustering to group them.

        Args:
            idea_canvas_data: Canvas.to_dict() from Stage 1
            similarity_threshold: Minimum Jaccard similarity to link ideas (0.0-1.0)
            min_cluster_size: Minimum number of ideas to form a cluster

        Returns:
            Enriched canvas data with cluster nodes and membership edges added
        """
        import copy

        result = copy.deepcopy(idea_canvas_data)
        nodes = result.get("nodes", [])
        edges = result.get("edges", [])

        if len(nodes) < min_cluster_size:
            return result

        # Extract tokenized text for each node
        node_tokens: list[tuple[str, frozenset[str]]] = []
        for node in nodes:
            nid = node.get("id", "")
            label = node.get("label", "")
            full_content = node.get("data", {}).get("full_content", "")
            text = f"{label} {full_content}"
            tokens = _tokenize(text)
            node_tokens.append((nid, tokens))

        # Build pairwise similarity matrix
        n = len(node_tokens)
        similarity: dict[tuple[int, int], float] = {}
        for i in range(n):
            for j in range(i + 1, n):
                sim = _jaccard_similarity(node_tokens[i][1], node_tokens[j][1])
                similarity[(i, j)] = sim

        # Agglomerative clustering: each node starts as its own cluster
        # cluster_id -> set of node indices
        clusters: dict[int, set[int]] = {i: {i} for i in range(n)}

        while True:
            # Find the most similar pair of clusters
            best_sim = -1.0
            best_pair: tuple[int, int] | None = None

            cluster_ids = sorted(clusters.keys())
            for ci_idx in range(len(cluster_ids)):
                for cj_idx in range(ci_idx + 1, len(cluster_ids)):
                    ci = cluster_ids[ci_idx]
                    cj = cluster_ids[cj_idx]
                    # Average linkage: mean similarity between all pairs
                    total = 0.0
                    count = 0
                    for ni in clusters[ci]:
                        for nj in clusters[cj]:
                            key = (min(ni, nj), max(ni, nj))
                            total += similarity.get(key, 0.0)
                            count += 1
                    avg_sim = total / max(count, 1)
                    if avg_sim > best_sim:
                        best_sim = avg_sim
                        best_pair = (ci, cj)

            if best_pair is None or best_sim < similarity_threshold:
                break

            # Merge the two clusters
            ci, cj = best_pair
            clusters[ci] = clusters[ci] | clusters[cj]
            del clusters[cj]

        # Create cluster nodes for clusters meeting min_cluster_size
        for cluster_id, member_indices in clusters.items():
            if len(member_indices) < min_cluster_size:
                continue

            # Generate cluster name from top terms
            cluster_name = _name_cluster([node_tokens[i][1] for i in member_indices])
            cluster_node_id = f"cluster-{uuid.uuid4().hex[:8]}"

            # Add cluster node
            nodes.append(
                {
                    "id": cluster_node_id,
                    "label": cluster_name,
                    "data": {
                        "idea_type": "cluster",
                        "member_count": len(member_indices),
                        "auto_generated": True,
                    },
                }
            )

            # Add membership edges
            for idx in member_indices:
                member_id = node_tokens[idx][0]
                edges.append(
                    {
                        "source": member_id,
                        "target": cluster_node_id,
                        "type": "member_of",
                    }
                )

        result["nodes"] = nodes
        result["edges"] = edges
        return result

    # =========================================================================
    # Goal conflict detection (2B)
    # =========================================================================

    def detect_goal_conflicts(
        self,
        goal_graph: GoalGraph,
    ) -> list[dict[str, Any]]:
        """Detect conflicts between goals.

        Checks for:
        1. Contradictory keywords (maximize vs minimize, increase vs decrease)
        2. Circular dependencies in goal graph
        3. Near-duplicate detection (goals with very similar titles)

        Args:
            goal_graph: GoalGraph to analyze

        Returns:
            List of conflict dicts with keys: type, severity, goal_ids,
            description, suggestion
        """
        conflicts: list[dict[str, Any]] = []

        goals = goal_graph.goals
        if not goals:
            return conflicts

        # 1. Contradictory keywords
        conflicts.extend(self._detect_contradictions(goals))

        # 2. Circular dependencies
        conflicts.extend(self._detect_circular_dependencies(goals))

        # 3. Near-duplicate detection
        conflicts.extend(self._detect_near_duplicates(goals))

        return conflicts

    def _detect_contradictions(self, goals: list[GoalNode]) -> list[dict[str, Any]]:
        """Find goals with contradictory action keywords."""
        conflicts: list[dict[str, Any]] = []

        for i in range(len(goals)):
            text_i = f"{goals[i].title} {goals[i].description}".lower()
            words_i = set(text_i.split())
            for j in range(i + 1, len(goals)):
                text_j = f"{goals[j].title} {goals[j].description}".lower()
                words_j = set(text_j.split())

                for word_a, word_b in _CONTRADICTORY_PAIRS:
                    if (word_a in words_i and word_b in words_j) or (
                        word_b in words_i and word_a in words_j
                    ):
                        conflicts.append(
                            {
                                "type": "contradiction",
                                "severity": "high",
                                "goal_ids": [goals[i].id, goals[j].id],
                                "description": (
                                    f"Goals may conflict: '{goals[i].title}' vs "
                                    f"'{goals[j].title}' (contradictory terms: "
                                    f"{word_a}/{word_b})"
                                ),
                                "suggestion": (
                                    "Review these goals to ensure they are not "
                                    "working against each other. Consider merging "
                                    "or prioritizing one over the other."
                                ),
                            }
                        )
                        break  # One contradiction per pair is enough

        return conflicts

    def _detect_circular_dependencies(self, goals: list[GoalNode]) -> list[dict[str, Any]]:
        """Detect cycles in the goal dependency graph using DFS."""
        conflicts: list[dict[str, Any]] = []

        # Build adjacency: goal_id -> list of dependency goal_ids
        dep_map: dict[str, list[str]] = {}
        goal_ids = {g.id for g in goals}
        for g in goals:
            # Only include deps that reference actual goals in this graph
            dep_map[g.id] = [d for d in g.dependencies if d in goal_ids]

        # DFS cycle detection
        WHITE, GRAY, BLACK = 0, 1, 2
        color: dict[str, int] = {gid: WHITE for gid in goal_ids}
        {gid: None for gid in goal_ids}

        def _dfs(node: str, path: list[str]) -> list[str] | None:
            """Return cycle path if found, else None."""
            color[node] = GRAY
            path.append(node)
            for dep in dep_map.get(node, []):
                if color.get(dep) == GRAY:
                    # Found a cycle — extract it
                    cycle_start = path.index(dep)
                    return path[cycle_start:]
                if color.get(dep) == WHITE:
                    result = _dfs(dep, path)
                    if result is not None:
                        return result
            path.pop()
            color[node] = BLACK
            return None

        seen_cycles: set[frozenset[str]] = set()
        for gid in goal_ids:
            if color[gid] == WHITE:
                cycle = _dfs(gid, [])
                if cycle is not None:
                    cycle_key = frozenset(cycle)
                    if cycle_key not in seen_cycles:
                        seen_cycles.add(cycle_key)
                        # Build goal title lookup
                        title_map = {g.id: g.title for g in goals}
                        cycle_desc = " -> ".join(title_map.get(c, c) for c in cycle)
                        conflicts.append(
                            {
                                "type": "circular_dependency",
                                "severity": "high",
                                "goal_ids": list(cycle),
                                "description": (f"Circular dependency detected: {cycle_desc}"),
                                "suggestion": (
                                    "Break the dependency cycle by removing at "
                                    "least one dependency link or reordering goals."
                                ),
                            }
                        )
                    # Reset colors for remaining nodes to find more cycles
                    for node_id in goal_ids:
                        if color[node_id] == GRAY:
                            color[node_id] = WHITE

        return conflicts

    def _detect_near_duplicates(self, goals: list[GoalNode]) -> list[dict[str, Any]]:
        """Find goals with very similar titles (potential duplicates)."""
        conflicts: list[dict[str, Any]] = []

        for i in range(len(goals)):
            tokens_i = _tokenize(goals[i].title)
            for j in range(i + 1, len(goals)):
                tokens_j = _tokenize(goals[j].title)
                sim = _jaccard_similarity(tokens_i, tokens_j)
                if sim > 0.6:
                    conflicts.append(
                        {
                            "type": "near_duplicate",
                            "severity": "medium",
                            "goal_ids": [goals[i].id, goals[j].id],
                            "description": (
                                f"Goals appear very similar "
                                f"(similarity: {sim:.1%}): "
                                f"'{goals[i].title}' vs '{goals[j].title}'"
                            ),
                            "suggestion": (
                                "Consider merging these goals into a single, "
                                "more comprehensive goal."
                            ),
                        }
                    )

        return conflicts

    # =========================================================================
    # Enhanced SMART scoring (2C)
    # =========================================================================

    def score_smart(self, goal: GoalNode) -> dict[str, float]:
        """Score a goal on all SMART dimensions.

        - Specific: Technical details and scope constraints
        - Measurable: Quantitative success criteria
        - Achievable: Scope-limited, concrete deliverable
        - Relevant: References domain-specific terms
        - Time-bound: Has timeline or deadline references

        Returns:
            Dict with keys: specific, measurable, achievable, relevant,
            time_bound, overall
        """
        text = f"{goal.title} {goal.description}"

        specific = _score_specificity(text)
        measurable = _score_measurability(text)
        achievable = _score_achievability(text)
        relevant = _score_relevance(text, goal.source_idea_ids)
        time_bound = _score_time_bound(text)

        overall = (
            0.25 * specific
            + 0.25 * measurable
            + 0.20 * achievable
            + 0.15 * relevant
            + 0.15 * time_bound
        )

        return {
            "specific": specific,
            "measurable": measurable,
            "achievable": achievable,
            "relevant": relevant,
            "time_bound": time_bound,
            "overall": overall,
        }

    def suggest_improvements(self, goal: GoalNode) -> list[str]:
        """Suggest improvements for low-scoring SMART dimensions.

        Analyzes each SMART dimension and provides actionable advice
        for dimensions scoring below 0.5.

        Args:
            goal: GoalNode to analyze

        Returns:
            List of improvement suggestion strings
        """
        scores = self.score_smart(goal)
        suggestions: list[str] = []
        if scores["specific"] < 0.5:
            suggestions.append("Add specific technical details or scope constraints")
        if scores["measurable"] < 0.5:
            suggestions.append(
                "Add quantitative success criteria (e.g., percentages, counts, response times)"
            )
        if scores["achievable"] < 0.5:
            suggestions.append("Narrow the scope to a concrete, single-step deliverable")
        if scores["time_bound"] < 0.5:
            suggestions.append("Add a timeline or deadline (e.g., 'within 2 sprints')")
        return suggestions


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
_STOP_WORDS = frozenset(
    {
        "the",
        "a",
        "an",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "could",
        "should",
        "may",
        "might",
        "can",
        "shall",
        "to",
        "of",
        "in",
        "for",
        "on",
        "with",
        "at",
        "by",
        "from",
        "as",
        "into",
        "through",
        "during",
        "before",
        "after",
        "above",
        "below",
        "between",
        "and",
        "but",
        "or",
        "not",
        "no",
        "this",
        "that",
        "these",
        "those",
        "it",
        "its",
    }
)

# Contradictory keyword pairs for conflict detection
_CONTRADICTORY_PAIRS: list[tuple[str, str]] = [
    ("maximize", "minimize"),
    ("increase", "decrease"),
    ("add", "remove"),
    ("enable", "disable"),
    ("expand", "reduce"),
    ("accelerate", "decelerate"),
    ("centralize", "decentralize"),
    ("simplify", "complicate"),
    ("automate", "manual"),
    ("open", "close"),
]

# Achievability patterns — scope-limiting words score higher
_ACHIEVABILITY_PATTERNS = [
    re.compile(r"\b(single|one|specific|particular|individual)\b", re.I),
    re.compile(r"\b(implement|add|create|build|fix|update|write)\b", re.I),
    re.compile(r"\b(module|component|function|endpoint|page|file)\b", re.I),
    re.compile(r"\b(step|phase|iteration|sprint|task)\b", re.I),
]

# Over-ambition patterns — broad scope reduces achievability
_OVERAMBITION_PATTERNS = [
    re.compile(r"\b(everything|all|entire|complete|total|whole)\b", re.I),
    re.compile(r"\b(transform|revolutionize|overhaul|rewrite)\b", re.I),
    re.compile(r"\b(always|never|every|universal)\b", re.I),
]

# Time-bound patterns
_TIME_BOUND_PATTERNS = [
    re.compile(r"\b(by|within|before|deadline)\b", re.I),
    re.compile(r"\b(sprint|week|month|quarter|year|day)\b", re.I),
    re.compile(r"\b(q[1-4]|h[12])\b", re.I),  # Q1, Q2, H1, H2
    re.compile(r"\b\d{4}[-/]\d{2}([-/]\d{2})?\b"),  # Date patterns
    re.compile(r"\b(timeline|schedule|milestone|phase)\b", re.I),
]

# Domain relevance patterns — technical and business terms
_RELEVANCE_PATTERNS = [
    re.compile(r"\b(api|database|server|client|service|endpoint|module)\b", re.I),
    re.compile(r"\b(user|customer|team|stakeholder|developer)\b", re.I),
    re.compile(r"\b(performance|security|reliability|scalability)\b", re.I),
    re.compile(r"\b(test|deploy|monitor|audit|review)\b", re.I),
    re.compile(r"\b(revenue|cost|budget|roi|efficiency)\b", re.I),
]


# =========================================================================
# Tokenization and similarity helpers
# =========================================================================


def _tokenize(text: str) -> frozenset[str]:
    """Tokenize text into a set of meaningful words.

    Lowercases, splits on non-alphanumeric, removes stop words
    and very short tokens.
    """
    words = re.split(r"[^a-zA-Z0-9]+", text.lower())
    return frozenset(w for w in words if w and len(w) > 1 and w not in _STOP_WORDS)


def _jaccard_similarity(a: frozenset[str], b: frozenset[str]) -> float:
    """Compute Jaccard similarity between two token sets."""
    if not a and not b:
        return 0.0
    intersection = len(a & b)
    union = len(a | b)
    if union == 0:
        return 0.0
    return intersection / union


def _name_cluster(token_sets: list[frozenset[str]]) -> str:
    """Generate a cluster name from the most common terms across members."""
    from collections import Counter

    term_counts: Counter[str] = Counter()
    for tokens in token_sets:
        term_counts.update(tokens)

    # Pick top 3 most common terms
    top_terms = [term for term, _ in term_counts.most_common(3)]
    if not top_terms:
        return "Unnamed cluster"
    return " / ".join(top_terms)


def _score_achievability(text: str) -> float:
    """Score how achievable/scoped a goal text appears (0.0 to 1.0).

    Higher scores for concrete, scoped tasks.
    Lower scores for overly ambitious language.
    """
    if not text:
        return 0.0

    positive = sum(1 for p in _ACHIEVABILITY_PATTERNS if p.search(text))
    negative = sum(1 for p in _OVERAMBITION_PATTERNS if p.search(text))

    score = positive / len(_ACHIEVABILITY_PATTERNS)
    penalty = negative / len(_OVERAMBITION_PATTERNS) * 0.5
    return max(0.0, min(1.0, score - penalty))


def _score_time_bound(text: str) -> float:
    """Score whether a goal has time references (0.0 to 1.0)."""
    if not text:
        return 0.0
    matches = sum(1 for p in _TIME_BOUND_PATTERNS if p.search(text))
    return min(1.0, matches / len(_TIME_BOUND_PATTERNS))


# =========================================================================
# Domain-specific goal templates
# =========================================================================

_DOMAIN_GOALS: dict[str, list[dict[str, str]]] = {
    "healthcare": [
        {
            "title": "Ensure HIPAA compliance for all patient data handling",
            "description": (
                "Verify that all data flows meet HIPAA Privacy and Security Rules. "
                "Implement access controls, audit logging, and encryption at rest."
            ),
            "type": "goal",
            "priority": "critical",
            "measurable": "Zero HIPAA violations; 100% of PHI access logged",
        },
        {
            "title": "Validate patient safety impact assessment",
            "description": (
                "Assess how the decision affects patient safety outcomes. "
                "Cross-reference with clinical guidelines and adverse event databases."
            ),
            "type": "risk",
            "priority": "critical",
            "measurable": "Patient safety risk score below threshold",
        },
        {
            "title": "Maintain clinical data integrity and provenance",
            "description": (
                "Ensure all clinical data used in decision-making has a clear "
                "provenance chain and meets data quality standards."
            ),
            "type": "principle",
            "priority": "high",
            "measurable": "100% of clinical data sources documented",
        },
    ],
    "financial": [
        {
            "title": "Complete risk assessment and exposure analysis",
            "description": (
                "Quantify financial risk exposure across all identified scenarios. "
                "Model worst-case, base-case, and best-case outcomes."
            ),
            "type": "goal",
            "priority": "critical",
            "measurable": "Risk-adjusted return calculated for all scenarios",
        },
        {
            "title": "Ensure regulatory compliance with financial standards",
            "description": (
                "Verify compliance with applicable financial regulations "
                "(SOX, Basel III, SEC requirements). Document all controls."
            ),
            "type": "goal",
            "priority": "critical",
            "measurable": "All applicable regulations mapped with control evidence",
        },
        {
            "title": "Establish audit trail for financial decision-making",
            "description": (
                "Create a complete, immutable audit trail documenting the "
                "rationale, data sources, and approvals for the financial decision."
            ),
            "type": "principle",
            "priority": "high",
            "measurable": "SOX-compliant audit trail generated",
        },
    ],
    "legal": [
        {
            "title": "Conduct thorough contract review and risk identification",
            "description": (
                "Review all contract clauses for potential risks, ambiguities, "
                "and unfavorable terms. Flag indemnification and liability provisions."
            ),
            "type": "goal",
            "priority": "critical",
            "measurable": "All contract clauses reviewed; risks categorized by severity",
        },
        {
            "title": "Complete legal due diligence checklist",
            "description": (
                "Execute comprehensive due diligence covering corporate structure, "
                "IP rights, litigation history, and regulatory status."
            ),
            "type": "goal",
            "priority": "critical",
            "measurable": "100% of due diligence items completed",
        },
        {
            "title": "Document legal basis and precedent for decision",
            "description": (
                "Establish the legal foundation for the decision including "
                "applicable statutes, case law precedents, and regulatory guidance."
            ),
            "type": "principle",
            "priority": "high",
            "measurable": "Legal basis documented with cited authorities",
        },
    ],
}


def _get_domain_goals(domain: str) -> list[dict[str, str]]:
    """Get domain-specific goal templates for the given vertical.

    Args:
        domain: Vertical domain name (e.g., "healthcare", "financial", "legal")

    Returns:
        List of goal template dicts, empty if domain not recognized
    """
    return _DOMAIN_GOALS.get(domain, [])


def _score_relevance(text: str, source_idea_ids: list[str] | None = None) -> float:
    """Score how relevant/domain-specific a goal is (0.0 to 1.0).

    Checks for domain-specific terminology. Having source ideas
    provides a small baseline boost (goal was derived from evidence).
    """
    if not text:
        return 0.0

    matches = sum(1 for p in _RELEVANCE_PATTERNS if p.search(text))
    base_score = matches / len(_RELEVANCE_PATTERNS)

    # Bonus for having source ideas (means it's grounded)
    if source_idea_ids:
        base_score += 0.1

    return min(1.0, base_score)
