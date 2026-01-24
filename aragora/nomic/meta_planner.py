"""Meta-Planner for debate-driven goal prioritization.

Takes high-level business objectives and uses multi-agent debate to
determine which areas should be improved first.

Usage:
    from aragora.nomic.meta_planner import MetaPlanner

    planner = MetaPlanner()
    goals = await planner.prioritize_work(
        objective="Maximize utility for SME businesses",
        available_tracks=[Track.SME, Track.QA],
    )

    for goal in goals:
        print(f"{goal.track}: {goal.description} ({goal.estimated_impact})")
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class Track(Enum):
    """Development tracks for domain-based routing."""

    SME = "sme"
    DEVELOPER = "developer"
    SELF_HOSTED = "self_hosted"
    QA = "qa"
    CORE = "core"


@dataclass
class PrioritizedGoal:
    """A prioritized improvement goal."""

    id: str
    track: Track
    description: str
    rationale: str
    estimated_impact: str  # high, medium, low
    priority: int  # 1 = highest
    focus_areas: List[str] = field(default_factory=list)
    file_hints: List[str] = field(default_factory=list)


@dataclass
class PlanningContext:
    """Context for meta-planning decisions."""

    recent_issues: List[str] = field(default_factory=list)
    test_failures: List[str] = field(default_factory=list)
    user_feedback: List[str] = field(default_factory=list)
    recent_changes: List[str] = field(default_factory=list)


@dataclass
class MetaPlannerConfig:
    """Configuration for MetaPlanner."""

    agents: List[str] = field(default_factory=lambda: ["claude", "gemini", "deepseek"])
    debate_rounds: int = 2
    max_goals: int = 5
    consensus_threshold: float = 0.6


class MetaPlanner:
    """Debate-driven goal prioritization.

    Uses multi-agent debate to determine which areas should be improved
    to best achieve a high-level objective.
    """

    def __init__(self, config: Optional[MetaPlannerConfig] = None):
        self.config = config or MetaPlannerConfig()

    async def prioritize_work(
        self,
        objective: str,
        available_tracks: Optional[List[Track]] = None,
        constraints: Optional[List[str]] = None,
        context: Optional[PlanningContext] = None,
    ) -> List[PrioritizedGoal]:
        """Use multi-agent debate to prioritize work.

        Args:
            objective: High-level business objective
            available_tracks: Which tracks can be worked on
            constraints: Constraints like "no breaking changes"
            context: Additional context (issues, feedback, etc.)

        Returns:
            List of prioritized goals ordered by priority
        """
        available_tracks = available_tracks or list(Track)
        constraints = constraints or []
        context = context or PlanningContext()

        logger.info(
            f"meta_planner_started objective={objective[:100]} tracks={[t.value for t in available_tracks]}"
        )

        try:
            from aragora.debate.orchestrator import Arena, DebateProtocol
            from aragora.core import Environment
            from aragora.agents import create_agent

            # Build debate topic
            topic = self._build_debate_topic(objective, available_tracks, constraints, context)

            # Create agents
            agents = []
            for agent_type in self.config.agents:
                try:
                    agent = create_agent(agent_type)  # type: ignore[arg-type]
                    agents.append(agent)
                except Exception as e:
                    logger.warning(f"Could not create agent {agent_type}: {e}")

            if not agents:
                logger.warning("No agents available, using heuristic prioritization")
                return self._heuristic_prioritize(objective, available_tracks)

            # Run debate
            env = Environment(task=topic)
            protocol = DebateProtocol(
                rounds=self.config.debate_rounds,
                consensus="weighted",
            )

            arena = Arena(env, agents, protocol)
            result = await arena.run()

            # Parse goals from debate result
            goals = self._parse_goals_from_debate(result, available_tracks, objective)

            logger.info(
                f"meta_planner_completed goal_count={len(goals)} objectives={[g.description[:50] for g in goals]}"
            )

            return goals

        except ImportError as e:
            logger.warning(f"Debate infrastructure not available: {e}")
            return self._heuristic_prioritize(objective, available_tracks)
        except Exception as e:
            logger.exception(f"Meta-planning failed: {e}")
            return self._heuristic_prioritize(objective, available_tracks)

    def _build_debate_topic(
        self,
        objective: str,
        tracks: List[Track],
        constraints: List[str],
        context: PlanningContext,
    ) -> str:
        """Build the debate topic for meta-planning."""
        track_names = ", ".join(t.value for t in tracks)

        topic = f"""You are planning improvements for the Aragora project.

OBJECTIVE: {objective}

AVAILABLE TRACKS (domains you can work on):
{track_names}

Track descriptions:
- SME: Small business features, dashboard, user workspace
- Developer: SDKs, API, documentation
- Self-Hosted: Docker, deployment, backup/restore
- QA: Tests, CI/CD, code quality
- Core: Debate engine, agents, memory (requires approval)

CONSTRAINTS:
{chr(10).join(f"- {c}" for c in constraints) if constraints else "- None specified"}

"""
        if context.recent_issues:
            topic += f"""
RECENT ISSUES:
{chr(10).join(f"- {issue}" for issue in context.recent_issues[:5])}
"""

        if context.test_failures:
            topic += f"""
FAILING TESTS:
{chr(10).join(f"- {failure}" for failure in context.test_failures[:5])}
"""

        topic += """
YOUR TASK:
Propose 3-5 specific improvement goals that would best achieve the objective.
For each goal, specify:
1. Which track it belongs to
2. A clear, actionable description
3. Why this should be prioritized (rationale)
4. Expected impact: high, medium, or low

Format your response as a numbered list with clear structure.
Consider dependencies and order goals by priority.
"""
        return topic

    def _parse_goals_from_debate(
        self,
        debate_result: Any,
        available_tracks: List[Track],
        objective: str,
    ) -> List[PrioritizedGoal]:
        """Parse prioritized goals from debate consensus."""
        goals = []

        # Get consensus text from debate result
        consensus_text = ""
        if hasattr(debate_result, "consensus") and debate_result.consensus:
            consensus_text = str(debate_result.consensus)
        elif hasattr(debate_result, "final_response"):
            consensus_text = str(debate_result.final_response)
        elif hasattr(debate_result, "responses") and debate_result.responses:
            consensus_text = str(debate_result.responses[-1])

        if not consensus_text:
            return self._heuristic_prioritize(objective, available_tracks)

        # Parse numbered items from the consensus
        lines = consensus_text.split("\n")
        current_goal: Dict[str, Any] = {}
        goal_id = 0

        for line in lines:
            line = line.strip()

            # Detect numbered items (1., 2., etc.) or bullet points
            if re.match(r"^[\d]+[\.\)]\s+", line) or re.match(r"^[-*]\s+", line):
                # Save previous goal if exists
                if current_goal.get("description"):
                    goals.append(self._build_goal(current_goal, goal_id, available_tracks))
                    goal_id += 1

                # Start new goal
                current_goal = {
                    "description": re.sub(r"^[\d]+[\.\)]\s+|^[-*]\s+", "", line),
                    "track": None,
                    "rationale": "",
                    "impact": "medium",
                }

            elif current_goal:
                # Parse track from line
                for track in Track:
                    if track.value.lower() in line.lower():
                        current_goal["track"] = track
                        break

                # Parse impact
                if "high" in line.lower() and "impact" in line.lower():
                    current_goal["impact"] = "high"
                elif "low" in line.lower() and "impact" in line.lower():
                    current_goal["impact"] = "low"

                # Accumulate rationale
                if "because" in line.lower() or "rationale" in line.lower():
                    current_goal["rationale"] = line

        # Don't forget last goal
        if current_goal.get("description"):
            goals.append(self._build_goal(current_goal, goal_id, available_tracks))

        # Limit to max goals
        goals = goals[: self.config.max_goals]

        # If no goals parsed, fall back to heuristics
        if not goals:
            return self._heuristic_prioritize(objective, available_tracks)

        return goals

    def _build_goal(
        self,
        goal_dict: Dict[str, Any],
        priority: int,
        available_tracks: List[Track],
    ) -> PrioritizedGoal:
        """Build a PrioritizedGoal from parsed data."""
        # Default track based on keywords if not explicitly set
        track = goal_dict.get("track")
        if not track:
            track = self._infer_track(goal_dict["description"], available_tracks)

        return PrioritizedGoal(
            id=f"goal_{priority}",
            track=track,
            description=goal_dict["description"],
            rationale=goal_dict.get("rationale", ""),
            estimated_impact=goal_dict.get("impact", "medium"),
            priority=priority + 1,
        )

    def _infer_track(self, description: str, available_tracks: List[Track]) -> Track:
        """Infer track from goal description."""
        desc_lower = description.lower()

        track_keywords = {
            Track.SME: ["dashboard", "user", "ui", "frontend", "workspace", "admin"],
            Track.DEVELOPER: ["sdk", "api", "documentation", "client", "package"],
            Track.SELF_HOSTED: ["docker", "deploy", "backup", "ops", "kubernetes"],
            Track.QA: ["test", "ci", "coverage", "quality", "e2e", "playwright"],
            Track.CORE: ["debate", "agent", "consensus", "arena", "memory"],
        }

        for track, keywords in track_keywords.items():
            if track in available_tracks:
                if any(kw in desc_lower for kw in keywords):
                    return track

        # Default to first available track
        return available_tracks[0] if available_tracks else Track.DEVELOPER

    def _heuristic_prioritize(
        self,
        objective: str,
        available_tracks: List[Track],
    ) -> List[PrioritizedGoal]:
        """Fallback heuristic prioritization when debate is unavailable."""
        goals = []
        obj_lower = objective.lower()

        # Generate goals based on objective keywords
        if "sme" in obj_lower or "small business" in obj_lower:
            if Track.SME in available_tracks:
                goals.append(
                    PrioritizedGoal(
                        id="goal_0",
                        track=Track.SME,
                        description="Improve dashboard usability for small business users",
                        rationale="Directly addresses SME utility objective",
                        estimated_impact="high",
                        priority=1,
                    )
                )

            if Track.QA in available_tracks:
                goals.append(
                    PrioritizedGoal(
                        id="goal_1",
                        track=Track.QA,
                        description="Add E2E tests for critical user flows",
                        rationale="Ensures reliability for SME users",
                        estimated_impact="medium",
                        priority=2,
                    )
                )

        # Add default goals for available tracks
        priority = len(goals) + 1
        for track in available_tracks:
            if not any(g.track == track for g in goals):
                goals.append(
                    PrioritizedGoal(
                        id=f"goal_{priority - 1}",
                        track=track,
                        description=f"Improve {track.value} track capabilities",
                        rationale="Supports overall project health",
                        estimated_impact="medium",
                        priority=priority,
                    )
                )
                priority += 1

        return goals[: self.config.max_goals]


__all__ = [
    "MetaPlanner",
    "MetaPlannerConfig",
    "PrioritizedGoal",
    "PlanningContext",
    "Track",
]
