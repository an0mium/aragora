"""
Review Cycle Pattern - Iterative refinement with convergence check.

The Review Cycle pattern iteratively refines output through multiple rounds
of review until convergence or a maximum iteration count. This is ideal for:
- Code review workflows
- Document editing
- Quality assurance processes
- Iterative improvement tasks

Structure:
    [Input] -> [Initial Draft] -> [Review] -> [Check Convergence] -+-> [Output]
                      ^                                |
                      +--------------------------------+ (needs refinement)

Configuration:
    - draft_agent: Agent for creating/refining drafts
    - review_agent: Agent for reviewing (can be same as draft)
    - max_iterations: Maximum refinement cycles
    - convergence_threshold: Score needed to exit loop (0.0-1.0)
    - review_criteria: What aspects to review
"""

from __future__ import annotations

from typing import List, Optional

from aragora.workflow.types import (
    WorkflowDefinition,
    TransitionRule,
    Position,
    NodeCategory,
    WorkflowCategory,
    VisualEdgeData,
    EdgeType,
)
from aragora.workflow.patterns.base import WorkflowPattern, PatternType


class ReviewCyclePattern(WorkflowPattern):
    """
    Iterative refinement with convergence check.

    Produces initial draft, reviews it, and refines until quality threshold
    is met or maximum iterations reached.

    Example:
        workflow = ReviewCyclePattern.create(
            name="Code Review Cycle",
            draft_agent="claude",
            review_agent="gpt4",
            task="Implement a rate limiter class",
            review_criteria=["correctness", "efficiency", "readability"],
            max_iterations=3,
            convergence_threshold=0.85,
        )
    """

    pattern_type = PatternType.REVIEW_CYCLE

    def __init__(
        self,
        name: str,
        agents: Optional[List[str]] = None,
        task: str = "",
        draft_agent: Optional[str] = None,
        review_agent: Optional[str] = None,
        max_iterations: int = 3,
        convergence_threshold: float = 0.85,
        review_criteria: Optional[List[str]] = None,
        draft_prompt: str = "",
        review_prompt: str = "",
        timeout_per_step: float = 120.0,
        **kwargs,
    ):
        super().__init__(name, agents, task, **kwargs)
        self.draft_agent = draft_agent or (agents[0] if agents else "claude")
        self.review_agent = review_agent or (agents[1] if agents and len(agents) > 1 else "gpt4")
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        self.review_criteria = review_criteria or ["quality", "completeness", "accuracy"]
        self.draft_prompt = draft_prompt
        self.review_prompt = review_prompt
        self.timeout_per_step = timeout_per_step

    def create_workflow(self) -> WorkflowDefinition:
        """Create a review cycle workflow definition."""
        workflow_id = self._generate_id("rc")
        steps = []
        transitions = []

        # Positions
        draft_x = 100
        review_x = 350
        check_x = 600
        output_x = 850
        y_pos = 200

        # Step 1: Initial/Refine Draft
        draft_prompt = self.draft_prompt or self._build_draft_prompt()
        draft_step = self._create_agent_step(
            step_id="draft",
            name="Create/Refine Draft",
            agent_type=self.draft_agent,
            prompt=draft_prompt,
            position=Position(x=draft_x, y=y_pos),
            timeout=self.timeout_per_step,
        )
        steps.append(draft_step)

        # Step 2: Review
        review_prompt = self.review_prompt or self._build_review_prompt()
        review_step = self._create_agent_step(
            step_id="review",
            name="Review",
            agent_type=self.review_agent,
            prompt=review_prompt,
            position=Position(x=review_x, y=y_pos),
            timeout=self.timeout_per_step,
        )
        review_step.config["system_prompt"] = (
            f"You are a critical reviewer. Evaluate based on: {', '.join(self.review_criteria)}. "
            "Provide specific, actionable feedback. Score from 0.0 to 1.0."
        )
        steps.append(review_step)

        # Step 3: Check convergence
        check_step = self._create_task_step(
            step_id="check_convergence",
            name="Check Convergence",
            task_type="function",
            config={
                "handler": "review_cycle_check",
                "args": {
                    "threshold": self.convergence_threshold,
                    "max_iterations": self.max_iterations,
                },
            },
            position=Position(x=check_x, y=y_pos),
            category=NodeCategory.CONTROL,
        )
        steps.append(check_step)

        # Step 4: Final output
        output_step = self._create_task_step(
            step_id="output",
            name="Final Output",
            task_type="transform",
            config={
                "transform": "outputs.get('draft', {}).get('response', '')",
                "output_format": "text",
            },
            position=Position(x=output_x, y=y_pos),
            category=NodeCategory.TASK,
        )
        steps.append(output_step)

        # Transitions
        draft_step.next_steps = ["review"]
        review_step.next_steps = ["check_convergence"]

        transitions.extend(
            [
                self._create_transition("draft", "review"),
                self._create_transition("review", "check_convergence"),
                # Conditional: converged -> output
                TransitionRule(
                    id="tr_converged",
                    from_step="check_convergence",
                    to_step="output",
                    condition="step_output.get('converged', False) == True",
                    priority=10,
                    label="Converged",
                    visual=VisualEdgeData(edge_type=EdgeType.CONDITIONAL, label="Converged"),
                ),
                # Conditional: not converged -> back to draft
                TransitionRule(
                    id="tr_refine",
                    from_step="check_convergence",
                    to_step="draft",
                    condition="step_output.get('converged', False) == False",
                    priority=5,
                    label="Refine",
                    visual=VisualEdgeData(edge_type=EdgeType.CONDITIONAL, label="Needs Work"),
                ),
            ]
        )

        return WorkflowDefinition(
            id=workflow_id,
            name=self.name,
            description=f"Review cycle: max {self.max_iterations} iterations, threshold {self.convergence_threshold}",
            steps=steps,
            transitions=transitions,
            entry_step="draft",
            category=self.config.get("category", WorkflowCategory.GENERAL),
            tags=["review_cycle", "iterative", "refinement"] + self.config.get("tags", []),
            metadata={
                "pattern": "review_cycle",
                "draft_agent": self.draft_agent,
                "review_agent": self.review_agent,
                "max_iterations": self.max_iterations,
                "convergence_threshold": self.convergence_threshold,
                "review_criteria": self.review_criteria,
            },
        )

    def _build_draft_prompt(self) -> str:
        """Build draft/refinement prompt."""
        return """Create or refine a response for this task.

Task: {task}

{feedback_section}

Provide your best response, addressing any feedback if given:"""

    def _build_review_prompt(self) -> str:
        """Build review prompt."""
        criteria_str = "\n".join([f"- {c}" for c in self.review_criteria])
        return f"""Review this draft based on the following criteria:
{criteria_str}

Original Task: {{task}}

Draft to Review:
{{step.draft}}

Provide:
1. Score (0.0 to 1.0) for overall quality
2. Specific feedback for each criterion
3. Concrete suggestions for improvement

Format your response as:
SCORE: <number>
FEEDBACK:
<detailed feedback>
SUGGESTIONS:
<improvement suggestions>"""


# Register review cycle handlers
def _register_review_cycle_handlers():
    """Register review cycle task handlers."""
    try:
        from aragora.workflow.nodes.task import register_task_handler
        import re

        async def review_cycle_check(context, threshold=0.85, max_iterations=3):
            """Check if review cycle should continue or converge."""
            review_result = context.step_outputs.get("review", {})
            response = review_result.get("response", "")

            # Track iteration count
            iteration = context.state.get("review_iteration", 0) + 1
            context.set_state("review_iteration", iteration)

            # Extract score from review
            score = 0.0
            score_match = re.search(r"SCORE:\s*([\d.]+)", response)
            if score_match:
                try:
                    score = float(score_match.group(1))
                except ValueError:
                    score = 0.0

            # Extract feedback for next iteration
            feedback = ""
            feedback_match = re.search(r"FEEDBACK:([\s\S]*?)(?:SUGGESTIONS:|$)", response)
            if feedback_match:
                feedback = feedback_match.group(1).strip()

            suggestions = ""
            suggestions_match = re.search(r"SUGGESTIONS:([\s\S]*?)$", response)
            if suggestions_match:
                suggestions = suggestions_match.group(1).strip()

            # Store feedback for next draft iteration
            context.set_state("last_feedback", feedback)
            context.set_state("last_suggestions", suggestions)
            context.set_state("last_score", score)

            # Check convergence
            converged = score >= threshold or iteration >= max_iterations

            return {
                "converged": converged,
                "score": score,
                "iteration": iteration,
                "max_iterations": max_iterations,
                "threshold": threshold,
                "reason": (
                    f"Score {score:.2f} >= threshold {threshold}"
                    if score >= threshold
                    else (
                        f"Max iterations ({max_iterations}) reached"
                        if iteration >= max_iterations
                        else f"Score {score:.2f} < threshold {threshold}, continuing"
                    )
                ),
            }

        register_task_handler("review_cycle_check", review_cycle_check)

    except ImportError:
        pass


_register_review_cycle_handlers()
