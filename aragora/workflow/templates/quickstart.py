"""
Quickstart Workflow Templates.

Minimal-configuration templates designed for rapid decision-making:
- Quick Yes/No decisions
- Pros & Cons analysis
- Risk assessment
- Brainstorming sessions

These templates prioritize speed and simplicity, completing in under 5 minutes
with minimal setup required.
"""

from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from aragora.workflow.types import (
    StepDefinition,
    WorkflowCategory,
    WorkflowDefinition,
)


def create_yes_no_workflow(
    question: str,
    context: Optional[str] = None,
    confidence_threshold: float = 0.7,
    agents: Optional[List[str]] = None,
) -> WorkflowDefinition:
    """Create a quick yes/no decision workflow.

    Fast binary decision-making using multi-agent debate.
    Designed to complete in under 2 minutes.

    Args:
        question: Yes/No question to answer
        context: Additional context for the decision
        confidence_threshold: Minimum confidence for auto-decision
        agents: Agents to use (defaults to claude + gpt-4)

    Returns:
        WorkflowDefinition for yes/no decision

    Example:
        workflow = create_yes_no_workflow(
            question="Should we launch the feature this week?",
            context="Testing is 95% complete, one minor bug remains",
        )
    """
    workflow_id = f"yesno_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    agent_list = agents or ["claude", "gpt-4"]

    return WorkflowDefinition(
        id=workflow_id,
        name=f"Quick Decision: {question[:40]}",
        description="Quick yes/no decision via agent debate",
        category=WorkflowCategory.GENERAL,
        tags=["quickstart", "yes-no", "fast", "decision"],
        inputs={
            "question": question,
            "context": context,
            "confidence_threshold": confidence_threshold,
        },
        steps=[
            StepDefinition(
                id="debate",
                name="Quick Debate",
                step_type="debate",
                config={
                    "agents": agent_list,
                    "topic": question,
                    "rounds": 2,
                    "format": "light",
                    "consensus_mode": "majority",
                },
                next_steps=["decide"],
            ),
            StepDefinition(
                id="decide",
                name="Render Decision",
                step_type="agent",
                config={
                    "agent_type": "claude",
                    "prompt_template": "Based on debate, answer YES or NO with confidence score",
                },
                next_steps=["store"],
            ),
            StepDefinition(
                id="store",
                name="Record Decision",
                step_type="memory_write",
                config={
                    "collection": "quick_decisions",
                },
                next_steps=[],
            ),
        ],
        entry_step="debate",
    )


def create_pros_cons_workflow(
    topic: str,
    context: Optional[str] = None,
    max_items: int = 5,
    weighted: bool = False,
) -> WorkflowDefinition:
    """Create a pros and cons analysis workflow.

    Structured analysis listing advantages and disadvantages
    with optional weighting for importance.

    Args:
        topic: Topic to analyze
        context: Additional context
        max_items: Maximum pros/cons per side
        weighted: Whether to weight items by importance

    Returns:
        WorkflowDefinition for pros/cons analysis

    Example:
        workflow = create_pros_cons_workflow(
            topic="Switching to remote-first work policy",
            weighted=True,
        )
    """
    workflow_id = f"proscons_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    return WorkflowDefinition(
        id=workflow_id,
        name=f"Pros & Cons: {topic[:40]}",
        description="Structured pros and cons analysis",
        category=WorkflowCategory.GENERAL,
        tags=["quickstart", "pros-cons", "analysis", "decision"],
        inputs={
            "topic": topic,
            "context": context,
            "max_items": max_items,
            "weighted": weighted,
        },
        steps=[
            StepDefinition(
                id="pros",
                name="List Pros",
                step_type="agent",
                config={
                    "agent_type": "claude",
                    "prompt_template": f"List top {max_items} advantages of: {topic}",
                },
                next_steps=["cons"],
            ),
            StepDefinition(
                id="cons",
                name="List Cons",
                step_type="agent",
                config={
                    "agent_type": "gpt-4",
                    "prompt_template": f"List top {max_items} disadvantages of: {topic}",
                },
                next_steps=["weight"] if weighted else ["synthesize"],
            ),
            StepDefinition(
                id="weight",
                name="Weight Items",
                step_type="agent",
                config={
                    "agent_type": "claude",
                    "prompt_template": "Assign importance weights (1-10) to each pro and con",
                },
                next_steps=["synthesize"],
            ),
            StepDefinition(
                id="synthesize",
                name="Synthesize Analysis",
                step_type="agent",
                config={
                    "agent_type": "claude",
                    "prompt_template": "Synthesize pros/cons into recommendation",
                },
                next_steps=["store"],
            ),
            StepDefinition(
                id="store",
                name="Store Analysis",
                step_type="memory_write",
                config={
                    "collection": "pros_cons_analyses",
                },
                next_steps=[],
            ),
        ],
        entry_step="pros",
    )


def create_risk_assessment_workflow(
    scenario: str,
    context: Optional[str] = None,
    risk_categories: Optional[List[str]] = None,
    include_mitigation: bool = True,
) -> WorkflowDefinition:
    """Create a risk assessment workflow.

    Quick risk identification and assessment using
    multi-agent analysis.

    Args:
        scenario: Scenario or decision to assess
        context: Additional context
        risk_categories: Categories to assess (defaults to standard set)
        include_mitigation: Whether to include mitigation strategies

    Returns:
        WorkflowDefinition for risk assessment

    Example:
        workflow = create_risk_assessment_workflow(
            scenario="Launch new product in Q1",
            risk_categories=["market", "technical", "financial"],
        )
    """
    workflow_id = f"risk_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    categories = risk_categories or ["operational", "financial", "reputational", "strategic"]

    return WorkflowDefinition(
        id=workflow_id,
        name=f"Risk Assessment: {scenario[:40]}",
        description="Quick risk identification and assessment",
        category=WorkflowCategory.GENERAL,
        tags=["quickstart", "risk", "assessment", "decision"],
        inputs={
            "scenario": scenario,
            "context": context,
            "categories": categories,
            "include_mitigation": include_mitigation,
        },
        steps=[
            StepDefinition(
                id="identify",
                name="Identify Risks",
                step_type="debate",
                config={
                    "agents": ["claude", "gpt-4"],
                    "topic": f"What are the risks of: {scenario}",
                    "rounds": 2,
                    "categories": categories,
                },
                next_steps=["score"],
            ),
            StepDefinition(
                id="score",
                name="Score Risks",
                step_type="agent",
                config={
                    "agent_type": "claude",
                    "prompt_template": "Score each risk: likelihood (1-5) x impact (1-5)",
                },
                next_steps=["mitigate"] if include_mitigation else ["summarize"],
            ),
            StepDefinition(
                id="mitigate",
                name="Mitigation Strategies",
                step_type="agent",
                config={
                    "agent_type": "claude",
                    "prompt_template": "Propose mitigation strategy for each high-priority risk",
                },
                next_steps=["summarize"],
            ),
            StepDefinition(
                id="summarize",
                name="Risk Summary",
                step_type="agent",
                config={
                    "agent_type": "claude",
                    "prompt_template": "Create executive summary of risk assessment",
                },
                next_steps=["store"],
            ),
            StepDefinition(
                id="store",
                name="Store Assessment",
                step_type="memory_write",
                config={
                    "collection": "risk_assessments",
                },
                next_steps=[],
            ),
        ],
        entry_step="identify",
    )


def create_brainstorm_workflow(
    topic: str,
    goal: Optional[str] = None,
    num_ideas: int = 10,
    prioritize: bool = True,
    perspectives: Optional[List[str]] = None,
) -> WorkflowDefinition:
    """Create a brainstorming session workflow.

    Multi-agent ideation session to generate creative solutions
    or ideas for a given topic.

    Args:
        topic: Topic to brainstorm
        goal: Specific goal for the session
        num_ideas: Target number of ideas
        prioritize: Whether to prioritize/rank ideas
        perspectives: Different perspectives to explore

    Returns:
        WorkflowDefinition for brainstorming

    Example:
        workflow = create_brainstorm_workflow(
            topic="Improve customer retention",
            goal="Reduce churn by 20%",
            num_ideas=15,
        )
    """
    workflow_id = f"brainstorm_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    viewpoints = perspectives or ["creative", "practical", "cost-effective", "innovative"]

    return WorkflowDefinition(
        id=workflow_id,
        name=f"Brainstorm: {topic[:40]}",
        description="Multi-agent creative ideation session",
        category=WorkflowCategory.GENERAL,
        tags=["quickstart", "brainstorm", "ideation", "creative"],
        inputs={
            "topic": topic,
            "goal": goal,
            "num_ideas": num_ideas,
            "prioritize": prioritize,
            "perspectives": viewpoints,
        },
        steps=[
            StepDefinition(
                id="diverge",
                name="Generate Ideas",
                step_type="parallel",
                config={
                    "sub_steps": [f"ideas_{p}" for p in viewpoints],
                    "perspectives": viewpoints,
                },
                next_steps=["combine"],
            ),
            StepDefinition(
                id="combine",
                name="Combine & Dedupe",
                step_type="agent",
                config={
                    "agent_type": "claude",
                    "prompt_template": f"Combine ideas, remove duplicates, ensure {num_ideas} unique ideas",
                },
                next_steps=["expand"],
            ),
            StepDefinition(
                id="expand",
                name="Expand Ideas",
                step_type="debate",
                config={
                    "agents": ["claude", "gpt-4"],
                    "topic": "Which ideas have the most potential?",
                    "rounds": 2,
                },
                next_steps=["prioritize"] if prioritize else ["store"],
            ),
            StepDefinition(
                id="prioritize",
                name="Prioritize Ideas",
                step_type="agent",
                config={
                    "agent_type": "claude",
                    "prompt_template": "Rank ideas by feasibility and impact",
                },
                next_steps=["store"],
            ),
            StepDefinition(
                id="store",
                name="Store Session",
                step_type="memory_write",
                config={
                    "collection": "brainstorm_sessions",
                },
                next_steps=[],
            ),
        ],
        entry_step="diverge",
    )


# =============================================================================
# Convenience Functions
# =============================================================================


def quick_decision(question: str) -> WorkflowDefinition:
    """Create the fastest possible yes/no decision.

    Args:
        question: Question to answer

    Returns:
        Minimal yes/no workflow
    """
    return create_yes_no_workflow(question=question)


def quick_analysis(topic: str) -> WorkflowDefinition:
    """Create a quick pros/cons analysis.

    Args:
        topic: Topic to analyze

    Returns:
        Streamlined pros/cons workflow
    """
    return create_pros_cons_workflow(topic=topic, max_items=3, weighted=False)


def quick_risks(scenario: str) -> WorkflowDefinition:
    """Create a quick risk check.

    Args:
        scenario: Scenario to assess

    Returns:
        Minimal risk assessment workflow
    """
    return create_risk_assessment_workflow(
        scenario=scenario,
        risk_categories=["operational", "financial"],
        include_mitigation=False,
    )


def quick_ideas(topic: str, count: int = 5) -> WorkflowDefinition:
    """Create a quick brainstorm.

    Args:
        topic: Topic to brainstorm
        count: Number of ideas to generate

    Returns:
        Fast brainstorming workflow
    """
    return create_brainstorm_workflow(
        topic=topic,
        num_ideas=count,
        prioritize=False,
        perspectives=["creative", "practical"],
    )


__all__ = [
    # Core templates
    "create_yes_no_workflow",
    "create_pros_cons_workflow",
    "create_risk_assessment_workflow",
    "create_brainstorm_workflow",
    # Convenience functions
    "quick_decision",
    "quick_analysis",
    "quick_risks",
    "quick_ideas",
]
