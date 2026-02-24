import logging

from aragora.task_brief import TaskBriefV1


logger = logging.getLogger(__name__)


def handle_ambiguous_task(task: str) -> TaskBriefV1:
    """
    Handles an ambiguous topic by generating plausible interpretations
    and returning a default TopicSpec for the meta-task.
    """
    logger.info("Ambiguous task '%s': defaulting to meta-level interpretation.", task)

    # For now, we will use a simple rule-based approach to structure the brief.
    # In the future, this could involve an LLM call to an 'analyst' agent.
    return TaskBriefV1(
        goal=f"Design a software architecture for: {task}",
        assumptions=[
            "The goal is to produce a high-level software architecture design.",
            "The design should be suitable for a multi-agent system.",
            "Key components and their interactions should be identified.",
        ],
        confidence=0.4,
        requires_user_confirmation=True,
    )
