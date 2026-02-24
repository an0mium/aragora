import logging

from aragora.topic_spec import TopicSpec

logger = logging.getLogger(__name__)


def handle_ambiguous_topic(topic: str) -> TopicSpec:
    """
    Handles an ambiguous topic by generating plausible interpretations
    and returning a default TopicSpec for the meta-task.
    """
    logger.info("Ambiguous topic '%s': defaulting to meta-level interpretation.", topic)

    # In a real implementation, this would involve prompting the user for clarification.
    # For now, we'll default to the most relevant meta-task.
    return TopicSpec(
        title="Improve Ambiguous Topic Handling",
        objective="Propose and implement a system to handle ambiguous or underspecified debate topics to improve the quality of debate outcomes.",
        assumptions=[
            "Ambiguous topics are a significant source of low-quality debates.",
            "A structured topic format is preferable to a simple string.",
            "The solution should be integrated into the existing debate startup process."
        ],
        non_goals=[
            "Solving all natural language understanding problems.",
            "Replacing the need for human-in-the-loop for highly nuanced topics."
        ],
        evaluation_criteria=[
            "The system correctly identifies and flags ambiguous topics.",
            "The proposed `TopicSpec` structure is adopted by the orchestration layer.",
            "A measurable reduction in debate failures or low-quality outcomes for previously ambiguous topics."
        ],
        context="The initial task was simply 'Topic', which is too vague to be actionable. This meta-task is to fix that class of problems."
    )
