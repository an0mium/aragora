"""
Context formatting functions for debate prompts.

Pure functions that format various context types for injection into agent prompts.
Extracted from PromptBuilder for improved modularity and testability.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from aragora.evidence.collector import EvidenceSnippet
    from aragora.pulse.types import TrendingTopic

logger = logging.getLogger(__name__)


def format_patterns_for_prompt(patterns: list[dict]) -> str:
    """Format learned patterns as prompt context for agents.

    This enables pattern-based learning: agents are warned about
    recurring issues from past debates before they make the same mistakes.

    Args:
        patterns: List of pattern dicts with 'category', 'pattern', 'occurrences'

    Returns:
        Formatted string to inject into debate context
    """
    if not patterns:
        return ""

    lines = ["## LEARNED PATTERNS (From Previous Debates)"]
    lines.append("Be especially careful about these recurring issues:\n")

    for p in patterns[:5]:  # Limit to top 5 patterns
        category = p.get("category", "general")
        pattern = p.get("pattern", "")
        occurrences = p.get("occurrences", 0)
        severity = p.get("avg_severity", 0)

        severity_label = ""
        if severity >= 0.7:
            severity_label = " [HIGH SEVERITY]"
        elif severity >= 0.4:
            severity_label = " [MEDIUM]"

        lines.append(f"- {category.upper()}{severity_label}: {pattern}")
        lines.append(f"  (Seen {occurrences} times)")

    lines.append("\nLearn from these past issues to improve your analysis.")
    return "\n".join(lines)


def format_successful_patterns(
    patterns: list[dict],
    limit: int = 3,
) -> str:
    """Format successful debate patterns for prompt injection.

    Highlights patterns that have led to good outcomes in past debates.

    Args:
        patterns: List of successful pattern dicts
        limit: Maximum number of patterns to include

    Returns:
        Formatted string with successful patterns
    """
    if not patterns:
        return ""

    lines = ["## SUCCESSFUL STRATEGIES (From Past Debates)"]
    lines.append("These approaches have worked well before:\n")

    for p in patterns[:limit]:
        strategy = p.get("strategy", "")
        success_rate = p.get("success_rate", 0)
        context = p.get("context", "")

        lines.append(f"- {strategy}")
        lines.append(f"  Success rate: {success_rate:.0%}")
        if context:
            lines.append(f"  Context: {context}")

    return "\n".join(lines)


def format_evidence_for_prompt(
    snippets: list["EvidenceSnippet"],
    max_snippets: int = 5,
    rlm_adapter: Optional[Any] = None,
    enable_rlm_hints: bool = True,
) -> str:
    """Format evidence snippets as citable references for agent prompts.

    Args:
        snippets: List of EvidenceSnippet objects
        max_snippets: Maximum number of snippets to include
        rlm_adapter: Optional RLM adapter for content formatting
        enable_rlm_hints: Whether to include RLM usage hints

    Returns:
        Formatted string with evidence citations
    """
    if not snippets:
        return ""

    lines = ["## AVAILABLE EVIDENCE"]
    lines.append("Reference these sources by ID when making factual claims:\n")

    for i, snippet in enumerate(snippets[:max_snippets], 1):
        evid_id = f"[EVID-{i}]"
        title = (
            (snippet.title[:80] if snippet.title else "Untitled")
            if hasattr(snippet, "title")
            else "Untitled"
        )
        source = getattr(snippet, "source", "Unknown") or "Unknown"

        # Get reliability score with safe fallback
        reliability = getattr(snippet, "reliability_score", 0.5)
        if not isinstance(reliability, (int, float)):
            reliability = 0.5

        # Format the snippet
        lines.append(f'{evid_id} "{title}" ({source})')
        lines.append(f"  Reliability: {reliability:.0%}")

        url = getattr(snippet, "url", None)
        if url:
            lines.append(f"  URL: {url}")

        # Include snippet content
        snippet_text = getattr(snippet, "snippet", None)
        if snippet_text:
            if rlm_adapter and len(snippet_text) > 200:
                # RLM pattern: register full content, show summary with hint
                content = rlm_adapter.format_for_prompt(
                    content=snippet_text,
                    max_chars=200,
                    content_type="evidence",
                    include_hint=enable_rlm_hints,
                )
            else:
                # Fallback: simple truncation
                content = snippet_text[:200]
                if len(snippet_text) > 200:
                    content += "..."
            lines.append(f"  > {content}")
        lines.append("")  # Blank line between snippets

    lines.append("When stating facts, cite evidence as [EVID-N]. Uncited claims may be challenged.")
    return "\n".join(lines)


def format_trending_for_prompt(
    topics: list["TrendingTopic"],
    task: str = "",
    max_topics: int = 3,
    use_relevance_filter: bool = True,
) -> str:
    """Format trending topics as context for agent prompts.

    Args:
        topics: List of TrendingTopic objects from Pulse system
        task: The debate task for relevance filtering
        max_topics: Maximum number of topics to include
        use_relevance_filter: Whether to filter for relevance to task

    Returns:
        Formatted trending context, or empty string if no topics
    """
    if not topics:
        return ""

    if use_relevance_filter and task:
        # Filter for relevance to task if possible
        task_lower = task.lower()
        relevant_topics = []

        for topic in topics[: max_topics * 2]:  # Get more for filtering
            # Simple relevance check - topic keywords in task or vice versa
            topic_text = topic.topic.lower() if hasattr(topic, "topic") else str(topic).lower()
            if any(word in task_lower for word in topic_text.split() if len(word) > 3):
                relevant_topics.append(topic)
            elif len(relevant_topics) < max_topics:
                relevant_topics.append(topic)

            if len(relevant_topics) >= max_topics:
                break

        if not relevant_topics:
            relevant_topics = topics[:max_topics]
    else:
        # No filtering - just take top N topics
        relevant_topics = topics[:max_topics]

    lines = ["## CURRENT TRENDING CONTEXT"]
    lines.append("These topics are currently trending and may provide timely context:\n")

    for topic in relevant_topics:
        topic_name = getattr(topic, "topic", str(topic))
        platform = getattr(topic, "platform", "unknown")
        volume = getattr(topic, "volume", 0)
        category = getattr(topic, "category", "general")

        lines.append(f"- **{topic_name}** ({platform})")
        if volume:
            lines.append(f"  Engagement: {volume:,} | Category: {category}")

    lines.append("")
    lines.append("Consider how current events may relate to the debate topic.")
    return "\n".join(lines)


def format_elo_ranking_context(
    agent_name: str,
    all_agent_names: list[str],
    ratings: dict[str, float],
    domain: str = "general",
    include_calibration: bool = True,
) -> str:
    """Format ELO ranking context for agent awareness.

    Args:
        agent_name: The current agent's name
        all_agent_names: Names of all agents in the debate
        ratings: Dict mapping agent names to ELO ratings
        domain: The debate domain for context
        include_calibration: Whether to include calibration hints

    Returns:
        Formatted ELO context string
    """
    if not ratings:
        return ""

    domain_suffix = f" ({domain})" if domain and domain != "general" else ""

    lines = [f"## Agent Rankings{domain_suffix}"]
    lines.append("Consider these rankings when weighing arguments:\n")

    # Sort by ELO for display
    sorted_ratings = sorted(
        [(name, rating) for name, rating in ratings.items()],
        key=lambda x: x[1],
        reverse=True,
    )

    for rank, (name, rating) in enumerate(sorted_ratings, 1):
        is_self = name == agent_name
        marker = " (you)" if is_self else ""
        lines.append(f"{rank}. {name}{marker}: {rating:.0f} ELO")

    lines.append("")
    lines.append("Higher-rated agents have demonstrated stronger performance in similar debates.")

    return "\n".join(lines)


def format_calibration_context(
    brier_score: float,
    is_overconfident: bool,
    is_underconfident: bool,
    total_predictions: int,
    min_predictions: int = 5,
    threshold: float = 0.25,
) -> str:
    """Format calibration feedback for an agent.

    Args:
        brier_score: The agent's Brier score (lower is better)
        is_overconfident: Whether agent tends to be overconfident
        is_underconfident: Whether agent tends to be underconfident
        total_predictions: Number of predictions in agent's history
        min_predictions: Minimum predictions needed for feedback
        threshold: Brier score threshold for providing feedback

    Returns:
        Formatted calibration context, or empty string if not needed
    """
    # Need minimum predictions for meaningful feedback
    if total_predictions < min_predictions:
        return ""

    # Only provide feedback for poorly calibrated agents
    if brier_score <= threshold:
        return ""

    lines = ["## Calibration Feedback"]
    lines.append(
        f"Your historical prediction accuracy needs improvement (Brier score: {brier_score:.2f})."
    )

    if is_overconfident:
        lines.append("You tend to be OVERCONFIDENT - your certainty often exceeds your accuracy.")
        lines.append("Consider expressing more uncertainty in your claims.")
    elif is_underconfident:
        lines.append(
            "You tend to be UNDERCONFIDENT - your accuracy is better than your expressed certainty."
        )
        lines.append("You can express more confidence in well-supported claims.")

    lines.append("\nAdjust your certainty levels in this debate accordingly.")

    return "\n".join(lines)


def format_belief_context(
    converged_beliefs: list[dict],
    limit: int = 3,
) -> str:
    """Format converged beliefs from belief network for context.

    Args:
        converged_beliefs: List of belief dicts with statement and confidence
        limit: Maximum number of beliefs to include

    Returns:
        Formatted belief context string
    """
    if not converged_beliefs:
        return ""

    lines = ["## ESTABLISHED BELIEFS"]
    lines.append("These positions have achieved network consensus:\n")

    for belief in converged_beliefs[:limit]:
        statement = belief.get("statement", "")
        confidence = belief.get("confidence", 0.0)
        support_count = belief.get("support_count", 0)

        lines.append(f"- {statement}")
        lines.append(f"  Confidence: {confidence:.0%} | Supporters: {support_count}")

    lines.append("\nConsider how your position aligns with or challenges these beliefs.")
    return "\n".join(lines)
