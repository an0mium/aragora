"""
TRUE RLM REPL helpers for debate context navigation.

Based on arXiv:2512.24601 "Recursive Language Models":
These helpers enable LLMs to programmatically navigate debate context
stored as Python variables in a REPL environment.

Key Insight from the paper:
"Long prompts should not be fed into the neural network directly but should
instead be treated as part of the environment that the LLM can symbolically
interact with."

Usage in TRUE RLM REPL:
    # Context is stored as a variable, not in the prompt
    debate = load_debate_context(debate_result)

    # LLM writes code like this to examine context:
    proposals = get_proposals_by_agent(debate, "claude")
    matching = search_debate(debate, r"consensus|agree")
    round3 = get_round(debate, 3)

    # Recursive calls for complex queries
    summary = RLM_M("What are the key disagreements?", subset=round3)
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from aragora.core import DebateResult


@dataclass
class DebateREPLContext:
    """
    Structured debate context for TRUE RLM REPL navigation.

    The debate content is stored in Python data structures that the
    LLM can query programmatically rather than stuffing into prompts.
    """

    # Debate metadata
    debate_id: str
    task: str
    total_rounds: int
    agent_names: list[str]

    # Round-indexed messages
    rounds: dict[int, list[dict[str, Any]]]  # round_num -> messages

    # Agent-indexed messages
    by_agent: dict[str, list[dict[str, Any]]]  # agent_name -> messages

    # Full message list (for searching)
    all_messages: list[dict[str, Any]]

    # Outcome data
    consensus_reached: bool
    final_answer: str | None
    confidence: float

    # Original debate result (for full access)
    _raw: Optional["DebateResult"] = None


def load_debate_context(debate_result: "DebateResult") -> DebateREPLContext:
    """
    Load a DebateResult into a structured context for REPL navigation.

    This converts the debate result into indexed data structures that
    the LLM can query efficiently using Python code.

    Args:
        debate_result: The DebateResult from a completed debate

    Returns:
        DebateREPLContext with indexed access to debate content

    Example in TRUE RLM REPL:
        >>> debate = load_debate_context(result)
        >>> print(f"Agents: {debate.agent_names}")
        >>> print(f"Rounds: {debate.total_rounds}")
    """
    # Index by round
    rounds: dict[int, list[dict[str, Any]]] = {}
    # Index by agent
    by_agent: dict[str, list[dict[str, Any]]] = {}
    # All messages flat list
    all_messages: list[dict[str, Any]] = []

    agent_names: set[str] = set()

    # Process messages from debate result
    messages = getattr(debate_result, "messages", [])
    for msg in messages:
        # Normalize message to dict
        if hasattr(msg, "model_dump"):
            msg_dict = msg.model_dump()
        elif hasattr(msg, "__dict__"):
            msg_dict = dict(msg.__dict__)
        elif isinstance(msg, dict):
            msg_dict = msg
        else:
            continue

        # Extract round number (default to 0)
        round_num = msg_dict.get("round", msg_dict.get("round_num", 0))

        # Extract agent name
        agent = msg_dict.get("agent", msg_dict.get("agent_name", "unknown"))
        agent_names.add(agent)

        # Index by round
        if round_num not in rounds:
            rounds[round_num] = []
        rounds[round_num].append(msg_dict)

        # Index by agent
        if agent not in by_agent:
            by_agent[agent] = []
        by_agent[agent].append(msg_dict)

        # Add to flat list
        all_messages.append(msg_dict)

    # Extract consensus info
    consensus_reached = getattr(debate_result, "consensus_reached", False)
    final_answer = getattr(debate_result, "final_answer", None)
    if final_answer is None:
        final_answer = getattr(debate_result, "answer", None)
    confidence = getattr(debate_result, "confidence", 0.0)

    # Build context
    return DebateREPLContext(
        debate_id=getattr(debate_result, "debate_id", ""),
        task=getattr(debate_result, "task", ""),
        total_rounds=len(rounds),
        agent_names=sorted(agent_names),
        rounds=rounds,
        by_agent=by_agent,
        all_messages=all_messages,
        consensus_reached=consensus_reached,
        final_answer=final_answer,
        confidence=confidence,
        _raw=debate_result,
    )


def get_round(context: DebateREPLContext, round_num: int) -> list[dict[str, Any]]:
    """
    Get all messages from a specific round.

    Args:
        context: The debate REPL context
        round_num: Round number to retrieve (1-indexed)

    Returns:
        List of messages from that round

    Example in TRUE RLM REPL:
        >>> round1 = get_round(debate, 1)
        >>> for msg in round1:
        ...     print(f"{msg['agent']}: {msg['content'][:100]}...")
    """
    return context.rounds.get(round_num, [])


def get_proposals_by_agent(
    context: DebateREPLContext,
    agent_name: str,
    round_num: int | None = None,
) -> list[dict[str, Any]]:
    """
    Get all messages/proposals from a specific agent.

    Args:
        context: The debate REPL context
        agent_name: Name of the agent
        round_num: Optional round filter

    Returns:
        List of messages from that agent

    Example in TRUE RLM REPL:
        >>> claude_msgs = get_proposals_by_agent(debate, "claude")
        >>> print(f"Claude made {len(claude_msgs)} proposals")
    """
    messages = context.by_agent.get(agent_name, [])
    if round_num is not None:
        messages = [m for m in messages if m.get("round", m.get("round_num")) == round_num]
    return messages


def search_debate(
    context: DebateREPLContext,
    pattern: str,
    case_insensitive: bool = True,
) -> list[dict[str, Any]]:
    """
    Search debate messages using regex pattern.

    This is the "grep" operation from the RLM paper - allows the LLM
    to narrow down relevant context using pattern matching.

    Args:
        context: The debate REPL context
        pattern: Regex pattern to match
        case_insensitive: Whether to ignore case (default True)

    Returns:
        List of messages matching the pattern

    Example in TRUE RLM REPL:
        >>> disagreements = search_debate(debate, r"disagree|oppose|however")
        >>> consensus = search_debate(debate, r"agree|consensus|support")
    """
    flags = re.IGNORECASE if case_insensitive else 0
    regex = re.compile(pattern, flags)

    matching = []
    for msg in context.all_messages:
        content = msg.get("content", "")
        if regex.search(content):
            matching.append(msg)

    return matching


def get_evidence_snippets(
    context: DebateREPLContext,
    keyword: str | None = None,
) -> list[dict[str, str]]:
    """
    Extract evidence snippets mentioned in the debate.

    Args:
        context: The debate REPL context
        keyword: Optional keyword to filter evidence

    Returns:
        List of evidence snippets with source info

    Example in TRUE RLM REPL:
        >>> evidence = get_evidence_snippets(debate, "research")
        >>> for e in evidence:
        ...     print(f"[{e['source']}] {e['snippet']}")
    """
    snippets = []

    for msg in context.all_messages:
        content = msg.get("content", "")
        # Look for evidence markers (quotes, citations)
        quote_pattern = r'"([^"]+)"'
        for match in re.finditer(quote_pattern, content):
            snippet = match.group(1)
            if keyword is None or keyword.lower() in snippet.lower():
                snippets.append(
                    {
                        "snippet": snippet,
                        "source": msg.get("agent", "unknown"),
                        "round": msg.get("round", msg.get("round_num", 0)),
                    }
                )

    return snippets


def get_critiques(
    context: DebateREPLContext,
    target_agent: str | None = None,
) -> list[dict[str, Any]]:
    """
    Get critique messages from the debate.

    Args:
        context: The debate REPL context
        target_agent: Optional agent being critiqued

    Returns:
        List of critique messages

    Example in TRUE RLM REPL:
        >>> critiques = get_critiques(debate, target_agent="gpt4")
        >>> print(f"Found {len(critiques)} critiques of GPT-4")
    """
    critique_markers = [
        r"critique",
        r"disagree",
        r"however",
        r"but I think",
        r"on the other hand",
        r"counterpoint",
        r"issue with",
        r"problem with",
    ]
    pattern = "|".join(critique_markers)

    critiques = search_debate(context, pattern)

    if target_agent:
        critiques = [c for c in critiques if target_agent.lower() in c.get("content", "").lower()]

    return critiques


def summarize_round(context: DebateREPLContext, round_num: int) -> str:
    """
    Get a textual summary of a single round.

    Args:
        context: The debate REPL context
        round_num: Round to summarize

    Returns:
        Text summary of the round

    Example in TRUE RLM REPL:
        >>> print(summarize_round(debate, 1))
        "Round 1: 3 agents participated. Main points: ..."
    """
    messages = get_round(context, round_num)
    if not messages:
        return f"Round {round_num}: No messages"

    agents = list({m.get("agent", "unknown") for m in messages})
    return (
        f"Round {round_num}: {len(agents)} agents participated ({', '.join(agents)}). "
        f"{len(messages)} messages total."
    )


def partition_debate(
    context: DebateREPLContext,
    partition_by: str = "round",
) -> dict[Any, list[dict[str, Any]]]:
    """
    Partition debate messages for parallel processing.

    This is the "partition-map" operation from the RLM paper -
    allows splitting context for recursive sub-calls.

    Args:
        context: The debate REPL context
        partition_by: "round" or "agent"

    Returns:
        Partitioned messages

    Example in TRUE RLM REPL:
        >>> partitions = partition_debate(debate, "round")
        >>> # Process each round in parallel
        >>> for round_num, msgs in partitions.items():
        ...     result = RLM_M("Summarize key points", subset=msgs)
    """
    if partition_by == "round":
        return dict(context.rounds)
    elif partition_by == "agent":
        return dict(context.by_agent)
    else:
        # Default: single partition
        return {0: context.all_messages}


# RLM Primitives (for use in REPL)


def RLM_M(query: str, subset: list[dict[str, Any]] | None = None) -> str:
    """
    Recursive RLM call for synthesizing debate message subsets.

    In TRUE RLM, this triggers a recursive LLM call on a subset of context.
    When called outside a TRUE RLM REPL environment, this function provides
    a heuristic-based synthesis of the debate messages based on the query.

    Args:
        query: The query to answer (used to guide synthesis)
        subset: Optional subset of messages to synthesize

    Returns:
        Synthesized answer based on the subset and query

    Example in TRUE RLM REPL:
        >>> # LLM writes code like this:
        >>> round1_summary = RLM_M("What was proposed?", subset=get_round(debate, 1))
        >>> round2_summary = RLM_M("What critiques were made?", subset=get_round(debate, 2))
        >>> FINAL(f"Round 1: {round1_summary}. Round 2: {round2_summary}")

    Note:
        When used within a TRUE RLM REPL environment (via RLMEnvironment),
        this placeholder is replaced by the actual runtime's _rlm_call method
        which invokes a sub-LM for proper synthesis.
    """
    if subset is None or len(subset) == 0:
        return f"No debate messages provided for query: {query}"

    # Extract query keywords for relevance scoring
    query_lower = query.lower()
    query_words = set(query_lower.split())

    # Score and sort messages by relevance to query
    scored_messages: list[tuple[float, dict[str, Any]]] = []
    for msg in subset:
        content = msg.get("content", "")
        content_lower = content.lower()
        # Score based on keyword matches
        keyword_score = sum(1 for word in query_words if word in content_lower)
        # Boost for longer, more substantial messages
        length_score = min(len(content) / 500, 1.0) * 0.2
        relevance_score = keyword_score * 0.8 + length_score
        scored_messages.append((relevance_score, msg))

    # Sort by relevance (highest first)
    scored_messages.sort(key=lambda x: x[0], reverse=True)

    # Group messages by agent
    by_agent: dict[str, list[dict[str, Any]]] = {}
    for _, msg in scored_messages:
        agent = msg.get("agent", msg.get("agent_name", "unknown"))
        if agent not in by_agent:
            by_agent[agent] = []
        by_agent[agent].append(msg)

    # Group messages by round
    by_round: dict[int, list[dict[str, Any]]] = {}
    for _, msg in scored_messages:
        round_num = msg.get("round", msg.get("round_num", 0))
        if round_num not in by_round:
            by_round[round_num] = []
        by_round[round_num].append(msg)

    # Build synthesis based on query patterns
    synthesis_parts: list[str] = []

    # Detect query intent
    is_proposal_query = any(kw in query_lower for kw in ["propos", "suggest", "recommend", "idea"])
    is_critique_query = any(
        kw in query_lower for kw in ["critique", "disagree", "issue", "problem", "concern"]
    )
    is_consensus_query = any(kw in query_lower for kw in ["consensus", "agree", "common", "shared"])
    is_summary_query = any(
        kw in query_lower for kw in ["summarize", "summary", "overview", "key", "main"]
    )
    is_round_query = any(kw in query_lower for kw in ["round"])

    # Build response based on query type
    if is_critique_query:
        # Focus on critique patterns
        critique_markers = ["disagree", "however", "but", "issue", "problem", "concern", "critique"]
        critiques = [
            msg
            for _, msg in scored_messages
            if any(marker in msg.get("content", "").lower() for marker in critique_markers)
        ]
        if critiques:
            synthesis_parts.append(f"Found {len(critiques)} critique(s):")
            for msg in critiques[:5]:  # Top 5
                agent = msg.get("agent", msg.get("agent_name", "unknown"))
                content = msg.get("content", "")
                synthesis_parts.append(f"  - {agent}: {_truncate_content(content, 100)}")
        else:
            synthesis_parts.append("No explicit critiques found in the provided messages.")

    elif is_consensus_query:
        # Look for agreement patterns (use regex word boundaries to avoid false positives
        # like "disagree" matching the "agree" pattern)
        agreement_patterns = [
            r"\bagree\b",
            r"\bconsensus\b",
            r"\bsupport\b",
            r"\bconcur\b",
            r"\balign\b",
        ]
        agreements = [
            msg
            for _, msg in scored_messages
            if any(
                re.search(pattern, msg.get("content", ""), re.IGNORECASE)
                for pattern in agreement_patterns
            )
        ]
        if agreements:
            synthesis_parts.append(f"Found {len(agreements)} agreement(s):")
            for msg in agreements[:5]:  # Top 5
                agent = msg.get("agent", msg.get("agent_name", "unknown"))
                content = msg.get("content", "")
                synthesis_parts.append(f"  - {agent}: {_truncate_content(content, 100)}")
        else:
            synthesis_parts.append("No explicit agreements found in the provided messages.")

    elif is_proposal_query:
        # Focus on proposals
        proposal_markers = ["propose", "suggest", "recommend", "should", "could"]
        proposals = [
            msg
            for _, msg in scored_messages
            if any(marker in msg.get("content", "").lower() for marker in proposal_markers)
        ]
        if proposals:
            synthesis_parts.append(f"Found {len(proposals)} proposal(s):")
            for msg in proposals[:5]:  # Top 5
                agent = msg.get("agent", msg.get("agent_name", "unknown"))
                content = msg.get("content", "")
                synthesis_parts.append(f"  - {agent}: {_truncate_content(content, 100)}")
        else:
            # Fall back to showing top messages
            synthesis_parts.append(f"Showing {min(5, len(scored_messages))} relevant message(s):")
            for _, msg in scored_messages[:5]:
                agent = msg.get("agent", msg.get("agent_name", "unknown"))
                content = msg.get("content", "")
                synthesis_parts.append(f"  - {agent}: {_truncate_content(content, 100)}")

    elif is_round_query and by_round:
        # Summarize by round
        synthesis_parts.append(
            f"Debate summary ({len(subset)} messages across {len(by_round)} round(s)):"
        )
        for round_num in sorted(by_round.keys()):
            msgs = by_round[round_num]
            agents = list({m.get("agent", m.get("agent_name", "unknown")) for m in msgs})
            synthesis_parts.append(
                f"\nRound {round_num} ({len(msgs)} messages from {', '.join(agents)}):"
            )
            for msg in msgs[:2]:  # Top 2 per round
                agent = msg.get("agent", msg.get("agent_name", "unknown"))
                content = msg.get("content", "")
                synthesis_parts.append(f"  - {agent}: {_truncate_content(content, 80)}")
            if len(msgs) > 2:
                synthesis_parts.append(f"  ... and {len(msgs) - 2} more")

    elif is_summary_query or True:  # Default to summary
        # Provide overview by agent
        total_messages = len(subset)
        synthesis_parts.append(
            f"Debate synthesis ({total_messages} messages from {len(by_agent)} agent(s)):"
        )

        for agent, msgs in by_agent.items():
            synthesis_parts.append(f"\n{agent} ({len(msgs)} messages):")
            # Show top 2 most relevant messages per agent
            for msg in msgs[:2]:
                content = msg.get("content", "")
                round_num = msg.get("round", msg.get("round_num", "?"))
                synthesis_parts.append(f"  - [Round {round_num}] {_truncate_content(content, 80)}")
            if len(msgs) > 2:
                synthesis_parts.append(f"  ... and {len(msgs) - 2} more")

    return "\n".join(synthesis_parts)


def _truncate_content(content: str, max_length: int) -> str:
    """Truncate content to max_length, preserving word boundaries."""
    if len(content) <= max_length:
        return content
    # Find last space before max_length
    truncated = content[:max_length]
    last_space = truncated.rfind(" ")
    if last_space > max_length * 0.5:
        return truncated[:last_space] + "..."
    return truncated + "..."


def FINAL(answer: str) -> str:
    """
    Signal final answer in RLM.

    From the paper: The LLM produces FINAL(<ans>) when ready to terminate.

    Args:
        answer: The final answer

    Returns:
        The answer (for use in expressions)

    Example in TRUE RLM REPL:
        >>> FINAL("The main disagreement was about X vs Y")
    """
    # Placeholder - the RLM runtime captures this
    return answer


def get_debate_helpers(include_rlm_primitives: bool = False) -> dict[str, Any]:
    """
    Get all debate REPL helpers as a dictionary.

    This is used to inject helpers into a TRUE RLM REPL environment.

    Args:
        include_rlm_primitives: If True, include RLM_M/FINAL placeholders.
            Defaults to False because RLMEnvironment provides proper
            implementations that should NOT be overwritten.

    Returns:
        Dictionary of helper functions

    Example:
        >>> from aragora.rlm.debate_helpers import get_debate_helpers
        >>> helpers = get_debate_helpers()
        >>> rlm_env.inject_helpers(helpers)
    """
    helpers = {
        # Context loading
        "load_debate_context": load_debate_context,
        "DebateREPLContext": DebateREPLContext,
        # Navigation
        "get_round": get_round,
        "get_proposals_by_agent": get_proposals_by_agent,
        "search_debate": search_debate,
        "get_evidence_snippets": get_evidence_snippets,
        "get_critiques": get_critiques,
        "summarize_round": summarize_round,
        "partition_debate": partition_debate,
    }
    # Only include RLM primitives if explicitly requested.
    # RLMEnvironment provides proper implementations of RLM_M and FINAL
    # that integrate with the agent callback system.
    if include_rlm_primitives:
        helpers["RLM_M"] = RLM_M
        helpers["FINAL"] = FINAL
    return helpers


__all__ = [
    "DebateREPLContext",
    "load_debate_context",
    "get_round",
    "get_proposals_by_agent",
    "search_debate",
    "get_evidence_snippets",
    "get_critiques",
    "summarize_round",
    "partition_debate",
    "RLM_M",
    "FINAL",
    "get_debate_helpers",
]
