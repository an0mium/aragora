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
    final_answer: Optional[str]
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
    round_num: Optional[int] = None,
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
    keyword: Optional[str] = None,
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
    target_agent: Optional[str] = None,
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


def RLM_M(query: str, subset: Optional[list[dict[str, Any]]] = None) -> str:
    """
    Recursive RLM call placeholder.

    In TRUE RLM, this triggers a recursive LLM call on a subset of context.
    This placeholder is replaced by the actual RLM runtime.

    Args:
        query: The query to answer
        subset: Optional subset of messages to query

    Returns:
        Answer from recursive call

    Example in TRUE RLM REPL:
        >>> # LLM writes code like this:
        >>> round1_summary = RLM_M("What was proposed?", subset=get_round(debate, 1))
        >>> round2_summary = RLM_M("What critiques were made?", subset=get_round(debate, 2))
        >>> FINAL(f"Round 1: {round1_summary}. Round 2: {round2_summary}")
    """
    # Placeholder - replaced by actual RLM runtime
    raise NotImplementedError(
        "RLM_M must be called within a TRUE RLM REPL environment. "
        "Install with: pip install aragora[rlm]"
    )


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


def get_debate_helpers() -> dict[str, Any]:
    """
    Get all debate REPL helpers as a dictionary.

    This is used to inject helpers into a TRUE RLM REPL environment.

    Returns:
        Dictionary of helper functions

    Example:
        >>> from aragora.rlm.debate_helpers import get_debate_helpers
        >>> helpers = get_debate_helpers()
        >>> rlm_env.inject_helpers(helpers)
    """
    return {
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
        # RLM primitives
        "RLM_M": RLM_M,
        "FINAL": FINAL,
    }


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
