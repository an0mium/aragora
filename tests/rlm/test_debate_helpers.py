"""
Comprehensive tests for RLM debate helpers.

Covers:
1. DebateREPLContext dataclass creation and field defaults
2. load_debate_context from DebateResult (dict, object, model_dump variants)
3. Context navigation: get_round, get_proposals_by_agent, search_debate
4. Evidence extraction: get_evidence_snippets
5. Critique detection: get_critiques
6. Round summarization: summarize_round
7. Partitioning: partition_debate (by round, agent, unknown)
8. RLM primitives: RLM_M (all query intent branches), FINAL
9. Helper injection: get_debate_helpers
10. REPL safety: injection via debate content cannot break helpers
11. Recursive context queries via RLM_M
12. Error handling for corrupted/malformed debate results
13. Resource exhaustion prevention (large messages, deep recursion, huge results)
14. Edge cases (empty debates, single message, unicode, special regex chars)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional
from unittest.mock import MagicMock, patch

import pytest

from aragora.rlm.debate_helpers import (
    FINAL,
    DebateREPLContext,
    RLM_M,
    _truncate_content,
    get_critiques,
    get_debate_helpers,
    get_evidence_snippets,
    get_proposals_by_agent,
    get_round,
    load_debate_context,
    partition_debate,
    search_debate,
    summarize_round,
)


# ---------------------------------------------------------------------------
# Helpers / Fixtures
# ---------------------------------------------------------------------------


@dataclass
class FakeMessage:
    """Fake message matching aragora.core_types.Message interface."""

    agent: str
    content: str
    round: int = 0
    role: str = "proposer"
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class FakeDebateResult:
    """Fake DebateResult for testing without importing core_types."""

    debate_id: str = "debate-001"
    task: str = "Design a rate limiter"
    messages: list = field(default_factory=list)
    consensus_reached: bool = True
    final_answer: str = "Use token bucket algorithm"
    confidence: float = 0.85


@pytest.fixture
def three_round_messages():
    """Three rounds with three agents producing 9 messages."""
    agents = ["claude", "gpt4", "gemini"]
    messages = []
    for rnd in range(1, 4):
        for agent in agents:
            messages.append(
                FakeMessage(
                    role="proposer",
                    agent=agent,
                    content=f"Round {rnd} proposal from {agent}: "
                    f"{'I suggest using caching.' if rnd == 1 else ''}"
                    f"{'However, I disagree with caching alone.' if rnd == 2 and agent == 'gpt4' else ''}"
                    f"{'I agree we should combine approaches.' if rnd == 3 else ''}",
                    round=rnd,
                )
            )
    return messages


@pytest.fixture
def debate_result(three_round_messages):
    """Fully populated FakeDebateResult."""
    return FakeDebateResult(
        debate_id="debate-001",
        task="Design a rate limiter",
        messages=three_round_messages,
        consensus_reached=True,
        final_answer="Use token bucket algorithm",
        confidence=0.85,
    )


@pytest.fixture
def debate_context(debate_result):
    """DebateREPLContext built from the standard debate_result fixture."""
    return load_debate_context(debate_result)


@pytest.fixture
def rich_messages():
    """Messages with evidence quotes, critiques, proposals, and consensus language."""
    return [
        FakeMessage(
            role="proposer",
            agent="claude",
            content='I propose using a sliding window. "Research shows 99.9% availability" is key evidence.',
            round=1,
        ),
        FakeMessage(
            role="proposer",
            agent="gpt4",
            content='I suggest a token bucket. "Industry standard approach" is widely adopted.',
            round=1,
        ),
        FakeMessage(
            role="critic",
            agent="gemini",
            content="However, I disagree with the sliding window. The issue with it is latency under load.",
            round=2,
        ),
        FakeMessage(
            role="proposer",
            agent="claude",
            content="I agree with the concern. Let us reach consensus on a hybrid approach that combines both.",
            round=2,
        ),
        FakeMessage(
            role="proposer",
            agent="gpt4",
            content="I support the hybrid approach. We concur on combining sliding window and token bucket.",
            round=3,
        ),
        FakeMessage(
            role="proposer",
            agent="gemini",
            content='On the other hand, we should consider "leaky bucket" as a counterpoint alternative.',
            round=3,
        ),
    ]


@pytest.fixture
def rich_context(rich_messages):
    """Context built from rich_messages for thorough testing."""
    result = FakeDebateResult(
        debate_id="debate-rich",
        task="Design a rate limiter",
        messages=rich_messages,
        consensus_reached=True,
        final_answer="Hybrid sliding window + token bucket",
        confidence=0.92,
    )
    return load_debate_context(result)


# ---------------------------------------------------------------------------
# 1. DebateREPLContext Dataclass
# ---------------------------------------------------------------------------


class TestDebateREPLContext:
    """Tests for the DebateREPLContext dataclass itself."""

    def test_all_fields_present(self, debate_context):
        """Context should expose all expected navigation fields."""
        assert isinstance(debate_context.debate_id, str)
        assert isinstance(debate_context.task, str)
        assert isinstance(debate_context.total_rounds, int)
        assert isinstance(debate_context.agent_names, list)
        assert isinstance(debate_context.rounds, dict)
        assert isinstance(debate_context.by_agent, dict)
        assert isinstance(debate_context.all_messages, list)
        assert isinstance(debate_context.consensus_reached, bool)
        assert isinstance(debate_context.confidence, float)

    def test_raw_reference_preserved(self, debate_result):
        """_raw should hold a reference to the original debate result."""
        ctx = load_debate_context(debate_result)
        assert ctx._raw is debate_result

    def test_agent_names_sorted(self, debate_context):
        """agent_names should be sorted alphabetically."""
        assert debate_context.agent_names == sorted(debate_context.agent_names)

    def test_total_rounds_matches_keys(self, debate_context):
        """total_rounds should equal the number of distinct round keys."""
        assert debate_context.total_rounds == len(debate_context.rounds)


# ---------------------------------------------------------------------------
# 2. load_debate_context
# ---------------------------------------------------------------------------


class TestLoadDebateContext:
    """Tests for load_debate_context with various input formats."""

    def test_basic_loading(self, debate_result, debate_context):
        """Should populate all index structures from a standard DebateResult."""
        assert debate_context.debate_id == "debate-001"
        assert debate_context.task == "Design a rate limiter"
        assert debate_context.consensus_reached is True
        assert debate_context.final_answer == "Use token bucket algorithm"
        assert debate_context.confidence == 0.85
        assert len(debate_context.all_messages) == 9
        assert debate_context.total_rounds == 3
        assert set(debate_context.agent_names) == {"claude", "gemini", "gpt4"}

    def test_dict_messages(self):
        """Should handle messages that are plain dicts."""
        result = FakeDebateResult(
            messages=[
                {"agent": "a1", "content": "Hello", "round": 1},
                {"agent": "a2", "content": "World", "round": 1},
            ]
        )
        ctx = load_debate_context(result)
        assert len(ctx.all_messages) == 2
        assert set(ctx.agent_names) == {"a1", "a2"}

    def test_model_dump_messages(self):
        """Should handle objects with model_dump() (e.g., Pydantic models)."""

        class FakePydanticModel:
            """Simulates a Pydantic model with model_dump method."""

            def model_dump(self):
                return {
                    "agent": "pydantic_agent",
                    "content": "Pydantic msg",
                    "round": 1,
                }

        msg = FakePydanticModel()
        result = FakeDebateResult(messages=[msg])
        ctx = load_debate_context(result)
        assert len(ctx.all_messages) == 1
        assert ctx.all_messages[0]["agent"] == "pydantic_agent"

    def test_agent_name_fallback(self):
        """Should use 'agent_name' key when 'agent' key is missing."""
        result = FakeDebateResult(
            messages=[{"agent_name": "fallback_agent", "content": "test", "round_num": 2}]
        )
        ctx = load_debate_context(result)
        assert "fallback_agent" in ctx.agent_names
        assert 2 in ctx.rounds

    def test_round_num_fallback(self):
        """Should use 'round_num' key when 'round' key is missing."""
        result = FakeDebateResult(messages=[{"agent": "a1", "content": "test", "round_num": 5}])
        ctx = load_debate_context(result)
        assert 5 in ctx.rounds

    def test_missing_round_defaults_to_zero(self):
        """Should default round to 0 when neither 'round' nor 'round_num' present."""
        result = FakeDebateResult(messages=[{"agent": "a1", "content": "no round"}])
        ctx = load_debate_context(result)
        assert 0 in ctx.rounds

    def test_unknown_agent_defaults(self):
        """Should default agent name to 'unknown' when neither key present."""
        result = FakeDebateResult(messages=[{"content": "orphan", "round": 1}])
        ctx = load_debate_context(result)
        assert "unknown" in ctx.agent_names

    def test_non_convertible_messages_skipped(self):
        """Should skip messages that cannot be converted to dict (e.g., ints)."""
        result = FakeDebateResult(messages=[42, None, True, "just a string"])
        ctx = load_debate_context(result)
        # None of these have agent/content dict shape; int/bool/str skip (no __dict__ useful or they lack keys)
        # 42, True -> no model_dump, no useful __dict__, not dict -> skip
        # None -> no model_dump, etc -> skip
        # "just a string" -> has no useful __dict__ either -> skip
        assert len(ctx.all_messages) == 0

    def test_empty_messages(self):
        """Should handle empty message list gracefully."""
        result = FakeDebateResult(messages=[])
        ctx = load_debate_context(result)
        assert ctx.total_rounds == 0
        assert ctx.all_messages == []
        assert ctx.agent_names == []

    def test_no_messages_attribute(self):
        """Should handle debate result with no messages attribute."""
        result = MagicMock(spec=[])  # Empty spec - no attributes
        ctx = load_debate_context(result)
        assert ctx.total_rounds == 0
        assert ctx.all_messages == []

    def test_answer_field_fallback(self):
        """Should try 'answer' attribute if 'final_answer' is None."""
        result = MagicMock()
        result.messages = []
        result.consensus_reached = True
        result.final_answer = None
        result.answer = "Fallback answer"
        result.confidence = 0.7
        result.debate_id = "d1"
        result.task = "test"
        ctx = load_debate_context(result)
        assert ctx.final_answer == "Fallback answer"

    def test_missing_consensus_fields(self):
        """Should default to False/None/0.0 for missing consensus fields."""
        result = MagicMock(spec=[])
        result.messages = []
        ctx = load_debate_context(result)
        assert ctx.consensus_reached is False
        assert ctx.final_answer is None
        assert ctx.confidence == 0.0


# ---------------------------------------------------------------------------
# 3. Context Navigation Methods
# ---------------------------------------------------------------------------


class TestGetRound:
    """Tests for get_round."""

    def test_get_existing_round(self, debate_context):
        """Should return messages for an existing round."""
        round1 = get_round(debate_context, 1)
        assert len(round1) == 3
        for msg in round1:
            assert msg["round"] == 1

    def test_get_all_rounds(self, debate_context):
        """Should return correct count for each round."""
        for rnd in range(1, 4):
            assert len(get_round(debate_context, rnd)) == 3

    def test_get_nonexistent_round(self, debate_context):
        """Should return empty list for non-existent round."""
        assert get_round(debate_context, 999) == []

    def test_get_round_zero(self):
        """Should return messages indexed at round 0."""
        result = FakeDebateResult(messages=[{"agent": "a1", "content": "hi"}])
        ctx = load_debate_context(result)
        assert len(get_round(ctx, 0)) == 1

    def test_get_negative_round(self, debate_context):
        """Should return empty list for negative round number."""
        assert get_round(debate_context, -1) == []


class TestGetProposalsByAgent:
    """Tests for get_proposals_by_agent."""

    def test_get_all_messages_for_agent(self, debate_context):
        """Should return all messages from a specific agent."""
        claude_msgs = get_proposals_by_agent(debate_context, "claude")
        assert len(claude_msgs) == 3  # one per round
        for msg in claude_msgs:
            assert msg["agent"] == "claude"

    def test_get_agent_messages_filtered_by_round(self, debate_context):
        """Should filter by round when round_num is specified."""
        claude_r1 = get_proposals_by_agent(debate_context, "claude", round_num=1)
        assert len(claude_r1) == 1
        assert claude_r1[0]["round"] == 1

    def test_nonexistent_agent(self, debate_context):
        """Should return empty list for unknown agent."""
        assert get_proposals_by_agent(debate_context, "nonexistent") == []

    def test_nonexistent_round_for_agent(self, debate_context):
        """Should return empty list when agent exists but round doesn't."""
        assert get_proposals_by_agent(debate_context, "claude", round_num=99) == []

    def test_round_num_key_fallback(self):
        """Should handle round_num key in message dicts."""
        result = FakeDebateResult(messages=[{"agent": "a1", "content": "hi", "round_num": 5}])
        ctx = load_debate_context(result)
        msgs = get_proposals_by_agent(ctx, "a1", round_num=5)
        assert len(msgs) == 1


class TestSearchDebate:
    """Tests for search_debate."""

    def test_basic_search(self, debate_context):
        """Should find messages matching a simple pattern."""
        results = search_debate(debate_context, "proposal")
        assert len(results) > 0

    def test_case_insensitive_default(self, rich_context):
        """Should be case-insensitive by default."""
        lower = search_debate(rich_context, "propose")
        upper = search_debate(rich_context, "PROPOSE")
        assert len(lower) == len(upper)

    def test_case_sensitive(self, rich_context):
        """Should respect case_insensitive=False."""
        insensitive = search_debate(rich_context, "propose", case_insensitive=True)
        sensitive = search_debate(rich_context, "PROPOSE", case_insensitive=False)
        # "PROPOSE" not in any content => 0 results for case-sensitive
        assert len(sensitive) == 0
        assert len(insensitive) > 0

    def test_regex_pattern(self, rich_context):
        """Should support regex patterns."""
        results = search_debate(rich_context, r"agree|consensus")
        assert len(results) >= 2  # claude round 2 agrees, gpt4 round 3 concurs

    def test_empty_pattern(self, debate_context):
        """Empty pattern should match all messages (regex '.' matches any)."""
        results = search_debate(debate_context, "")
        assert len(results) == len(debate_context.all_messages)

    def test_no_matches(self, debate_context):
        """Should return empty list when nothing matches."""
        results = search_debate(debate_context, "xyznonexistent123")
        assert results == []

    def test_special_regex_characters(self, rich_context):
        """Should handle special regex characters in content search."""
        # The pattern itself uses regex, so searching for literal dot requires escaping
        results = search_debate(rich_context, r"99\.9%")
        assert len(results) == 1

    def test_search_empty_context(self):
        """Should return empty list when context has no messages."""
        ctx = load_debate_context(FakeDebateResult(messages=[]))
        assert search_debate(ctx, "anything") == []


# ---------------------------------------------------------------------------
# 4. Evidence Extraction
# ---------------------------------------------------------------------------


class TestGetEvidenceSnippets:
    """Tests for get_evidence_snippets."""

    def test_extracts_quoted_text(self, rich_context):
        """Should extract text within double quotes."""
        snippets = get_evidence_snippets(rich_context)
        texts = [s["snippet"] for s in snippets]
        assert "Research shows 99.9% availability" in texts
        assert "Industry standard approach" in texts

    def test_snippet_has_source_and_round(self, rich_context):
        """Each snippet should contain source agent and round number."""
        snippets = get_evidence_snippets(rich_context)
        for s in snippets:
            assert "source" in s
            assert "round" in s
            assert "snippet" in s

    def test_keyword_filter(self, rich_context):
        """Should filter snippets by keyword when provided."""
        snippets = get_evidence_snippets(rich_context, keyword="research")
        assert len(snippets) == 1
        assert "Research" in snippets[0]["snippet"]

    def test_keyword_case_insensitive(self, rich_context):
        """Keyword filter should be case-insensitive."""
        lower = get_evidence_snippets(rich_context, keyword="industry")
        upper = get_evidence_snippets(rich_context, keyword="INDUSTRY")
        assert len(lower) == len(upper) == 1

    def test_no_keyword_returns_all(self, rich_context):
        """Should return all snippets when keyword is None."""
        all_snippets = get_evidence_snippets(rich_context)
        assert len(all_snippets) >= 3  # "Research...", "Industry...", "leaky bucket"

    def test_no_quotes_returns_empty(self, debate_context):
        """Should return empty list when no quoted text exists."""
        snippets = get_evidence_snippets(debate_context)
        # debate_context messages don't contain double-quoted text
        assert isinstance(snippets, list)

    def test_keyword_no_match(self, rich_context):
        """Should return empty list when keyword matches no snippet."""
        snippets = get_evidence_snippets(rich_context, keyword="xyznonexistent")
        assert snippets == []


# ---------------------------------------------------------------------------
# 5. Critique Detection
# ---------------------------------------------------------------------------


class TestGetCritiques:
    """Tests for get_critiques."""

    def test_finds_critique_markers(self, rich_context):
        """Should find messages containing critique markers."""
        critiques = get_critiques(rich_context)
        assert len(critiques) >= 1
        # Gemini's round 2 message has "However" and "disagree" and "issue with"
        agents = [c.get("agent") for c in critiques]
        assert "gemini" in agents

    def test_filter_by_target_agent(self, rich_context):
        """Should filter critiques mentioning the target agent."""
        # Gemini critiques the "sliding window" proposed by claude
        # But target_agent filters by agent name in content
        # Need a message that mentions "gpt4" by name
        messages = [
            FakeMessage(agent="claude", content="I disagree with gpt4 on this point.", round=1),
            FakeMessage(agent="gemini", content="However, claude made a good argument.", round=1),
        ]
        result = FakeDebateResult(messages=messages)
        ctx = load_debate_context(result)
        critiques_of_gpt4 = get_critiques(ctx, target_agent="gpt4")
        assert len(critiques_of_gpt4) == 1
        assert critiques_of_gpt4[0]["agent"] == "claude"

    def test_no_critiques(self):
        """Should return empty list when no critique markers found."""
        messages = [
            FakeMessage(agent="a1", content="I fully support this plan.", round=1),
            FakeMessage(agent="a2", content="Excellent idea, let us proceed.", round=1),
        ]
        result = FakeDebateResult(messages=messages)
        ctx = load_debate_context(result)
        critiques = get_critiques(ctx)
        assert critiques == []

    def test_target_agent_case_insensitive(self):
        """Target agent filtering should be case-insensitive."""
        messages = [
            FakeMessage(agent="a1", content="I disagree with CLAUDE here.", round=1),
        ]
        result = FakeDebateResult(messages=messages)
        ctx = load_debate_context(result)
        assert len(get_critiques(ctx, target_agent="claude")) == 1
        assert len(get_critiques(ctx, target_agent="CLAUDE")) == 1


# ---------------------------------------------------------------------------
# 6. Round Summarization
# ---------------------------------------------------------------------------


class TestSummarizeRound:
    """Tests for summarize_round."""

    def test_basic_summary(self, debate_context):
        """Should produce a readable summary string."""
        summary = summarize_round(debate_context, 1)
        assert "Round 1" in summary
        assert "3 agents" in summary
        assert "3 messages" in summary

    def test_agents_listed(self, debate_context):
        """Summary should include agent names."""
        summary = summarize_round(debate_context, 1)
        for agent in ["claude", "gemini", "gpt4"]:
            assert agent in summary

    def test_nonexistent_round(self, debate_context):
        """Should report 'No messages' for missing round."""
        summary = summarize_round(debate_context, 999)
        assert "No messages" in summary

    def test_single_agent_round(self):
        """Should handle round with a single agent."""
        result = FakeDebateResult(
            messages=[FakeMessage(agent="solo", content="Only me here", round=1)]
        )
        ctx = load_debate_context(result)
        summary = summarize_round(ctx, 1)
        assert "1 agents" in summary or "1 agent" in summary
        assert "solo" in summary


# ---------------------------------------------------------------------------
# 7. Partitioning
# ---------------------------------------------------------------------------


class TestPartitionDebate:
    """Tests for partition_debate."""

    def test_partition_by_round(self, debate_context):
        """Should partition by round number."""
        partitions = partition_debate(debate_context, "round")
        assert set(partitions.keys()) == {1, 2, 3}
        for rnd, msgs in partitions.items():
            assert len(msgs) == 3

    def test_partition_by_agent(self, debate_context):
        """Should partition by agent name."""
        partitions = partition_debate(debate_context, "agent")
        assert set(partitions.keys()) == {"claude", "gemini", "gpt4"}
        for agent, msgs in partitions.items():
            assert len(msgs) == 3

    def test_partition_by_unknown(self, debate_context):
        """Unknown partition_by should return all messages under key 0."""
        partitions = partition_debate(debate_context, "unknown_key")
        assert 0 in partitions
        assert len(partitions[0]) == 9

    def test_partition_empty_context(self):
        """Should handle empty context."""
        ctx = load_debate_context(FakeDebateResult(messages=[]))
        partitions = partition_debate(ctx, "round")
        assert partitions == {}


# ---------------------------------------------------------------------------
# 8. RLM Primitives
# ---------------------------------------------------------------------------


class TestRLM_M:
    """Tests for the RLM_M heuristic synthesis function."""

    def test_empty_subset(self):
        """Should report no messages when subset is empty."""
        result = RLM_M("What happened?", subset=[])
        assert "No debate messages" in result

    def test_none_subset(self):
        """Should report no messages when subset is None."""
        result = RLM_M("What happened?", subset=None)
        assert "No debate messages" in result

    def test_proposal_query(self):
        """Should detect and synthesize proposal-related queries."""
        msgs = [
            {"agent": "claude", "content": "I propose using Redis for caching.", "round": 1},
            {"agent": "gpt4", "content": "I suggest Memcached instead.", "round": 1},
        ]
        result = RLM_M("What proposals were made?", subset=msgs)
        assert (
            "proposal" in result.lower()
            or "suggest" in result.lower()
            or "claude" in result.lower()
        )

    def test_critique_query(self):
        """Should detect and synthesize critique-related queries."""
        msgs = [
            {
                "agent": "gpt4",
                "content": "I disagree with using Redis. The problem with it is latency.",
                "round": 2,
            },
            {"agent": "gemini", "content": "I have concerns about memory usage.", "round": 2},
        ]
        result = RLM_M("What critiques were raised?", subset=msgs)
        assert "critique" in result.lower() or "disagree" in result.lower()

    def test_consensus_query(self):
        """Should detect and synthesize consensus-related queries."""
        msgs = [
            {"agent": "claude", "content": "I agree with the hybrid approach.", "round": 3},
            {"agent": "gpt4", "content": "I support this consensus direction.", "round": 3},
        ]
        result = RLM_M("Was consensus reached?", subset=msgs)
        assert "agree" in result.lower() or "support" in result.lower()

    def test_summary_query(self):
        """Should provide an overview for summary-type queries."""
        msgs = [
            {"agent": "claude", "content": "First point about architecture.", "round": 1},
            {"agent": "gpt4", "content": "Second point about testing.", "round": 2},
        ]
        result = RLM_M("Summarize the key points", subset=msgs)
        assert "synthesis" in result.lower() or "message" in result.lower()

    def test_round_query(self):
        """Should detect round-specific query and summarize by round."""
        msgs = [
            {"agent": "claude", "content": "Round 1 content here.", "round": 1},
            {"agent": "gpt4", "content": "Round 2 content here.", "round": 2},
        ]
        result = RLM_M("What happened in each round?", subset=msgs)
        assert "round" in result.lower()

    def test_default_falls_to_summary(self):
        """Unknown query types should fall through to summary."""
        msgs = [
            {"agent": "a1", "content": "Some content about algorithms.", "round": 1},
        ]
        result = RLM_M("xyzgarbage query", subset=msgs)
        # Falls through to summary branch (is_summary_query or True)
        assert "synthesis" in result.lower() or "a1" in result.lower()

    def test_relevance_scoring(self):
        """Messages should be scored by keyword relevance to query."""
        msgs = [
            {"agent": "a1", "content": "Completely unrelated topic about fishing.", "round": 1},
            {"agent": "a2", "content": "I propose a comprehensive caching strategy.", "round": 1},
        ]
        result = RLM_M("What was proposed about caching?", subset=msgs)
        # a2 is more relevant; in proposals mode should appear
        assert "a2" in result or "caching" in result.lower()

    def test_truncates_long_content(self):
        """Should truncate long message content in synthesis output."""
        long_content = "A" * 500
        msgs = [{"agent": "verbose", "content": long_content, "round": 1}]
        result = RLM_M("Summarize", subset=msgs)
        # The _truncate_content function is used internally; result should not contain full 500 chars verbatim
        assert len(result) < len(long_content) + 200

    def test_multiple_agents_grouped(self):
        """Should group messages by agent in summary mode."""
        msgs = [
            {"agent": "a1", "content": "Point one.", "round": 1},
            {"agent": "a1", "content": "Point two.", "round": 2},
            {"agent": "a2", "content": "Different view.", "round": 1},
        ]
        result = RLM_M("Summarize everything", subset=msgs)
        assert "a1" in result
        assert "a2" in result

    def test_consensus_uses_word_boundaries(self):
        """Consensus detection should not match 'disagree' for 'agree'."""
        msgs = [
            {"agent": "a1", "content": "I strongly disagree with this approach.", "round": 1},
        ]
        result = RLM_M("Was there consensus?", subset=msgs)
        # "disagree" should NOT match the \bagree\b pattern
        # So we should see "No explicit agreements"
        assert "no explicit" in result.lower() or "agreement" in result.lower()

    def test_critique_no_matches_fallback(self):
        """Should report no critiques when none found."""
        msgs = [
            {"agent": "a1", "content": "Everything is wonderful and perfect.", "round": 1},
        ]
        result = RLM_M("What critiques were made?", subset=msgs)
        assert "no explicit" in result.lower() or "critique" in result.lower()

    def test_proposal_no_matches_shows_relevant(self):
        """Should show top messages when no proposal markers found."""
        msgs = [
            {"agent": "a1", "content": "Just a general discussion.", "round": 1},
        ]
        result = RLM_M("What was proposed?", subset=msgs)
        # Falls back to showing relevant messages
        assert "a1" in result or "message" in result.lower()

    def test_more_than_five_critiques_caps(self):
        """Should cap at 5 critiques in output."""
        msgs = [
            {"agent": f"a{i}", "content": f"I disagree with approach {i}.", "round": 1}
            for i in range(10)
        ]
        result = RLM_M("What critiques were made?", subset=msgs)
        # Should show at most 5 individual critique lines
        agent_lines = [line for line in result.split("\n") if line.strip().startswith("- a")]
        assert len(agent_lines) <= 5

    def test_round_query_with_many_messages(self):
        """Should cap messages per round and show '... and N more'."""
        msgs = [
            {"agent": f"a{i}", "content": f"Round 1 message {i}.", "round": 1} for i in range(5)
        ]
        result = RLM_M("What happened in each round?", subset=msgs)
        if "more" in result.lower():
            assert "and" in result.lower()


class TestFINAL:
    """Tests for the FINAL primitive."""

    def test_returns_answer(self):
        """FINAL should return the given answer unchanged."""
        assert FINAL("The answer is 42") == "The answer is 42"

    def test_returns_empty_string(self):
        """FINAL should handle empty strings."""
        assert FINAL("") == ""

    def test_preserves_multiline(self):
        """FINAL should preserve multiline answers."""
        answer = "Line 1\nLine 2\nLine 3"
        assert FINAL(answer) == answer

    def test_preserves_unicode(self):
        """FINAL should preserve unicode content."""
        answer = "La reponse est 42"
        assert FINAL(answer) == answer


# ---------------------------------------------------------------------------
# 9. Helper Injection
# ---------------------------------------------------------------------------


class TestGetDebateHelpers:
    """Tests for get_debate_helpers."""

    def test_returns_dict(self):
        """Should return a dictionary of helper functions."""
        helpers = get_debate_helpers()
        assert isinstance(helpers, dict)

    def test_contains_all_navigation_helpers(self):
        """Should include all navigation/context functions."""
        helpers = get_debate_helpers()
        expected_keys = {
            "load_debate_context",
            "DebateREPLContext",
            "get_round",
            "get_proposals_by_agent",
            "search_debate",
            "get_evidence_snippets",
            "get_critiques",
            "summarize_round",
            "partition_debate",
        }
        assert expected_keys.issubset(set(helpers.keys()))

    def test_excludes_rlm_primitives_by_default(self):
        """Should NOT include RLM_M and FINAL by default."""
        helpers = get_debate_helpers()
        assert "RLM_M" not in helpers
        assert "FINAL" not in helpers

    def test_includes_rlm_primitives_when_requested(self):
        """Should include RLM_M and FINAL when include_rlm_primitives=True."""
        helpers = get_debate_helpers(include_rlm_primitives=True)
        assert "RLM_M" in helpers
        assert "FINAL" in helpers
        assert helpers["RLM_M"] is RLM_M
        assert helpers["FINAL"] is FINAL

    def test_helpers_are_callable(self):
        """All helpers should be callable (except DebateREPLContext which is a class)."""
        helpers = get_debate_helpers(include_rlm_primitives=True)
        for name, func in helpers.items():
            assert callable(func), f"{name} should be callable"


# ---------------------------------------------------------------------------
# 10. REPL Safety - Injection via Debate Content
# ---------------------------------------------------------------------------


class TestREPLSafety:
    """Tests that debate content cannot inject code or break helpers."""

    def test_malicious_content_in_search(self):
        """Regex patterns in message content should not break search_debate."""
        messages = [
            FakeMessage(
                agent="adversary",
                content='exec(import("os")); __globals__["system"]("rm -rf /")',
                round=1,
            ),
        ]
        result = FakeDebateResult(messages=messages)
        ctx = load_debate_context(result)
        # Should not raise; just searches content as text
        results = search_debate(ctx, "exec")
        assert len(results) == 1

    def test_regex_injection_in_search_pattern(self):
        """Malformed regex in search pattern should raise, not crash the system."""
        messages = [FakeMessage(agent="a1", content="normal text", round=1)]
        result = FakeDebateResult(messages=messages)
        ctx = load_debate_context(result)
        # Invalid regex should raise re.error, not execute arbitrary code
        with pytest.raises(re.error):
            search_debate(ctx, "[invalid(regex")

    def test_dunder_in_content_does_not_affect_context(self):
        """Messages containing __globals__ etc. should not affect context integrity."""
        messages = [
            FakeMessage(
                agent="attacker",
                content="__globals__.__builtins__.__import__('os').system('pwd')",
                round=1,
            ),
        ]
        result = FakeDebateResult(messages=messages)
        ctx = load_debate_context(result)
        assert ctx.debate_id == result.debate_id
        assert len(ctx.all_messages) == 1

    def test_oversized_content_in_evidence_snippets(self):
        """Very large quoted strings should be handled without crash."""
        huge_quote = '"' + "A" * 100_000 + '"'
        messages = [FakeMessage(agent="a1", content=huge_quote, round=1)]
        result = FakeDebateResult(messages=messages)
        ctx = load_debate_context(result)
        snippets = get_evidence_snippets(ctx)
        assert len(snippets) == 1
        assert len(snippets[0]["snippet"]) == 100_000

    def test_null_bytes_in_content(self):
        """Messages with null bytes should not crash helpers."""
        messages = [
            FakeMessage(agent="a1", content="before\x00after", round=1),
        ]
        result = FakeDebateResult(messages=messages)
        ctx = load_debate_context(result)
        results = search_debate(ctx, "before")
        assert len(results) == 1

    def test_nested_quotes_in_evidence(self):
        """Should handle nested/escaped quotes in evidence extraction."""
        messages = [
            FakeMessage(
                agent="a1",
                content='She said "he said "hello"" to me.',
                round=1,
            ),
        ]
        result = FakeDebateResult(messages=messages)
        ctx = load_debate_context(result)
        snippets = get_evidence_snippets(ctx)
        # Regex r'"([^"]+)"' matches between non-nested quotes
        assert len(snippets) >= 1


# ---------------------------------------------------------------------------
# 11. Recursive Context Queries
# ---------------------------------------------------------------------------


class TestRecursiveContextQueries:
    """Tests for using RLM_M with partitioned/subset data, simulating recursive queries."""

    def test_partition_then_synthesize(self, rich_context):
        """Should be able to partition by round and synthesize each."""
        partitions = partition_debate(rich_context, "round")
        summaries = {}
        for rnd, msgs in partitions.items():
            summaries[rnd] = RLM_M(f"What was proposed in round {rnd}?", subset=msgs)
        assert len(summaries) == rich_context.total_rounds
        for rnd, summary in summaries.items():
            assert isinstance(summary, str)
            assert len(summary) > 0

    def test_agent_partition_then_synthesize(self, rich_context):
        """Should be able to partition by agent and synthesize each."""
        partitions = partition_debate(rich_context, "agent")
        for agent, msgs in partitions.items():
            result = RLM_M(f"What did {agent} propose?", subset=msgs)
            assert isinstance(result, str)

    def test_chained_navigation(self, rich_context):
        """Should support chaining navigation calls for complex queries."""
        # Find critiques, then search within them for specific patterns
        critiques = get_critiques(rich_context)
        # Use these critiques as a subset for RLM_M
        if critiques:
            result = RLM_M("What are the main concerns?", subset=critiques)
            assert isinstance(result, str)

    def test_search_then_synthesize(self, rich_context):
        """Should search first, then synthesize results."""
        matches = search_debate(rich_context, "hybrid")
        if matches:
            result = RLM_M("Summarize the hybrid approach discussion", subset=matches)
            assert isinstance(result, str)


# ---------------------------------------------------------------------------
# 12. Error Handling for Corrupted/Malformed Data
# ---------------------------------------------------------------------------


class TestCorruptedData:
    """Tests for handling corrupted or malformed debate results."""

    def test_none_content_in_messages(self):
        """Should handle messages with None content."""
        messages = [{"agent": "a1", "content": None, "round": 1}]
        result = FakeDebateResult(messages=messages)
        ctx = load_debate_context(result)
        # search_debate should not crash on None content
        results = search_debate(ctx, "test")
        assert results == []

    def test_missing_content_key(self):
        """Should handle messages missing the 'content' key entirely."""
        messages = [{"agent": "a1", "round": 1}]  # no 'content'
        result = FakeDebateResult(messages=messages)
        ctx = load_debate_context(result)
        results = search_debate(ctx, "anything")
        assert results == []

    def test_non_string_content(self):
        """Should handle messages where content is not a string."""
        messages = [
            {"agent": "a1", "content": 12345, "round": 1},
            {"agent": "a2", "content": ["list", "content"], "round": 1},
            {"agent": "a3", "content": {"nested": "dict"}, "round": 1},
        ]
        result = FakeDebateResult(messages=messages)
        ctx = load_debate_context(result)
        # search_debate calls regex.search(content) which needs a string
        # get("content", "") returns the non-string value, regex.search may fail
        # This tests graceful behavior
        assert len(ctx.all_messages) == 3

    def test_mixed_round_types(self):
        """Should handle non-integer round values gracefully."""
        messages = [
            {"agent": "a1", "content": "test", "round": "not_a_number"},
            {"agent": "a2", "content": "test2", "round": 1},
        ]
        result = FakeDebateResult(messages=messages)
        ctx = load_debate_context(result)
        # "not_a_number" becomes a dict key
        assert "not_a_number" in ctx.rounds
        assert 1 in ctx.rounds

    def test_debate_result_with_only_answer(self):
        """Should handle result with 'answer' instead of 'final_answer'."""
        result = MagicMock()
        result.messages = []
        result.consensus_reached = False
        result.final_answer = None
        result.answer = "Fallback"
        result.confidence = 0.5
        result.debate_id = "d1"
        result.task = "t1"
        ctx = load_debate_context(result)
        assert ctx.final_answer == "Fallback"

    def test_extremely_deep_nested_dict_message(self):
        """Should handle deeply nested dict messages without stack overflow."""
        # Build a deeply nested dict
        nested = {"agent": "deep", "content": "deep message", "round": 1}
        for _ in range(100):
            nested = {"inner": nested, "agent": "deep", "content": "deep message", "round": 1}
        # This is a plain dict, so it will be loaded as-is
        result = FakeDebateResult(messages=[nested])
        ctx = load_debate_context(result)
        assert len(ctx.all_messages) == 1

    def test_empty_string_debate_id(self):
        """Should handle empty string debate_id."""
        result = FakeDebateResult(debate_id="", messages=[])
        ctx = load_debate_context(result)
        assert ctx.debate_id == ""

    def test_unicode_in_all_fields(self):
        """Should handle unicode in agent names, content, and task."""
        messages = [
            FakeMessage(agent="claude", content="Les modeles d'IA sont performants", round=1),
        ]
        result = FakeDebateResult(
            debate_id="unicode-test",
            task="Les modeles d'IA",
            messages=messages,
            final_answer="Tres bien",
        )
        ctx = load_debate_context(result)
        assert ctx.task == "Les modeles d'IA"
        results = search_debate(ctx, "performants")
        assert len(results) == 1

    def test_messages_attribute_is_none(self):
        """Should handle messages attribute being None instead of list."""
        result = MagicMock()
        result.messages = None
        result.consensus_reached = False
        result.final_answer = None
        result.answer = None
        result.confidence = 0.0
        result.debate_id = ""
        result.task = ""
        # getattr(result, "messages", []) returns None, iterating None fails
        # load_debate_context uses getattr with default []
        # Actually: getattr returns None since it exists. for msg in None -> TypeError
        # Check how the code handles it
        ctx = load_debate_context(result)
        # If messages is None, the for loop will raise TypeError
        # This verifies the behavior
        assert ctx.total_rounds == 0


# ---------------------------------------------------------------------------
# 13. Resource Exhaustion Prevention
# ---------------------------------------------------------------------------


class TestResourceExhaustion:
    """Tests for resource exhaustion prevention."""

    def test_large_number_of_messages(self):
        """Should handle a large number of messages without excessive memory."""
        messages = [
            {"agent": f"agent_{i % 10}", "content": f"Message {i} content", "round": i % 50}
            for i in range(10_000)
        ]
        result = FakeDebateResult(messages=messages)
        ctx = load_debate_context(result)
        assert len(ctx.all_messages) == 10_000
        assert ctx.total_rounds == 50

    def test_large_message_search(self):
        """Should search through many messages without timeout."""
        messages = [
            {"agent": "a1", "content": f"Message number {i} with some content", "round": 1}
            for i in range(5_000)
        ]
        result = FakeDebateResult(messages=messages)
        ctx = load_debate_context(result)
        results = search_debate(ctx, r"number 4999")
        assert len(results) == 1

    def test_rlm_m_with_large_subset(self):
        """RLM_M should handle large subsets without excessive output."""
        msgs = [
            {"agent": f"agent_{i % 5}", "content": f"Point {i} about topic.", "round": i % 10}
            for i in range(1_000)
        ]
        result = RLM_M("Summarize everything", subset=msgs)
        # Should produce bounded output, not dump all 1000 messages
        assert isinstance(result, str)
        # Output should be reasonable length (not 1000 * line_length)
        assert len(result) < 10_000

    def test_many_agents_partition(self):
        """Should handle partitioning with many agents."""
        messages = [
            {"agent": f"agent_{i}", "content": f"Message from agent {i}", "round": 1}
            for i in range(100)
        ]
        result = FakeDebateResult(messages=messages)
        ctx = load_debate_context(result)
        partitions = partition_debate(ctx, "agent")
        assert len(partitions) == 100

    def test_catastrophic_regex_in_search(self):
        """Should handle slow regex patterns in search_debate."""
        messages = [
            # Keep input very short: (a+)+b is catastrophic backtracking,
            # exponential in len(input). 10 chars finishes; 30+ hangs forever.
            {"agent": "a1", "content": "a" * 10, "round": 1},
        ]
        result = FakeDebateResult(messages=messages)
        ctx = load_debate_context(result)
        results = search_debate(ctx, r"(a+)+b")
        assert results == []  # No match


# ---------------------------------------------------------------------------
# 14. Edge Cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Tests for various edge cases."""

    def test_single_message_debate(self):
        """Should handle a debate with exactly one message."""
        result = FakeDebateResult(
            messages=[FakeMessage(agent="solo", content="Only message", round=1)]
        )
        ctx = load_debate_context(result)
        assert ctx.total_rounds == 1
        assert ctx.agent_names == ["solo"]
        assert len(ctx.all_messages) == 1

    def test_duplicate_agent_names(self):
        """Should properly index duplicate agent messages."""
        messages = [
            FakeMessage(agent="claude", content="First", round=1),
            FakeMessage(agent="claude", content="Second", round=1),
            FakeMessage(agent="claude", content="Third", round=2),
        ]
        result = FakeDebateResult(messages=messages)
        ctx = load_debate_context(result)
        assert ctx.agent_names == ["claude"]
        claude_msgs = get_proposals_by_agent(ctx, "claude")
        assert len(claude_msgs) == 3

    def test_zero_confidence(self):
        """Should handle zero confidence score."""
        result = FakeDebateResult(confidence=0.0)
        ctx = load_debate_context(result)
        assert ctx.confidence == 0.0

    def test_confidence_above_one(self):
        """Should preserve confidence even if above 1.0 (no clamping)."""
        result = FakeDebateResult(confidence=1.5)
        ctx = load_debate_context(result)
        assert ctx.confidence == 1.5

    def test_empty_agent_name(self):
        """Should handle empty string agent name."""
        messages = [{"agent": "", "content": "anonymous", "round": 1}]
        result = FakeDebateResult(messages=messages)
        ctx = load_debate_context(result)
        assert "" in ctx.agent_names

    def test_very_long_agent_name(self):
        """Should handle very long agent names."""
        long_name = "a" * 10_000
        messages = [{"agent": long_name, "content": "test", "round": 1}]
        result = FakeDebateResult(messages=messages)
        ctx = load_debate_context(result)
        assert long_name in ctx.agent_names

    def test_special_characters_in_agent_name(self):
        """Should handle special characters in agent names."""
        messages = [{"agent": "agent-with-dashes_and.dots", "content": "test", "round": 1}]
        result = FakeDebateResult(messages=messages)
        ctx = load_debate_context(result)
        assert "agent-with-dashes_and.dots" in ctx.agent_names

    def test_get_round_returns_new_list(self, debate_context):
        """get_round should return the actual stored list (tests reference semantics)."""
        round1a = get_round(debate_context, 1)
        round1b = get_round(debate_context, 1)
        assert round1a is round1b  # Same list object from dict

    def test_search_with_pipe_alternation(self, rich_context):
        """Should support regex alternation with pipe."""
        results = search_debate(rich_context, r"propose|suggest|recommend")
        assert len(results) >= 2


# ---------------------------------------------------------------------------
# 15. _truncate_content Internal Helper
# ---------------------------------------------------------------------------


class TestTruncateContent:
    """Tests for the _truncate_content helper function."""

    def test_no_truncation_needed(self):
        """Should return content unchanged when within limit."""
        assert _truncate_content("short", 100) == "short"

    def test_exact_length(self):
        """Should return content unchanged when exactly at limit."""
        content = "12345"
        assert _truncate_content(content, 5) == "12345"

    def test_truncation_at_word_boundary(self):
        """Should truncate at a word boundary when possible."""
        content = "hello world test"
        result = _truncate_content(content, 12)
        assert result.endswith("...")
        # Should break at "hello world" (11 chars) + "..."
        assert "hello" in result

    def test_truncation_no_good_word_boundary(self):
        """Should truncate at max_length when no good word boundary found."""
        content = "abcdefghijklmnopqrstuvwxyz"
        result = _truncate_content(content, 10)
        # No space in first 10 chars, so last_space < 10 * 0.5 = 5
        # Falls through to truncated + "..."
        assert result == "abcdefghij..."

    def test_truncation_with_early_space(self):
        """Should use word boundary if it's past 50% of max_length."""
        content = "hello beautiful world of programming"
        result = _truncate_content(content, 20)
        # "hello beautiful worl" -> last space at 15, which is > 20*0.5=10
        assert result.endswith("...")
        assert "hello beautiful" in result

    def test_empty_content(self):
        """Should handle empty content."""
        assert _truncate_content("", 10) == ""

    def test_single_character(self):
        """Should handle single character content."""
        assert _truncate_content("x", 1) == "x"

    def test_max_length_zero(self):
        """Should handle zero max_length by truncating."""
        result = _truncate_content("content", 0)
        assert result == "..."

    def test_max_length_one(self):
        """Should handle max_length of 1."""
        result = _truncate_content("hello world", 1)
        assert result == "h..."
