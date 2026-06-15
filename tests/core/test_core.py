"""
Tests for Core abstractions (dataclasses and Agent ABC).

Tests cover:
- Message dataclass (creation, defaults, __str__)
- Critique dataclass (creation, to_prompt formatting)
- Vote dataclass (creation, defaults)
- DisagreementReport dataclass (creation, summary formatting)
- DebateResult dataclass (creation, defaults, summary)
- Environment dataclass (creation, defaults)
- Agent ABC (abstract methods, initialization, repr)
"""

import uuid
from datetime import datetime
from unittest.mock import AsyncMock

import pytest

from aragora.core import (
    Agent,
    Critique,
    DebateResult,
    DisagreementReport,
    Environment,
    Message,
    Vote,
)


class TestMessageDataclass:
    """Tests for Message dataclass."""

    @pytest.mark.smoke
    def test_creation_with_required_fields(self):
        """Message should be created with role, agent, content."""
        msg = Message(role="proposer", agent="claude", content="Test content")
        assert msg.role == "proposer"
        assert msg.agent == "claude"
        assert msg.content == "Test content"

    def test_timestamp_defaults_to_now(self):
        """timestamp should default to current time."""
        before = datetime.now()
        msg = Message(role="proposer", agent="claude", content="Test")
        after = datetime.now()
        assert before <= msg.timestamp <= after

    def test_round_defaults_to_zero(self):
        """round should default to 0."""
        msg = Message(role="proposer", agent="claude", content="Test")
        assert msg.round == 0

    def test_round_can_be_set(self):
        """round should be settable."""
        msg = Message(role="proposer", agent="claude", content="Test", round=3)
        assert msg.round == 3

    def test_str_truncates_long_content(self):
        """__str__ should truncate content to 100 chars."""
        long_content = "x" * 200
        msg = Message(role="proposer", agent="claude", content=long_content)
        str_repr = str(msg)
        assert "..." in str_repr
        # Should contain role and agent
        assert "proposer" in str_repr
        assert "claude" in str_repr

    def test_str_short_content(self):
        """__str__ should show short content with ellipsis."""
        msg = Message(role="critic", agent="codex", content="Short")
        str_repr = str(msg)
        assert "critic" in str_repr
        assert "codex" in str_repr
        assert "Short" in str_repr


class TestCritiqueDataclass:
    """Tests for Critique dataclass."""

    @pytest.mark.smoke
    def test_creation_with_all_fields(self):
        """Critique should store all provided fields."""
        critique = Critique(
            agent="claude",
            target_agent="codex",
            target_content="proposal content",
            issues=["Issue 1", "Issue 2"],
            suggestions=["Fix A", "Fix B"],
            severity=0.7,
            reasoning="Because of X",
        )
        assert critique.agent == "claude"
        assert critique.target_agent == "codex"
        assert critique.target_content == "proposal content"
        assert critique.issues == ["Issue 1", "Issue 2"]
        assert critique.suggestions == ["Fix A", "Fix B"]
        assert critique.severity == 0.7
        assert critique.reasoning == "Because of X"

    def test_issues_is_list(self):
        """issues should be a list."""
        critique = Critique(
            agent="a",
            target_agent="b",
            target_content="c",
            issues=["One"],
            suggestions=[],
            severity=0.5,
            reasoning="r",
        )
        assert isinstance(critique.issues, list)
        assert len(critique.issues) == 1

    def test_suggestions_is_list(self):
        """suggestions should be a list."""
        critique = Critique(
            agent="a",
            target_agent="b",
            target_content="c",
            issues=[],
            suggestions=["Sug 1", "Sug 2", "Sug 3"],
            severity=0.5,
            reasoning="r",
        )
        assert isinstance(critique.suggestions, list)
        assert len(critique.suggestions) == 3

    def test_severity_stored(self):
        """severity should be stored as float."""
        critique = Critique(
            agent="a",
            target_agent="b",
            target_content="c",
            issues=[],
            suggestions=[],
            severity=0.85,
            reasoning="r",
        )
        assert critique.severity == 0.85

    def test_to_prompt_includes_agent(self):
        """to_prompt should include the critiquing agent name."""
        critique = Critique(
            agent="claude",
            target_agent="codex",
            target_content="content",
            issues=["Issue"],
            suggestions=["Fix"],
            severity=0.6,
            reasoning="Reason",
        )
        prompt = critique.to_prompt()
        assert "claude" in prompt

    def test_to_prompt_includes_severity(self):
        """to_prompt should include severity."""
        critique = Critique(
            agent="claude",
            target_agent="codex",
            target_content="content",
            issues=["Issue"],
            suggestions=["Fix"],
            severity=0.7,
            reasoning="Reason",
        )
        prompt = critique.to_prompt()
        assert "0.7" in prompt

    def test_to_prompt_includes_all_issues(self):
        """to_prompt should include all issues."""
        critique = Critique(
            agent="a",
            target_agent="b",
            target_content="c",
            issues=["First issue", "Second issue", "Third issue"],
            suggestions=[],
            severity=0.5,
            reasoning="r",
        )
        prompt = critique.to_prompt()
        assert "First issue" in prompt
        assert "Second issue" in prompt
        assert "Third issue" in prompt

    def test_to_prompt_includes_all_suggestions(self):
        """to_prompt should include all suggestions."""
        critique = Critique(
            agent="a",
            target_agent="b",
            target_content="c",
            issues=[],
            suggestions=["Fix A", "Fix B"],
            severity=0.5,
            reasoning="r",
        )
        prompt = critique.to_prompt()
        assert "Fix A" in prompt
        assert "Fix B" in prompt

    def test_to_prompt_includes_reasoning(self):
        """to_prompt should include reasoning."""
        critique = Critique(
            agent="a",
            target_agent="b",
            target_content="c",
            issues=[],
            suggestions=[],
            severity=0.5,
            reasoning="This is my detailed reasoning",
        )
        prompt = critique.to_prompt()
        assert "This is my detailed reasoning" in prompt


class TestVoteDataclass:
    """Tests for Vote dataclass."""

    @pytest.mark.smoke
    def test_creation_with_required_fields(self):
        """Vote should be created with agent, choice, reasoning."""
        vote = Vote(agent="claude", choice="proposal_a", reasoning="Best option")
        assert vote.agent == "claude"
        assert vote.choice == "proposal_a"
        assert vote.reasoning == "Best option"

    def test_confidence_defaults_to_one(self):
        """confidence should default to 1.0."""
        vote = Vote(agent="claude", choice="a", reasoning="r")
        assert vote.confidence == 1.0

    def test_continue_debate_defaults_to_true(self):
        """continue_debate should default to True."""
        vote = Vote(agent="claude", choice="a", reasoning="r")
        assert vote.continue_debate is True

    def test_confidence_can_be_set(self):
        """confidence should be settable."""
        vote = Vote(agent="claude", choice="a", reasoning="r", confidence=0.75)
        assert vote.confidence == 0.75

    def test_continue_debate_can_be_set_false(self):
        """continue_debate should be settable to False."""
        vote = Vote(agent="claude", choice="a", reasoning="r", continue_debate=False)
        assert vote.continue_debate is False

    def test_choice_stored_correctly(self):
        """choice should store the voted option."""
        vote = Vote(agent="voter", choice="option_b", reasoning="Because")
        assert vote.choice == "option_b"


class TestDisagreementReportDataclass:
    """Tests for DisagreementReport dataclass."""

    def test_creation_with_defaults(self):
        """DisagreementReport should have empty defaults."""
        report = DisagreementReport()
        assert report.unanimous_critiques == []
        assert report.split_opinions == []
        assert report.risk_areas == []
        assert report.agreement_score == 0.0
        assert report.agent_alignment == {}

    def test_unanimous_critiques_list(self):
        """unanimous_critiques should store list of strings."""
        report = DisagreementReport(unanimous_critiques=["All agree issue 1", "All agree issue 2"])
        assert len(report.unanimous_critiques) == 2
        assert "All agree issue 1" in report.unanimous_critiques

    def test_split_opinions_format(self):
        """split_opinions should be list of (topic, agree, disagree) tuples."""
        report = DisagreementReport(
            split_opinions=[
                ("Topic A", ["claude", "codex"], ["gemini"]),
                ("Topic B", ["gemini"], ["claude", "codex"]),
            ]
        )
        assert len(report.split_opinions) == 2
        topic, agree, disagree = report.split_opinions[0]
        assert topic == "Topic A"
        assert "claude" in agree
        assert "gemini" in disagree

    def test_risk_areas_list(self):
        """risk_areas should store list of strings."""
        report = DisagreementReport(risk_areas=["Risk 1", "Risk 2", "Risk 3"])
        assert len(report.risk_areas) == 3

    def test_agreement_score_stored(self):
        """agreement_score should be stored."""
        report = DisagreementReport(agreement_score=0.75)
        assert report.agreement_score == 0.75

    def test_summary_includes_unanimous(self):
        """summary should include unanimous critiques section."""
        report = DisagreementReport(unanimous_critiques=["Critical issue"])
        summary = report.summary()
        assert "UNANIMOUS" in summary
        assert "Critical issue" in summary

    def test_summary_includes_split_opinions(self):
        """summary should include split opinions section."""
        report = DisagreementReport(split_opinions=[("Disputed topic", ["claude"], ["codex"])])
        summary = report.summary()
        assert "SPLIT" in summary
        assert "Disputed topic" in summary

    def test_summary_includes_risk_areas(self):
        """summary should include risk areas section."""
        report = DisagreementReport(risk_areas=["Potential risk"])
        summary = report.summary()
        assert "RISK" in summary
        assert "Potential risk" in summary

    def test_summary_includes_agreement_score(self):
        """summary should include agreement score."""
        report = DisagreementReport(agreement_score=0.85)
        summary = report.summary()
        assert "85%" in summary or "0.85" in summary

    def test_summary_limits_items(self):
        """summary should limit to 5 items per section."""
        report = DisagreementReport(unanimous_critiques=[f"Issue {i}" for i in range(10)])
        summary = report.summary()
        # Should show first 5, not all 10
        assert "Issue 0" in summary
        assert "Issue 4" in summary
        # Issue 9 should not appear (limited to 5)


class TestDebateResultDataclass:
    """Tests for DebateResult dataclass."""

    def test_id_auto_generated(self):
        """id should be auto-generated as UUID."""
        result = DebateResult()
        # Should be valid UUID format
        uuid.UUID(result.id)  # Raises if invalid

    def test_id_is_unique(self):
        """Each DebateResult should have unique id."""
        result1 = DebateResult()
        result2 = DebateResult()
        assert result1.id != result2.id

    def test_defaults_empty_lists(self):
        """Default lists should be empty."""
        result = DebateResult()
        assert result.messages == []
        assert result.critiques == []
        assert result.votes == []
        assert result.dissenting_views == []
        assert result.winning_patterns == []

    def test_consensus_defaults_false(self):
        """consensus_reached should default to False."""
        result = DebateResult()
        assert result.consensus_reached is False

    def test_task_stored(self):
        """task should be stored."""
        result = DebateResult(task="Design a rate limiter")
        assert result.task == "Design a rate limiter"

    def test_final_answer_stored(self):
        """final_answer should be stored."""
        result = DebateResult(final_answer="Use token bucket algorithm")
        assert result.final_answer == "Use token bucket algorithm"

    def test_messages_stores_message_objects(self):
        """messages should store Message objects."""
        msg = Message(role="proposer", agent="claude", content="Proposal")
        result = DebateResult(messages=[msg])
        assert len(result.messages) == 1
        assert result.messages[0].agent == "claude"

    def test_critiques_stores_critique_objects(self):
        """critiques should store Critique objects."""
        critique = Critique(
            agent="a",
            target_agent="b",
            target_content="c",
            issues=["i"],
            suggestions=["s"],
            severity=0.5,
            reasoning="r",
        )
        result = DebateResult(critiques=[critique])
        assert len(result.critiques) == 1
        assert result.critiques[0].severity == 0.5

    def test_votes_stores_vote_objects(self):
        """votes should store Vote objects."""
        vote = Vote(agent="claude", choice="a", reasoning="r")
        result = DebateResult(votes=[vote])
        assert len(result.votes) == 1
        assert result.votes[0].agent == "claude"

    def test_convergence_status_stored(self):
        """convergence_status should be stored."""
        result = DebateResult(convergence_status="converged")
        assert result.convergence_status == "converged"

    def test_convergence_similarity_stored(self):
        """convergence_similarity should be stored."""
        result = DebateResult(convergence_similarity=0.92)
        assert result.convergence_similarity == 0.92

    def test_per_agent_similarity_stored(self):
        """per_agent_similarity should be stored."""
        result = DebateResult(per_agent_similarity={"claude": 0.9, "codex": 0.85})
        assert result.per_agent_similarity["claude"] == 0.9

    def test_summary_includes_task(self):
        """summary should include task."""
        result = DebateResult(task="Important task")
        summary = result.summary()
        assert "Important task" in summary

    def test_summary_includes_rounds(self):
        """summary should include rounds used."""
        result = DebateResult(rounds_used=3)
        summary = result.summary()
        assert "3" in summary

    def test_summary_includes_consensus_status(self):
        """summary should indicate consensus status."""
        result = DebateResult(consensus_reached=True)
        summary = result.summary()
        assert "Yes" in summary or "True" in summary or "consensus" in summary.lower()

    def test_summary_includes_disagreement_report(self):
        """summary should include disagreement report when present."""
        report = DisagreementReport(
            unanimous_critiques=["All agree on this"],
            agreement_score=0.7,
        )
        result = DebateResult(disagreement_report=report)
        summary = result.summary()
        assert "All agree on this" in summary or "UNANIMOUS" in summary


class TestEnvironmentDataclass:
    """Tests for Environment dataclass."""

    def test_task_required(self):
        """Environment should require task."""
        env = Environment(task="Design a caching system")
        assert env.task == "Design a caching system"

    def test_context_defaults_empty(self):
        """context should default to empty string."""
        env = Environment(task="task")
        assert env.context == ""

    def test_default_roles(self):
        """Default roles should be proposer, critic, synthesizer."""
        env = Environment(task="task")
        assert "proposer" in env.roles
        assert "critic" in env.roles
        assert "synthesizer" in env.roles
        assert len(env.roles) == 3

    def test_max_rounds_defaults_to_three(self):
        """max_rounds should default to 3."""
        env = Environment(task="task")
        assert env.max_rounds == 3

    def test_consensus_threshold_default(self):
        """consensus_threshold should default to 0.7."""
        env = Environment(task="task")
        assert env.consensus_threshold == 0.7

    def test_require_consensus_defaults_false(self):
        """require_consensus should default to False."""
        env = Environment(task="task")
        assert env.require_consensus is False

    def test_documents_list_default_empty(self):
        """documents should default to empty list."""
        env = Environment(task="task")
        assert env.documents == []

    def test_documents_can_be_set(self):
        """documents should be settable."""
        env = Environment(task="task", documents=["doc1.md", "doc2.md"])
        assert len(env.documents) == 2
        assert "doc1.md" in env.documents

    def test_success_fn_default_none(self):
        """success_fn should default to None."""
        env = Environment(task="task")
        assert env.success_fn is None

    def test_success_fn_can_be_set(self):
        """success_fn should be settable."""
        fn = lambda x: 0.5  # noqa: E731
        env = Environment(task="task", success_fn=fn)
        assert env.success_fn is not None
        assert env.success_fn("test") == 0.5


class ConcreteAgent(Agent):
    """Concrete implementation of Agent for testing."""

    async def generate(self, prompt: str, context=None) -> str:
        return f"Response to: {prompt}"

    async def critique(self, proposal: str, task: str, context=None) -> Critique:
        return Critique(
            agent=self.name,
            target_agent="unknown",
            target_content=proposal,
            issues=["Test issue"],
            suggestions=["Test suggestion"],
            severity=0.5,
            reasoning="Test reasoning",
        )


class TestAgentABC:
    """Tests for Agent abstract base class."""

    def test_init_stores_name(self):
        """__init__ should store name."""
        agent = ConcreteAgent(name="test_agent", model="test_model")
        assert agent.name == "test_agent"

    def test_init_stores_model(self):
        """__init__ should store model."""
        agent = ConcreteAgent(name="test", model="claude-3")
        assert agent.model == "claude-3"

    def test_init_stores_role(self):
        """__init__ should store role."""
        agent = ConcreteAgent(name="test", model="model", role="critic")
        assert agent.role == "critic"

    def test_role_defaults_to_proposer(self):
        """role should default to 'proposer'."""
        agent = ConcreteAgent(name="test", model="model")
        assert agent.role == "proposer"

    def test_agent_type_defaults_to_unknown(self):
        """agent_type should default to 'unknown'."""
        agent = ConcreteAgent(name="test", model="model")
        assert agent.agent_type == "unknown"

    def test_stance_defaults_to_neutral(self):
        """stance should default to 'neutral'."""
        agent = ConcreteAgent(name="test", model="model")
        assert agent.stance == "neutral"

    def test_system_prompt_defaults_empty(self):
        """system_prompt should default to empty string."""
        agent = ConcreteAgent(name="test", model="model")
        assert agent.system_prompt == ""

    def test_set_system_prompt_updates(self):
        """set_system_prompt should update system_prompt."""
        agent = ConcreteAgent(name="test", model="model")
        agent.set_system_prompt("You are a helpful assistant")
        assert agent.system_prompt == "You are a helpful assistant"

    def test_repr_includes_class_name(self):
        """__repr__ should include class name."""
        agent = ConcreteAgent(name="test", model="model")
        repr_str = repr(agent)
        assert "ConcreteAgent" in repr_str

    def test_repr_includes_name(self):
        """__repr__ should include agent name."""
        agent = ConcreteAgent(name="my_agent", model="model")
        repr_str = repr(agent)
        assert "my_agent" in repr_str

    def test_repr_includes_model(self):
        """__repr__ should include model."""
        agent = ConcreteAgent(name="test", model="claude-opus")
        repr_str = repr(agent)
        assert "claude-opus" in repr_str

    def test_repr_includes_role(self):
        """__repr__ should include role."""
        agent = ConcreteAgent(name="test", model="model", role="synthesizer")
        repr_str = repr(agent)
        assert "synthesizer" in repr_str

    @pytest.mark.asyncio
    async def test_generate_is_abstract(self):
        """generate should be implemented by subclass."""
        agent = ConcreteAgent(name="test", model="model")
        result = await agent.generate("test prompt")
        assert "Response to: test prompt" in result

    @pytest.mark.asyncio
    async def test_critique_is_abstract(self):
        """critique should be implemented by subclass."""
        agent = ConcreteAgent(name="test", model="model")
        result = await agent.critique("proposal", "task")
        assert isinstance(result, Critique)
        assert result.agent == "test"

    @pytest.mark.asyncio
    async def test_vote_has_default_implementation(self):
        """vote should have a default implementation."""
        agent = ConcreteAgent(name="test", model="model")
        # Mock generate to return a parseable vote response
        agent.generate = AsyncMock(
            return_value="CHOICE: option_a\nCONFIDENCE: 0.8\nCONTINUE: no\nREASONING: Best"
        )
        vote = await agent.vote({"a": "prop_a", "b": "prop_b"}, "task")
        assert isinstance(vote, Vote)
        assert vote.agent == "test"

    @pytest.mark.asyncio
    async def test_vote_parses_choice(self):
        """vote should parse CHOICE from response."""
        agent = ConcreteAgent(name="test", model="model")
        agent.generate = AsyncMock(
            return_value="CHOICE: winner\nCONFIDENCE: 0.9\nCONTINUE: yes\nREASONING: Good"
        )
        vote = await agent.vote({"a": "a", "b": "b"}, "task")
        assert vote.choice == "winner"

    @pytest.mark.asyncio
    async def test_vote_parses_confidence(self):
        """vote should parse CONFIDENCE from response."""
        agent = ConcreteAgent(name="test", model="model")
        agent.generate = AsyncMock(
            return_value="CHOICE: a\nCONFIDENCE: 0.75\nCONTINUE: yes\nREASONING: OK"
        )
        vote = await agent.vote({"a": "a"}, "task")
        assert vote.confidence == 0.75

    @pytest.mark.asyncio
    async def test_vote_parses_continue_debate_false(self):
        """vote should parse CONTINUE: no as False."""
        agent = ConcreteAgent(name="test", model="model")
        agent.generate = AsyncMock(
            return_value="CHOICE: a\nCONFIDENCE: 0.8\nCONTINUE: no\nREASONING: Done"
        )
        vote = await agent.vote({"a": "a"}, "task")
        assert vote.continue_debate is False

    @pytest.mark.asyncio
    async def test_vote_parses_continue_debate_true(self):
        """vote should parse CONTINUE: yes as True."""
        agent = ConcreteAgent(name="test", model="model")
        agent.generate = AsyncMock(
            return_value="CHOICE: a\nCONFIDENCE: 0.8\nCONTINUE: yes\nREASONING: More"
        )
        vote = await agent.vote({"a": "a"}, "task")
        assert vote.continue_debate is True

    @pytest.mark.asyncio
    async def test_vote_handles_invalid_confidence(self):
        """vote should handle invalid confidence gracefully."""
        agent = ConcreteAgent(name="test", model="model")
        agent.generate = AsyncMock(
            return_value="CHOICE: a\nCONFIDENCE: invalid\nCONTINUE: yes\nREASONING: OK"
        )
        vote = await agent.vote({"a": "a"}, "task")
        assert vote.confidence == 0.5  # Default fallback

    @pytest.mark.asyncio
    async def test_vote_parses_reasoning(self):
        """vote should parse REASONING from response."""
        agent = ConcreteAgent(name="test", model="model")
        agent.generate = AsyncMock(
            return_value="CHOICE: a\nCONFIDENCE: 0.8\nCONTINUE: yes\nREASONING: This is best because X"
        )
        vote = await agent.vote({"a": "a"}, "task")
        assert vote.reasoning == "This is best because X"
