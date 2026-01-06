"""Integration tests for Evidence collection in debate prompts."""

import pytest
from unittest.mock import MagicMock, AsyncMock
from dataclasses import dataclass
from typing import Optional

from aragora.debate.prompt_builder import PromptBuilder
from aragora.debate.protocol import DebateProtocol
from aragora.core import Environment


# Mock EvidenceSnippet and EvidencePack for testing
@dataclass
class MockEvidenceSnippet:
    """Mock evidence snippet for testing."""
    id: str = "test-snippet"
    source: str = "web"
    title: str = "Test Article"
    snippet: str = "This is test evidence content."
    url: str = "https://example.com/article"
    reliability_score: float = 0.8
    metadata: dict = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class MockEvidencePack:
    """Mock evidence pack for testing."""
    topic_keywords: list = None
    snippets: list = None
    search_timestamp: str = ""
    total_searched: int = 0

    def __post_init__(self):
        if self.topic_keywords is None:
            self.topic_keywords = []
        if self.snippets is None:
            self.snippets = []

    def to_context_string(self) -> str:
        """Format as context string."""
        if not self.snippets:
            return "No evidence available."
        lines = []
        for s in self.snippets:
            lines.append(f"[{s.source}] {s.title}: {s.snippet[:100]}")
        return "\n".join(lines)


class TestPromptBuilderEvidence:
    """Tests for evidence integration in PromptBuilder."""

    @pytest.fixture
    def protocol(self):
        """Create basic debate protocol."""
        return DebateProtocol(rounds=3)

    @pytest.fixture
    def env(self):
        """Create basic environment."""
        return Environment(task="Discuss AI safety measures")

    @pytest.fixture
    def sample_evidence_pack(self):
        """Create sample evidence pack."""
        return MockEvidencePack(
            topic_keywords=["AI", "safety"],
            snippets=[
                MockEvidenceSnippet(
                    id="evid-1",
                    source="academic",
                    title="AI Safety Research Overview",
                    snippet="Recent research shows that AI alignment is critical for safe deployment. Key areas include interpretability and robustness.",
                    url="https://arxiv.org/paper",
                    reliability_score=0.9,
                ),
                MockEvidenceSnippet(
                    id="evid-2",
                    source="web",
                    title="Industry Guidelines on AI Safety",
                    snippet="Major tech companies have adopted voluntary AI safety guidelines including red-teaming and safety testing.",
                    url="https://techpolicy.org/guidelines",
                    reliability_score=0.75,
                ),
                MockEvidenceSnippet(
                    id="evid-3",
                    source="github",
                    title="AI Safety Toolkit Repository",
                    snippet="Open-source tools for evaluating AI model safety including bias detection and adversarial testing.",
                    url="https://github.com/safety-toolkit",
                    reliability_score=0.65,
                ),
            ],
        )

    @pytest.fixture
    def mock_agent(self):
        """Create mock agent."""
        agent = MagicMock()
        agent.name = "test_agent"
        agent.role = "proposer"
        return agent

    def test_prompt_builder_accepts_evidence_pack(self, protocol, env, sample_evidence_pack):
        """Test PromptBuilder constructor accepts evidence_pack parameter."""
        builder = PromptBuilder(
            protocol=protocol,
            env=env,
            evidence_pack=sample_evidence_pack,
        )

        assert builder.evidence_pack is not None
        assert len(builder.evidence_pack.snippets) == 3

    def test_prompt_builder_without_evidence(self, protocol, env):
        """Test PromptBuilder works without evidence_pack."""
        builder = PromptBuilder(protocol=protocol, env=env)

        assert builder.evidence_pack is None

    def test_set_evidence_pack(self, protocol, env, sample_evidence_pack):
        """Test set_evidence_pack updates evidence."""
        builder = PromptBuilder(protocol=protocol, env=env)
        assert builder.evidence_pack is None

        builder.set_evidence_pack(sample_evidence_pack)
        assert builder.evidence_pack is not None
        assert len(builder.evidence_pack.snippets) == 3

    def test_format_evidence_empty(self, protocol, env):
        """Test format_evidence_for_prompt with no evidence."""
        builder = PromptBuilder(protocol=protocol, env=env)

        result = builder.format_evidence_for_prompt()
        assert result == ""

    def test_format_evidence_with_snippets(self, protocol, env, sample_evidence_pack):
        """Test format_evidence_for_prompt with evidence."""
        builder = PromptBuilder(
            protocol=protocol, env=env, evidence_pack=sample_evidence_pack
        )

        result = builder.format_evidence_for_prompt()

        assert "## AVAILABLE EVIDENCE" in result
        assert "[EVID-1]" in result
        assert "[EVID-2]" in result
        assert "[EVID-3]" in result

    def test_format_evidence_includes_titles(self, protocol, env, sample_evidence_pack):
        """Test evidence formatting includes titles."""
        builder = PromptBuilder(
            protocol=protocol, env=env, evidence_pack=sample_evidence_pack
        )

        result = builder.format_evidence_for_prompt()

        assert "AI Safety Research Overview" in result
        assert "Industry Guidelines" in result

    def test_format_evidence_includes_sources(self, protocol, env, sample_evidence_pack):
        """Test evidence formatting includes sources."""
        builder = PromptBuilder(
            protocol=protocol, env=env, evidence_pack=sample_evidence_pack
        )

        result = builder.format_evidence_for_prompt()

        assert "academic" in result
        assert "web" in result
        assert "github" in result

    def test_format_evidence_includes_reliability(self, protocol, env, sample_evidence_pack):
        """Test evidence formatting includes reliability scores."""
        builder = PromptBuilder(
            protocol=protocol, env=env, evidence_pack=sample_evidence_pack
        )

        result = builder.format_evidence_for_prompt()

        assert "Reliability:" in result
        assert "90%" in result  # 0.9 reliability

    def test_format_evidence_includes_urls(self, protocol, env, sample_evidence_pack):
        """Test evidence formatting includes URLs."""
        builder = PromptBuilder(
            protocol=protocol, env=env, evidence_pack=sample_evidence_pack
        )

        result = builder.format_evidence_for_prompt()

        assert "URL:" in result
        assert "arxiv.org" in result

    def test_format_evidence_includes_snippet_content(self, protocol, env, sample_evidence_pack):
        """Test evidence formatting includes snippet content."""
        builder = PromptBuilder(
            protocol=protocol, env=env, evidence_pack=sample_evidence_pack
        )

        result = builder.format_evidence_for_prompt()

        assert "alignment" in result.lower()
        assert ">" in result  # Blockquote marker

    def test_format_evidence_includes_citation_instruction(self, protocol, env, sample_evidence_pack):
        """Test evidence formatting includes citation instructions."""
        builder = PromptBuilder(
            protocol=protocol, env=env, evidence_pack=sample_evidence_pack
        )

        result = builder.format_evidence_for_prompt()

        assert "cite evidence as [EVID-N]" in result.lower() or "EVID-N" in result

    def test_format_evidence_respects_max_snippets(self, protocol, env, sample_evidence_pack):
        """Test format_evidence_for_prompt respects max_snippets parameter."""
        builder = PromptBuilder(
            protocol=protocol, env=env, evidence_pack=sample_evidence_pack
        )

        result = builder.format_evidence_for_prompt(max_snippets=2)

        assert "[EVID-1]" in result
        assert "[EVID-2]" in result
        assert "[EVID-3]" not in result  # Should be excluded


class TestProposalPromptEvidence:
    """Tests for evidence in proposal prompts."""

    @pytest.fixture
    def builder_with_evidence(self, protocol, env, sample_evidence_pack):
        """Create builder with evidence."""
        return PromptBuilder(
            protocol=DebateProtocol(rounds=3),
            env=Environment(task="Test task"),
            evidence_pack=sample_evidence_pack,
        )

    @pytest.fixture
    def protocol(self):
        return DebateProtocol(rounds=3)

    @pytest.fixture
    def env(self):
        return Environment(task="Test task")

    @pytest.fixture
    def sample_evidence_pack(self):
        return MockEvidencePack(
            snippets=[
                MockEvidenceSnippet(
                    title="Evidence Title",
                    snippet="Evidence content",
                    reliability_score=0.8,
                )
            ]
        )

    @pytest.fixture
    def mock_agent(self):
        agent = MagicMock()
        agent.name = "test_agent"
        agent.role = "proposer"
        return agent

    def test_proposal_prompt_includes_evidence(self, builder_with_evidence, mock_agent):
        """Test build_proposal_prompt includes evidence section."""
        prompt = builder_with_evidence.build_proposal_prompt(mock_agent)

        assert "## AVAILABLE EVIDENCE" in prompt
        assert "[EVID-1]" in prompt

    def test_proposal_prompt_without_evidence(self, protocol, env, mock_agent):
        """Test build_proposal_prompt works without evidence."""
        builder = PromptBuilder(protocol=protocol, env=env)
        prompt = builder.build_proposal_prompt(mock_agent)

        assert "## AVAILABLE EVIDENCE" not in prompt
        assert "You are acting as a" in prompt


class TestRevisionPromptEvidence:
    """Tests for evidence in revision prompts."""

    @pytest.fixture
    def builder_with_evidence(self):
        return PromptBuilder(
            protocol=DebateProtocol(rounds=3),
            env=Environment(task="Test task"),
            evidence_pack=MockEvidencePack(
                snippets=[
                    MockEvidenceSnippet(
                        title="Revision Evidence",
                        snippet="Supporting content for revision",
                        reliability_score=0.85,
                    )
                ]
            ),
        )

    @pytest.fixture
    def mock_agent(self):
        agent = MagicMock()
        agent.name = "test_agent"
        agent.role = "proposer"
        return agent

    @pytest.fixture
    def mock_critique(self):
        critique = MagicMock()
        critique.agent = "critic_agent"
        critique.issues = ["Issue 1", "Issue 2"]
        critique.to_prompt = MagicMock(return_value="Critique: Your argument lacks evidence.")
        return critique

    def test_revision_prompt_includes_evidence(self, builder_with_evidence, mock_agent, mock_critique):
        """Test build_revision_prompt includes evidence section."""
        prompt = builder_with_evidence.build_revision_prompt(
            mock_agent, "Original proposal", [mock_critique]
        )

        assert "## AVAILABLE EVIDENCE" in prompt
        assert "[EVID-1]" in prompt

    def test_revision_prompt_citation_instruction(self, builder_with_evidence, mock_agent, mock_critique):
        """Test revision prompt includes citation instructions."""
        prompt = builder_with_evidence.build_revision_prompt(
            mock_agent, "Original proposal", [mock_critique]
        )

        assert "EVID-N" in prompt or "evidence citations" in prompt.lower()


class TestJudgePromptEvidence:
    """Tests for evidence in judge/synthesizer prompts."""

    @pytest.fixture
    def builder_with_evidence(self):
        return PromptBuilder(
            protocol=DebateProtocol(rounds=3),
            env=Environment(task="Test task"),
            evidence_pack=MockEvidencePack(
                snippets=[
                    MockEvidenceSnippet(
                        title="Judge Evidence 1",
                        snippet="Evidence for synthesis",
                        reliability_score=0.9,
                    ),
                    MockEvidenceSnippet(
                        title="Judge Evidence 2",
                        snippet="More evidence",
                        reliability_score=0.8,
                    ),
                ]
            ),
        )

    @pytest.fixture
    def mock_critique(self):
        critique = MagicMock()
        critique.agent = "agent1"
        critique.issues = ["Issue"]
        return critique

    def test_judge_prompt_includes_evidence(self, builder_with_evidence, mock_critique):
        """Test build_judge_prompt includes evidence section."""
        proposals = {"agent1": "Proposal 1", "agent2": "Proposal 2"}
        prompt = builder_with_evidence.build_judge_prompt(
            proposals, "Test task", [mock_critique]
        )

        assert "## AVAILABLE EVIDENCE" in prompt
        assert "[EVID-1]" in prompt

    def test_judge_prompt_citation_instruction(self, builder_with_evidence, mock_critique):
        """Test judge prompt includes citation instruction."""
        proposals = {"agent1": "Proposal 1", "agent2": "Proposal 2"}
        prompt = builder_with_evidence.build_judge_prompt(
            proposals, "Test task", [mock_critique]
        )

        assert "EVID-N" in prompt or "Reference evidence" in prompt


class TestEvidenceEdgeCases:
    """Tests for evidence edge cases."""

    @pytest.fixture
    def protocol(self):
        return DebateProtocol(rounds=3)

    @pytest.fixture
    def env(self):
        return Environment(task="Test task")

    def test_empty_snippets_list(self, protocol, env):
        """Test with empty snippets list."""
        evidence_pack = MockEvidencePack(snippets=[])
        builder = PromptBuilder(protocol=protocol, env=env, evidence_pack=evidence_pack)

        result = builder.format_evidence_for_prompt()
        assert result == ""

    def test_snippet_without_url(self, protocol, env):
        """Test snippet without URL."""
        evidence_pack = MockEvidencePack(
            snippets=[
                MockEvidenceSnippet(
                    title="No URL Snippet",
                    snippet="Content without URL",
                    url="",  # Empty URL
                    reliability_score=0.7,
                )
            ]
        )
        builder = PromptBuilder(protocol=protocol, env=env, evidence_pack=evidence_pack)

        result = builder.format_evidence_for_prompt()
        assert "[EVID-1]" in result
        assert "URL:" not in result  # Should not include empty URL

    def test_long_snippet_truncation(self, protocol, env):
        """Test long snippet content is truncated."""
        long_content = "A" * 500  # 500 character content
        evidence_pack = MockEvidencePack(
            snippets=[
                MockEvidenceSnippet(
                    title="Long Content",
                    snippet=long_content,
                    reliability_score=0.8,
                )
            ]
        )
        builder = PromptBuilder(protocol=protocol, env=env, evidence_pack=evidence_pack)

        result = builder.format_evidence_for_prompt()
        assert "..." in result  # Should be truncated
        assert len(result) < len(long_content) + 500  # Much shorter than full content

    def test_special_characters_in_snippet(self, protocol, env):
        """Test special characters in snippet content."""
        evidence_pack = MockEvidencePack(
            snippets=[
                MockEvidenceSnippet(
                    title='Title with "quotes" and <tags>',
                    snippet="Content with special chars: & < > \"",
                    reliability_score=0.8,
                )
            ]
        )
        builder = PromptBuilder(protocol=protocol, env=env, evidence_pack=evidence_pack)

        result = builder.format_evidence_for_prompt()
        assert "[EVID-1]" in result
        # Should not crash and should contain some version of the content

    def test_missing_reliability_score(self, protocol, env):
        """Test snippet with missing reliability_score attribute."""
        # Create a minimal mock without reliability_score
        snippet = MagicMock()
        snippet.title = "Test Title"
        snippet.source = "web"
        snippet.snippet = "Test content"
        snippet.url = "https://example.com"
        # Don't set reliability_score - let hasattr return False

        evidence_pack = MagicMock()
        evidence_pack.snippets = [snippet]

        builder = PromptBuilder(protocol=protocol, env=env, evidence_pack=evidence_pack)

        result = builder.format_evidence_for_prompt()
        assert "[EVID-1]" in result
        assert "Reliability:" in result  # Should use default 0.5
