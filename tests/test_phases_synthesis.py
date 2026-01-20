"""
Tests for the SynthesisGenerator class.

Tests synthesis prompt building and proposal combination fallback.
"""

from dataclasses import dataclass, field
from typing import Any, Optional
from unittest.mock import MagicMock, AsyncMock, patch

import pytest


@dataclass
class MockEnv:
    """Mock Environment for testing."""

    task: str = "Test debate topic"


@dataclass
class MockResult:
    """Mock DebateResult for testing."""

    confidence: float = 0.8
    winner: Optional[str] = None
    synthesis: Optional[str] = None
    final_answer: Optional[str] = None
    debate_id: Optional[str] = None
    export_links: dict = field(default_factory=dict)


@dataclass
class MockCritique:
    """Mock Critique for testing."""

    agent: str
    target: str
    summary: str = ""


@dataclass
class MockContext:
    """Mock DebateContext for testing."""

    proposals: dict = field(default_factory=dict)
    env: Optional[MockEnv] = None
    result: Optional[MockResult] = None
    context_messages: list = field(default_factory=list)
    critiques: list = field(default_factory=list)
    debate_id: Optional[str] = None


class TestSynthesisPromptBuilding:
    """Tests for synthesis prompt building."""

    def test_builds_prompt_with_proposals(self):
        """Should include proposals in the synthesis prompt."""
        from aragora.debate.phases.synthesis_generator import SynthesisGenerator

        generator = SynthesisGenerator()

        ctx = MockContext(
            proposals={"alice": "Proposal A content", "bob": "Proposal B content"},
            env=MockEnv(task="Design a secure API"),
            result=MockResult(),
        )

        prompt = generator._build_synthesis_prompt(ctx)

        assert "Design a secure API" in prompt
        assert "alice" in prompt
        assert "Proposal A content" in prompt
        assert "bob" in prompt
        assert "Proposal B content" in prompt

    def test_builds_prompt_with_critiques(self):
        """Should include critiques in the synthesis prompt."""
        from aragora.debate.phases.synthesis_generator import SynthesisGenerator

        generator = SynthesisGenerator()

        ctx = MockContext(
            proposals={"alice": "Proposal A"},
            env=MockEnv(task="Test task"),
            result=MockResult(),
            critiques=[
                MockCritique("bob", "alice", "Good point about X"),
                MockCritique("alice", "bob", "Consider Y instead"),
            ],
        )

        prompt = generator._build_synthesis_prompt(ctx)

        assert "bob on alice" in prompt or "alice on bob" in prompt

    def test_handles_missing_critiques(self):
        """Should handle case with no critiques."""
        from aragora.debate.phases.synthesis_generator import SynthesisGenerator

        generator = SynthesisGenerator()

        ctx = MockContext(
            proposals={"alice": "Proposal A"},
            env=MockEnv(task="Test task"),
            result=MockResult(),
            critiques=[],
        )

        prompt = generator._build_synthesis_prompt(ctx)

        assert "No critiques recorded" in prompt

    def test_truncates_long_proposals(self):
        """Should truncate very long proposals in prompt."""
        from aragora.debate.phases.synthesis_generator import SynthesisGenerator

        generator = SynthesisGenerator()

        long_proposal = "x" * 5000  # 5000 chars
        ctx = MockContext(
            proposals={"alice": long_proposal},
            env=MockEnv(task="Test task"),
            result=MockResult(),
        )

        prompt = generator._build_synthesis_prompt(ctx)

        # Should be truncated to 1500 chars
        assert len(prompt) < 5000 + 1000  # Some buffer for prompt text


class TestProposalCombination:
    """Tests for proposal combination fallback."""

    def test_combines_with_winner(self):
        """Should prioritize winner's proposal when available."""
        from aragora.debate.phases.synthesis_generator import SynthesisGenerator

        generator = SynthesisGenerator()

        ctx = MockContext(
            proposals={
                "alice": "Alice proposal content",
                "bob": "Bob proposal content",
                "charlie": "Charlie proposal content",
            },
            env=MockEnv(task="Test topic"),
            result=MockResult(winner="alice"),
        )

        synthesis = generator._combine_proposals_as_synthesis(ctx)

        assert "Winning Position (alice)" in synthesis
        assert "Alice proposal content" in synthesis
        assert "Other Perspectives" in synthesis

    def test_combines_without_winner(self):
        """Should combine all proposals when no winner."""
        from aragora.debate.phases.synthesis_generator import SynthesisGenerator

        generator = SynthesisGenerator()

        ctx = MockContext(
            proposals={
                "alice": "Alice proposal content",
                "bob": "Bob proposal content",
            },
            env=MockEnv(task="Test topic"),
            result=MockResult(winner=None),
        )

        synthesis = generator._combine_proposals_as_synthesis(ctx)

        assert "Combined Perspectives" in synthesis
        assert "alice" in synthesis
        assert "bob" in synthesis

    def test_includes_task_in_combined(self):
        """Should include the task in the combined synthesis."""
        from aragora.debate.phases.synthesis_generator import SynthesisGenerator

        generator = SynthesisGenerator()

        ctx = MockContext(
            proposals={"alice": "Proposal"},
            env=MockEnv(task="Important debate question"),
            result=MockResult(),
        )

        synthesis = generator._combine_proposals_as_synthesis(ctx)

        assert "Important debate question" in synthesis

    def test_limits_proposal_length(self):
        """Should limit individual proposal lengths in synthesis."""
        from aragora.debate.phases.synthesis_generator import SynthesisGenerator

        generator = SynthesisGenerator()

        long_proposal = "x" * 5000
        ctx = MockContext(
            proposals={"alice": long_proposal},
            env=MockEnv(task="Test"),
            result=MockResult(winner="alice"),
        )

        synthesis = generator._combine_proposals_as_synthesis(ctx)

        # Winner proposal limited to 2000 chars
        assert len(synthesis) < 5000

    def test_handles_missing_env(self):
        """Should handle missing environment gracefully."""
        from aragora.debate.phases.synthesis_generator import SynthesisGenerator

        generator = SynthesisGenerator()

        ctx = MockContext(
            proposals={"alice": "Proposal"},
            env=None,
            result=MockResult(),
        )

        synthesis = generator._combine_proposals_as_synthesis(ctx)

        assert "the debate topic" in synthesis


class TestExportLinks:
    """Tests for export link generation."""

    def test_generates_all_export_formats(self):
        """Should generate links for all export formats."""
        from aragora.debate.phases.synthesis_generator import SynthesisGenerator

        generator = SynthesisGenerator()

        ctx = MockContext(
            proposals={"alice": "Proposal"},
            env=MockEnv(),
            result=MockResult(debate_id="test-debate-123"),
            debate_id="test-debate-123",
        )

        generator._generate_export_links(ctx)

        links = ctx.result.export_links
        assert "json" in links
        assert "markdown" in links
        assert "html" in links
        assert "txt" in links
        assert "csv_summary" in links
        assert "csv_messages" in links

    def test_includes_debate_id_in_links(self):
        """Should include debate ID in export URLs."""
        from aragora.debate.phases.synthesis_generator import SynthesisGenerator

        generator = SynthesisGenerator()

        ctx = MockContext(
            proposals={"alice": "Proposal"},
            env=MockEnv(),
            result=MockResult(debate_id="abc-123"),
            debate_id="abc-123",
        )

        generator._generate_export_links(ctx)

        for format_name, url in ctx.result.export_links.items():
            assert "abc-123" in url

    def test_skips_without_debate_id(self):
        """Should skip link generation without debate ID."""
        from aragora.debate.phases.synthesis_generator import SynthesisGenerator

        generator = SynthesisGenerator()

        ctx = MockContext(
            proposals={"alice": "Proposal"},
            env=MockEnv(),
            result=MockResult(debate_id=None),
            debate_id=None,
        )

        generator._generate_export_links(ctx)

        assert ctx.result.export_links == {}


class TestEventEmission:
    """Tests for synthesis event emission."""

    def test_calls_on_synthesis_hook(self):
        """Should call on_synthesis hook when provided."""
        from aragora.debate.phases.synthesis_generator import SynthesisGenerator

        on_synthesis = MagicMock()
        generator = SynthesisGenerator(hooks={"on_synthesis": on_synthesis})

        ctx = MockContext(
            proposals={"alice": "Proposal"},
            env=MockEnv(),
            result=MockResult(confidence=0.85),
        )

        generator._emit_synthesis_events(ctx, "Test synthesis", "opus")

        on_synthesis.assert_called_once()
        call_kwargs = on_synthesis.call_args[1]
        assert call_kwargs["content"] == "Test synthesis"
        assert call_kwargs["confidence"] == 0.85

    def test_calls_on_message_hook(self):
        """Should call on_message hook for backwards compatibility."""
        from aragora.debate.phases.synthesis_generator import SynthesisGenerator

        on_message = MagicMock()
        mock_protocol = MagicMock()
        mock_protocol.rounds = 3

        generator = SynthesisGenerator(
            protocol=mock_protocol,
            hooks={"on_message": on_message},
        )

        ctx = MockContext(
            proposals={"alice": "Proposal"},
            env=MockEnv(),
            result=MockResult(),
        )

        generator._emit_synthesis_events(ctx, "Test synthesis", "opus")

        on_message.assert_called_once()
        call_kwargs = on_message.call_args[1]
        assert call_kwargs["agent"] == "synthesis-agent"
        assert call_kwargs["role"] == "synthesis"
        assert call_kwargs["round_num"] == 4  # rounds + 1

    def test_calls_spectator_notification(self):
        """Should notify spectator when callback provided."""
        from aragora.debate.phases.synthesis_generator import SynthesisGenerator

        notify = MagicMock()
        generator = SynthesisGenerator(notify_spectator=notify)

        ctx = MockContext(
            proposals={"alice": "Proposal"},
            env=MockEnv(),
            result=MockResult(confidence=0.9),
        )

        generator._emit_synthesis_events(ctx, "Test synthesis", "sonnet")

        notify.assert_called_once()
        call_args = notify.call_args[0]
        assert call_args[0] == "synthesis"
        call_kwargs = notify.call_args[1]
        assert "sonnet" in call_kwargs["details"]

    def test_handles_hook_errors_gracefully(self):
        """Should handle hook errors without raising."""
        from aragora.debate.phases.synthesis_generator import SynthesisGenerator

        failing_hook = MagicMock(side_effect=Exception("Hook error"))
        generator = SynthesisGenerator(hooks={"on_synthesis": failing_hook})

        ctx = MockContext(
            proposals={"alice": "Proposal"},
            env=MockEnv(),
            result=MockResult(),
        )

        # Should not raise
        generator._emit_synthesis_events(ctx, "Test synthesis", "opus")


class TestMandatorySynthesisAsync:
    """Tests for the async generate_mandatory_synthesis method."""

    @pytest.mark.asyncio
    async def test_returns_false_without_proposals(self):
        """Should return False when no proposals available."""
        from aragora.debate.phases.synthesis_generator import SynthesisGenerator

        generator = SynthesisGenerator()

        ctx = MockContext(
            proposals={},
            env=MockEnv(),
            result=MockResult(),
        )

        result = await generator.generate_mandatory_synthesis(ctx)

        assert result is False

    @pytest.mark.asyncio
    async def test_falls_back_to_combined_on_errors(self):
        """Should fall back to combined proposals when LLM fails."""
        from aragora.debate.phases.synthesis_generator import SynthesisGenerator

        generator = SynthesisGenerator()

        ctx = MockContext(
            proposals={"alice": "Alice proposal", "bob": "Bob proposal"},
            env=MockEnv(task="Test question"),
            result=MockResult(winner="alice"),
        )

        # Mock the AnthropicAPIAgent import to fail
        with patch.dict("sys.modules", {"aragora.agents.api_agents.anthropic": None}):
            with patch(
                "aragora.debate.phases.synthesis_generator.SynthesisGenerator._build_synthesis_prompt",
                side_effect=ImportError("No module"),
            ):
                # Should still succeed with fallback
                result = await generator.generate_mandatory_synthesis(ctx)

        # Combined fallback should have been used
        assert ctx.result.synthesis is not None
        assert ctx.result.final_answer is not None

    @pytest.mark.asyncio
    async def test_sets_synthesis_and_final_answer(self):
        """Should set both synthesis and final_answer on result."""
        from aragora.debate.phases.synthesis_generator import SynthesisGenerator

        generator = SynthesisGenerator()

        ctx = MockContext(
            proposals={"alice": "Alice proposal"},
            env=MockEnv(task="Test question"),
            result=MockResult(),
        )

        # Force fallback to combined proposals
        with patch(
            "aragora.debate.phases.synthesis_generator.SynthesisGenerator._build_synthesis_prompt",
            return_value="Mock prompt",
        ):
            # Patch import to simulate LLM failure
            import sys

            original_modules = sys.modules.copy()

            # This will cause ImportError and trigger fallback
            with patch.object(
                generator, "_combine_proposals_as_synthesis", return_value="Combined synthesis"
            ):
                # Simulate all LLM attempts failing by patching the imports
                sys.modules["aragora.agents.api_agents.anthropic"] = None
                try:
                    result = await generator.generate_mandatory_synthesis(ctx)
                finally:
                    sys.modules.update(original_modules)

        assert ctx.result.synthesis is not None
        assert ctx.result.final_answer == ctx.result.synthesis
