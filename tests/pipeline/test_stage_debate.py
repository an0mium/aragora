"""Tests for the stage transition debate module."""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, AsyncMock, patch

from aragora.pipeline.stage_debate import (
    StageTransitionDebater,
    TransitionDebateResult,
)


class TestTransitionDebateResult:
    """Tests for TransitionDebateResult dataclass."""

    def test_defaults(self):
        result = TransitionDebateResult(
            transition_id="td-1",
            from_stage="ideas",
            to_stage="goals",
        )
        assert result.verdict == "proceed"
        assert result.confidence == 0.7
        assert result.rationale == ""
        assert result.dissenting_views == []
        assert result.suggestions == []
        assert result.receipt is None

    def test_to_dict(self):
        result = TransitionDebateResult(
            transition_id="td-1",
            from_stage="ideas",
            to_stage="goals",
            verdict="revise",
            confidence=0.8,
            rationale="Goals need refinement",
            suggestions=["Merge goals 1 and 2"],
        )
        d = result.to_dict()
        assert d["transition_id"] == "td-1"
        assert d["verdict"] == "revise"
        assert d["confidence"] == 0.8
        assert d["from_stage"] == "ideas"
        assert d["to_stage"] == "goals"
        assert "Merge goals 1 and 2" in d["suggestions"]


class TestVerdictParsing:
    """Tests for verdict parsing from debate output."""

    @pytest.fixture
    def debater(self):
        return StageTransitionDebater()

    def test_parse_proceed(self, debater):
        text = "The goals look well-formed.\n\nVERDICT: PROCEED"
        verdict, confidence, rationale = debater._parse_verdict(text)
        assert verdict == "proceed"
        assert confidence == 0.8

    def test_parse_revise(self, debater):
        text = "Goal 3 is too vague.\n\nVERDICT: REVISE — need more specificity"
        verdict, confidence, rationale = debater._parse_verdict(text)
        assert verdict == "revise"
        assert confidence == 0.7

    def test_parse_block(self, debater):
        text = "Critical dependency missing.\n\nVERDICT: BLOCK"
        verdict, confidence, rationale = debater._parse_verdict(text)
        assert verdict == "block"
        assert confidence == 0.8

    def test_parse_no_verdict_defaults_proceed(self, debater):
        text = "Everything looks reasonable but I'm not sure."
        verdict, confidence, rationale = debater._parse_verdict(text)
        assert verdict == "proceed"
        assert confidence == 0.5

    def test_parse_case_insensitive(self, debater):
        text = "verdict: proceed"
        verdict, _, _ = debater._parse_verdict(text)
        assert verdict == "proceed"

    def test_parse_extracts_rationale(self, debater):
        text = "The approach has merit.\nGoals are well-aligned.\n\nVERDICT: PROCEED"
        verdict, _, rationale = debater._parse_verdict(text)
        assert verdict == "proceed"
        assert "well-aligned" in rationale


class TestSuggestionExtraction:
    """Tests for extracting suggestions from debate output."""

    @pytest.fixture
    def debater(self):
        return StageTransitionDebater()

    def test_extract_dash_suggestions(self, debater):
        text = (
            "Analysis:\n"
            "- Should merge goals 1 and 2\n"
            "- Consider adding a testing phase\n"
            "- Recommend parallel execution\n"
            "VERDICT: REVISE"
        )
        suggestions = debater._extract_suggestions(text)
        assert len(suggestions) == 3
        assert "merge goals 1 and 2" in suggestions[0]

    def test_extract_numbered_suggestions(self, debater):
        text = "1. Add error handling\n2. Split the deploy step\n3. Add rollback\nVERDICT: REVISE"
        suggestions = debater._extract_suggestions(text)
        assert len(suggestions) == 3

    def test_no_suggestions(self, debater):
        text = "Everything looks good.\nVERDICT: PROCEED"
        suggestions = debater._extract_suggestions(text)
        assert suggestions == []

    def test_max_five_suggestions(self, debater):
        text = "\n".join(f"- Should do thing {i}" for i in range(10))
        suggestions = debater._extract_suggestions(text)
        assert len(suggestions) <= 5


class TestFallbackDebate:
    """Tests for the structural fallback when Arena is unavailable."""

    @pytest.fixture
    def debater(self):
        return StageTransitionDebater()

    def test_fallback_returns_proceed(self, debater):
        result = debater._fallback_debate("any task")
        assert "PROCEED" in result["final_answer"]
        assert result["consensus_reached"] is True
        assert result["confidence"] == 0.5


class TestReceiptGeneration:
    """Tests for transition debate receipt generation."""

    @pytest.fixture
    def debater(self):
        return StageTransitionDebater(enable_receipts=True)

    def test_generates_receipt(self, debater):
        td_result = TransitionDebateResult(
            transition_id="td-test",
            from_stage="ideas",
            to_stage="goals",
            verdict="proceed",
            confidence=0.8,
            rationale="Goals are solid",
        )
        receipt = debater._generate_transition_receipt(
            td_result,
            task="Review the transition",
            debate_result={"consensus_reached": True, "participants": ["a1", "a2"]},
        )
        assert receipt["receipt_id"] == "receipt-td-test"
        assert receipt["type"] == "stage_transition"
        assert receipt["verdict"] == "proceed"
        assert receipt["confidence"] == 0.8
        assert receipt["content_hash"]  # SHA-256 hash exists
        assert receipt["transition"]["from_stage"] == "ideas"
        assert receipt["transition"]["to_stage"] == "goals"


class TestDebateIdeasToGoals:
    """Tests for the Ideas → Goals transition debate."""

    @pytest.mark.asyncio
    async def test_debate_ideas_to_goals_with_arena(self):
        debater = StageTransitionDebater(debate_rounds=2)

        mock_result = MagicMock()
        mock_result.final_answer = "Goals are well-formed.\n\nVERDICT: PROCEED"
        mock_result.consensus_reached = True
        mock_result.confidence = 0.85
        mock_result.id = "debate-123"
        mock_result.dissenting_views = []
        mock_result.participants = ["agent-1", "agent-2"]

        mock_arena = AsyncMock()
        mock_arena.run.return_value = mock_result

        with patch(
            "aragora.pipeline.stage_debate.StageTransitionDebater._execute_arena_debate"
        ) as mock_exec:
            mock_exec.return_value = {
                "final_answer": "Goals are well-formed.\n\nVERDICT: PROCEED",
                "consensus_reached": True,
                "confidence": 0.85,
                "debate_id": "debate-123",
                "dissenting_views": [],
                "participants": ["agent-1", "agent-2"],
            }

            result = await debater.debate_ideas_to_goals(
                ideas=["improve UX", "add caching"],
                proposed_goals=[
                    {"title": "Redesign navigation", "priority": "high"},
                    {"title": "Add Redis cache", "priority": "medium"},
                ],
            )

        assert result.verdict == "proceed"
        assert result.confidence == 0.8
        assert result.from_stage == "ideas"
        assert result.to_stage == "goals"
        assert result.duration_ms > 0

    @pytest.mark.asyncio
    async def test_debate_fallback_on_arena_failure(self):
        debater = StageTransitionDebater()

        with patch(
            "aragora.pipeline.stage_debate.StageTransitionDebater._execute_arena_debate",
            side_effect=RuntimeError("Arena unavailable"),
        ):
            result = await debater.debate_ideas_to_goals(
                ideas=["idea1"],
                proposed_goals=[{"title": "goal1"}],
            )

        # Should gracefully fallback
        assert result.verdict == "proceed"
        assert result.confidence == 0.5
        assert "unavailable" in result.rationale.lower()


class TestDebateGoalsToActions:
    """Tests for the Goals → Actions transition debate."""

    @pytest.mark.asyncio
    async def test_debate_goals_to_actions(self):
        debater = StageTransitionDebater()

        with patch(
            "aragora.pipeline.stage_debate.StageTransitionDebater._execute_arena_debate"
        ) as mock_exec:
            mock_exec.return_value = {
                "final_answer": (
                    "Action plan needs revision.\n"
                    "- Should add a testing step\n"
                    "- Consider parallel execution\n\n"
                    "VERDICT: REVISE"
                ),
                "consensus_reached": True,
                "confidence": 0.7,
                "debate_id": "debate-456",
                "dissenting_views": ["Agent-2 thinks plan is fine"],
                "participants": ["agent-1", "agent-2"],
            }

            result = await debater.debate_goals_to_actions(
                goals=[{"title": "Redesign nav", "priority": "high"}],
                proposed_actions=[
                    {"name": "Research UX patterns", "step_type": "research"},
                    {"name": "Implement new nav", "step_type": "task"},
                ],
            )

        assert result.verdict == "revise"
        assert result.from_stage == "goals"
        assert result.to_stage == "actions"
        assert len(result.suggestions) >= 1


class TestDebateActionsToOrchestration:
    """Tests for the Actions → Orchestration transition debate."""

    @pytest.mark.asyncio
    async def test_debate_actions_to_orchestration(self):
        debater = StageTransitionDebater()

        with patch(
            "aragora.pipeline.stage_debate.StageTransitionDebater._execute_arena_debate"
        ) as mock_exec:
            mock_exec.return_value = {
                "final_answer": "Assignments look optimal.\n\nVERDICT: PROCEED",
                "consensus_reached": True,
                "confidence": 0.9,
                "debate_id": "debate-789",
                "dissenting_views": [],
                "participants": ["agent-1", "agent-2"],
            }

            result = await debater.debate_actions_to_orchestration(
                actions=[
                    {"name": "Research UX patterns"},
                    {"name": "Implement new nav"},
                ],
                proposed_assignments=[
                    {"name": "Research UX patterns", "agent_id": "agent-researcher"},
                    {"name": "Implement new nav", "agent_id": "agent-implementer"},
                ],
            )

        assert result.verdict == "proceed"
        assert result.from_stage == "actions"
        assert result.to_stage == "orchestration"


class TestDebaterConfiguration:
    """Tests for debater configuration options."""

    def test_default_config(self):
        debater = StageTransitionDebater()
        assert debater.debate_rounds == 2
        assert debater.num_agents == 2
        assert debater.enable_receipts is True
        assert debater.timeout_seconds == 300

    def test_custom_config(self):
        debater = StageTransitionDebater(
            debate_rounds=3,
            num_agents=3,
            enable_receipts=False,
            timeout_seconds=600,
        )
        assert debater.debate_rounds == 3
        assert debater.num_agents == 3
        assert debater.enable_receipts is False
        assert debater.timeout_seconds == 600

    @pytest.mark.asyncio
    async def test_receipts_disabled(self):
        debater = StageTransitionDebater(enable_receipts=False)

        with patch(
            "aragora.pipeline.stage_debate.StageTransitionDebater._execute_arena_debate"
        ) as mock_exec:
            mock_exec.return_value = {
                "final_answer": "VERDICT: PROCEED",
                "consensus_reached": True,
                "confidence": 0.8,
                "debate_id": None,
                "dissenting_views": [],
                "participants": [],
            }

            result = await debater.debate_ideas_to_goals(
                ideas=["idea"],
                proposed_goals=[{"title": "goal"}],
            )

        assert result.receipt is None


class TestPipelineIntegration:
    """Tests for integration with IdeaToExecutionPipeline."""

    def test_pipeline_config_has_transition_debate_flags(self):
        from aragora.pipeline.idea_to_execution import PipelineConfig

        config = PipelineConfig()
        assert config.enable_transition_debates is False
        assert config.transition_debate_rounds == 2

    def test_pipeline_config_enables_debates(self):
        from aragora.pipeline.idea_to_execution import PipelineConfig

        config = PipelineConfig(
            enable_transition_debates=True,
            transition_debate_rounds=3,
        )
        assert config.enable_transition_debates is True
        assert config.transition_debate_rounds == 3
