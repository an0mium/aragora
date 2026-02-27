"""Tests for PromptToSpecEngine state machine."""

import pytest
from unittest.mock import AsyncMock, patch
from aragora.prompt_engine.types import (
    EngineStage,
    UserProfile,
    ResearchSource,
)


class TestStartSession:
    async def test_creates_session_at_intake(self):
        from aragora.prompt_engine.engine import PromptToSpecEngine

        engine = PromptToSpecEngine()
        state = await engine.start_session(
            raw_prompt="make onboarding better",
            profile=UserProfile.FOUNDER,
        )
        assert state.stage == EngineStage.INTAKE
        assert state.raw_prompt == "make onboarding better"
        assert state.profile == UserProfile.FOUNDER
        assert state.session_id

    async def test_uses_default_research_sources(self):
        from aragora.prompt_engine.engine import PromptToSpecEngine

        engine = PromptToSpecEngine()
        state = await engine.start_session(raw_prompt="test", profile=UserProfile.CTO)
        assert ResearchSource.KNOWLEDGE_MOUND in state.research_sources
        assert ResearchSource.CODEBASE in state.research_sources

    async def test_custom_research_sources(self):
        from aragora.prompt_engine.engine import PromptToSpecEngine

        engine = PromptToSpecEngine()
        state = await engine.start_session(
            raw_prompt="test",
            profile=UserProfile.FOUNDER,
            research_sources=[ResearchSource.WEB],
        )
        assert state.research_sources == [ResearchSource.WEB]


class TestDecompose:
    async def test_transitions_to_decompose(self):
        from aragora.prompt_engine.engine import PromptToSpecEngine

        engine = PromptToSpecEngine()
        state = await engine.start_session("improve error handling", UserProfile.CTO)

        with patch.object(engine, "_call_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = {
                "intent_type": "improvement",
                "domains": ["error-handling", "reliability"],
                "ambiguities": ["which errors?"],
                "assumptions": ["python backend"],
                "scope_estimate": "medium",
            }
            state = await engine.decompose(state.session_id)

        assert state.stage == EngineStage.DECOMPOSE
        assert state.intent is not None
        assert state.intent.intent_type == "improvement"
        assert "error-handling" in state.intent.domains

    async def test_decompose_requires_intake_stage(self):
        from aragora.prompt_engine.engine import PromptToSpecEngine

        engine = PromptToSpecEngine()
        state = await engine.start_session("test", UserProfile.FOUNDER)
        engine._sessions[state.session_id].stage = EngineStage.RESEARCH
        with pytest.raises(ValueError, match="Expected stage"):
            await engine.decompose(state.session_id)


class TestInterrogate:
    async def test_generates_questions(self):
        from aragora.prompt_engine.engine import PromptToSpecEngine

        engine = PromptToSpecEngine()
        state = await engine.start_session("add auth", UserProfile.CTO)

        with patch.object(engine, "_call_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = {
                "intent_type": "feature",
                "domains": ["auth"],
                "ambiguities": ["auth method?"],
                "assumptions": [],
                "scope_estimate": "large",
            }
            state = await engine.decompose(state.session_id)

            mock_llm.return_value = {
                "questions": [
                    {
                        "id": "q1",
                        "question": "Which auth method?",
                        "why_it_matters": "Determines architecture",
                        "options": [
                            {"label": "jwt", "description": "JWT tokens", "tradeoff": "Stateless"},
                            {
                                "label": "session",
                                "description": "Server sessions",
                                "tradeoff": "Stateful",
                            },
                        ],
                        "default_option": "jwt",
                        "impact": "high",
                    }
                ]
            }
            state = await engine.generate_questions(state.session_id)

        assert state.stage == EngineStage.INTERROGATE
        assert len(state.questions) == 1
        assert state.questions[0].id == "q1"

    async def test_answer_question(self):
        from aragora.prompt_engine.engine import PromptToSpecEngine

        engine = PromptToSpecEngine()
        state = await engine.start_session("add auth", UserProfile.FOUNDER)

        with patch.object(engine, "_call_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = {
                "intent_type": "feature",
                "domains": ["auth"],
                "ambiguities": [],
                "assumptions": [],
                "scope_estimate": "medium",
            }
            await engine.decompose(state.session_id)
            mock_llm.return_value = {
                "questions": [
                    {
                        "id": "q1",
                        "question": "Which auth?",
                        "why_it_matters": "Architecture",
                        "impact": "high",
                        "options": [
                            {"label": "jwt", "description": "JWT", "tradeoff": "Stateless"}
                        ],
                        "default_option": "jwt",
                    }
                ]
            }
            await engine.generate_questions(state.session_id)

        state = await engine.answer_question(state.session_id, "q1", "jwt")
        assert state.answers["q1"] == "jwt"

    async def test_founder_gets_fewer_questions(self):
        from aragora.prompt_engine.types import PROFILE_DEFAULTS

        depth = PROFILE_DEFAULTS["founder"]["interrogation_depth"]
        assert depth == "quick"


class TestGetSession:
    async def test_get_existing_session(self):
        from aragora.prompt_engine.engine import PromptToSpecEngine

        engine = PromptToSpecEngine()
        state = await engine.start_session("test", UserProfile.FOUNDER)
        retrieved = engine.get_session(state.session_id)
        assert retrieved.session_id == state.session_id

    async def test_get_nonexistent_session_raises(self):
        from aragora.prompt_engine.engine import PromptToSpecEngine

        engine = PromptToSpecEngine()
        with pytest.raises(KeyError):
            engine.get_session("nonexistent")

    async def test_delete_session(self):
        from aragora.prompt_engine.engine import PromptToSpecEngine

        engine = PromptToSpecEngine()
        state = await engine.start_session("test", UserProfile.FOUNDER)
        engine.delete_session(state.session_id)
        with pytest.raises(KeyError):
            engine.get_session(state.session_id)


class TestConfidenceGate:
    def test_high_complexity_triggers_debate(self):
        from aragora.prompt_engine.engine import should_validate

        assert (
            should_validate(
                estimated_complexity="high", confidence=0.95, auto_execute_threshold=0.8
            )
            is True
        )

    def test_low_confidence_triggers_debate(self):
        from aragora.prompt_engine.engine import should_validate

        assert (
            should_validate(
                estimated_complexity="small", confidence=0.5, auto_execute_threshold=0.8
            )
            is True
        )

    def test_high_confidence_low_complexity_skips_debate(self):
        from aragora.prompt_engine.engine import should_validate

        assert (
            should_validate(
                estimated_complexity="small", confidence=0.9, auto_execute_threshold=0.8
            )
            is False
        )

    def test_medium_complexity_with_sufficient_confidence_skips(self):
        from aragora.prompt_engine.engine import should_validate

        assert (
            should_validate(
                estimated_complexity="medium", confidence=0.91, auto_execute_threshold=0.9
            )
            is False
        )
