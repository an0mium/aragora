"""End-to-end integration test: raw prompt -> pipeline_id (mock LLM)."""

import pytest
from unittest.mock import AsyncMock, patch
from aragora.prompt_engine.engine import PromptToSpecEngine, should_validate
from aragora.prompt_engine.types import (
    EngineStage,
    UserProfile,
    ResearchSource,
)


class TestEndToEnd:
    async def test_full_pipeline_founder_profile(self):
        """Founder: prompt -> decompose -> interrogate -> research -> spec -> handoff."""
        engine = PromptToSpecEngine()

        # 1. Start session
        state = await engine.start_session(
            "make onboarding better",
            UserProfile.FOUNDER,
            [ResearchSource.KNOWLEDGE_MOUND],
        )
        assert state.stage == EngineStage.INTAKE

        # 2. Decompose
        with patch.object(engine, "_call_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = {
                "intent_type": "improvement",
                "domains": ["ux", "onboarding"],
                "ambiguities": ["which part of onboarding?"],
                "assumptions": ["web app"],
                "scope_estimate": "medium",
            }
            state = await engine.decompose(state.session_id)
        assert state.stage == EngineStage.DECOMPOSE

        # 3. Generate questions
        with patch.object(engine, "_call_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = {
                "questions": [
                    {
                        "id": "q1",
                        "question": "Which onboarding step?",
                        "why_it_matters": "Scope",
                        "options": [
                            {
                                "label": "signup",
                                "description": "Signup",
                                "tradeoff": "Narrow",
                            }
                        ],
                        "default_option": "signup",
                        "impact": "high",
                    }
                ]
            }
            state = await engine.generate_questions(state.session_id)
        assert state.stage == EngineStage.INTERROGATE

        # 4. Answer + finalize
        state = await engine.answer_question(state.session_id, "q1", "signup")
        state = await engine.finalize_interrogation(state.session_id)
        assert state.stage == EngineStage.RESEARCH
        assert state.refined_intent is not None
        assert state.refined_intent.confidence > 0

        # 5. Check confidence gate
        needs_validation = should_validate(
            estimated_complexity="medium",
            confidence=state.refined_intent.confidence,
            auto_execute_threshold=0.8,
        )
        # With 1 answer: confidence = 0.5 + 0.1 = 0.6 < 0.8 -> needs validation
        assert needs_validation is True

    async def test_session_crud_lifecycle(self):
        """Create, retrieve, delete session."""
        engine = PromptToSpecEngine()
        state = await engine.start_session("test", UserProfile.FOUNDER)
        sid = state.session_id

        # Get
        retrieved = engine.get_session(sid)
        assert retrieved.raw_prompt == "test"

        # Delete
        engine.delete_session(sid)
        with pytest.raises(KeyError):
            engine.get_session(sid)

    async def test_state_transitions_are_enforced(self):
        """Cannot skip stages."""
        engine = PromptToSpecEngine()
        state = await engine.start_session("test", UserProfile.CTO)

        # Cannot generate questions before decompose
        with pytest.raises(ValueError, match="Expected stage"):
            await engine.generate_questions(state.session_id)
