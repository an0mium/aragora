"""Shared fixtures for prompt engine tests."""

import pytest
from aragora.prompt_engine.types import (
    EngineStage,
    UserProfile,
    ResearchSource,
    SessionState,
    PromptIntent,
    ClarifyingQuestion,
    QuestionOption,
    RefinedIntent,
)


@pytest.fixture
def sample_intent():
    return PromptIntent(
        raw_prompt="make onboarding better",
        intent_type="improvement",
        domains=["ux", "onboarding"],
        ambiguities=["what does 'better' mean?"],
        assumptions=["web app onboarding"],
        scope_estimate="medium",
    )


@pytest.fixture
def sample_session(sample_intent):
    return SessionState(
        session_id="test-sess-1",
        stage=EngineStage.INTAKE,
        profile=UserProfile.FOUNDER,
        research_sources=[ResearchSource.KNOWLEDGE_MOUND, ResearchSource.CODEBASE],
        raw_prompt="make onboarding better",
    )


@pytest.fixture
def sample_questions():
    return [
        ClarifyingQuestion(
            id="q1",
            question="What aspect of onboarding needs improvement?",
            why_it_matters="Determines scope of changes",
            options=[
                QuestionOption("signup", "Signup flow", "Narrow scope"),
                QuestionOption("tutorial", "Tutorial system", "Broader scope"),
            ],
            default_option="signup",
            impact="high",
        ),
    ]
