"""Tests for prompt engine type definitions."""

import json
import pytest
from datetime import datetime, timezone


class TestEngineStage:
    def test_all_stages_exist(self):
        from aragora.prompt_engine.types import EngineStage

        stages = ["intake", "decompose", "interrogate", "research", "spec", "validate", "handoff"]
        for s in stages:
            assert EngineStage(s) == s

    def test_stage_ordering(self):
        from aragora.prompt_engine.types import EngineStage

        ordered = list(EngineStage)
        assert ordered[0] == EngineStage.INTAKE
        assert ordered[-1] == EngineStage.HANDOFF


class TestUserProfile:
    def test_all_profiles_exist(self):
        from aragora.prompt_engine.types import UserProfile

        for p in ["founder", "cto", "business", "team"]:
            assert UserProfile(p) == p


class TestResearchSource:
    def test_all_sources_exist(self):
        from aragora.prompt_engine.types import ResearchSource

        for s in ["km", "codebase", "obsidian", "web"]:
            assert ResearchSource(s) == s


class TestPromptIntent:
    def test_creation(self):
        from aragora.prompt_engine.types import PromptIntent

        intent = PromptIntent(
            raw_prompt="make onboarding better",
            intent_type="improvement",
            domains=["ux", "onboarding"],
            ambiguities=["what does 'better' mean?"],
            assumptions=["web app onboarding"],
            scope_estimate="medium",
        )
        assert intent.raw_prompt == "make onboarding better"
        assert intent.intent_type == "improvement"
        assert intent.enriched_dump is None


class TestClarifyingQuestion:
    def test_creation(self):
        from aragora.prompt_engine.types import ClarifyingQuestion, QuestionOption

        q = ClarifyingQuestion(
            id="q1",
            question="What aspect of onboarding?",
            why_it_matters="Determines scope",
            options=[
                QuestionOption(label="signup", description="Signup flow", tradeoff="Narrow scope"),
                QuestionOption(label="tutorial", description="Tutorial", tradeoff="Broader scope"),
            ],
            default_option="signup",
            impact="high",
        )
        assert q.id == "q1"
        assert len(q.options) == 2


class TestRefinedIntent:
    def test_creation(self):
        from aragora.prompt_engine.types import RefinedIntent, PromptIntent

        intent = PromptIntent(
            raw_prompt="test",
            intent_type="feature",
            domains=[],
            ambiguities=[],
            assumptions=[],
            scope_estimate="small",
        )
        refined = RefinedIntent(intent=intent, answers={"q1": "signup"}, confidence=0.85)
        assert refined.confidence == 0.85


class TestResearchReport:
    def test_creation_empty(self):
        from aragora.prompt_engine.types import ResearchReport, ResearchSource

        report = ResearchReport(
            km_precedents=[],
            codebase_context=[],
            obsidian_notes=[],
            web_results=[],
            sources_used=[ResearchSource.KNOWLEDGE_MOUND],
        )
        assert len(report.sources_used) == 1


class TestSessionState:
    def test_creation_minimal(self):
        from aragora.prompt_engine.types import (
            SessionState,
            EngineStage,
            UserProfile,
            ResearchSource,
        )

        state = SessionState(
            session_id="sess-1",
            stage=EngineStage.INTAKE,
            profile=UserProfile.FOUNDER,
            research_sources=[ResearchSource.KNOWLEDGE_MOUND, ResearchSource.CODEBASE],
            raw_prompt="make onboarding better",
        )
        assert state.session_id == "sess-1"
        assert state.stage == EngineStage.INTAKE
        assert state.intent is None
        assert state.spec is None
        assert isinstance(state.created_at, datetime)

    def test_to_dict_roundtrip(self):
        from aragora.prompt_engine.types import (
            SessionState,
            EngineStage,
            UserProfile,
            ResearchSource,
        )

        state = SessionState(
            session_id="sess-2",
            stage=EngineStage.DECOMPOSE,
            profile=UserProfile.CTO,
            research_sources=[ResearchSource.WEB],
            raw_prompt="add rate limiting",
        )
        d = state.to_dict()
        restored = SessionState.from_dict(d)
        assert restored.session_id == "sess-2"
        assert restored.stage == EngineStage.DECOMPOSE
        assert restored.profile == UserProfile.CTO

    def test_full_state_roundtrip(self):
        from aragora.prompt_engine.types import (
            SessionState,
            EngineStage,
            UserProfile,
            ResearchSource,
            PromptIntent,
            ClarifyingQuestion,
            QuestionOption,
            RefinedIntent,
        )

        state = SessionState(
            session_id="s2",
            stage=EngineStage.INTERROGATE,
            profile=UserProfile.CTO,
            research_sources=[ResearchSource.KNOWLEDGE_MOUND, ResearchSource.WEB],
            raw_prompt="add auth",
            intent=PromptIntent(
                raw_prompt="add auth",
                intent_type="feature",
                domains=["auth"],
                ambiguities=["method?"],
                assumptions=["backend"],
                scope_estimate="large",
            ),
            questions=[
                ClarifyingQuestion(
                    id="q1",
                    question="Which auth?",
                    why_it_matters="Architecture",
                    options=[QuestionOption("jwt", "JWT tokens", "Stateless")],
                    default_option="jwt",
                    impact="high",
                ),
            ],
            answers={"q1": "jwt"},
        )
        d = state.to_dict()
        json_str = json.dumps(d)
        restored = SessionState.from_dict(json.loads(json_str))
        assert restored.session_id == "s2"
        assert restored.intent.intent_type == "feature"
        assert len(restored.questions) == 1
        assert restored.answers["q1"] == "jwt"

    def test_provenance_hash_preserved(self):
        from aragora.prompt_engine.types import SessionState, EngineStage, UserProfile

        state = SessionState(
            session_id="s3",
            stage=EngineStage.INTAKE,
            profile=UserProfile.FOUNDER,
            research_sources=[],
            raw_prompt="test",
        )
        original_hash = state.provenance_hash
        d = state.to_dict()
        restored = SessionState.from_dict(d)
        assert restored.provenance_hash == original_hash

    def test_json_serializable(self):
        from aragora.prompt_engine.types import (
            SessionState,
            EngineStage,
            UserProfile,
            ResearchSource,
        )

        state = SessionState(
            session_id="s4",
            stage=EngineStage.SPEC,
            profile=UserProfile.BUSINESS,
            research_sources=[ResearchSource.OBSIDIAN],
            raw_prompt="improve UX",
        )
        json_str = json.dumps(state.to_dict())
        assert '"session_id": "s4"' in json_str


class TestProfileDefaults:
    def test_founder_defaults(self):
        from aragora.prompt_engine.types import PROFILE_DEFAULTS

        f = PROFILE_DEFAULTS["founder"]
        assert f["interrogation_depth"] == "quick"
        assert f["auto_execute_threshold"] == 0.8

    def test_team_always_requires_approval(self):
        from aragora.prompt_engine.types import PROFILE_DEFAULTS

        t = PROFILE_DEFAULTS["team"]
        assert t["auto_execute_threshold"] == 1.0
        assert t["require_approval"] is True
