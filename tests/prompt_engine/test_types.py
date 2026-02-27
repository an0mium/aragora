"""Tests for prompt engine type definitions."""

from __future__ import annotations

import pytest

from aragora.prompt_engine.types import (
    AmbiguityLevel,
    ClarifyingQuestion,
    IntentType,
    PromptIntent,
    ResearchReport,
    RiskItem,
    Specification,
    SpecificationStatus,
    UserProfile,
)


class TestUserProfile:
    def test_all_profiles_exist(self) -> None:
        assert UserProfile.FOUNDER == "founder"
        assert UserProfile.CTO == "cto"
        assert UserProfile.BUSINESS == "business"
        assert UserProfile.TEAM == "team"

    def test_default_config_exists_for_each_profile(self) -> None:
        for profile in UserProfile:
            config = profile.default_config()
            assert "interrogation_depth" in config
            assert "auto_execute_threshold" in config
            assert "autonomy_level" in config


class TestIntentType:
    def test_intent_types(self) -> None:
        assert IntentType.FEATURE == "feature"
        assert IntentType.IMPROVEMENT == "improvement"
        assert IntentType.INVESTIGATION == "investigation"
        assert IntentType.FIX == "fix"
        assert IntentType.STRATEGIC == "strategic"


class TestPromptIntent:
    def test_create_intent(self) -> None:
        intent = PromptIntent(
            raw_prompt="make onboarding better",
            intent_type=IntentType.IMPROVEMENT,
            domains=["onboarding", "ux"],
            ambiguities=["What aspect of onboarding?"],
            assumptions=["Web app onboarding, not mobile"],
            scope_estimate="medium",
        )
        assert intent.intent_type == IntentType.IMPROVEMENT
        assert len(intent.domains) == 2
        assert len(intent.ambiguities) == 1

    def test_intent_to_dict(self) -> None:
        intent = PromptIntent(
            raw_prompt="test",
            intent_type=IntentType.FIX,
            domains=[],
            ambiguities=[],
            assumptions=[],
            scope_estimate="small",
        )
        d = intent.to_dict()
        assert d["intent_type"] == "fix"
        assert d["scope_estimate"] == "small"


class TestClarifyingQuestion:
    def test_create_question(self) -> None:
        q = ClarifyingQuestion(
            question="What aspect of onboarding needs improvement?",
            why_it_matters="Determines scope: UI changes vs backend flow vs copy",
            options=[
                {"label": "Sign-up flow", "description": "Registration and initial setup"},
                {"label": "First-run experience", "description": "What users see after sign-up"},
            ],
            default_option="Sign-up flow",
            impact_level="high",
        )
        assert q.impact_level == "high"
        assert len(q.options) == 2


class TestSpecification:
    def test_create_spec(self) -> None:
        spec = Specification(
            title="Improve onboarding sign-up flow",
            problem_statement="Users drop off during registration",
            proposed_solution="Simplify to 2-step sign-up",
            implementation_plan=[],
            risk_register=[],
            success_criteria=["Drop-off rate < 20%"],
            estimated_effort="medium",
            status=SpecificationStatus.DRAFT,
        )
        assert spec.status == SpecificationStatus.DRAFT
        assert len(spec.success_criteria) == 1

    def test_spec_to_dict(self) -> None:
        spec = Specification(
            title="Test",
            problem_statement="Problem",
            proposed_solution="Solution",
            implementation_plan=[],
            risk_register=[],
            success_criteria=[],
            estimated_effort="small",
            status=SpecificationStatus.DRAFT,
        )
        d = spec.to_dict()
        assert d["status"] == "draft"
        assert "provenance_chain" in d


class TestRiskItem:
    def test_create_risk(self) -> None:
        risk = RiskItem(
            description="Token leakage",
            likelihood="low",
            impact="high",
            mitigation="Use PKCE flow",
        )
        d = risk.to_dict()
        assert d["likelihood"] == "low"


class TestResearchReport:
    def test_empty_report(self) -> None:
        report = ResearchReport()
        d = report.to_dict()
        assert d["codebase_findings"] == []
        assert d["past_decisions"] == []
