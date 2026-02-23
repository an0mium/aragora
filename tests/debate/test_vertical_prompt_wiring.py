"""Tests for vertical weight profile wiring into debate prompts."""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock


def _make_protocol(**overrides):
    """Create a mock DebateProtocol."""
    proto = MagicMock()
    proto.rounds = 3
    proto.asymmetric_stances = False
    proto.agreement_intensity = None
    proto.enforce_language = False
    proto.response_language = "en"
    proto.audience_injection = None
    proto.enable_privacy_anonymization = False
    for k, v in overrides.items():
        setattr(proto, k, v)
    return proto


def _make_env(task="Evaluate patient treatment options"):
    env = MagicMock()
    env.task = task
    env.context = ""
    return env


def _make_agent(name="claude_proposer", role="proposer"):
    agent = MagicMock()
    agent.name = name
    agent.role = role
    return agent


class TestGetVerticalContext:
    """Test PromptContextMixin.get_vertical_context()."""

    def test_no_vertical_returns_empty(self):
        from aragora.debate.prompt_builder import PromptBuilder

        pb = PromptBuilder(protocol=_make_protocol(), env=_make_env())
        assert pb.get_vertical_context() == ""

    def test_unknown_profile_returns_empty(self):
        from aragora.debate.prompt_builder import PromptBuilder

        pb = PromptBuilder(protocol=_make_protocol(), env=_make_env(), vertical="nonexistent")
        assert pb.get_vertical_context() == ""

    def test_healthcare_hipaa_profile(self):
        from aragora.debate.prompt_builder import PromptBuilder

        pb = PromptBuilder(protocol=_make_protocol(), env=_make_env(), vertical="healthcare_hipaa")
        ctx = pb.get_vertical_context()
        assert "Healthcare Hipaa" in ctx
        assert "ACCURACY" in ctx
        assert "SAFETY" in ctx
        # Creativity should be de-emphasized (weight = 0.0)
        assert "CREATIVITY" in ctx
        assert "De-emphasized" in ctx

    def test_financial_audit_profile(self):
        from aragora.debate.prompt_builder import PromptBuilder

        pb = PromptBuilder(protocol=_make_protocol(), env=_make_env(), vertical="financial_audit")
        ctx = pb.get_vertical_context()
        assert "Financial Audit" in ctx
        assert "ACCURACY" in ctx
        assert "30%" in ctx  # accuracy is 30% for financial_audit

    def test_legal_contract_profile(self):
        from aragora.debate.prompt_builder import PromptBuilder

        pb = PromptBuilder(protocol=_make_protocol(), env=_make_env(), vertical="legal_contract")
        ctx = pb.get_vertical_context()
        assert "Legal Contract" in ctx
        assert "COMPLETENESS" in ctx
        assert "25%" in ctx  # completeness is 25% for legal_contract

    def test_compliance_sox_profile(self):
        from aragora.debate.prompt_builder import PromptBuilder

        pb = PromptBuilder(protocol=_make_protocol(), env=_make_env(), vertical="compliance_sox")
        ctx = pb.get_vertical_context()
        assert "Compliance Sox" in ctx
        assert "ACCURACY" in ctx
        assert "COMPLETENESS" in ctx

    def test_general_profile_not_in_weight_profiles(self):
        """General/generic verticals without a weight profile return empty."""
        from aragora.debate.prompt_builder import PromptBuilder

        pb = PromptBuilder(protocol=_make_protocol(), env=_make_env(), vertical="general")
        assert pb.get_vertical_context() == ""


class TestVerticalInPromptAssembly:
    """Test that vertical context is injected into assembled prompts."""

    def test_proposal_prompt_includes_vertical(self):
        from aragora.debate.prompt_builder import PromptBuilder

        pb = PromptBuilder(
            protocol=_make_protocol(),
            env=_make_env(),
            vertical="healthcare_hipaa",
        )
        agent = _make_agent()
        prompt = pb.build_proposal_prompt(agent)
        assert "ACCURACY" in prompt
        assert "SAFETY" in prompt

    def test_proposal_prompt_no_vertical_no_section(self):
        from aragora.debate.prompt_builder import PromptBuilder

        pb = PromptBuilder(protocol=_make_protocol(), env=_make_env())
        agent = _make_agent()
        prompt = pb.build_proposal_prompt(agent)
        assert "Evaluation Profile" not in prompt

    def test_revision_prompt_includes_vertical(self):
        from aragora.debate.prompt_builder import PromptBuilder

        pb = PromptBuilder(
            protocol=_make_protocol(),
            env=_make_env(),
            vertical="financial_audit",
        )
        agent = _make_agent()
        critique = MagicMock()
        critique.to_prompt.return_value = "Needs more evidence"
        prompt = pb.build_revision_prompt(agent, "original text", [critique])
        assert "ACCURACY" in prompt
        assert "Financial Audit" in prompt


class TestVerticalPassthrough:
    """Test that vertical is properly passed through Arena initialization."""

    def test_prompt_builder_receives_vertical_param(self):
        """PromptBuilder.__init__ accepts vertical kwarg."""
        from aragora.debate.prompt_builder import PromptBuilder

        pb = PromptBuilder(
            protocol=_make_protocol(),
            env=_make_env(),
            vertical="healthcare_clinical",
        )
        assert pb.vertical == "healthcare_clinical"

    def test_prompt_builder_default_vertical_is_none(self):
        from aragora.debate.prompt_builder import PromptBuilder

        pb = PromptBuilder(protocol=_make_protocol(), env=_make_env())
        assert pb.vertical is None


class TestWeightProfileDimensions:
    """Test that all 7 vertical profiles produce valid context."""

    PROFILES = [
        "healthcare_hipaa",
        "healthcare_clinical",
        "financial_audit",
        "financial_risk",
        "legal_contract",
        "legal_due_diligence",
        "compliance_sox",
    ]

    @pytest.mark.parametrize("profile", PROFILES)
    def test_profile_produces_nonempty_context(self, profile):
        from aragora.debate.prompt_builder import PromptBuilder

        pb = PromptBuilder(
            protocol=_make_protocol(),
            env=_make_env(),
            vertical=profile,
        )
        ctx = pb.get_vertical_context()
        assert ctx, f"Profile {profile} should produce non-empty context"
        assert "Evaluation Profile" in ctx
        assert "%" in ctx  # Should contain percentage weights
