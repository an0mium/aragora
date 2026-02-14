"""Tests for deliberation template injection into debate prompts."""

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import pytest

from aragora.debate.protocol import DebateProtocol


@dataclass
class FakeAgent:
    name: str = "test_agent"
    role: str = "analyst"


@dataclass
class FakeEnvironment:
    task: str = "Evaluate the proposal"
    context: str = ""


def _make_prompt_builder(protocol: DebateProtocol | None = None):
    """Create a minimal PromptBuilder for testing."""
    from aragora.debate.prompt_builder import PromptBuilder

    proto = protocol or DebateProtocol()
    env = FakeEnvironment()
    return PromptBuilder(protocol=proto, env=env)


class TestDeliberationTemplateField:
    """Test the deliberation_template field on DebateProtocol."""

    def test_default_is_none(self):
        protocol = DebateProtocol()
        assert protocol.deliberation_template is None

    def test_can_set_template_name(self):
        protocol = DebateProtocol(deliberation_template="code_review")
        assert protocol.deliberation_template == "code_review"


class TestGetDeliberationTemplateContext:
    """Test the get_deliberation_template_context method on PromptContextMixin."""

    def test_returns_empty_when_not_set(self):
        builder = _make_prompt_builder(DebateProtocol(deliberation_template=None))
        result = builder.get_deliberation_template_context()
        assert result == ""

    def test_returns_empty_when_template_not_found(self):
        builder = _make_prompt_builder(DebateProtocol(deliberation_template="nonexistent_template"))
        with patch(
            "aragora.deliberation.templates.registry.get_template",
            return_value=None,
        ):
            result = builder.get_deliberation_template_context()
            assert result == ""

    def test_returns_context_with_system_prompt_additions(self):
        from aragora.deliberation.templates.base import (
            DeliberationTemplate,
            TemplateCategory,
        )

        template = DeliberationTemplate(
            name="code_review",
            description="Review code changes",
            category=TemplateCategory.CODE,
            system_prompt_additions="Focus on security vulnerabilities and performance.",
            personas=["Security Expert", "Performance Engineer"],
        )

        builder = _make_prompt_builder(DebateProtocol(deliberation_template="code_review"))
        with patch(
            "aragora.deliberation.templates.registry.get_template",
            return_value=template,
        ):
            result = builder.get_deliberation_template_context()
            assert "DELIBERATION TEMPLATE: code_review" in result
            assert "code" in result
            assert "Review code changes" in result
            assert "Focus on security vulnerabilities and performance." in result
            assert "Security Expert" in result
            assert "Performance Engineer" in result

    def test_returns_context_without_system_prompt(self):
        from aragora.deliberation.templates.base import (
            DeliberationTemplate,
            TemplateCategory,
        )

        template = DeliberationTemplate(
            name="general_review",
            description="General review template",
            category=TemplateCategory.GENERAL,
            system_prompt_additions=None,
            personas=[],
        )

        builder = _make_prompt_builder(DebateProtocol(deliberation_template="general_review"))
        with patch(
            "aragora.deliberation.templates.registry.get_template",
            return_value=template,
        ):
            result = builder.get_deliberation_template_context()
            assert "DELIBERATION TEMPLATE: general_review" in result
            assert "General review template" in result
            # No system_prompt_additions or personas sections
            assert "Assigned Personas" not in result

    def test_personas_formatted_correctly(self):
        from aragora.deliberation.templates.base import (
            DeliberationTemplate,
            TemplateCategory,
        )

        template = DeliberationTemplate(
            name="legal_review",
            description="Legal contract review",
            category=TemplateCategory.LEGAL,
            personas=["Contract Lawyer", "Risk Analyst", "Compliance Officer"],
        )

        builder = _make_prompt_builder(DebateProtocol(deliberation_template="legal_review"))
        with patch(
            "aragora.deliberation.templates.registry.get_template",
            return_value=template,
        ):
            result = builder.get_deliberation_template_context()
            assert "- Contract Lawyer" in result
            assert "- Risk Analyst" in result
            assert "- Compliance Officer" in result

    def test_handles_import_error_gracefully(self):
        builder = _make_prompt_builder(DebateProtocol(deliberation_template="some_template"))
        with patch(
            "aragora.deliberation.templates.registry.get_template",
            side_effect=ImportError("no module"),
        ):
            result = builder.get_deliberation_template_context()
            assert result == ""


class TestTemplateInjectionInPrompts:
    """Test that template context is injected into built prompts."""

    def test_proposal_prompt_includes_template_context(self):
        from aragora.deliberation.templates.base import (
            DeliberationTemplate,
            TemplateCategory,
        )

        template = DeliberationTemplate(
            name="code_review",
            description="Review code for quality",
            category=TemplateCategory.CODE,
            system_prompt_additions="Check for OWASP Top 10 vulnerabilities.",
            personas=["Security Auditor"],
        )

        builder = _make_prompt_builder(DebateProtocol(deliberation_template="code_review"))
        agent = FakeAgent()

        with patch(
            "aragora.deliberation.templates.registry.get_template",
            return_value=template,
        ):
            prompt = builder.build_proposal_prompt(agent)
            assert "DELIBERATION TEMPLATE: code_review" in prompt
            assert "Check for OWASP Top 10 vulnerabilities." in prompt
            assert "Security Auditor" in prompt

    def test_proposal_prompt_no_template_when_none(self):
        builder = _make_prompt_builder(DebateProtocol(deliberation_template=None))
        agent = FakeAgent()
        prompt = builder.build_proposal_prompt(agent)
        assert "DELIBERATION TEMPLATE" not in prompt

    def test_revision_prompt_includes_template_context(self):
        from aragora.debate.prompt_builder import PromptBuilder
        from aragora.deliberation.templates.base import (
            DeliberationTemplate,
            TemplateCategory,
        )

        template = DeliberationTemplate(
            name="finance_audit",
            description="Financial audit review",
            category=TemplateCategory.FINANCE,
            system_prompt_additions="Verify SOX compliance controls.",
            personas=["Auditor", "CFO Advisor"],
        )

        builder = _make_prompt_builder(DebateProtocol(deliberation_template="finance_audit"))
        agent = FakeAgent()

        # Create a mock critique
        critique = MagicMock()
        critique.to_prompt.return_value = "Consider edge cases."

        with patch(
            "aragora.deliberation.templates.registry.get_template",
            return_value=template,
        ):
            prompt = builder.build_revision_prompt(
                agent, "Original proposal text", [critique]
            )
            assert "DELIBERATION TEMPLATE: finance_audit" in prompt
            assert "Verify SOX compliance controls." in prompt
            assert "Auditor" in prompt

    def test_revision_prompt_no_template_when_none(self):
        builder = _make_prompt_builder(DebateProtocol(deliberation_template=None))
        agent = FakeAgent()

        critique = MagicMock()
        critique.to_prompt.return_value = "Fix this."

        prompt = builder.build_revision_prompt(agent, "Original", [critique])
        assert "DELIBERATION TEMPLATE" not in prompt

    def test_invalid_template_name_produces_no_section(self):
        builder = _make_prompt_builder(
            DebateProtocol(deliberation_template="does_not_exist")
        )
        agent = FakeAgent()

        with patch(
            "aragora.deliberation.templates.registry.get_template",
            return_value=None,
        ):
            prompt = builder.build_proposal_prompt(agent)
            assert "DELIBERATION TEMPLATE" not in prompt
