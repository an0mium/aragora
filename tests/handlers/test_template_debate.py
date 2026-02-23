"""Tests for template-driven debate creation."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from aragora.server.debate_controller import DebateRequest


class TestDebateRequestTemplate:
    """Tests for template support in DebateRequest.from_dict()."""

    def test_template_applies_defaults(self):
        """Template defaults are applied when user doesn't specify values."""
        request = DebateRequest.from_dict(
            {
                "question": "Should we hire this VP candidate?",
                "template": "hiring_decision",
            }
        )
        assert request.template_name == "hiring_decision"
        # hiring_decision has max_rounds=4 and default_agents
        assert request.rounds == 4
        assert request.metadata.get("template_name") == "hiring_decision"

    def test_template_user_overrides_rounds(self):
        """User-specified rounds override template defaults."""
        request = DebateRequest.from_dict(
            {
                "question": "Review this code",
                "template": "code_review",
                "rounds": 7,
            }
        )
        assert request.template_name == "code_review"
        # User specified 7 rounds, should override template's 3
        assert request.rounds == 7

    def test_template_user_overrides_agents(self):
        """User-specified agents override template defaults."""
        request = DebateRequest.from_dict(
            {
                "question": "Review this contract",
                "template": "contract_review",
                "agents": ["anthropic-api", "gemini"],
            }
        )
        assert request.template_name == "contract_review"
        assert request.agents_str == ["anthropic-api", "gemini"]

    def test_unknown_template_raises_error(self):
        """Unknown template name raises ValueError."""
        with pytest.raises(ValueError, match="Unknown template"):
            DebateRequest.from_dict(
                {
                    "question": "Test question",
                    "template": "nonexistent_template_xyz",
                }
            )

    def test_no_template_works_normally(self):
        """Requests without template work as before."""
        request = DebateRequest.from_dict(
            {
                "question": "What is the best approach?",
            }
        )
        assert request.template_name is None
        assert "template_name" not in (request.metadata or {})

    def test_template_applies_consensus_threshold(self):
        """Template consensus threshold is used as fallback."""
        request = DebateRequest.from_dict(
            {
                "question": "Audit these financial statements",
                "template": "financial_audit",
            }
        )
        # financial_audit has consensus_threshold=0.85
        assert request.consensus == "0.85"

    def test_template_user_overrides_consensus(self):
        """User-specified consensus overrides template default."""
        request = DebateRequest.from_dict(
            {
                "question": "Audit these statements",
                "template": "financial_audit",
                "consensus": "majority",
            }
        )
        assert request.consensus == "majority"

    def test_template_applies_default_agents(self):
        """Template default agents are applied when no agents specified."""
        request = DebateRequest.from_dict(
            {
                "question": "Review this security posture",
                "template": "security_audit",
            }
        )
        # security_audit has default_agents
        assert "anthropic-api" in request.agents_str
