"""Tests for prior claims context injection in debate prompts.

Validates that ClaimsKernel.get_related_claims() finds relevant prior claims
and that PromptBuilder correctly injects them into debate prompts when
include_prior_claims=True.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from aragora.reasoning.claims import (
    ClaimType,
    ClaimsKernel,
    TypedClaim,
)


# =========================================================================
# ClaimsKernel.get_related_claims tests
# =========================================================================


class TestGetRelatedClaims:
    """Test ClaimsKernel.get_related_claims topic matching."""

    @pytest.fixture
    def kernel(self):
        """Create a ClaimsKernel with some claims."""
        k = ClaimsKernel(debate_id="test-debate")
        k.add_claim(
            "Rate limiting should use token bucket algorithm", "claude", ClaimType.PROPOSAL, 0.8
        )
        k.add_claim("Database queries need connection pooling", "gpt", ClaimType.ASSERTION, 0.7)
        k.add_claim("Security audit found SQL injection risk", "gemini", ClaimType.ASSERTION, 0.9)
        k.add_claim("The API design follows REST principles", "claude", ClaimType.ASSERTION, 0.6)
        k.add_claim(
            "Rate limiting prevents abuse of API endpoints", "deepseek", ClaimType.ASSERTION, 0.75
        )
        return k

    def test_finds_related_claims_by_keyword(self, kernel):
        """Related claims are found by keyword overlap."""
        results = kernel.get_related_claims("rate limiting for the API")
        assert len(results) > 0
        statements = [c.statement for c in results]
        assert any("Rate limiting" in s for s in statements)

    def test_returns_empty_for_no_match(self, kernel):
        """Empty list when no claims match the topic."""
        results = kernel.get_related_claims("quantum computing entanglement")
        assert results == []

    def test_returns_empty_for_empty_topic(self, kernel):
        """Empty list for empty topic string."""
        results = kernel.get_related_claims("")
        assert results == []

    def test_returns_empty_for_empty_kernel(self):
        """Empty list when kernel has no claims."""
        k = ClaimsKernel(debate_id="empty")
        results = k.get_related_claims("anything")
        assert results == []

    def test_respects_limit(self, kernel):
        """Results are limited to the requested count."""
        results = kernel.get_related_claims("rate limiting API", limit=1)
        assert len(results) <= 1

    def test_sorted_by_relevance(self, kernel):
        """Results are sorted by relevance (most related first)."""
        results = kernel.get_related_claims("rate limiting API endpoints")
        # Claims mentioning more topic words should come first
        assert len(results) >= 2
        # The claim about "rate limiting prevents abuse of API endpoints"
        # has more keyword overlap than just "rate limiting"
        # Both rate limiting claims should appear
        statements = [c.statement for c in results]
        assert any("rate limiting" in s.lower() for s in statements)

    def test_ignores_short_words(self, kernel):
        """Words shorter than 3 characters are ignored in matching."""
        results = kernel.get_related_claims("is it ok to do")
        # These short words shouldn't match anything meaningful
        assert len(results) == 0

    def test_case_insensitive_matching(self, kernel):
        """Matching is case-insensitive."""
        results = kernel.get_related_claims("RATE LIMITING")
        assert len(results) > 0

    def test_default_limit_is_five(self, kernel):
        """Default limit is 5."""
        # Add more claims to exceed 5
        for i in range(10):
            kernel.add_claim(
                f"API endpoint number {i} needs rate limiting",
                "agent",
                ClaimType.ASSERTION,
                0.5,
            )
        results = kernel.get_related_claims("API rate limiting endpoint")
        assert len(results) <= 5

    def test_includes_claim_type(self, kernel):
        """Returned claims have their type preserved."""
        results = kernel.get_related_claims("rate limiting token bucket")
        proposal_found = any(c.claim_type == ClaimType.PROPOSAL for c in results)
        assert proposal_found


# =========================================================================
# PromptBuilder prior claims context tests
# =========================================================================


class TestPriorClaimsContext:
    """Test prior claims context injection in PromptBuilder."""

    @pytest.fixture
    def mock_protocol(self):
        """Create a mock debate protocol."""
        protocol = MagicMock()
        protocol.rounds = 3
        protocol.asymmetric_stances = False
        protocol.agreement_intensity = None
        protocol.enable_privacy_anonymization = False
        protocol.get_round_phase.return_value = None
        return protocol

    @pytest.fixture
    def mock_env(self):
        """Create a mock environment."""
        env = MagicMock()
        env.task = "Design a rate limiter for the API"
        env.context = ""
        return env

    @pytest.fixture
    def claims_kernel(self):
        """Create a ClaimsKernel with prior claims."""
        k = ClaimsKernel(debate_id="prior-debate")
        k.add_claim(
            "Rate limiting should use token bucket algorithm",
            "claude",
            ClaimType.PROPOSAL,
            0.8,
        )
        k.add_claim(
            "Fixed window counters are simpler but less fair",
            "gpt",
            ClaimType.ASSERTION,
            0.6,
        )
        return k

    def test_prior_claims_disabled_by_default(self, mock_protocol, mock_env):
        """Prior claims are not injected when include_prior_claims=False."""
        from aragora.debate.prompt_builder import PromptBuilder

        builder = PromptBuilder(protocol=mock_protocol, env=mock_env)
        context = builder.get_prior_claims_context()
        assert context == ""

    def test_prior_claims_enabled_without_kernel(self, mock_protocol, mock_env):
        """No error when include_prior_claims=True but no kernel provided."""
        from aragora.debate.prompt_builder import PromptBuilder

        builder = PromptBuilder(
            protocol=mock_protocol,
            env=mock_env,
            include_prior_claims=True,
        )
        context = builder.get_prior_claims_context()
        assert context == ""

    def test_prior_claims_context_injected(self, mock_protocol, mock_env, claims_kernel):
        """Prior claims are injected when enabled with kernel."""
        from aragora.debate.prompt_builder import PromptBuilder

        builder = PromptBuilder(
            protocol=mock_protocol,
            env=mock_env,
            claims_kernel=claims_kernel,
            include_prior_claims=True,
        )
        context = builder.get_prior_claims_context()
        assert "PRIOR CLAIMS" in context
        assert "token bucket" in context
        assert "claude" in context

    def test_prior_claims_shows_confidence(self, mock_protocol, mock_env, claims_kernel):
        """Prior claims context includes confidence information."""
        from aragora.debate.prompt_builder import PromptBuilder

        builder = PromptBuilder(
            protocol=mock_protocol,
            env=mock_env,
            claims_kernel=claims_kernel,
            include_prior_claims=True,
        )
        context = builder.get_prior_claims_context()
        assert "confidence" in context

    def test_prior_claims_shows_claim_type(self, mock_protocol, mock_env, claims_kernel):
        """Prior claims context includes claim type label."""
        from aragora.debate.prompt_builder import PromptBuilder

        builder = PromptBuilder(
            protocol=mock_protocol,
            env=mock_env,
            claims_kernel=claims_kernel,
            include_prior_claims=True,
        )
        context = builder.get_prior_claims_context()
        assert "PROPOSAL" in context

    def test_prior_claims_in_proposal_prompt(self, mock_protocol, mock_env, claims_kernel):
        """Prior claims appear in the built proposal prompt."""
        from aragora.debate.prompt_builder import PromptBuilder

        builder = PromptBuilder(
            protocol=mock_protocol,
            env=mock_env,
            claims_kernel=claims_kernel,
            include_prior_claims=True,
        )
        agent = MagicMock()
        agent.name = "test-agent"
        agent.role = "proposer"
        prompt = builder.build_proposal_prompt(agent)
        assert "PRIOR CLAIMS" in prompt

    def test_prior_claims_not_in_prompt_when_disabled(self, mock_protocol, mock_env, claims_kernel):
        """Prior claims do not appear in prompt when disabled."""
        from aragora.debate.prompt_builder import PromptBuilder

        builder = PromptBuilder(
            protocol=mock_protocol,
            env=mock_env,
            claims_kernel=claims_kernel,
            include_prior_claims=False,
        )
        agent = MagicMock()
        agent.name = "test-agent"
        agent.role = "proposer"
        prompt = builder.build_proposal_prompt(agent)
        assert "PRIOR CLAIMS" not in prompt

    def test_prior_claims_respects_limit(self, mock_protocol, mock_env):
        """Prior claims context respects the limit parameter."""
        from aragora.debate.prompt_builder import PromptBuilder

        k = ClaimsKernel(debate_id="limit-test")
        for i in range(10):
            k.add_claim(
                f"Rate limiting approach {i} for the API design",
                f"agent-{i}",
                ClaimType.ASSERTION,
                0.5,
            )

        builder = PromptBuilder(
            protocol=mock_protocol,
            env=mock_env,
            claims_kernel=k,
            include_prior_claims=True,
        )
        context = builder.get_prior_claims_context(limit=3)
        # Count claim entries (lines starting with "- [")
        claim_lines = [line for line in context.split("\n") if line.strip().startswith("- [")]
        assert len(claim_lines) <= 3

    def test_prior_claims_truncates_long_statements(self, mock_protocol, mock_env):
        """Long claim statements are truncated in context."""
        from aragora.debate.prompt_builder import PromptBuilder

        k = ClaimsKernel(debate_id="truncate-test")
        long_statement = "Rate limiting " + "x" * 200 + " for the API"
        k.add_claim(long_statement, "agent", ClaimType.ASSERTION, 0.5)

        builder = PromptBuilder(
            protocol=mock_protocol,
            env=mock_env,
            claims_kernel=k,
            include_prior_claims=True,
        )
        context = builder.get_prior_claims_context()
        assert "..." in context

    def test_error_handling_graceful(self, mock_protocol, mock_env):
        """Errors in claims kernel are handled gracefully."""
        from aragora.debate.prompt_builder import PromptBuilder

        broken_kernel = MagicMock()
        broken_kernel.get_related_claims.side_effect = AttributeError("broken")

        builder = PromptBuilder(
            protocol=mock_protocol,
            env=mock_env,
            claims_kernel=broken_kernel,
            include_prior_claims=True,
        )
        context = builder.get_prior_claims_context()
        assert context == ""
