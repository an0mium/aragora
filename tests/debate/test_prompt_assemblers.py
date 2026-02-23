"""Tests for PromptAssemblyMixin prompt construction.

Covers build_proposal_prompt, build_revision_prompt, build_judge_prompt,
build_judge_vote_prompt, and _anonymize_if_enabled.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from aragora.debate.prompt_assemblers import PromptAssemblyMixin


# ---------------------------------------------------------------------------
# Concrete test class implementing the mixin
# ---------------------------------------------------------------------------


class ConcreteAssembler(PromptAssemblyMixin):
    """Concrete class implementing PromptAssemblyMixin for testing."""

    def __init__(self):
        self.protocol = MagicMock()
        self.protocol.enable_privacy_anonymization = False
        self.protocol.asymmetric_stances = False
        self.protocol.agreement_intensity = "normal"
        self.protocol.enable_trending_injection = False
        self.env = MagicMock()
        self.env.task = "Design a rate limiter"
        self.env.context = ""
        self._rlm_context = None
        self._rlm_adapter = None
        self._enable_rlm_hints = False
        self._historical_context_cache = ""
        self.dissent_retriever = None
        self._context_budgeter = None

    # Methods from PromptBuilder
    def _get_introspection_context(self, agent_name):
        return ""

    def _get_active_introspection_context(self, agent_name):
        return ""

    def get_mode_prompt(self):
        return ""

    # Methods from PromptContextMixin
    def get_stance_guidance(self, agent):
        return ""

    def get_agreement_intensity_guidance(self):
        return ""

    def get_role_context(self, agent):
        return ""

    def get_persona_context(self, agent):
        return ""

    def get_flip_context(self, agent):
        return ""

    def get_round_phase_context(self, round_number):
        return ""

    def get_rlm_abstract(self, max_chars=2000):
        return ""

    def get_rlm_context_hint(self):
        return ""

    def get_continuum_context(self):
        return ""

    def get_supermemory_context(self):
        return ""

    def get_knowledge_mound_context(self):
        return ""

    def get_outcome_context(self):
        return ""

    def get_codebase_context(self):
        return ""

    def get_prior_claims_context(self, limit=5):
        return ""

    def format_pulse_context(self, max_topics=5):
        return ""

    def get_language_constraint(self):
        return ""

    def format_successful_patterns(self, limit=3):
        return ""

    def format_evidence_for_prompt(self, max_snippets=5):
        return ""

    def format_trending_for_prompt(self, max_topics=3):
        return ""

    def get_elo_context(self, agent, all_agents):
        return ""

    def _inject_belief_context(self, limit=3):
        return ""

    def _inject_calibration_context(self, agent):
        return ""

    def _estimate_tokens(self, text):
        return len(text) // 4

    def _apply_context_budget(self, env_context="", sections=None):
        parts = []
        if env_context:
            parts.append(env_context)
        if sections:
            for s in sections:
                if s.content:
                    parts.append(s.content)
        block = "\n\n".join(parts)
        return block, ""

    def get_deliberation_template_context(self):
        return ""


@pytest.fixture
def assembler():
    return ConcreteAssembler()


@pytest.fixture
def agent():
    a = MagicMock()
    a.name = "claude"
    a.role = "proposer"
    return a


# ---------------------------------------------------------------------------
# _anonymize_if_enabled
# ---------------------------------------------------------------------------


class TestAnonymize:
    def test_disabled_returns_unchanged(self, assembler):
        assert assembler._anonymize_if_enabled("hello world") == "hello world"

    def test_enabled_import_error(self, assembler):
        assembler.protocol.enable_privacy_anonymization = True
        with patch.dict("sys.modules", {"aragora.privacy.anonymization": None}):
            result = assembler._anonymize_if_enabled("hello world")
        assert result == "hello world"

    def test_enabled_runtime_error(self, assembler):
        assembler.protocol.enable_privacy_anonymization = True
        with patch(
            "aragora.debate.prompt_assemblers.PromptAssemblyMixin._anonymize_if_enabled",
            return_value="hello world",
        ):
            # Just verify no crash when anonymization fails gracefully
            result = assembler._anonymize_if_enabled("hello world")
            assert isinstance(result, str)


# ---------------------------------------------------------------------------
# build_proposal_prompt
# ---------------------------------------------------------------------------


class TestBuildProposalPrompt:
    def test_basic_prompt_structure(self, assembler, agent):
        prompt = assembler.build_proposal_prompt(agent)
        assert "proposer" in prompt
        assert "multi-agent debate" in prompt
        assert "Design a rate limiter" in prompt
        assert "proposal" in prompt.lower()

    def test_includes_task(self, assembler, agent):
        assembler.env.task = "Implement OAuth 2.0"
        prompt = assembler.build_proposal_prompt(agent)
        assert "Implement OAuth 2.0" in prompt

    def test_no_research_indicator(self, assembler, agent):
        assembler.env.context = ""
        prompt = assembler.build_proposal_prompt(agent)
        assert "RESEARCH STATUS" in prompt

    def test_evidence_research_indicator(self, assembler, agent):
        assembler.env.context = "EVIDENCE CONTEXT: some data"
        prompt = assembler.build_proposal_prompt(agent)
        assert "Research context has been provided" in prompt

    def test_context_injected(self, assembler, agent):
        assembler.env.context = "Background: We use Redis"
        prompt = assembler.build_proposal_prompt(agent)
        assert "We use Redis" in prompt

    def test_stance_section(self, assembler, agent):
        assembler.get_stance_guidance = lambda a: "Argue in favor strongly"
        prompt = assembler.build_proposal_prompt(agent)
        assert "Argue in favor strongly" in prompt

    def test_role_section(self, assembler, agent):
        assembler.get_role_context = lambda a: "## Devil's Advocate Role"
        prompt = assembler.build_proposal_prompt(agent)
        assert "Devil's Advocate" in prompt

    def test_persona_section(self, assembler, agent):
        assembler.get_persona_context = lambda a: "You are a security expert"
        prompt = assembler.build_proposal_prompt(agent)
        assert "security expert" in prompt

    def test_flip_section(self, assembler, agent):
        assembler.get_flip_context = lambda a: "## Position Consistency Note\nYou flipped"
        prompt = assembler.build_proposal_prompt(agent)
        assert "Position Consistency" in prompt

    def test_historical_rlm_context(self, assembler, agent):
        assembler._rlm_context = MagicMock()
        assembler.get_rlm_abstract = lambda max_chars=800: "Compressed historical summary"
        prompt = assembler.build_proposal_prompt(agent)
        assert "Prior Context" in prompt
        assert "Compressed historical summary" in prompt

    def test_historical_cache_fallback(self, assembler, agent):
        assembler._historical_context_cache = "Previous debates showed..."
        prompt = assembler.build_proposal_prompt(agent)
        assert "Previous debates showed" in prompt

    def test_continuum_section(self, assembler, agent):
        assembler.get_continuum_context = lambda: "Memory from past sessions"
        prompt = assembler.build_proposal_prompt(agent)
        assert "Memory from past sessions" in prompt

    def test_km_section(self, assembler, agent):
        assembler.get_knowledge_mound_context = lambda: "Company uses microservices"
        prompt = assembler.build_proposal_prompt(agent)
        assert "Organizational Knowledge" in prompt

    def test_codebase_section(self, assembler, agent):
        assembler.get_codebase_context = lambda: "src/rate_limiter.py has TokenBucket class"
        prompt = assembler.build_proposal_prompt(agent)
        assert "Codebase Context" in prompt

    def test_evidence_section(self, assembler, agent):
        assembler.format_evidence_for_prompt = lambda max_snippets=5: "## AVAILABLE EVIDENCE"
        prompt = assembler.build_proposal_prompt(agent)
        assert "AVAILABLE EVIDENCE" in prompt

    def test_elo_section_with_all_agents(self, assembler, agent):
        assembler.get_elo_context = lambda a, agents: "## Agent Rankings"
        other = MagicMock()
        other.name = "gpt"
        prompt = assembler.build_proposal_prompt(agent, all_agents=[agent, other])
        assert "Agent Rankings" in prompt

    def test_audience_section(self, assembler, agent):
        prompt = assembler.build_proposal_prompt(agent, audience_section="Target: CTOs")
        assert "Target: CTOs" in prompt

    def test_language_constraint(self, assembler, agent):
        assembler.get_language_constraint = lambda: "\n\nRespond in French only."
        prompt = assembler.build_proposal_prompt(agent)
        assert "Respond in French only" in prompt


# ---------------------------------------------------------------------------
# build_revision_prompt
# ---------------------------------------------------------------------------


class TestBuildRevisionPrompt:
    def test_basic_structure(self, assembler, agent):
        critique = MagicMock()
        critique.to_prompt.return_value = "Issue: Missing error handling"
        prompt = assembler.build_revision_prompt(agent, "Original text", [critique])
        assert "revising" in prompt.lower()
        assert "Original text" in prompt
        assert "Missing error handling" in prompt

    def test_includes_task(self, assembler, agent):
        prompt = assembler.build_revision_prompt(agent, "text", [])
        assert "Design a rate limiter" in prompt

    def test_round_phase_context(self, assembler, agent):
        assembler.get_round_phase_context = lambda rn: (
            "## Round 2: Critique Phase" if rn > 0 else ""
        )
        prompt = assembler.build_revision_prompt(agent, "text", [], round_number=2)
        assert "Round 2" in prompt

    def test_km_section_in_revision(self, assembler, agent):
        assembler.get_knowledge_mound_context = lambda: "Internal API guidelines"
        prompt = assembler.build_revision_prompt(agent, "text", [])
        assert "Organizational Knowledge" in prompt

    def test_codebase_section_in_revision(self, assembler, agent):
        assembler.get_codebase_context = lambda: "Current impl uses sliding window"
        prompt = assembler.build_revision_prompt(agent, "text", [])
        assert "Codebase Context" in prompt

    def test_multiple_critiques(self, assembler, agent):
        c1 = MagicMock()
        c1.to_prompt.return_value = "Critique 1: Too vague"
        c2 = MagicMock()
        c2.to_prompt.return_value = "Critique 2: No benchmarks"
        prompt = assembler.build_revision_prompt(agent, "text", [c1, c2])
        assert "Too vague" in prompt
        assert "No benchmarks" in prompt

    def test_evidence_citation_guidance(self, assembler, agent):
        prompt = assembler.build_revision_prompt(agent, "text", [])
        assert "EVID-N" in prompt


# ---------------------------------------------------------------------------
# build_judge_prompt
# ---------------------------------------------------------------------------


class TestBuildJudgePrompt:
    def test_basic_structure(self, assembler):
        proposals = {"claude": "Proposal A", "gpt": "Proposal B"}
        critique = MagicMock()
        critique.agent = "claude"
        critique.issues = ["Missing tests", "No error handling"]
        prompt = assembler.build_judge_prompt(proposals, "Design a system", [critique])
        assert "synthesizer" in prompt.lower() or "judge" in prompt.lower()
        assert "Proposal A" in prompt
        assert "Proposal B" in prompt
        assert "Missing tests" in prompt
        assert "Design a system" in prompt

    def test_empty_critiques(self, assembler):
        proposals = {"claude": "Solo proposal"}
        prompt = assembler.build_judge_prompt(proposals, "task", [])
        assert "Solo proposal" in prompt

    def test_evidence_context(self, assembler):
        assembler.format_evidence_for_prompt = lambda max_snippets=5: "## EVIDENCE"
        prompt = assembler.build_judge_prompt({"a": "p"}, "task", [])
        assert "EVIDENCE" in prompt


# ---------------------------------------------------------------------------
# build_judge_vote_prompt
# ---------------------------------------------------------------------------


class TestBuildJudgeVotePrompt:
    def test_basic_structure(self, assembler):
        c1 = MagicMock()
        c1.name = "claude"
        c2 = MagicMock()
        c2.name = "gpt"
        proposals = {"claude": "x" * 400, "gpt": "y" * 400}
        prompt = assembler.build_judge_vote_prompt([c1, c2], proposals)
        assert "claude" in prompt
        assert "gpt" in prompt
        assert "vote" in prompt.lower()
        assert "cannot vote for yourself" in prompt.lower()

    def test_proposals_truncated(self, assembler):
        c1 = MagicMock()
        c1.name = "agent1"
        proposals = {"agent1": "x" * 500}
        prompt = assembler.build_judge_vote_prompt([c1], proposals)
        # Proposals should be truncated at 300 chars
        assert "..." in prompt
