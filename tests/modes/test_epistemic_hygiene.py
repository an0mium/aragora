"""Tests for Epistemic Hygiene Mode and integration with debate infrastructure."""

import pytest

from aragora.modes.base import ModeRegistry
from aragora.modes.builtin.epistemic_hygiene import EpistemicHygieneMode


class TestEpistemicHygieneMode:
    """Tests for the EpistemicHygieneMode class."""

    def setup_method(self):
        """Clear registry before each test."""
        ModeRegistry.clear()

    def test_mode_creation(self):
        """EpistemicHygieneMode can be created with defaults."""
        mode = EpistemicHygieneMode()
        assert mode.name == "epistemic_hygiene"
        assert mode.description

    def test_auto_registration(self):
        """Mode auto-registers in ModeRegistry on instantiation."""
        EpistemicHygieneMode()
        found = ModeRegistry.get("epistemic_hygiene")
        assert found is not None
        assert found.name == "epistemic_hygiene"

    def test_system_prompt_contains_alternatives(self):
        """System prompt requires alternatives section."""
        mode = EpistemicHygieneMode()
        prompt = mode.get_system_prompt()
        assert "alternative" in prompt.lower()
        assert "ALTERNATIVES CONSIDERED" in prompt

    def test_system_prompt_contains_falsifiability(self):
        """System prompt requires falsifiability section."""
        mode = EpistemicHygieneMode()
        prompt = mode.get_system_prompt()
        assert "falsif" in prompt.lower()
        assert "FALSIFIABILITY" in prompt

    def test_system_prompt_contains_confidence(self):
        """System prompt requires confidence levels."""
        mode = EpistemicHygieneMode()
        prompt = mode.get_system_prompt()
        assert "confidence" in prompt.lower()
        assert "CONFIDENCE LEVELS" in prompt

    def test_system_prompt_contains_unknowns(self):
        """System prompt requires explicit unknowns."""
        mode = EpistemicHygieneMode()
        prompt = mode.get_system_prompt()
        assert "unknown" in prompt.lower()
        assert "EXPLICIT UNKNOWNS" in prompt

    def test_system_prompt_mentions_penalties(self):
        """System prompt warns about consensus scoring penalties."""
        mode = EpistemicHygieneMode()
        prompt = mode.get_system_prompt()
        assert "penalt" in prompt.lower() or "reduced weight" in prompt.lower()

    def test_tool_groups_include_debate(self):
        """Mode grants debate tool access."""
        from aragora.modes.tool_groups import ToolGroup

        mode = EpistemicHygieneMode()
        assert ToolGroup.DEBATE in mode.tool_groups

    def test_tool_groups_include_read(self):
        """Mode grants read tool access."""
        from aragora.modes.tool_groups import ToolGroup

        mode = EpistemicHygieneMode()
        assert ToolGroup.READ in mode.tool_groups

    def test_mode_available_via_top_level_import(self):
        """EpistemicHygieneMode is importable from aragora.modes."""
        from aragora.modes import EpistemicHygieneMode as EHM

        assert EHM is EpistemicHygieneMode


class TestProtocolFlagActivation:
    """Tests for protocol flag behavior with epistemic hygiene."""

    def test_protocol_flags_default_off(self):
        """Epistemic hygiene flags default to off."""
        from aragora.debate.protocol import DebateProtocol

        p = DebateProtocol()
        assert p.enable_epistemic_hygiene is False

    def test_protocol_flags_enable_all(self):
        """with_epistemic_hygiene enables all sub-flags."""
        from aragora.debate.protocol import DebateProtocol

        p = DebateProtocol.with_epistemic_hygiene()
        assert p.enable_epistemic_hygiene is True
        assert p.epistemic_require_falsifiers is True
        assert p.epistemic_require_confidence is True
        assert p.epistemic_require_unknowns is True
        assert p.epistemic_min_alternatives >= 1

    def test_protocol_penalty_configurable(self):
        """Penalty value can be customized."""
        from aragora.debate.protocol import DebateProtocol

        p = DebateProtocol.with_epistemic_hygiene(penalty=0.25)
        assert p.epistemic_hygiene_penalty == 0.25

    def test_protocol_min_alternatives_configurable(self):
        """Minimum alternatives can be customized."""
        from aragora.debate.protocol import DebateProtocol

        p = DebateProtocol.with_epistemic_hygiene(min_alternatives=3)
        assert p.epistemic_min_alternatives == 3


class TestPromptInjection:
    """Tests for epistemic hygiene prompt injection."""

    def test_proposal_prompt_when_enabled(self):
        """Proposal prompt is injected when epistemic hygiene is enabled."""
        from aragora.debate.epistemic_hygiene import get_epistemic_proposal_prompt
        from aragora.debate.protocol import DebateProtocol

        protocol = DebateProtocol.with_epistemic_hygiene()
        prompt = get_epistemic_proposal_prompt(protocol)
        assert "EPISTEMIC HYGIENE REQUIREMENTS" in prompt
        assert "ALTERNATIVES CONSIDERED" in prompt
        assert "FALSIFIABILITY" in prompt
        assert "CONFIDENCE LEVELS" in prompt
        assert "EXPLICIT UNKNOWNS" in prompt

    def test_proposal_prompt_when_disabled(self):
        """Proposal prompt is empty when epistemic hygiene is disabled."""
        from aragora.debate.epistemic_hygiene import get_epistemic_proposal_prompt
        from aragora.debate.protocol import DebateProtocol

        protocol = DebateProtocol()
        prompt = get_epistemic_proposal_prompt(protocol)
        assert prompt == ""

    def test_revision_prompt_when_enabled(self):
        """Revision prompt is injected when epistemic hygiene is enabled."""
        from aragora.debate.epistemic_hygiene import get_epistemic_revision_prompt
        from aragora.debate.protocol import DebateProtocol

        protocol = DebateProtocol.with_epistemic_hygiene()
        prompt = get_epistemic_revision_prompt(protocol)
        assert "EPISTEMIC HYGIENE REQUIREMENTS" in prompt

    def test_revision_prompt_when_disabled(self):
        """Revision prompt is empty when epistemic hygiene is disabled."""
        from aragora.debate.epistemic_hygiene import get_epistemic_revision_prompt
        from aragora.debate.protocol import DebateProtocol

        protocol = DebateProtocol()
        prompt = get_epistemic_revision_prompt(protocol)
        assert prompt == ""

    def test_min_alternatives_in_prompt(self):
        """Prompt includes the configured minimum alternatives count."""
        from aragora.debate.epistemic_hygiene import get_epistemic_proposal_prompt
        from aragora.debate.protocol import DebateProtocol

        protocol = DebateProtocol.with_epistemic_hygiene(min_alternatives=3)
        prompt = get_epistemic_proposal_prompt(protocol)
        assert "3 alternative" in prompt


class TestComplianceDetection:
    """Tests for epistemic compliance detection in agent responses."""

    def test_detects_alternatives(self):
        """Score detects alternatives in response text."""
        from aragora.debate.epistemic_hygiene import score_response

        text = """
        ## ALTERNATIVES CONSIDERED
        Alternative approach: Use a queue-based system. Rejected because it adds latency.
        """
        score = score_response(text, agent="test")
        assert score.has_alternatives is True

    def test_detects_falsifiers(self):
        """Score detects falsifiers in response text."""
        from aragora.debate.epistemic_hygiene import score_response

        text = """
        ## FALSIFIABILITY
        Claim: The cache improves latency. Falsified if: p99 latency increases after deployment.
        """
        score = score_response(text, agent="test")
        assert score.has_falsifiers is True

    def test_detects_confidence(self):
        """Score detects confidence levels in response text."""
        from aragora.debate.epistemic_hygiene import score_response

        text = """
        ## CONFIDENCE LEVELS
        Claim: The system can handle 10k RPS. Confidence: 0.85
        """
        score = score_response(text, agent="test")
        assert score.has_confidence is True

    def test_detects_unknowns(self):
        """Score detects explicit unknowns in response text."""
        from aragora.debate.epistemic_hygiene import score_response

        text = """
        ## EXPLICIT UNKNOWNS
        I do not know the exact memory requirements for the proposed solution.
        The impact on cold-start performance is unclear.
        """
        score = score_response(text, agent="test")
        assert score.has_unknowns is True

    def test_full_compliance_score(self):
        """Fully compliant response gets score 1.0."""
        from aragora.debate.epistemic_hygiene import score_response

        text = """
        ## ALTERNATIVES CONSIDERED
        Alternative approach: microservices. Rejected because of operational overhead.

        ## FALSIFIABILITY
        Claim: Monolith is simpler. Falsified if: deployment frequency drops below 2x/week.

        ## CONFIDENCE LEVELS
        Confidence: 0.9 for the core architectural claim.

        ## EXPLICIT UNKNOWNS
        Uncertain about long-term scaling beyond 100k users.
        """
        score = score_response(text, agent="test")
        assert score.score == 1.0
        assert score.missing == []

    def test_missing_all_sections(self):
        """Response missing all sections gets score 0.0."""
        from aragora.debate.epistemic_hygiene import score_response

        text = "I think we should use a monolith architecture because it is simpler."
        score = score_response(text, agent="test")
        assert score.score == 0.0
        assert len(score.missing) == 4

    def test_partial_compliance(self):
        """Partial compliance gets fractional score."""
        from aragora.debate.epistemic_hygiene import score_response

        text = """
        Alternative approach: Use a different database. We could also use MongoDB.
        I do not know what the memory requirements are.
        """
        score = score_response(text, agent="test")
        assert 0.0 < score.score < 1.0
        assert score.has_alternatives is True
        assert score.has_unknowns is True

    def test_natural_language_falsifier_detection(self):
        """Detects natural language falsifier patterns."""
        from aragora.debate.epistemic_hygiene import score_response

        text = "This would be wrong if the data shows a negative trend."
        score = score_response(text)
        assert score.has_falsifiers is True

    def test_score_to_dict(self):
        """EpistemicScore serializes to dictionary."""
        from aragora.debate.epistemic_hygiene import score_response

        score = score_response("test text", agent="claude", round_number=3)
        d = score.to_dict()
        assert d["agent"] == "claude"
        assert d["round_number"] == 3
        assert "score" in d
        assert "missing" in d


class TestConsensusPenalty:
    """Tests for epistemic hygiene consensus penalty calculation."""

    def test_no_penalty_when_disabled(self):
        """No penalty when epistemic hygiene is disabled."""
        from aragora.debate.epistemic_hygiene import compute_epistemic_penalty, score_response
        from aragora.debate.protocol import DebateProtocol

        protocol = DebateProtocol()
        score = score_response("no sections at all")
        penalty = compute_epistemic_penalty(score, protocol)
        assert penalty == 0.0

    def test_full_penalty_when_all_missing(self):
        """Full penalty when all required sections are missing."""
        from aragora.debate.epistemic_hygiene import compute_epistemic_penalty, score_response
        from aragora.debate.protocol import DebateProtocol

        protocol = DebateProtocol.with_epistemic_hygiene()
        score = score_response("just a plain response with nothing special")
        penalty = compute_epistemic_penalty(score, protocol)
        assert penalty == pytest.approx(protocol.epistemic_hygiene_penalty)

    def test_zero_penalty_when_fully_compliant(self):
        """Zero penalty when all sections are present."""
        from aragora.debate.epistemic_hygiene import compute_epistemic_penalty, score_response
        from aragora.debate.protocol import DebateProtocol

        protocol = DebateProtocol.with_epistemic_hygiene()
        text = """
        Alternative approach: X. Rejected because of Y.
        Falsified if: Z happens.
        Confidence: 0.8
        Uncertain about W.
        """
        score = score_response(text)
        penalty = compute_epistemic_penalty(score, protocol)
        assert penalty == 0.0

    def test_partial_penalty(self):
        """Partial compliance gets proportional penalty."""
        from aragora.debate.epistemic_hygiene import compute_epistemic_penalty, score_response
        from aragora.debate.protocol import DebateProtocol

        protocol = DebateProtocol.with_epistemic_hygiene(penalty=0.20)
        # Only has alternatives and unknowns (2 of 4 dimensions)
        text = """
        Alternative approach: Use Redis. Rejected because of cost.
        I do not know the scaling limits.
        """
        score = score_response(text)
        penalty = compute_epistemic_penalty(score, protocol)
        # 2 of 4 missing → 50% of 0.20 = 0.10
        assert penalty == pytest.approx(0.10)


class TestEpistemicHygieneTracker:
    """Tests for per-debate epistemic hygiene tracking."""

    def test_record_and_retrieve_scores(self):
        """Tracker records and retrieves scores."""
        from aragora.debate.epistemic_hygiene import EpistemicHygieneTracker, score_response

        tracker = EpistemicHygieneTracker(debate_id="test-123")
        score = score_response("Alternative approach: X", agent="claude", round_number=1)
        tracker.record(score)

        assert len(tracker.scores) == 1
        agent_scores = tracker.get_agent_scores("claude")
        assert len(agent_scores) == 1

    def test_agent_average(self):
        """Tracker computes correct agent average."""
        from aragora.debate.epistemic_hygiene import EpistemicHygieneTracker, EpistemicScore

        tracker = EpistemicHygieneTracker(debate_id="test")
        tracker.record(
            EpistemicScore(
                has_alternatives=True,
                has_falsifiers=True,
                has_confidence=True,
                has_unknowns=True,
                agent="a",
                round_number=1,
            )
        )
        tracker.record(
            EpistemicScore(
                has_alternatives=False,
                has_falsifiers=False,
                has_confidence=False,
                has_unknowns=False,
                agent="a",
                round_number=2,
            )
        )
        assert tracker.get_agent_average("a") == pytest.approx(0.5)

    def test_debate_average(self):
        """Tracker computes correct debate average."""
        from aragora.debate.epistemic_hygiene import EpistemicHygieneTracker, EpistemicScore

        tracker = EpistemicHygieneTracker(debate_id="test")
        tracker.record(
            EpistemicScore(
                has_alternatives=True,
                has_falsifiers=True,
                has_confidence=True,
                has_unknowns=True,
                agent="a",
            )
        )
        tracker.record(
            EpistemicScore(
                has_alternatives=True,
                has_falsifiers=False,
                has_confidence=False,
                has_unknowns=False,
                agent="b",
            )
        )
        avg = tracker.get_debate_average()
        assert avg == pytest.approx(0.625)  # (1.0 + 0.25) / 2

    def test_summary(self):
        """Tracker generates summary statistics."""
        from aragora.debate.epistemic_hygiene import EpistemicHygieneTracker, EpistemicScore

        tracker = EpistemicHygieneTracker(debate_id="sum-test")
        tracker.record(
            EpistemicScore(
                has_alternatives=True,
                has_falsifiers=True,
                has_confidence=True,
                has_unknowns=True,
                agent="claude",
                round_number=1,
            )
        )
        summary = tracker.summary()
        assert summary["debate_id"] == "sum-test"
        assert summary["total_scores"] == 1
        assert "claude" in summary["agents"]
        assert summary["agents"]["claude"]["fully_compliant"] == 1


class TestDebateControllerIntegration:
    """Tests for debate controller epistemic hygiene integration."""

    def test_normalize_mode_epistemic(self):
        """Mode normalization recognizes epistemic hygiene variants."""
        from aragora.server.debate_controller import _normalize_mode

        assert _normalize_mode("epistemic_hygiene") == "epistemic_hygiene"
        assert _normalize_mode("epistemic") == "epistemic_hygiene"
        assert _normalize_mode("hygiene") == "epistemic_hygiene"
        assert _normalize_mode("Epistemic-Hygiene") == "epistemic_hygiene"

    def test_append_epistemic_prompt(self):
        """Epistemic prompt is appended to context."""
        from aragora.server.debate_controller import _append_epistemic_hygiene_prompt

        result = _append_epistemic_hygiene_prompt("Existing context")
        assert "Existing context" in result
        assert "Epistemic hygiene protocol" in result

    def test_append_epistemic_prompt_no_duplicate(self):
        """Epistemic prompt is not duplicated if already present."""
        from aragora.server.debate_controller import (
            _EPISTEMIC_HYGIENE_PROMPT,
            _append_epistemic_hygiene_prompt,
        )

        result = _append_epistemic_hygiene_prompt(_EPISTEMIC_HYGIENE_PROMPT)
        assert result.count("Epistemic hygiene protocol") == 1

    def test_ensure_metadata_sets_mode(self):
        """Metadata helper sets mode and settlement scaffolding."""
        from aragora.server.debate_controller import _ensure_epistemic_hygiene_metadata

        metadata: dict = {}
        _ensure_epistemic_hygiene_metadata(metadata)
        assert metadata["mode"] == "epistemic_hygiene"
        assert metadata["epistemic_hygiene"] is True
        assert "settlement" in metadata
        assert metadata["settlement"]["status"] == "needs_definition"

    def test_debate_request_from_dict_with_mode(self):
        """DebateRequest.from_dict correctly parses epistemic_hygiene mode."""
        from aragora.server.debate_controller import DebateRequest

        data = {
            "question": "Should we use microservices?",
            "mode": "epistemic_hygiene",
        }
        req = DebateRequest.from_dict(data)
        assert req.mode == "epistemic_hygiene"
        assert req.metadata["epistemic_hygiene"] is True

    def test_debate_request_from_dict_with_flag(self):
        """DebateRequest.from_dict recognizes epistemic_hygiene boolean flag."""
        from aragora.server.debate_controller import DebateRequest

        data = {
            "question": "Should we use microservices?",
            "epistemic_hygiene": True,
        }
        req = DebateRequest.from_dict(data)
        assert req.mode == "epistemic_hygiene"
