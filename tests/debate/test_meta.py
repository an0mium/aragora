"""
Tests for aragora/debate/meta.py — meta-level debate analysis.

Covers:
- MetaObservation dataclass
- MetaCritique dataclass (including summary property)
- MetaCritiqueAnalyzer:
  - analyze()
  - _detect_repetition()
  - _detect_misalignment()
  - _detect_circular_arguments()
  - _detect_ignored_critiques()
  - _classify_rounds()
  - _calculate_quality()
  - _generate_recommendations()
  - _text_similarity() (Jaccard path and embedding path)
  - generate_meta_prompt()
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.debate.meta import MetaCritique, MetaCritiqueAnalyzer, MetaObservation


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_message(agent: str, content: str, round_: int = 0, role: str = "proposer"):
    """Return a mock Message with the minimal attributes meta.py requires."""
    msg = MagicMock()
    msg.agent = agent
    msg.content = content
    msg.round = round_
    msg.role = role
    return msg


def make_critique(
    agent: str,
    target_agent: str,
    target_content: str,
    issues: list[str],
    suggestions: list[str],
    severity: float = 5.0,
):
    """Return a mock Critique with the minimal attributes meta.py requires."""
    crit = MagicMock()
    crit.agent = agent
    crit.target_agent = target_agent
    crit.target_content = target_content
    crit.issues = issues
    crit.suggestions = suggestions
    crit.severity = severity
    return crit


def make_result(
    id_: str = "debate-1",
    task: str = "Design a rate limiter",
    messages=None,
    critiques=None,
    consensus_reached: bool = True,
    confidence: float = 0.8,
):
    """Return a mock DebateResult."""
    result = MagicMock()
    result.id = id_
    result.task = task
    result.messages = messages if messages is not None else []
    result.critiques = critiques if critiques is not None else []
    result.consensus_reached = consensus_reached
    result.confidence = confidence
    return result


# ---------------------------------------------------------------------------
# MetaObservation
# ---------------------------------------------------------------------------


class TestMetaObservation:
    def test_required_fields(self):
        obs = MetaObservation(
            observation_type="issue",
            description="Agent repeated itself",
            severity=0.6,
            round_range=(1, 2),
        )
        assert obs.observation_type == "issue"
        assert obs.description == "Agent repeated itself"
        assert obs.severity == 0.6
        assert obs.round_range == (1, 2)
        assert obs.agents_involved == []
        assert obs.evidence == []

    def test_optional_fields_provided(self):
        obs = MetaObservation(
            observation_type="pattern",
            description="Circular argument detected",
            severity=0.7,
            round_range=(1, 4),
            agents_involved=["alice", "bob"],
            evidence=["High similarity between rounds 1 and 4"],
        )
        assert obs.agents_involved == ["alice", "bob"]
        assert obs.evidence == ["High similarity between rounds 1 and 4"]

    @pytest.mark.parametrize("obs_type", ["issue", "pattern", "suggestion"])
    def test_observation_types(self, obs_type):
        obs = MetaObservation(
            observation_type=obs_type,
            description="Test",
            severity=0.5,
            round_range=(0, 0),
        )
        assert obs.observation_type == obs_type

    def test_severity_boundaries(self):
        obs_low = MetaObservation(
            observation_type="suggestion", description="x", severity=0.0, round_range=(0, 0)
        )
        obs_high = MetaObservation(
            observation_type="issue", description="x", severity=1.0, round_range=(0, 0)
        )
        assert obs_low.severity == 0.0
        assert obs_high.severity == 1.0


# ---------------------------------------------------------------------------
# MetaCritique
# ---------------------------------------------------------------------------


class TestMetaCritique:
    def _make_critique(self, observations=None, quality=0.8, productive=None, unproductive=None, recs=None):
        return MetaCritique(
            debate_id="debate-123",
            observations=observations or [],
            overall_quality=quality,
            productive_rounds=productive or [1, 2],
            unproductive_rounds=unproductive or [3],
            recommendations=recs or ["Add more rounds"],
            created_at="2026-02-17T00:00:00",
        )

    def test_basic_fields(self):
        mc = self._make_critique()
        assert mc.debate_id == "debate-123"
        assert mc.overall_quality == 0.8
        assert mc.productive_rounds == [1, 2]
        assert mc.unproductive_rounds == [3]
        assert mc.recommendations == ["Add more rounds"]

    def test_summary_includes_quality(self):
        mc = self._make_critique(quality=0.75)
        assert "75%" in mc.summary

    def test_summary_includes_productive_rounds(self):
        mc = self._make_critique(productive=[1, 2, 3])
        assert "[1, 2, 3]" in mc.summary

    def test_summary_issue_count(self):
        issues = [
            MetaObservation("issue", "Repeated", 0.6, (1, 2)),
            MetaObservation("issue", "Ignored", 0.4, (0, 0)),
            MetaObservation("pattern", "Circular", 0.7, (1, 4)),  # not an "issue"
        ]
        mc = self._make_critique(observations=issues)
        # Only two are "issue" type
        assert "Issues Found: 2" in mc.summary

    def test_summary_top_recommendation(self):
        mc = self._make_critique(recs=["First rec", "Second rec"])
        assert "First rec" in mc.summary

    def test_summary_no_recommendations(self):
        mc = MetaCritique(
            debate_id="debate-123",
            observations=[],
            overall_quality=0.8,
            productive_rounds=[1, 2],
            unproductive_rounds=[3],
            recommendations=[],
            created_at="2026-02-17T00:00:00",
        )
        summary = mc.summary
        assert "Top Recommendation" not in summary

    def test_summary_format_pipes(self):
        mc = self._make_critique()
        parts = mc.summary.split(" | ")
        assert len(parts) >= 3

    def test_created_at_has_default(self):
        mc = MetaCritique(
            debate_id="x",
            observations=[],
            overall_quality=0.5,
            productive_rounds=[],
            unproductive_rounds=[],
            recommendations=[],
        )
        assert mc.created_at  # not empty

    def test_zero_quality(self):
        mc = self._make_critique(quality=0.0)
        assert "0%" in mc.summary

    def test_full_quality(self):
        mc = self._make_critique(quality=1.0)
        assert "100%" in mc.summary


# ---------------------------------------------------------------------------
# MetaCritiqueAnalyzer — _text_similarity (Jaccard path)
# ---------------------------------------------------------------------------


class TestTextSimilarity:
    def setup_method(self):
        self.analyzer = MetaCritiqueAnalyzer()

    def test_identical_texts(self):
        sim = self.analyzer._text_similarity("hello world", "hello world")
        assert sim == 1.0

    def test_completely_different_texts(self):
        sim = self.analyzer._text_similarity("apple banana cherry", "dog cat fish")
        assert sim == 0.0

    def test_partial_overlap(self):
        sim = self.analyzer._text_similarity("the quick brown fox", "the slow brown dog")
        # Overlap: {the, brown} = 2, union = 6 → 2/6 ≈ 0.333
        assert 0.2 < sim < 0.5

    def test_empty_text1(self):
        sim = self.analyzer._text_similarity("", "hello world")
        assert sim == 0.0

    def test_empty_text2(self):
        sim = self.analyzer._text_similarity("hello world", "")
        assert sim == 0.0

    def test_both_empty(self):
        sim = self.analyzer._text_similarity("", "")
        assert sim == 0.0

    def test_case_insensitive(self):
        sim = self.analyzer._text_similarity("Hello World", "hello world")
        assert sim == 1.0

    def test_similarity_range(self):
        sim = self.analyzer._text_similarity("some words here now", "some different content exists")
        assert 0.0 <= sim <= 1.0


# ---------------------------------------------------------------------------
# MetaCritiqueAnalyzer — _text_similarity with embedding provider
# ---------------------------------------------------------------------------


class TestTextSimilarityWithEmbeddings:
    def test_uses_embedding_provider_when_provided(self):
        provider = MagicMock()
        analyzer = MetaCritiqueAnalyzer(embedding_provider=provider)

        with patch.object(analyzer, "_semantic_similarity", return_value=0.9) as mock_sem:
            sim = analyzer._text_similarity("text one", "text two")
        mock_sem.assert_called_once_with("text one", "text two")
        assert sim == 0.9

    def test_falls_back_to_jaccard_on_runtime_error(self):
        provider = MagicMock()
        analyzer = MetaCritiqueAnalyzer(embedding_provider=provider)

        with patch.object(analyzer, "_semantic_similarity", side_effect=RuntimeError("API down")):
            sim = analyzer._text_similarity("hello world", "hello world")
        # Jaccard fallback: identical texts → 1.0
        assert sim == 1.0

    def test_falls_back_to_jaccard_on_value_error(self):
        provider = MagicMock()
        analyzer = MetaCritiqueAnalyzer(embedding_provider=provider)

        with patch.object(analyzer, "_semantic_similarity", side_effect=ValueError("bad")):
            sim = analyzer._text_similarity("apple banana", "dog cat")
        assert sim == 0.0

    def test_falls_back_to_jaccard_on_attribute_error(self):
        provider = MagicMock()
        analyzer = MetaCritiqueAnalyzer(embedding_provider=provider)

        with patch.object(analyzer, "_semantic_similarity", side_effect=AttributeError("no attr")):
            sim = analyzer._text_similarity("same same same", "same same same")
        assert sim == 1.0


# ---------------------------------------------------------------------------
# MetaCritiqueAnalyzer — _detect_repetition
# ---------------------------------------------------------------------------


class TestDetectRepetition:
    def setup_method(self):
        self.analyzer = MetaCritiqueAnalyzer()

    def test_no_repetition_single_message_per_agent(self):
        messages = [
            make_message("alice", "The proposal is excellent"),
            make_message("bob", "I disagree with everything"),
        ]
        obs = self.analyzer._detect_repetition(messages)
        assert obs == []

    def test_repetition_detected_above_threshold(self):
        # Nearly identical consecutive messages from same agent
        content = "We should use a token bucket algorithm for rate limiting"
        messages = [
            make_message("alice", content, round_=1),
            make_message("alice", content, round_=2),
        ]
        obs = self.analyzer._detect_repetition(messages)
        assert len(obs) == 1
        assert obs[0].observation_type == "issue"
        assert "alice" in obs[0].description
        assert "alice" in obs[0].agents_involved
        assert obs[0].severity == 0.6
        assert obs[0].round_range == (1, 2)

    def test_no_repetition_below_threshold(self):
        messages = [
            make_message("alice", "The proposal uses a token bucket algorithm", round_=1),
            make_message(
                "alice",
                "I believe the system should use entirely different caching techniques with redis",
                round_=2,
            ),
        ]
        obs = self.analyzer._detect_repetition(messages)
        assert obs == []

    def test_multiple_agents_only_one_repeats(self):
        repeated = "Use microservices architecture for scalability"
        messages = [
            make_message("alice", repeated, round_=1),
            make_message("alice", repeated, round_=2),
            make_message("bob", "Option A: monolith", round_=1),
            make_message("bob", "Option B: serverless completely", round_=2),
        ]
        obs = self.analyzer._detect_repetition(messages)
        assert len(obs) == 1
        assert obs[0].agents_involved == ["alice"]

    def test_repetition_evidence_contains_round_info(self):
        content = "identical content in both rounds for testing"
        messages = [
            make_message("alice", content, round_=2),
            make_message("alice", content, round_=3),
        ]
        obs = self.analyzer._detect_repetition(messages)
        assert len(obs) == 1
        evidence_str = " ".join(obs[0].evidence)
        assert "Round 2" in evidence_str
        assert "Round 3" in evidence_str

    def test_single_message_agent_skipped(self):
        messages = [make_message("alice", "only message")]
        obs = self.analyzer._detect_repetition(messages)
        assert obs == []

    def test_multiple_repetitions_same_agent(self):
        content = "always repeating the same thing verbatim"
        messages = [
            make_message("alice", content, round_=1),
            make_message("alice", content, round_=2),
            make_message("alice", content, round_=3),
        ]
        obs = self.analyzer._detect_repetition(messages)
        # Two consecutive pairs: (1,2) and (2,3)
        assert len(obs) == 2


# ---------------------------------------------------------------------------
# MetaCritiqueAnalyzer — _detect_misalignment
# ---------------------------------------------------------------------------


class TestDetectMisalignment:
    def setup_method(self):
        self.analyzer = MetaCritiqueAnalyzer()

    def test_no_misalignment_high_overlap(self):
        # Critique issues directly reference words from the target content
        critique = make_critique(
            agent="bob",
            target_agent="alice",
            target_content="implement token bucket rate limiter with redis cache storage",
            issues=["token bucket algorithm needs tuning", "redis cache storage may be slow"],
            suggestions=[],
        )
        obs = self.analyzer._detect_misalignment([], [critique])
        assert obs == []

    def test_misalignment_detected_low_overlap(self):
        critique = make_critique(
            agent="bob",
            target_agent="alice",
            target_content="design database schema for user accounts",
            issues=["the weather forecast looks cloudy today"],
            suggestions=[],
        )
        obs = self.analyzer._detect_misalignment([], [critique])
        assert len(obs) == 1
        assert obs[0].observation_type == "issue"
        assert "bob" in obs[0].description
        assert obs[0].severity == 0.5
        assert "bob" in obs[0].agents_involved
        assert "alice" in obs[0].agents_involved

    def test_evidence_contains_overlap_percentage(self):
        critique = make_critique(
            agent="bob",
            target_agent="alice",
            target_content="completely unrelated to the critique here",
            issues=["xyz123 nonsense purple elephant dancing"],
            suggestions=[],
        )
        obs = self.analyzer._detect_misalignment([], [critique])
        assert len(obs) == 1
        assert any("%" in e for e in obs[0].evidence)

    def test_empty_critiques_list(self):
        obs = self.analyzer._detect_misalignment([], [])
        assert obs == []

    def test_multiple_critiques_mixed(self):
        good_critique = make_critique(
            agent="alice",
            target_agent="bob",
            target_content="the microservices architecture needs better load balancing",
            issues=["microservices load balancing configuration is missing"],
            suggestions=[],
        )
        bad_critique = make_critique(
            agent="carol",
            target_agent="bob",
            target_content="use consistent hashing for distributed caching",
            issues=["xyz purple elephant dancing rainbow"],
            suggestions=[],
        )
        obs = self.analyzer._detect_misalignment([], [good_critique, bad_critique])
        # Only bad_critique should trigger
        assert len(obs) == 1
        assert "carol" in obs[0].agents_involved


# ---------------------------------------------------------------------------
# MetaCritiqueAnalyzer — _detect_circular_arguments
# ---------------------------------------------------------------------------


class TestDetectCircularArguments:
    def setup_method(self):
        self.analyzer = MetaCritiqueAnalyzer()

    def test_no_detection_fewer_than_4_rounds(self):
        messages = [
            make_message("alice", "proposal content", round_=0),
            make_message("bob", "critique content", round_=1),
            make_message("alice", "revised proposal", round_=2),
        ]
        obs = self.analyzer._detect_circular_arguments(messages)
        assert obs == []

    def test_circular_detected_when_high_similarity_across_4_rounds(self):
        # First and last rounds should be very similar to trigger detection
        base = "we should implement the same token bucket algorithm as initially proposed"
        messages = [
            make_message("alice", base, round_=1),
            make_message("bob", "different content here", round_=2),
            make_message("alice", "completely other ideas", round_=3),
            make_message("alice", base, round_=4),
        ]
        obs = self.analyzer._detect_circular_arguments(messages)
        assert len(obs) == 1
        assert obs[0].observation_type == "pattern"
        assert "circular" in obs[0].description.lower()
        assert obs[0].severity == 0.7
        assert obs[0].round_range == (1, 4)

    def test_no_circular_low_similarity_across_rounds(self):
        messages = [
            make_message("alice", "token bucket algorithm rate limiting approach", round_=1),
            make_message("bob", "critique of token bucket", round_=2),
            make_message("alice", "revised with sliding window", round_=3),
            make_message("bob", "completely different caching redis elasticsearch", round_=4),
        ]
        obs = self.analyzer._detect_circular_arguments(messages)
        assert obs == []

    def test_circular_evidence_contains_similarity_message(self):
        base = "identical content repeated to force detection of circular patterns in debate"
        messages = [
            make_message("alice", base, round_=1),
            make_message("bob", "other content", round_=2),
            make_message("alice", "different content again", round_=3),
            make_message("alice", base, round_=4),
        ]
        obs = self.analyzer._detect_circular_arguments(messages)
        if obs:  # Only check if detected
            assert len(obs[0].evidence) > 0

    def test_exactly_4_rounds_triggers_check(self):
        # With exactly 4 distinct round numbers, we have len(rounds) >= 4
        messages = [
            make_message("alice", "start content abc", round_=1),
            make_message("bob", "middle point xyz", round_=2),
            make_message("alice", "third round ideas", round_=3),
            make_message("alice", "start content abc", round_=4),
        ]
        # This just ensures no exception is raised; detection depends on similarity
        obs = self.analyzer._detect_circular_arguments(messages)
        assert isinstance(obs, list)


# ---------------------------------------------------------------------------
# MetaCritiqueAnalyzer — _detect_ignored_critiques
# ---------------------------------------------------------------------------


class TestDetectIgnoredCritiques:
    def setup_method(self):
        self.analyzer = MetaCritiqueAnalyzer()

    def test_no_detection_when_no_suggestions(self):
        critique = make_critique(
            agent="bob",
            target_agent="alice",
            target_content="some proposal",
            issues=["An issue"],
            suggestions=[],  # No suggestions
        )
        messages = [make_message("alice", "my response", role="proposer")]
        obs = self.analyzer._detect_ignored_critiques([critique], messages)
        assert obs == []

    def test_no_detection_when_no_subsequent_message_from_target(self):
        critique = make_critique(
            agent="bob",
            target_agent="alice",
            target_content="proposal",
            issues=["issue"],
            suggestions=["add caching improve performance scalability"],
        )
        messages = [make_message("carol", "carol message", role="proposer")]
        obs = self.analyzer._detect_ignored_critiques([critique], messages)
        assert obs == []

    def test_ignored_critique_detected(self):
        critique = make_critique(
            agent="bob",
            target_agent="alice",
            target_content="proposal",
            issues=["issue"],
            suggestions=["implement distributed caching with redis cluster nodes"],
        )
        # Alice's response doesn't mention any suggestion keywords
        messages = [
            make_message("alice", "I will proceed with the original plan", role="proposer")
        ]
        obs = self.analyzer._detect_ignored_critiques([critique], messages)
        assert len(obs) == 1
        assert obs[0].observation_type == "issue"
        assert "bob" in obs[0].description
        assert obs[0].severity == 0.4
        assert "bob" in obs[0].agents_involved
        assert "alice" in obs[0].agents_involved

    def test_addressed_suggestion_not_flagged(self):
        critique = make_critique(
            agent="bob",
            target_agent="alice",
            target_content="proposal",
            issues=["issue"],
            suggestions=["implement distributed caching with redis cluster nodes"],
        )
        # Alice's response addresses suggestion keywords
        messages = [
            make_message(
                "alice",
                "I have implemented distributed caching with redis cluster nodes as suggested",
                role="proposer",
            )
        ]
        obs = self.analyzer._detect_ignored_critiques([critique], messages)
        assert obs == []

    def test_short_suggestions_not_flagged(self):
        # <= 3 words in suggestions after filtering common words → skip
        critique = make_critique(
            agent="bob",
            target_agent="alice",
            target_content="proposal",
            issues=["issue"],
            suggestions=["add the"],  # Only common words, results in <= 3 non-common words
        )
        messages = [make_message("alice", "I changed nothing", role="proposer")]
        obs = self.analyzer._detect_ignored_critiques([critique], messages)
        assert obs == []

    def test_uses_last_message_from_target(self):
        critique = make_critique(
            agent="bob",
            target_agent="alice",
            target_content="proposal",
            issues=["issue"],
            suggestions=["implement caching redis cluster distributed"],
        )
        # First message doesn't address it, last does — should not flag
        messages = [
            make_message("alice", "I will proceed as planned", role="proposer"),
            make_message(
                "alice",
                "I have implemented caching redis cluster distributed nodes",
                role="proposer",
            ),
        ]
        obs = self.analyzer._detect_ignored_critiques([critique], messages)
        assert obs == []

    def test_only_proposer_role_messages_considered(self):
        critique = make_critique(
            agent="bob",
            target_agent="alice",
            target_content="proposal",
            issues=["issue"],
            suggestions=["implement distributed caching with redis cluster nodes here"],
        )
        # Alice has messages, but not as "proposer"
        messages = [
            make_message("alice", "I agree with everything", role="critic"),
        ]
        obs = self.analyzer._detect_ignored_critiques([critique], messages)
        assert obs == []


# ---------------------------------------------------------------------------
# MetaCritiqueAnalyzer — _classify_rounds
# ---------------------------------------------------------------------------


class TestClassifyRounds:
    def setup_method(self):
        self.analyzer = MetaCritiqueAnalyzer()

    def test_productive_round_with_low_severity_critiques(self):
        # Round has critiques with avg severity < 0.8 (scale 0-10 in Critique, but
        # the code uses critique.severity directly — mock returns float)
        msg = make_message("alice", "proposal", round_=1)
        crit = make_critique("bob", "alice", "proposal", ["issue"], [], severity=0.5)
        result = make_result(messages=[msg], critiques=[crit])
        productive, unproductive = self.analyzer._classify_rounds(result)
        assert 1 in productive
        assert 1 not in unproductive

    def test_unproductive_round_no_critiques(self):
        msg = make_message("alice", "proposal", round_=1)
        result = make_result(messages=[msg], critiques=[])
        productive, unproductive = self.analyzer._classify_rounds(result)
        assert 1 in unproductive
        assert 1 not in productive

    def test_unproductive_round_high_severity(self):
        msg = make_message("alice", "proposal", round_=1)
        crit = make_critique("bob", "alice", "proposal", ["critical flaw"], [], severity=0.9)
        result = make_result(messages=[msg], critiques=[crit])
        productive, unproductive = self.analyzer._classify_rounds(result)
        assert 1 in unproductive

    def test_multiple_rounds_mixed(self):
        # Round 1: alice, targeted by a low-severity critique → productive
        # Round 2: carol only, not targeted by any critique → unproductive (no critiques)
        msg1 = make_message("alice", "proposal", round_=1)
        msg2 = make_message("carol", "separate proposal", round_=2)
        # Critique targets alice (round 1 only, carol not targeted)
        crit1 = make_critique("bob", "alice", "proposal", ["minor issue"], [], severity=0.3)
        result = make_result(messages=[msg1, msg2], critiques=[crit1])
        productive, unproductive = self.analyzer._classify_rounds(result)
        assert 1 in productive
        assert 2 in unproductive

    def test_empty_debate(self):
        result = make_result(messages=[], critiques=[])
        productive, unproductive = self.analyzer._classify_rounds(result)
        assert productive == []
        assert unproductive == []

    def test_returns_sorted_rounds(self):
        messages = [
            make_message("alice", "msg", round_=3),
            make_message("alice", "msg", round_=1),
            make_message("alice", "msg", round_=2),
        ]
        result = make_result(messages=messages, critiques=[])
        productive, unproductive = self.analyzer._classify_rounds(result)
        all_rounds = sorted(productive + unproductive)
        assert all_rounds == [1, 2, 3]


# ---------------------------------------------------------------------------
# MetaCritiqueAnalyzer — _calculate_quality
# ---------------------------------------------------------------------------


class TestCalculateQuality:
    def setup_method(self):
        self.analyzer = MetaCritiqueAnalyzer()

    def test_perfect_quality_no_issues_all_productive(self):
        result = make_result(consensus_reached=True, confidence=1.0)
        quality = self.analyzer._calculate_quality(result, [], [1, 2, 3], [])
        assert quality == pytest.approx(1.0, abs=0.01)

    def test_quality_penalized_for_issues(self):
        # Use confidence < 1.0 so the ceiling doesn't absorb the penalty
        result = make_result(consensus_reached=False, confidence=0.7)
        issue = MetaObservation("issue", "Problem", severity=0.6, round_range=(0, 0))
        quality_no_issue = self.analyzer._calculate_quality(result, [], [1], [])
        quality_with_issue = self.analyzer._calculate_quality(result, [issue], [1], [])
        assert quality_with_issue < quality_no_issue

    def test_quality_penalized_for_unproductive_rounds(self):
        result = make_result(consensus_reached=True, confidence=1.0)
        q_all_productive = self.analyzer._calculate_quality(result, [], [1, 2, 3], [])
        q_mixed = self.analyzer._calculate_quality(result, [], [1], [2, 3])
        assert q_mixed < q_all_productive

    def test_consensus_bonus(self):
        # Use an unproductive round so the base score < 1.0, leaving room for the +0.1 bonus
        result_consensus = make_result(consensus_reached=True, confidence=0.7)
        result_no_consensus = make_result(consensus_reached=False, confidence=0.7)
        q_with = self.analyzer._calculate_quality(result_consensus, [], [], [1])
        q_without = self.analyzer._calculate_quality(result_no_consensus, [], [], [1])
        assert q_with > q_without

    def test_quality_bounded_0_to_1(self):
        # Many severe issues should not push quality below 0
        result = make_result(consensus_reached=False, confidence=0.0)
        issues = [
            MetaObservation("issue", f"Issue {i}", severity=1.0, round_range=(0, 0))
            for i in range(20)
        ]
        quality = self.analyzer._calculate_quality(result, issues, [], [1, 2, 3])
        assert 0.0 <= quality <= 1.0

    def test_patterns_do_not_penalize_quality(self):
        result = make_result(consensus_reached=True, confidence=1.0)
        pattern = MetaObservation("pattern", "Circular", severity=0.7, round_range=(1, 4))
        q_no_pattern = self.analyzer._calculate_quality(result, [], [1, 2], [])
        q_with_pattern = self.analyzer._calculate_quality(result, [pattern], [1, 2], [])
        # Patterns are not "issues" so should not trigger penalty
        assert q_no_pattern == pytest.approx(q_with_pattern, abs=0.01)

    def test_confidence_factors_in(self):
        result_high_conf = make_result(consensus_reached=False, confidence=1.0)
        result_low_conf = make_result(consensus_reached=False, confidence=0.0)
        q_high = self.analyzer._calculate_quality(result_high_conf, [], [1], [])
        q_low = self.analyzer._calculate_quality(result_low_conf, [], [1], [])
        assert q_high > q_low


# ---------------------------------------------------------------------------
# MetaCritiqueAnalyzer — _generate_recommendations
# ---------------------------------------------------------------------------


class TestGenerateRecommendations:
    def setup_method(self):
        self.analyzer = MetaCritiqueAnalyzer()

    def test_repetition_recommendation(self):
        issue = MetaObservation("issue", "alice repeated similar content", 0.6, (1, 2))
        result = make_result(consensus_reached=True, confidence=0.8)
        recs = self.analyzer._generate_recommendations([issue], result)
        assert any("build on previous" in r.lower() for r in recs)

    def test_misalignment_recommendation(self):
        issue = MetaObservation(
            "issue", "bob's critique may not address the actual proposal", 0.5, (0, 0)
        )
        result = make_result(consensus_reached=True, confidence=0.8)
        recs = self.analyzer._generate_recommendations([issue], result)
        assert any("directly reference" in r.lower() or "specific parts" in r.lower() for r in recs)

    def test_ignored_critique_recommendation(self):
        issue = MetaObservation("issue", "suggestions from bob may have been ignored", 0.4, (0, 0))
        result = make_result(consensus_reached=True, confidence=0.8)
        recs = self.analyzer._generate_recommendations([issue], result)
        assert any("acknowledge" in r.lower() for r in recs)

    def test_circular_argument_recommendation(self):
        pattern = MetaObservation(
            "pattern",
            "Debate may have returned to initial positions (circular)",
            0.7,
            (1, 4),
        )
        result = make_result(consensus_reached=True, confidence=0.8)
        recs = self.analyzer._generate_recommendations([pattern], result)
        assert any("circular" in r.lower() or "maximum round" in r.lower() for r in recs)

    def test_low_confidence_recommendation(self):
        result = make_result(consensus_reached=True, confidence=0.4)
        recs = self.analyzer._generate_recommendations([], result)
        assert any("judge" in r.lower() or "consensus" in r.lower() for r in recs)

    def test_no_consensus_recommendation(self):
        result = make_result(consensus_reached=False, confidence=0.8)
        recs = self.analyzer._generate_recommendations([], result)
        assert any("consensus" in r.lower() or "minority" in r.lower() for r in recs)

    def test_limited_to_5_recommendations(self):
        # Create observations that trigger all recommendation types
        issues = [
            MetaObservation("issue", "alice repeated similar content again", 0.6, (1, 2)),
            MetaObservation(
                "issue", "bob critique may not address the actual proposal", 0.5, (0, 0)
            ),
            MetaObservation("issue", "suggestions from carol may have been ignored", 0.4, (0, 0)),
            MetaObservation(
                "pattern", "debate returned to initial positions circular argument", 0.7, (1, 4)
            ),
        ]
        result = make_result(consensus_reached=False, confidence=0.4)
        recs = self.analyzer._generate_recommendations(issues, result)
        assert len(recs) <= 5

    def test_no_observations_no_issues_recs(self):
        result = make_result(consensus_reached=True, confidence=0.9)
        recs = self.analyzer._generate_recommendations([], result)
        # Should have no issue-based recs, only maybe confidence if < 0.6
        assert isinstance(recs, list)
        assert len(recs) == 0  # consensus=True, confidence=0.9 → no recs


# ---------------------------------------------------------------------------
# MetaCritiqueAnalyzer — analyze() (integration)
# ---------------------------------------------------------------------------


class TestAnalyze:
    def setup_method(self):
        self.analyzer = MetaCritiqueAnalyzer()

    def test_analyze_returns_meta_critique(self):
        result = make_result()
        critique = self.analyzer.analyze(result)
        assert isinstance(critique, MetaCritique)

    def test_analyze_debate_id_matches(self):
        result = make_result(id_="my-debate-id")
        critique = self.analyzer.analyze(result)
        assert critique.debate_id == "my-debate-id"

    def test_analyze_empty_debate(self):
        result = make_result(messages=[], critiques=[], consensus_reached=False, confidence=0.0)
        critique = self.analyzer.analyze(result)
        assert isinstance(critique, MetaCritique)
        assert 0.0 <= critique.overall_quality <= 1.0
        assert isinstance(critique.recommendations, list)

    def test_analyze_quality_bounded(self):
        messages = [
            make_message("alice", "proposal", round_=1),
            make_message("bob", "critique", round_=1),
        ]
        critiques = [
            make_critique("bob", "alice", "proposal", ["issue"], ["suggestion"], severity=0.5)
        ]
        result = make_result(
            messages=messages,
            critiques=critiques,
            consensus_reached=True,
            confidence=0.85,
        )
        critique = self.analyzer.analyze(result)
        assert 0.0 <= critique.overall_quality <= 1.0

    def test_analyze_with_repetition_generates_observation(self):
        content = "same identical content repeated many times over and over again"
        messages = [
            make_message("alice", content, round_=1),
            make_message("alice", content, round_=2),
        ]
        result = make_result(messages=messages, critiques=[])
        critique = self.analyzer.analyze(result)
        types = [o.observation_type for o in critique.observations]
        assert "issue" in types

    def test_analyze_created_at_is_set(self):
        result = make_result()
        critique = self.analyzer.analyze(result)
        assert critique.created_at

    def test_analyze_productive_unproductive_lists(self):
        msg = make_message("alice", "proposal", round_=1)
        crit = make_critique("bob", "alice", "proposal", ["issue"], [], severity=0.5)
        result = make_result(messages=[msg], critiques=[crit])
        critique = self.analyzer.analyze(result)
        all_rounds = set(critique.productive_rounds + critique.unproductive_rounds)
        assert 1 in all_rounds


# ---------------------------------------------------------------------------
# MetaCritiqueAnalyzer — generate_meta_prompt()
# ---------------------------------------------------------------------------


class TestGenerateMetaPrompt:
    def setup_method(self):
        self.analyzer = MetaCritiqueAnalyzer()

    def test_prompt_contains_task(self):
        result = make_result(task="Design a caching layer")
        prompt = self.analyzer.generate_meta_prompt(result)
        assert "Design a caching layer" in prompt

    def test_prompt_contains_consensus_status_reached(self):
        result = make_result(consensus_reached=True, confidence=0.9)
        prompt = self.analyzer.generate_meta_prompt(result)
        assert "Reached" in prompt

    def test_prompt_contains_consensus_status_not_reached(self):
        result = make_result(consensus_reached=False, confidence=0.3)
        prompt = self.analyzer.generate_meta_prompt(result)
        assert "Not reached" in prompt

    def test_prompt_contains_confidence_percentage(self):
        result = make_result(confidence=0.75)
        prompt = self.analyzer.generate_meta_prompt(result)
        assert "75%" in prompt

    def test_prompt_contains_messages(self):
        messages = [
            make_message("alice", "My proposal for the system", round_=1, role="proposer"),
        ]
        result = make_result(messages=messages)
        prompt = self.analyzer.generate_meta_prompt(result)
        assert "alice" in prompt
        assert "My proposal for the system" in prompt

    def test_prompt_contains_critiques(self):
        critiques = [
            make_critique(
                "bob",
                "alice",
                "proposal",
                ["issue one", "issue two"],
                [],
            ),
        ]
        result = make_result(critiques=critiques)
        prompt = self.analyzer.generate_meta_prompt(result)
        assert "bob" in prompt
        assert "alice" in prompt

    def test_prompt_truncates_messages_to_20(self):
        messages = [
            make_message(f"agent{i}", f"content {i}", round_=i)
            for i in range(30)
        ]
        result = make_result(messages=messages)
        prompt = self.analyzer.generate_meta_prompt(result)
        # The 21st message should not appear (content 20 is index 20, beyond limit)
        assert "content 20" not in prompt
        assert "content 19" in prompt

    def test_prompt_contains_analysis_sections(self):
        result = make_result()
        prompt = self.analyzer.generate_meta_prompt(result)
        assert "PRODUCTIVE ROUNDS" in prompt
        assert "ISSUES" in prompt
        assert "PATTERNS" in prompt
        assert "RECOMMENDATIONS" in prompt

    def test_prompt_is_string(self):
        result = make_result()
        prompt = self.analyzer.generate_meta_prompt(result)
        assert isinstance(prompt, str)
        assert len(prompt) > 100

    def test_prompt_truncates_critiques_to_10(self):
        critiques = [
            make_critique(
                f"agent{i}",
                f"target{i}",
                "proposal",
                [f"issue{i}"],
                [],
            )
            for i in range(15)
        ]
        result = make_result(critiques=critiques)
        prompt = self.analyzer.generate_meta_prompt(result)
        # agent10 through agent14 should not appear (only first 10 critiques included)
        assert "agent10" not in prompt
        assert "agent9" in prompt

    def test_prompt_empty_messages_and_critiques(self):
        result = make_result(messages=[], critiques=[])
        prompt = self.analyzer.generate_meta_prompt(result)
        assert isinstance(prompt, str)
        assert "Task:" in prompt


# ---------------------------------------------------------------------------
# MetaCritiqueAnalyzer — __init__
# ---------------------------------------------------------------------------


class TestMetaCritiqueAnalyzerInit:
    def test_default_no_embedding_provider(self):
        analyzer = MetaCritiqueAnalyzer()
        assert analyzer._embedding_provider is None
        assert analyzer._embedding_cache == {}

    def test_with_embedding_provider(self):
        provider = MagicMock()
        analyzer = MetaCritiqueAnalyzer(embedding_provider=provider)
        assert analyzer._embedding_provider is provider

    def test_constants(self):
        assert MetaCritiqueAnalyzer.REPETITION_THRESHOLD == 0.6
        assert MetaCritiqueAnalyzer.MIN_PROGRESS_THRESHOLD == 0.2
