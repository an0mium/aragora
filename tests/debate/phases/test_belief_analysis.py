"""Tests for aragora.debate.phases.belief_analysis — DebateBeliefAnalyzer."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

import aragora.debate.phases.belief_analysis as mod
from aragora.debate.phases.belief_analysis import (
    BeliefAnalysisResult,
    DebateBeliefAnalyzer,
    _load_belief_classes,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _msg(role: str, agent: str, content: str) -> SimpleNamespace:
    """Create a lightweight stand-in for Message."""
    return SimpleNamespace(role=role, agent=agent, content=content)


def _claim(statement: str, confidence: float = 0.5, claim_id: str | None = None) -> SimpleNamespace:
    """Create a lightweight stand-in for a claim object."""
    ns = SimpleNamespace(statement=statement, confidence=confidence)
    if claim_id is not None:
        ns.claim_id = claim_id
    return ns


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def reset_globals():
    """Reset module-level lazy-loaded classes between tests."""
    orig_bn = mod._BeliefNetwork
    orig_bpa = mod._BeliefPropagationAnalyzer
    mod._BeliefNetwork = None
    mod._BeliefPropagationAnalyzer = None
    yield
    mod._BeliefNetwork = orig_bn
    mod._BeliefPropagationAnalyzer = orig_bpa


@pytest.fixture()
def mock_bn_class():
    """Return a MagicMock that acts as the BeliefNetwork class."""
    cls = MagicMock(name="BeliefNetwork")
    instance = MagicMock(name="bn_instance")
    instance.nodes = {}
    cls.return_value = instance
    return cls


@pytest.fixture()
def mock_bpa_class():
    """Return a MagicMock that acts as the BeliefPropagationAnalyzer class."""
    cls = MagicMock(name="BeliefPropagationAnalyzer")
    instance = MagicMock(name="bpa_instance")
    instance.identify_debate_cruxes.return_value = []
    instance.suggest_evidence_targets.return_value = []
    cls.return_value = instance
    return cls


@pytest.fixture()
def inject_mocks(mock_bn_class, mock_bpa_class):
    """Inject mock belief classes into the module globals."""
    mod._BeliefNetwork = mock_bn_class
    mod._BeliefPropagationAnalyzer = mock_bpa_class
    return mock_bn_class, mock_bpa_class


# ---------------------------------------------------------------------------
# BeliefAnalysisResult
# ---------------------------------------------------------------------------


class TestBeliefAnalysisResult:
    def test_defaults(self):
        r = BeliefAnalysisResult()
        assert r.cruxes == []
        assert r.evidence_suggestions == []
        assert r.network_size == 0
        assert r.analysis_error is None

    def test_custom_values(self):
        r = BeliefAnalysisResult(
            cruxes=[{"claim": "X"}],
            evidence_suggestions=["find Y"],
            network_size=5,
            analysis_error="oops",
        )
        assert r.cruxes == [{"claim": "X"}]
        assert r.evidence_suggestions == ["find Y"]
        assert r.network_size == 5
        assert r.analysis_error == "oops"


# ---------------------------------------------------------------------------
# _load_belief_classes
# ---------------------------------------------------------------------------


class TestLoadBeliefClasses:
    def test_success_sets_globals_and_returns_tuple(self):
        fake_bn = MagicMock(name="BN")
        fake_bpa = MagicMock(name="BPA")

        with patch.dict(
            "sys.modules",
            {
                "aragora.reasoning.belief": MagicMock(
                    BeliefNetwork=fake_bn,
                    BeliefPropagationAnalyzer=fake_bpa,
                ),
            },
        ):
            result = _load_belief_classes()

        assert result == (fake_bn, fake_bpa)
        assert mod._BeliefNetwork is fake_bn
        assert mod._BeliefPropagationAnalyzer is fake_bpa

    def test_import_error_returns_none_tuple(self):
        with patch(
            "aragora.debate.phases.belief_analysis._BeliefPropagationAnalyzer",
            None,
        ):
            # Force the global to None so the guard passes, then fail the import.
            mod._BeliefPropagationAnalyzer = None
            mod._BeliefNetwork = None
            with patch.dict("sys.modules", {"aragora.reasoning.belief": None}):
                result = _load_belief_classes()

        assert result == (None, None)

    def test_idempotent_returns_cached(self):
        """Second call should skip the import and return cached values."""
        fake_bn = MagicMock(name="BN")
        fake_bpa = MagicMock(name="BPA")
        mod._BeliefNetwork = fake_bn
        mod._BeliefPropagationAnalyzer = fake_bpa

        result = _load_belief_classes()
        assert result == (fake_bn, fake_bpa)


# ---------------------------------------------------------------------------
# DebateBeliefAnalyzer — init
# ---------------------------------------------------------------------------


class TestDebateBeliefAnalyzerInit:
    def test_defaults(self):
        a = DebateBeliefAnalyzer()
        assert a.propagation_iterations == 3
        assert a.crux_threshold == 0.6
        assert a.max_claims == 20
        assert a._network is None

    def test_custom_values(self):
        a = DebateBeliefAnalyzer(
            propagation_iterations=5,
            crux_threshold=0.8,
            max_claims=10,
        )
        assert a.propagation_iterations == 5
        assert a.crux_threshold == 0.8
        assert a.max_claims == 10


# ---------------------------------------------------------------------------
# analyze_messages
# ---------------------------------------------------------------------------


class TestAnalyzeMessages:
    def test_belief_module_not_available(self):
        """When belief classes are not loadable, returns analysis_error."""
        analyzer = DebateBeliefAnalyzer()
        msgs = [_msg("proposer", "alice", "claim A")]

        # Block the real import so _load_belief_classes fails
        with patch.dict("sys.modules", {"aragora.reasoning.belief": None}):
            result = analyzer.analyze_messages(msgs)

        assert result.analysis_error == "Belief module not available"
        assert result.cruxes == []

    def test_empty_messages(self, inject_mocks):
        analyzer = DebateBeliefAnalyzer()
        result = analyzer.analyze_messages([])
        assert result.analysis_error is None
        assert result.cruxes == []
        assert result.network_size == 0

    def test_filters_for_proposer_and_critic_roles(self, inject_mocks):
        mock_bn_cls, _bpa = inject_mocks
        instance = mock_bn_cls.return_value
        instance.nodes = {"n1": True}

        analyzer = DebateBeliefAnalyzer()
        msgs = [
            _msg("proposer", "alice", "claim A"),
            _msg("critic", "bob", "critique B"),
            _msg("synthesizer", "charlie", "synthesis C"),
            _msg("judge", "dave", "verdict D"),
        ]

        analyzer.analyze_messages(msgs)

        # Only proposer and critic messages should be added
        assert instance.add_claim.call_count == 2

    def test_respects_max_claims(self, inject_mocks):
        mock_bn_cls, _bpa = inject_mocks
        instance = mock_bn_cls.return_value
        instance.nodes = {"n1": True}

        analyzer = DebateBeliefAnalyzer(max_claims=2)
        msgs = [
            _msg("proposer", "a", "claim 1"),
            _msg("proposer", "b", "claim 2"),
            _msg("proposer", "c", "claim 3"),
        ]

        analyzer.analyze_messages(msgs)
        assert instance.add_claim.call_count == 2

    def test_claim_id_format(self, inject_mocks):
        mock_bn_cls, _bpa = inject_mocks
        instance = mock_bn_cls.return_value
        instance.nodes = {"n1": True}

        content = "This is a test claim"
        msg = _msg("proposer", "alice", content)
        expected_hash = hash(content[:100])
        expected_id = f"alice_{expected_hash}"

        analyzer = DebateBeliefAnalyzer()
        analyzer.analyze_messages([msg])

        call_kwargs = instance.add_claim.call_args
        assert call_kwargs[1]["claim_id"] == expected_id

    def test_network_size_reflects_node_count(self, inject_mocks):
        mock_bn_cls, _bpa = inject_mocks
        instance = mock_bn_cls.return_value
        instance.nodes = {"a": 1, "b": 2, "c": 3}

        analyzer = DebateBeliefAnalyzer()
        msgs = [_msg("proposer", "alice", "X")]
        result = analyzer.analyze_messages(msgs)
        assert result.network_size == 3

    def test_network_size_without_nodes_attr(self, inject_mocks):
        mock_bn_cls, _bpa = inject_mocks
        instance = mock_bn_cls.return_value
        del instance.nodes  # remove the attribute

        analyzer = DebateBeliefAnalyzer()
        msgs = [_msg("proposer", "alice", "X")]
        result = analyzer.analyze_messages(msgs)
        assert result.network_size == 0

    def test_propagate_called_and_cruxes_identified(self, inject_mocks):
        mock_bn_cls, mock_bpa_cls = inject_mocks
        bn_instance = mock_bn_cls.return_value
        bpa_instance = mock_bpa_cls.return_value

        bn_instance.nodes = {"n1": True, "n2": True}
        bpa_instance.identify_debate_cruxes.return_value = [
            {"claim": "X", "uncertainty": 0.9},
        ]
        bpa_instance.suggest_evidence_targets.return_value = ["ev1"]

        analyzer = DebateBeliefAnalyzer()
        msgs = [_msg("proposer", "alice", "claim A")]
        result = analyzer.analyze_messages(msgs)

        bn_instance.propagate.assert_called_once()
        mock_bpa_cls.assert_called_once_with(bn_instance)
        bpa_instance.identify_debate_cruxes.assert_called_once_with(top_k=3)
        assert result.cruxes == [{"claim": "X", "uncertainty": 0.9}]

    def test_evidence_suggestions_limited_to_top_k(self, inject_mocks):
        mock_bn_cls, mock_bpa_cls = inject_mocks
        bn_instance = mock_bn_cls.return_value
        bpa_instance = mock_bpa_cls.return_value

        bn_instance.nodes = {"n1": True}
        bpa_instance.suggest_evidence_targets.return_value = [
            "ev1",
            "ev2",
            "ev3",
            "ev4",
            "ev5",
        ]

        analyzer = DebateBeliefAnalyzer()
        msgs = [_msg("proposer", "alice", "claim")]
        result = analyzer.analyze_messages(msgs, top_k_evidence=2)

        assert result.evidence_suggestions == ["ev1", "ev2"]

    def test_custom_top_k_cruxes_passed(self, inject_mocks):
        mock_bn_cls, mock_bpa_cls = inject_mocks
        bn_instance = mock_bn_cls.return_value
        bpa_instance = mock_bpa_cls.return_value
        bn_instance.nodes = {"n1": True}

        analyzer = DebateBeliefAnalyzer()
        msgs = [_msg("critic", "bob", "critique")]
        analyzer.analyze_messages(msgs, top_k_cruxes=7)

        bpa_instance.identify_debate_cruxes.assert_called_once_with(top_k=7)

    def test_propagation_iterations_passed_to_network(self, inject_mocks):
        mock_bn_cls, _bpa = inject_mocks
        bn_instance = mock_bn_cls.return_value
        bn_instance.nodes = {"n1": True}

        analyzer = DebateBeliefAnalyzer(propagation_iterations=10)
        msgs = [_msg("proposer", "alice", "claim")]
        analyzer.analyze_messages(msgs)

        mock_bn_cls.assert_called_once_with(max_iterations=10)

    def test_content_truncated_to_500(self, inject_mocks):
        mock_bn_cls, _bpa = inject_mocks
        bn_instance = mock_bn_cls.return_value
        bn_instance.nodes = {"n1": True}

        long_content = "A" * 1000
        msg = _msg("proposer", "alice", long_content)

        analyzer = DebateBeliefAnalyzer()
        analyzer.analyze_messages([msg])

        call_kwargs = bn_instance.add_claim.call_args[1]
        assert call_kwargs["statement"] == "A" * 500

    def test_runtime_error_caught(self, inject_mocks):
        mock_bn_cls, _bpa = inject_mocks
        mock_bn_cls.return_value.nodes = {"n1": True}
        mock_bn_cls.return_value.propagate.side_effect = RuntimeError("boom")

        analyzer = DebateBeliefAnalyzer()
        msgs = [_msg("proposer", "alice", "claim")]
        result = analyzer.analyze_messages(msgs)

        assert result.analysis_error == "Belief analysis failed: RuntimeError"

    def test_attribute_error_caught(self, inject_mocks):
        mock_bn_cls, _bpa = inject_mocks
        mock_bn_cls.return_value.nodes = {"n1": True}
        mock_bn_cls.return_value.propagate.side_effect = AttributeError("bad attr")

        analyzer = DebateBeliefAnalyzer()
        msgs = [_msg("proposer", "alice", "claim")]
        result = analyzer.analyze_messages(msgs)

        assert result.analysis_error == "Belief analysis failed: AttributeError"

    def test_no_propagation_when_network_empty(self, inject_mocks):
        mock_bn_cls, mock_bpa_cls = inject_mocks
        bn_instance = mock_bn_cls.return_value
        bn_instance.nodes = {}

        analyzer = DebateBeliefAnalyzer()
        # Only a synthesizer message, so no claims added, network is empty
        msgs = [_msg("synthesizer", "charlie", "summary")]
        result = analyzer.analyze_messages(msgs)

        bn_instance.propagate.assert_not_called()
        mock_bpa_cls.assert_not_called()
        assert result.network_size == 0

    def test_add_claim_kwargs(self, inject_mocks):
        """Verify all keyword arguments passed to network.add_claim."""
        mock_bn_cls, _bpa = inject_mocks
        bn_instance = mock_bn_cls.return_value
        bn_instance.nodes = {"n1": True}

        msg = _msg("critic", "bob", "this is wrong")
        analyzer = DebateBeliefAnalyzer()
        analyzer.analyze_messages([msg])

        call_kwargs = bn_instance.add_claim.call_args[1]
        assert call_kwargs["author"] == "bob"
        assert call_kwargs["initial_confidence"] == 0.5
        assert call_kwargs["statement"] == "this is wrong"


# ---------------------------------------------------------------------------
# analyze_claims
# ---------------------------------------------------------------------------


class TestAnalyzeClaims:
    def test_bpa_not_available(self):
        """When BPA is not loadable, returns analysis_error."""
        analyzer = DebateBeliefAnalyzer()
        claims = [_claim("statement A")]

        # Block the real import so _load_belief_classes fails
        with patch.dict("sys.modules", {"aragora.reasoning.belief": None}):
            result = analyzer.analyze_claims(claims)

        assert result.analysis_error == "Belief module not available"
        assert result.cruxes == []

    def test_empty_claims(self, inject_mocks):
        analyzer = DebateBeliefAnalyzer()
        result = analyzer.analyze_claims([])
        assert result.analysis_error is None
        assert result.cruxes == []
        assert result.network_size == 0

    def test_respects_max_claims(self, inject_mocks):
        _bn, mock_bpa_cls = inject_mocks
        bpa_instance = mock_bpa_cls.return_value

        analyzer = DebateBeliefAnalyzer(max_claims=2)
        claims = [_claim(f"claim {i}") for i in range(5)]
        analyzer.analyze_claims(claims)

        assert bpa_instance.add_claim.call_count == 2

    def test_uses_crux_threshold(self, inject_mocks):
        _bn, mock_bpa_cls = inject_mocks
        bpa_instance = mock_bpa_cls.return_value

        analyzer = DebateBeliefAnalyzer(crux_threshold=0.9)
        claims = [_claim("statement")]
        analyzer.analyze_claims(claims)

        bpa_instance.identify_debate_cruxes.assert_called_once_with(threshold=0.9)

    def test_uses_claim_attributes(self, inject_mocks):
        _bn, mock_bpa_cls = inject_mocks
        bpa_instance = mock_bpa_cls.return_value

        claim = _claim("important statement", confidence=0.75, claim_id="my_claim_1")

        analyzer = DebateBeliefAnalyzer()
        analyzer.analyze_claims([claim])

        call_kwargs = bpa_instance.add_claim.call_args[1]
        assert call_kwargs["claim_id"] == "my_claim_1"
        assert call_kwargs["statement"] == "important statement"
        assert call_kwargs["prior"] == 0.75

    def test_claim_without_claim_id_uses_hash(self, inject_mocks):
        _bn, mock_bpa_cls = inject_mocks
        bpa_instance = mock_bpa_cls.return_value

        claim = _claim("a statement")  # no claim_id attribute

        analyzer = DebateBeliefAnalyzer()
        analyzer.analyze_claims([claim])

        call_kwargs = bpa_instance.add_claim.call_args[1]
        expected_id = str(hash("a statement"[:50]))
        assert call_kwargs["claim_id"] == expected_id

    def test_claim_without_confidence_defaults_to_05(self, inject_mocks):
        _bn, mock_bpa_cls = inject_mocks
        bpa_instance = mock_bpa_cls.return_value

        claim = SimpleNamespace(statement="a statement")
        # No confidence attribute at all

        analyzer = DebateBeliefAnalyzer()
        analyzer.analyze_claims([claim])

        call_kwargs = bpa_instance.add_claim.call_args[1]
        assert call_kwargs["prior"] == 0.5

    def test_network_size_incremented_per_claim(self, inject_mocks):
        analyzer = DebateBeliefAnalyzer()
        claims = [_claim("A"), _claim("B"), _claim("C")]
        result = analyzer.analyze_claims(claims)
        assert result.network_size == 3

    def test_runtime_error_caught(self, inject_mocks):
        _bn, mock_bpa_cls = inject_mocks
        mock_bpa_cls.return_value.add_claim.side_effect = RuntimeError("broken")

        analyzer = DebateBeliefAnalyzer()
        claims = [_claim("bad claim")]
        result = analyzer.analyze_claims(claims)

        assert result.analysis_error == "Belief analysis failed: RuntimeError"

    def test_cruxes_returned(self, inject_mocks):
        _bn, mock_bpa_cls = inject_mocks
        bpa_instance = mock_bpa_cls.return_value
        bpa_instance.identify_debate_cruxes.return_value = [
            {"claim": "key point", "uncertainty": 0.85},
        ]

        analyzer = DebateBeliefAnalyzer()
        claims = [_claim("key point")]
        result = analyzer.analyze_claims(claims)

        assert len(result.cruxes) == 1
        assert result.cruxes[0]["claim"] == "key point"

    def test_only_checks_bpa_not_bn(self):
        """analyze_claims only gates on BPA availability, not BN."""
        mock_bpa_cls = MagicMock(name="BPA")
        bpa_instance = MagicMock()
        bpa_instance.identify_debate_cruxes.return_value = []
        mock_bpa_cls.return_value = bpa_instance

        # BN is None but BPA is set
        mod._BeliefNetwork = None
        mod._BeliefPropagationAnalyzer = mock_bpa_cls

        analyzer = DebateBeliefAnalyzer()
        claims = [_claim("a claim")]
        result = analyzer.analyze_claims(claims)

        # Should not error — BPA is available
        assert result.analysis_error is None
