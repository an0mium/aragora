"""Tests for the argument structure verifier.

Tests cover:
- ArgumentVerificationResult dataclass and properties
- VerifiedChain, InvalidChain, UnsupportedConclusion, ContradictionPair dataclasses
- ArgumentStructureVerifier initialization and strategy selection
- Cycle detection in argument graphs
- Unsupported conclusion detection
- Contradiction detection (explicit and implicit)
- Argument chain extraction and verification
- Fallback to Z3 for arithmetic claims
- FormalVerificationManager.verify_argument_structure integration
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from aragora.verification.argument_verifier import (
    ArgumentStructureVerifier,
    ArgumentVerificationResult,
    ContradictionPair,
    InvalidChain,
    UnsupportedConclusion,
    VerifiedChain,
    VerificationStrategy,
)
from aragora.verification.formal import (
    FormalLanguage,
    FormalProofResult,
    FormalProofStatus,
    FormalVerificationManager,
    LeanBackend,
    Z3Backend,
)
from aragora.visualization.mapper import (
    ArgumentCartographer,
    ArgumentEdge,
    ArgumentNode,
    EdgeRelation,
    NodeType,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_graph() -> ArgumentCartographer:
    """Create an empty ArgumentCartographer for testing."""
    graph = ArgumentCartographer()
    graph.set_debate_context("test-debate-1", "Test topic")
    return graph


def _add_node(
    graph: ArgumentCartographer,
    node_id: str,
    agent: str = "agent-1",
    node_type: NodeType = NodeType.PROPOSAL,
    summary: str = "Test claim",
    round_num: int = 0,
    full_content: str | None = None,
) -> ArgumentNode:
    """Add a node directly to the graph for testing."""
    node = ArgumentNode(
        id=node_id,
        agent=agent,
        node_type=node_type,
        summary=summary,
        round_num=round_num,
        timestamp=1000.0,
        full_content=full_content or summary,
    )
    graph.nodes[node_id] = node
    return node


def _add_edge(
    graph: ArgumentCartographer,
    source_id: str,
    target_id: str,
    relation: EdgeRelation = EdgeRelation.SUPPORTS,
    weight: float = 1.0,
) -> ArgumentEdge:
    """Add an edge directly to the graph for testing."""
    edge = ArgumentEdge(
        source_id=source_id,
        target_id=target_id,
        relation=relation,
        weight=weight,
    )
    graph.edges.append(edge)
    return edge


# ---------------------------------------------------------------------------
# Dataclass Tests
# ---------------------------------------------------------------------------

class TestArgumentVerificationResult:
    """Tests for ArgumentVerificationResult dataclass."""

    def test_empty_result_is_sound(self):
        """An empty result with no findings should be sound."""
        result = ArgumentVerificationResult()
        assert result.is_sound is True

    def test_result_with_invalid_chains_is_not_sound(self):
        """Result with invalid chains should not be sound."""
        result = ArgumentVerificationResult(
            invalid_chains=[
                InvalidChain(
                    chain_id="c1",
                    name="c1",
                    premise_node_ids=["p1"],
                    conclusion_node_id="q1",
                    reason="non-sequitur",
                )
            ]
        )
        assert result.is_sound is False

    def test_result_with_contradictions_is_not_sound(self):
        """Result with contradictions should not be sound."""
        result = ArgumentVerificationResult(
            contradictions=[
                ContradictionPair(
                    node_id_a="a",
                    node_id_b="b",
                    summary_a="X",
                    summary_b="not X",
                    explanation="direct",
                )
            ]
        )
        assert result.is_sound is False

    def test_result_with_unsupported_is_not_sound(self):
        """Result with unsupported conclusions should not be sound."""
        result = ArgumentVerificationResult(
            unsupported_conclusions=[
                UnsupportedConclusion(
                    node_id="n1",
                    summary="claim",
                    agent="agent",
                    reason="no premises",
                )
            ]
        )
        assert result.is_sound is False

    def test_result_with_circular_deps_is_not_sound(self):
        """Result with circular dependencies should not be sound."""
        result = ArgumentVerificationResult(
            circular_dependencies=[["a", "b", "a"]]
        )
        assert result.is_sound is False

    def test_soundness_score_perfect(self):
        """Perfect soundness for all valid chains, no issues."""
        result = ArgumentVerificationResult(
            valid_chains=[
                VerifiedChain(
                    chain_id="c1", name="c1",
                    premise_node_ids=["p1"],
                    conclusion_node_id="q1",
                    confidence=0.9,
                )
            ],
            total_nodes_analyzed=5,
        )
        assert result.soundness_score == 1.0

    def test_soundness_score_mixed(self):
        """Mixed valid/invalid chains should produce intermediate score."""
        result = ArgumentVerificationResult(
            valid_chains=[
                VerifiedChain(
                    chain_id="c1", name="c1",
                    premise_node_ids=["p1"],
                    conclusion_node_id="q1",
                    confidence=0.9,
                )
            ],
            invalid_chains=[
                InvalidChain(
                    chain_id="c2", name="c2",
                    premise_node_ids=["p2"],
                    conclusion_node_id="q2",
                    reason="invalid",
                )
            ],
            total_nodes_analyzed=5,
        )
        score = result.soundness_score
        assert 0.0 < score < 1.0

    def test_to_dict_serialization(self):
        """to_dict should produce a complete serializable dictionary."""
        result = ArgumentVerificationResult(
            total_nodes_analyzed=3,
            total_chains_checked=1,
        )
        d = result.to_dict()
        assert "valid_chains" in d
        assert "invalid_chains" in d
        assert "unsupported_conclusions" in d
        assert "contradictions" in d
        assert "circular_dependencies" in d
        assert "is_sound" in d
        assert "soundness_score" in d
        assert d["total_nodes_analyzed"] == 3

    def test_soundness_score_no_data(self):
        """Soundness score with no data should be 1.0 (nothing to fail)."""
        result = ArgumentVerificationResult()
        assert result.soundness_score == 1.0


class TestVerifiedChain:
    """Tests for VerifiedChain dataclass."""

    def test_to_dict(self):
        chain = VerifiedChain(
            chain_id="vc1",
            name="valid_chain",
            premise_node_ids=["p1", "p2"],
            conclusion_node_id="c1",
            confidence=0.85,
        )
        d = chain.to_dict()
        assert d["chain_id"] == "vc1"
        assert d["confidence"] == 0.85
        assert d["proof_result"] is None

    def test_to_dict_with_proof_result(self):
        proof = FormalProofResult(
            status=FormalProofStatus.PROOF_FOUND,
            language=FormalLanguage.LEAN4,
        )
        chain = VerifiedChain(
            chain_id="vc2",
            name="chain_with_proof",
            premise_node_ids=["p1"],
            conclusion_node_id="c1",
            proof_result=proof,
            confidence=0.9,
        )
        d = chain.to_dict()
        assert d["proof_result"] is not None
        assert d["proof_result"]["status"] == "proof_found"


class TestInvalidChain:
    """Tests for InvalidChain dataclass."""

    def test_to_dict(self):
        chain = InvalidChain(
            chain_id="ic1",
            name="bad_chain",
            premise_node_ids=["p1"],
            conclusion_node_id="c1",
            reason="non-sequitur",
        )
        d = chain.to_dict()
        assert d["reason"] == "non-sequitur"


class TestContradictionPair:
    """Tests for ContradictionPair dataclass."""

    def test_to_dict(self):
        pair = ContradictionPair(
            node_id_a="a",
            node_id_b="b",
            summary_a="The sky is blue",
            summary_b="The sky is not blue",
            explanation="direct contradiction",
        )
        d = pair.to_dict()
        assert d["node_id_a"] == "a"
        assert d["explanation"] == "direct contradiction"


class TestUnsupportedConclusion:
    """Tests for UnsupportedConclusion dataclass."""

    def test_to_dict(self):
        uc = UnsupportedConclusion(
            node_id="n1",
            summary="Unsupported claim",
            agent="agent-x",
            reason="No premises",
        )
        d = uc.to_dict()
        assert d["agent"] == "agent-x"


# ---------------------------------------------------------------------------
# Verifier Initialization
# ---------------------------------------------------------------------------

class TestArgumentStructureVerifierInit:
    """Tests for ArgumentStructureVerifier initialization."""

    def test_default_initialization(self):
        """Should create with default backends and strategy."""
        verifier = ArgumentStructureVerifier()
        assert verifier._strategy == VerificationStrategy.LEAN_WITH_Z3_FALLBACK
        assert isinstance(verifier._lean_backend, LeanBackend)
        assert isinstance(verifier._z3_backend, Z3Backend)

    def test_custom_strategy(self):
        """Should accept a custom verification strategy."""
        verifier = ArgumentStructureVerifier(strategy=VerificationStrategy.Z3_ONLY)
        assert verifier._strategy == VerificationStrategy.Z3_ONLY

    def test_custom_backends(self):
        """Should accept custom backend instances."""
        lean = LeanBackend(sandbox_timeout=10.0)
        z3 = Z3Backend(cache_size=50)
        verifier = ArgumentStructureVerifier(lean_backend=lean, z3_backend=z3)
        assert verifier._lean_backend is lean
        assert verifier._z3_backend is z3


# ---------------------------------------------------------------------------
# Cycle Detection
# ---------------------------------------------------------------------------

class TestCycleDetection:
    """Tests for circular dependency detection."""

    def test_no_cycles_in_linear_graph(self):
        """A linear graph A -> B -> C should have no cycles."""
        graph = _make_graph()
        _add_node(graph, "a", node_type=NodeType.EVIDENCE)
        _add_node(graph, "b", node_type=NodeType.EVIDENCE)
        _add_node(graph, "c", node_type=NodeType.PROPOSAL, round_num=1)
        _add_edge(graph, "a", "b", EdgeRelation.SUPPORTS)
        _add_edge(graph, "b", "c", EdgeRelation.SUPPORTS)

        verifier = ArgumentStructureVerifier()
        cycles = verifier._detect_cycles(graph)
        assert len(cycles) == 0

    def test_cycle_detected(self):
        """A cycle A -> B -> A should be detected."""
        graph = _make_graph()
        _add_node(graph, "a", node_type=NodeType.EVIDENCE)
        _add_node(graph, "b", node_type=NodeType.EVIDENCE)
        _add_edge(graph, "a", "b", EdgeRelation.SUPPORTS)
        _add_edge(graph, "b", "a", EdgeRelation.SUPPORTS)

        verifier = ArgumentStructureVerifier()
        cycles = verifier._detect_cycles(graph)
        assert len(cycles) >= 1
        # The cycle should contain both nodes
        cycle_nodes = set()
        for cycle in cycles:
            cycle_nodes.update(cycle)
        assert "a" in cycle_nodes
        assert "b" in cycle_nodes

    def test_only_support_edges_considered(self):
        """Non-SUPPORTS edges should not create cycles."""
        graph = _make_graph()
        _add_node(graph, "a", node_type=NodeType.PROPOSAL)
        _add_node(graph, "b", node_type=NodeType.CRITIQUE)
        # A refutes B and B refutes A - not a logical cycle
        _add_edge(graph, "a", "b", EdgeRelation.REFUTES)
        _add_edge(graph, "b", "a", EdgeRelation.REFUTES)

        verifier = ArgumentStructureVerifier()
        cycles = verifier._detect_cycles(graph)
        assert len(cycles) == 0

    def test_empty_graph_no_cycles(self):
        """An empty graph should have no cycles."""
        graph = _make_graph()
        verifier = ArgumentStructureVerifier()
        cycles = verifier._detect_cycles(graph)
        assert len(cycles) == 0


# ---------------------------------------------------------------------------
# Unsupported Conclusion Detection
# ---------------------------------------------------------------------------

class TestUnsupportedConclusionDetection:
    """Tests for finding unsupported conclusions."""

    def test_unsupported_proposal(self):
        """A proposal in round > 0 with no SUPPORTS edges should be flagged."""
        graph = _make_graph()
        _add_node(graph, "p1", node_type=NodeType.PROPOSAL, round_num=1,
                  summary="We should adopt X")

        verifier = ArgumentStructureVerifier()
        unsupported = verifier._find_unsupported_conclusions(graph)
        assert len(unsupported) == 1
        assert unsupported[0].node_id == "p1"

    def test_supported_proposal_not_flagged(self):
        """A proposal with supporting evidence should not be flagged."""
        graph = _make_graph()
        _add_node(graph, "e1", node_type=NodeType.EVIDENCE,
                  summary="Data shows X works")
        _add_node(graph, "p1", node_type=NodeType.PROPOSAL, round_num=1,
                  summary="We should adopt X")
        _add_edge(graph, "e1", "p1", EdgeRelation.SUPPORTS)

        verifier = ArgumentStructureVerifier()
        unsupported = verifier._find_unsupported_conclusions(graph)
        assert len(unsupported) == 0

    def test_initial_proposal_not_flagged(self):
        """A proposal in round 0 is the initial claim - not flagged."""
        graph = _make_graph()
        _add_node(graph, "p0", node_type=NodeType.PROPOSAL, round_num=0,
                  summary="Initial proposal")

        verifier = ArgumentStructureVerifier()
        unsupported = verifier._find_unsupported_conclusions(graph)
        assert len(unsupported) == 0

    def test_evidence_nodes_not_checked(self):
        """Evidence and critique nodes are not checked for support."""
        graph = _make_graph()
        _add_node(graph, "e1", node_type=NodeType.EVIDENCE, round_num=1)
        _add_node(graph, "c1", node_type=NodeType.CRITIQUE, round_num=1)

        verifier = ArgumentStructureVerifier()
        unsupported = verifier._find_unsupported_conclusions(graph)
        assert len(unsupported) == 0


# ---------------------------------------------------------------------------
# Contradiction Detection
# ---------------------------------------------------------------------------

class TestContradictionDetection:
    """Tests for detecting contradictory premises."""

    def test_explicit_refutation_between_proposals(self):
        """Two proposals connected by a REFUTES edge are contradictory."""
        graph = _make_graph()
        _add_node(graph, "p1", agent="alice", node_type=NodeType.PROPOSAL,
                  summary="We should use Python")
        _add_node(graph, "p2", agent="bob", node_type=NodeType.REBUTTAL,
                  summary="We should not use Python")
        _add_edge(graph, "p2", "p1", EdgeRelation.REFUTES)

        verifier = ArgumentStructureVerifier()
        contradictions = verifier._detect_contradictions(graph)
        assert len(contradictions) == 1
        pair = contradictions[0]
        node_ids = {pair.node_id_a, pair.node_id_b}
        assert "p1" in node_ids
        assert "p2" in node_ids

    def test_opposing_support_and_refute(self):
        """One node supports and another refutes the same target = contradiction."""
        graph = _make_graph()
        _add_node(graph, "target", node_type=NodeType.PROPOSAL)
        _add_node(graph, "supporter", agent="alice", node_type=NodeType.EVIDENCE,
                  summary="Evidence for")
        _add_node(graph, "refuter", agent="bob", node_type=NodeType.EVIDENCE,
                  summary="Evidence against")
        _add_edge(graph, "supporter", "target", EdgeRelation.SUPPORTS)
        _add_edge(graph, "refuter", "target", EdgeRelation.REFUTES)

        verifier = ArgumentStructureVerifier()
        contradictions = verifier._detect_contradictions(graph)
        assert len(contradictions) >= 1

    def test_no_contradiction_in_agreement(self):
        """Two nodes both supporting the same target are not contradictions."""
        graph = _make_graph()
        _add_node(graph, "target", node_type=NodeType.PROPOSAL)
        _add_node(graph, "s1", node_type=NodeType.EVIDENCE, summary="Sup 1")
        _add_node(graph, "s2", node_type=NodeType.EVIDENCE, summary="Sup 2")
        _add_edge(graph, "s1", "target", EdgeRelation.SUPPORTS)
        _add_edge(graph, "s2", "target", EdgeRelation.SUPPORTS)

        verifier = ArgumentStructureVerifier()
        contradictions = verifier._detect_contradictions(graph)
        assert len(contradictions) == 0

    def test_refutation_between_non_claim_nodes_ignored(self):
        """Refutation between VOTE or CONSENSUS nodes should not be flagged."""
        graph = _make_graph()
        _add_node(graph, "v1", node_type=NodeType.VOTE, summary="vote yes")
        _add_node(graph, "v2", node_type=NodeType.VOTE, summary="vote no")
        _add_edge(graph, "v1", "v2", EdgeRelation.REFUTES)

        verifier = ArgumentStructureVerifier()
        contradictions = verifier._detect_contradictions(graph)
        assert len(contradictions) == 0


# ---------------------------------------------------------------------------
# Argument Chain Extraction
# ---------------------------------------------------------------------------

class TestChainExtraction:
    """Tests for extracting argument chains from graphs."""

    def test_single_chain_extracted(self):
        """A simple evidence -> proposal chain should be extracted."""
        graph = _make_graph()
        _add_node(graph, "e1", node_type=NodeType.EVIDENCE)
        _add_node(graph, "p1", node_type=NodeType.PROPOSAL, round_num=1)
        _add_edge(graph, "e1", "p1", EdgeRelation.SUPPORTS)

        verifier = ArgumentStructureVerifier()
        chains = verifier._extract_argument_chains(graph)
        assert len(chains) == 1
        premise_ids, conclusion_id, name = chains[0]
        assert premise_ids == ["e1"]
        assert conclusion_id == "p1"

    def test_multi_premise_chain(self):
        """Multiple premises supporting one conclusion should form one chain."""
        graph = _make_graph()
        _add_node(graph, "e1", node_type=NodeType.EVIDENCE, summary="Fact A")
        _add_node(graph, "e2", node_type=NodeType.EVIDENCE, summary="Fact B")
        _add_node(graph, "p1", node_type=NodeType.PROPOSAL, round_num=1)
        _add_edge(graph, "e1", "p1", EdgeRelation.SUPPORTS)
        _add_edge(graph, "e2", "p1", EdgeRelation.SUPPORTS)

        verifier = ArgumentStructureVerifier()
        chains = verifier._extract_argument_chains(graph)
        assert len(chains) == 1
        premise_ids = chains[0][0]
        assert set(premise_ids) == {"e1", "e2"}

    def test_no_chains_without_support_edges(self):
        """Graph with only REFUTES edges should yield no chains."""
        graph = _make_graph()
        _add_node(graph, "a", node_type=NodeType.PROPOSAL)
        _add_node(graph, "b", node_type=NodeType.CRITIQUE)
        _add_edge(graph, "b", "a", EdgeRelation.REFUTES)

        verifier = ArgumentStructureVerifier()
        chains = verifier._extract_argument_chains(graph)
        assert len(chains) == 0


# ---------------------------------------------------------------------------
# Full Verification (with mocked backends)
# ---------------------------------------------------------------------------

class TestFullVerification:
    """Tests for the full verify() pipeline with mocked backends."""

    @pytest.mark.asyncio
    async def test_verify_empty_graph(self):
        """Verifying an empty graph should return a sound result."""
        graph = _make_graph()
        verifier = ArgumentStructureVerifier()
        result = await verifier.verify(graph)
        assert result.is_sound is True
        assert result.total_nodes_analyzed == 0

    @pytest.mark.asyncio
    async def test_verify_valid_chain_with_z3(self):
        """A valid chain should be verified when Z3 returns PROOF_FOUND."""
        graph = _make_graph()
        _add_node(graph, "e1", node_type=NodeType.EVIDENCE,
                  summary="x > 0", full_content="x is greater than zero")
        _add_node(graph, "p1", node_type=NodeType.PROPOSAL, round_num=1,
                  summary="x >= 0", full_content="x is non-negative")
        _add_edge(graph, "e1", "p1", EdgeRelation.SUPPORTS)

        mock_z3 = MagicMock(spec=Z3Backend)
        mock_z3.is_available = True
        mock_z3.language = FormalLanguage.Z3_SMT
        mock_z3.can_verify.return_value = True
        mock_z3.translate = AsyncMock(return_value="(declare-const x Int)\n(assert (not (=> (> x 0) (>= x 0))))\n(check-sat)")
        mock_z3.prove = AsyncMock(return_value=FormalProofResult(
            status=FormalProofStatus.PROOF_FOUND,
            language=FormalLanguage.Z3_SMT,
            proof_text="QED (negation is unsatisfiable)",
            translation_confidence=0.9,
        ))

        mock_lean = MagicMock(spec=LeanBackend)
        mock_lean.is_available = False
        mock_lean.language = FormalLanguage.LEAN4

        verifier = ArgumentStructureVerifier(
            lean_backend=mock_lean,
            z3_backend=mock_z3,
            strategy=VerificationStrategy.LEAN_WITH_Z3_FALLBACK,
        )
        result = await verifier.verify(graph)
        assert len(result.valid_chains) == 1
        assert result.valid_chains[0].conclusion_node_id == "p1"

    @pytest.mark.asyncio
    async def test_verify_invalid_chain(self):
        """An invalid chain should be reported when proof fails."""
        graph = _make_graph()
        _add_node(graph, "e1", node_type=NodeType.EVIDENCE,
                  summary="The sky is blue")
        _add_node(graph, "p1", node_type=NodeType.PROPOSAL, round_num=1,
                  summary="Therefore cats can fly")
        _add_edge(graph, "e1", "p1", EdgeRelation.SUPPORTS)

        mock_z3 = MagicMock(spec=Z3Backend)
        mock_z3.is_available = True
        mock_z3.language = FormalLanguage.Z3_SMT
        mock_z3.translate = AsyncMock(return_value=None)

        mock_lean = MagicMock(spec=LeanBackend)
        mock_lean.is_available = False
        mock_lean.language = FormalLanguage.LEAN4

        verifier = ArgumentStructureVerifier(
            lean_backend=mock_lean,
            z3_backend=mock_z3,
            strategy=VerificationStrategy.LEAN_WITH_Z3_FALLBACK,
        )
        result = await verifier.verify(graph)
        assert len(result.invalid_chains) == 1

    @pytest.mark.asyncio
    async def test_verify_detects_all_issues(self):
        """A graph with multiple issues should report all of them."""
        graph = _make_graph()
        # Cycle: a <-> b (supports)
        _add_node(graph, "a", node_type=NodeType.EVIDENCE)
        _add_node(graph, "b", node_type=NodeType.EVIDENCE)
        _add_edge(graph, "a", "b", EdgeRelation.SUPPORTS)
        _add_edge(graph, "b", "a", EdgeRelation.SUPPORTS)

        # Unsupported conclusion
        _add_node(graph, "unsp", node_type=NodeType.PROPOSAL, round_num=1,
                  summary="Unsupported claim")

        # Contradiction
        _add_node(graph, "x", agent="alice", node_type=NodeType.PROPOSAL,
                  summary="X is true")
        _add_node(graph, "y", agent="bob", node_type=NodeType.REBUTTAL,
                  summary="X is false")
        _add_edge(graph, "y", "x", EdgeRelation.REFUTES)

        mock_z3 = MagicMock(spec=Z3Backend)
        mock_z3.is_available = True
        mock_z3.language = FormalLanguage.Z3_SMT
        mock_z3.translate = AsyncMock(return_value=None)

        mock_lean = MagicMock(spec=LeanBackend)
        mock_lean.is_available = False
        mock_lean.language = FormalLanguage.LEAN4

        verifier = ArgumentStructureVerifier(
            lean_backend=mock_lean,
            z3_backend=mock_z3,
            strategy=VerificationStrategy.LEAN_WITH_Z3_FALLBACK,
        )
        result = await verifier.verify(graph)

        assert len(result.circular_dependencies) >= 1
        assert len(result.unsupported_conclusions) >= 1
        assert len(result.contradictions) >= 1
        assert result.is_sound is False

    @pytest.mark.asyncio
    async def test_z3_fallback_when_lean_unavailable(self):
        """When Lean is unavailable, Z3 should be used as fallback."""
        graph = _make_graph()
        _add_node(graph, "e1", node_type=NodeType.EVIDENCE, summary="P1")
        _add_node(graph, "c1", node_type=NodeType.PROPOSAL, round_num=1, summary="C1")
        _add_edge(graph, "e1", "c1", EdgeRelation.SUPPORTS)

        mock_lean = MagicMock(spec=LeanBackend)
        mock_lean.is_available = False
        mock_lean.language = FormalLanguage.LEAN4

        mock_z3 = MagicMock(spec=Z3Backend)
        mock_z3.is_available = True
        mock_z3.language = FormalLanguage.Z3_SMT
        mock_z3.translate = AsyncMock(return_value="(check-sat)")
        mock_z3.prove = AsyncMock(return_value=FormalProofResult(
            status=FormalProofStatus.PROOF_FOUND,
            language=FormalLanguage.Z3_SMT,
            translation_confidence=0.8,
        ))

        verifier = ArgumentStructureVerifier(
            lean_backend=mock_lean,
            z3_backend=mock_z3,
            strategy=VerificationStrategy.LEAN_WITH_Z3_FALLBACK,
        )
        result = await verifier.verify(graph)
        assert len(result.valid_chains) == 1
        # Verify Z3 was called
        mock_z3.translate.assert_awaited()

    @pytest.mark.asyncio
    async def test_lean_only_strategy(self):
        """LEAN_ONLY strategy should not try Z3."""
        graph = _make_graph()
        _add_node(graph, "e1", node_type=NodeType.EVIDENCE, summary="P")
        _add_node(graph, "c1", node_type=NodeType.PROPOSAL, round_num=1, summary="C")
        _add_edge(graph, "e1", "c1", EdgeRelation.SUPPORTS)

        mock_lean = MagicMock(spec=LeanBackend)
        mock_lean.is_available = True
        mock_lean.language = FormalLanguage.LEAN4
        mock_lean.translate = AsyncMock(return_value="theorem t : True := trivial")
        mock_lean.prove = AsyncMock(return_value=FormalProofResult(
            status=FormalProofStatus.PROOF_FOUND,
            language=FormalLanguage.LEAN4,
            translation_confidence=0.85,
        ))

        mock_z3 = MagicMock(spec=Z3Backend)
        mock_z3.is_available = True
        mock_z3.language = FormalLanguage.Z3_SMT

        verifier = ArgumentStructureVerifier(
            lean_backend=mock_lean,
            z3_backend=mock_z3,
            strategy=VerificationStrategy.LEAN_ONLY,
        )
        result = await verifier.verify(graph)
        assert len(result.valid_chains) == 1
        mock_z3.translate.assert_not_called()

    @pytest.mark.asyncio
    async def test_z3_only_strategy(self):
        """Z3_ONLY strategy should not try Lean."""
        graph = _make_graph()
        _add_node(graph, "e1", node_type=NodeType.EVIDENCE, summary="P")
        _add_node(graph, "c1", node_type=NodeType.PROPOSAL, round_num=1, summary="C")
        _add_edge(graph, "e1", "c1", EdgeRelation.SUPPORTS)

        mock_lean = MagicMock(spec=LeanBackend)
        mock_lean.is_available = True
        mock_lean.language = FormalLanguage.LEAN4

        mock_z3 = MagicMock(spec=Z3Backend)
        mock_z3.is_available = True
        mock_z3.language = FormalLanguage.Z3_SMT
        mock_z3.translate = AsyncMock(return_value="(check-sat)")
        mock_z3.prove = AsyncMock(return_value=FormalProofResult(
            status=FormalProofStatus.PROOF_FOUND,
            language=FormalLanguage.Z3_SMT,
            translation_confidence=0.75,
        ))

        verifier = ArgumentStructureVerifier(
            lean_backend=mock_lean,
            z3_backend=mock_z3,
            strategy=VerificationStrategy.Z3_ONLY,
        )
        result = await verifier.verify(graph)
        assert len(result.valid_chains) == 1
        mock_lean.translate.assert_not_called()


# ---------------------------------------------------------------------------
# Implication Claim Building
# ---------------------------------------------------------------------------

class TestBuildImplicationClaim:
    """Tests for building formal implication claims."""

    def test_single_premise(self):
        verifier = ArgumentStructureVerifier()
        claim = verifier._build_implication_claim(
            ["All humans are mortal"], "Socrates is mortal"
        )
        assert "If All humans are mortal" in claim
        assert "then Socrates is mortal" in claim

    def test_multiple_premises(self):
        verifier = ArgumentStructureVerifier()
        claim = verifier._build_implication_claim(
            ["All humans are mortal", "Socrates is human"],
            "Socrates is mortal",
        )
        assert "and" in claim
        assert "then Socrates is mortal" in claim

    def test_long_text_truncated(self):
        verifier = ArgumentStructureVerifier()
        long_premise = "A" * 300
        claim = verifier._build_implication_claim([long_premise], "conclusion")
        assert "..." in claim
        # Truncated to 200 chars plus "..."
        assert len(claim) < 300 + 100


# ---------------------------------------------------------------------------
# Manager Integration
# ---------------------------------------------------------------------------

class TestFormalVerificationManagerIntegration:
    """Tests for FormalVerificationManager.verify_argument_structure."""

    @pytest.mark.asyncio
    async def test_manager_verify_argument_structure(self):
        """Manager should delegate to ArgumentStructureVerifier."""
        graph = _make_graph()
        # Just an empty graph to test the integration path
        manager = FormalVerificationManager()

        with patch(
            "aragora.verification.argument_verifier.ArgumentStructureVerifier.verify",
            new_callable=AsyncMock,
        ) as mock_verify:
            mock_verify.return_value = ArgumentVerificationResult(
                total_nodes_analyzed=0,
            )
            result = await manager.verify_argument_structure(graph)
            mock_verify.assert_awaited_once()
            assert isinstance(result, ArgumentVerificationResult)

    @pytest.mark.asyncio
    async def test_manager_passes_backends(self):
        """Manager should pass its own Lean/Z3 backends to the verifier."""
        graph = _make_graph()
        manager = FormalVerificationManager()

        # The manager has backends[0]=Z3, backends[1]=Lean
        with patch(
            "aragora.verification.argument_verifier.ArgumentStructureVerifier.__init__",
            return_value=None,
        ) as mock_init, patch(
            "aragora.verification.argument_verifier.ArgumentStructureVerifier.verify",
            new_callable=AsyncMock,
            return_value=ArgumentVerificationResult(),
        ):
            await manager.verify_argument_structure(graph)
            # Check that init was called with the manager's backends
            call_kwargs = mock_init.call_args[1]
            assert isinstance(call_kwargs["lean_backend"], LeanBackend)
            assert isinstance(call_kwargs["z3_backend"], Z3Backend)


# ---------------------------------------------------------------------------
# VerificationStrategy Enum
# ---------------------------------------------------------------------------

class TestVerificationStrategy:
    """Tests for VerificationStrategy enum values."""

    def test_all_strategies(self):
        assert VerificationStrategy.LEAN_ONLY.value == "lean_only"
        assert VerificationStrategy.Z3_ONLY.value == "z3_only"
        assert VerificationStrategy.LEAN_WITH_Z3_FALLBACK.value == "lean_with_z3_fallback"
        assert VerificationStrategy.AUTO.value == "auto"
