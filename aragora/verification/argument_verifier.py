"""
Argument Structure Verifier - Lean4 formal verification for debate argument chains.

Takes argument graphs (from ArgumentCartographer) and verifies the logical
structure of reasoning chains: whether conclusions follow from stated premises,
whether there are circular dependencies, and whether premises contradict each other.

Uses LeanBackend for structural proofs and Z3Backend as a fallback for decidable
fragments (arithmetic, propositional logic).

Usage:
    from aragora.verification.argument_verifier import ArgumentStructureVerifier

    verifier = ArgumentStructureVerifier()
    result = await verifier.verify(argument_graph)

    for chain in result.valid_chains:
        print(f"Valid: {chain.name}")
    for chain in result.invalid_chains:
        print(f"Invalid: {chain.name} - {chain.reason}")
"""

from __future__ import annotations

__all__ = [
    "ArgumentStructureVerifier",
    "ArgumentVerificationResult",
    "VerifiedChain",
    "InvalidChain",
    "UnsupportedConclusion",
    "ContradictionPair",
]

import hashlib
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from aragora.verification.formal import (
    FormalProofResult,
    FormalProofStatus,
    LeanBackend,
    Z3Backend,
)
from aragora.visualization.mapper import (
    ArgumentCartographer,
    EdgeRelation,
    NodeType,
)

logger = logging.getLogger(__name__)


class VerificationStrategy(Enum):
    """Strategy for selecting verification backend."""

    LEAN_ONLY = "lean_only"
    Z3_ONLY = "z3_only"
    LEAN_WITH_Z3_FALLBACK = "lean_with_z3_fallback"  # Default
    AUTO = "auto"  # Choose based on claim content


@dataclass
class VerifiedChain:
    """A verified argument chain where the conclusion follows from premises."""

    chain_id: str
    name: str
    premise_node_ids: list[str]
    conclusion_node_id: str
    proof_result: FormalProofResult | None = None
    confidence: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "chain_id": self.chain_id,
            "name": self.name,
            "premise_node_ids": self.premise_node_ids,
            "conclusion_node_id": self.conclusion_node_id,
            "confidence": self.confidence,
            "proof_result": self.proof_result.to_dict() if self.proof_result else None,
        }


@dataclass
class InvalidChain:
    """An argument chain where the conclusion does not follow from premises."""

    chain_id: str
    name: str
    premise_node_ids: list[str]
    conclusion_node_id: str
    reason: str
    proof_result: FormalProofResult | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "chain_id": self.chain_id,
            "name": self.name,
            "premise_node_ids": self.premise_node_ids,
            "conclusion_node_id": self.conclusion_node_id,
            "reason": self.reason,
            "proof_result": self.proof_result.to_dict() if self.proof_result else None,
        }


@dataclass
class UnsupportedConclusion:
    """A conclusion node that has no supporting premises."""

    node_id: str
    summary: str
    agent: str
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "summary": self.summary,
            "agent": self.agent,
            "reason": self.reason,
        }


@dataclass
class ContradictionPair:
    """A pair of claims that contradict each other."""

    node_id_a: str
    node_id_b: str
    summary_a: str
    summary_b: str
    explanation: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id_a": self.node_id_a,
            "node_id_b": self.node_id_b,
            "summary_a": self.summary_a,
            "summary_b": self.summary_b,
            "explanation": self.explanation,
        }


@dataclass
class ArgumentVerificationResult:
    """Complete result of argument structure verification."""

    valid_chains: list[VerifiedChain] = field(default_factory=list)
    invalid_chains: list[InvalidChain] = field(default_factory=list)
    unsupported_conclusions: list[UnsupportedConclusion] = field(default_factory=list)
    contradictions: list[ContradictionPair] = field(default_factory=list)
    circular_dependencies: list[list[str]] = field(default_factory=list)

    # Metadata
    total_nodes_analyzed: int = 0
    total_chains_checked: int = 0
    verification_time_ms: float = 0.0

    @property
    def is_sound(self) -> bool:
        """True if all chains are valid and no contradictions exist."""
        return (
            len(self.invalid_chains) == 0
            and len(self.contradictions) == 0
            and len(self.unsupported_conclusions) == 0
            and len(self.circular_dependencies) == 0
        )

    @property
    def soundness_score(self) -> float:
        """Score from 0.0 to 1.0 representing argument soundness.

        Computed as:
        - valid_chains / total_chains (weighted 0.4)
        - 1 - contradictions / total_nodes (weighted 0.3)
        - 1 - unsupported / total_nodes (weighted 0.3)
        """
        total_chains = len(self.valid_chains) + len(self.invalid_chains)
        if total_chains == 0 and self.total_nodes_analyzed == 0:
            return 1.0  # No arguments to check

        chain_score = len(self.valid_chains) / total_chains if total_chains > 0 else 1.0

        nodes = max(self.total_nodes_analyzed, 1)
        contradiction_score = max(0.0, 1.0 - len(self.contradictions) / nodes)
        unsupported_score = max(0.0, 1.0 - len(self.unsupported_conclusions) / nodes)

        return round(
            chain_score * 0.4 + contradiction_score * 0.3 + unsupported_score * 0.3,
            3,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "valid_chains": [c.to_dict() for c in self.valid_chains],
            "invalid_chains": [c.to_dict() for c in self.invalid_chains],
            "unsupported_conclusions": [u.to_dict() for u in self.unsupported_conclusions],
            "contradictions": [c.to_dict() for c in self.contradictions],
            "circular_dependencies": self.circular_dependencies,
            "is_sound": self.is_sound,
            "soundness_score": self.soundness_score,
            "total_nodes_analyzed": self.total_nodes_analyzed,
            "total_chains_checked": self.total_chains_checked,
            "verification_time_ms": self.verification_time_ms,
        }


class ArgumentStructureVerifier:
    """
    Verifies the logical structure of debate argument graphs.

    Extracts logical claims from an ArgumentCartographer graph, translates
    argument chains into formal propositions, and uses Lean4/Z3 to verify
    that conclusions follow from stated premises.

    Detects:
    - Valid argument chains (conclusion follows from premises)
    - Invalid chains (logical gaps, non-sequiturs)
    - Unsupported conclusions (no premises)
    - Circular dependencies (A supports B supports A)
    - Contradictory premises (P and not-P both asserted)
    """

    def __init__(
        self,
        lean_backend: LeanBackend | None = None,
        z3_backend: Z3Backend | None = None,
        strategy: VerificationStrategy = VerificationStrategy.LEAN_WITH_Z3_FALLBACK,
    ):
        """
        Initialize the argument structure verifier.

        Args:
            lean_backend: Optional LeanBackend instance. Created if None.
            z3_backend: Optional Z3Backend instance. Created if None.
            strategy: Which verification backend(s) to use.
        """
        self._lean_backend = lean_backend or LeanBackend()
        self._z3_backend = z3_backend or Z3Backend()
        self._strategy = strategy

    async def verify(
        self,
        graph: ArgumentCartographer,
        timeout_seconds: float = 30.0,
    ) -> ArgumentVerificationResult:
        """
        Verify the logical structure of an argument graph.

        Performs four checks:
        1. Detect circular dependencies in the argument graph
        2. Find unsupported conclusions (no supporting premises)
        3. Detect contradictory premises
        4. Verify argument chains (conclusion follows from premises)

        Args:
            graph: ArgumentCartographer with populated nodes and edges.
            timeout_seconds: Maximum time for each proof attempt.

        Returns:
            ArgumentVerificationResult with all findings.
        """
        import time

        start = time.time()

        result = ArgumentVerificationResult(
            total_nodes_analyzed=len(graph.nodes),
        )

        if not graph.nodes:
            return result

        # Step 1: Detect circular dependencies
        cycles = self._detect_cycles(graph)
        result.circular_dependencies = cycles

        # Step 2: Find unsupported conclusions
        unsupported = self._find_unsupported_conclusions(graph)
        result.unsupported_conclusions = unsupported

        # Step 3: Detect contradictions
        contradictions = self._detect_contradictions(graph)
        result.contradictions = contradictions

        # Step 4: Extract and verify argument chains
        chains = self._extract_argument_chains(graph)
        result.total_chains_checked = len(chains)

        for chain_premises, chain_conclusion, chain_name in chains:
            verified = await self._verify_chain(
                graph, chain_premises, chain_conclusion, chain_name, timeout_seconds
            )
            if isinstance(verified, VerifiedChain):
                result.valid_chains.append(verified)
            else:
                result.invalid_chains.append(verified)

        result.verification_time_ms = (time.time() - start) * 1000
        return result

    def _detect_cycles(self, graph: ArgumentCartographer) -> list[list[str]]:
        """
        Detect circular dependencies in the argument graph.

        Uses DFS-based cycle detection on support edges.
        Returns list of cycles, where each cycle is a list of node IDs.
        """
        # Build adjacency for SUPPORTS edges (the logical dependency direction)
        adj: dict[str, list[str]] = defaultdict(list)
        for edge in graph.edges:
            if edge.relation == EdgeRelation.SUPPORTS:
                # source supports target means source is a premise for target
                adj[edge.source_id].append(edge.target_id)

        cycles: list[list[str]] = []
        visited: set[str] = set()
        rec_stack: set[str] = set()

        def _dfs(node: str, path: list[str]) -> None:
            visited.add(node)
            rec_stack.add(node)
            path.append(node)

            for neighbor in adj.get(node, []):
                if neighbor not in visited:
                    _dfs(neighbor, path)
                elif neighbor in rec_stack:
                    # Found a cycle: extract cycle from path
                    cycle_start = path.index(neighbor)
                    cycle = path[cycle_start:] + [neighbor]
                    cycles.append(cycle)

            path.pop()
            rec_stack.discard(node)

        for node_id in graph.nodes:
            if node_id not in visited:
                _dfs(node_id, [])

        return cycles

    def _find_unsupported_conclusions(
        self, graph: ArgumentCartographer
    ) -> list[UnsupportedConclusion]:
        """
        Find conclusion/proposal nodes that have no supporting premises.

        A conclusion is unsupported if it is a PROPOSAL or CONSENSUS node
        with no incoming SUPPORTS edges.
        """
        # Build set of nodes that receive support
        supported_nodes: set[str] = set()
        for edge in graph.edges:
            if edge.relation == EdgeRelation.SUPPORTS:
                supported_nodes.add(edge.target_id)

        unsupported: list[UnsupportedConclusion] = []
        for node_id, node in graph.nodes.items():
            # Only check proposals and consensus nodes
            if node.node_type not in (NodeType.PROPOSAL, NodeType.CONSENSUS):
                continue

            # A proposal in the first round is allowed to be unsupported
            # (it's the initial claim)
            if node.node_type == NodeType.PROPOSAL and node.round_num == 0:
                continue

            if node_id not in supported_nodes:
                unsupported.append(
                    UnsupportedConclusion(
                        node_id=node_id,
                        summary=node.summary,
                        agent=node.agent,
                        reason="No supporting evidence or premises found",
                    )
                )

        return unsupported

    def _detect_contradictions(self, graph: ArgumentCartographer) -> list[ContradictionPair]:
        """
        Detect contradictory premises in the argument graph.

        Two nodes contradict if:
        1. One REFUTES the other (explicit contradiction via edge), OR
        2. Both are proposals/assertions from the same round that
           have opposing SUPPORTS/REFUTES relationships to a common target.
        """
        contradictions: list[ContradictionPair] = []
        seen_pairs: set[tuple[str, str]] = set()

        # Method 1: Explicit refutation edges between proposals/evidence
        for edge in graph.edges:
            if edge.relation != EdgeRelation.REFUTES:
                continue

            source = graph.nodes.get(edge.source_id)
            target = graph.nodes.get(edge.target_id)
            if not source or not target:
                continue

            # Only flag as contradiction if both are claim-bearing nodes
            claim_types = {NodeType.PROPOSAL, NodeType.EVIDENCE, NodeType.REBUTTAL}
            if source.node_type not in claim_types or target.node_type not in claim_types:
                continue

            a, b = sorted([edge.source_id, edge.target_id])
            pair_key = (a, b)
            if pair_key in seen_pairs:
                continue
            seen_pairs.add(pair_key)

            contradictions.append(
                ContradictionPair(
                    node_id_a=edge.source_id,
                    node_id_b=edge.target_id,
                    summary_a=source.summary,
                    summary_b=target.summary,
                    explanation=(
                        f"'{source.agent}' refutes '{target.agent}': "
                        f"'{source.summary[:60]}' vs '{target.summary[:60]}'"
                    ),
                )
            )

        # Method 2: Opposing support/refute to same target
        target_supporters: dict[str, list[str]] = defaultdict(list)
        target_refuters: dict[str, list[str]] = defaultdict(list)

        for edge in graph.edges:
            if edge.relation == EdgeRelation.SUPPORTS:
                target_supporters[edge.target_id].append(edge.source_id)
            elif edge.relation == EdgeRelation.REFUTES:
                target_refuters[edge.target_id].append(edge.source_id)

        for target_id in set(target_supporters) & set(target_refuters):
            for supporter_id in target_supporters[target_id]:
                for refuter_id in target_refuters[target_id]:
                    pair_key = tuple(sorted([supporter_id, refuter_id]))
                    if pair_key in seen_pairs:
                        continue
                    seen_pairs.add(pair_key)

                    sup_node = graph.nodes.get(supporter_id)
                    ref_node = graph.nodes.get(refuter_id)
                    if not sup_node or not ref_node:
                        continue

                    contradictions.append(
                        ContradictionPair(
                            node_id_a=supporter_id,
                            node_id_b=refuter_id,
                            summary_a=sup_node.summary,
                            summary_b=ref_node.summary,
                            explanation=(
                                f"Opposing positions on '{target_id}': "
                                f"'{sup_node.agent}' supports while "
                                f"'{ref_node.agent}' refutes"
                            ),
                        )
                    )

        return contradictions

    def _extract_argument_chains(
        self, graph: ArgumentCartographer
    ) -> list[tuple[list[str], str, str]]:
        """
        Extract argument chains from the graph.

        An argument chain is a set of premise nodes connected via SUPPORTS
        edges to a conclusion node (typically a PROPOSAL or CONSENSUS).

        Returns list of (premise_node_ids, conclusion_node_id, chain_name).
        """
        chains: list[tuple[list[str], str, str]] = []

        # Build reverse support map: target -> [sources that support it]
        supporters: dict[str, list[str]] = defaultdict(list)
        for edge in graph.edges:
            if edge.relation == EdgeRelation.SUPPORTS:
                supporters[edge.target_id].append(edge.source_id)

        # Find conclusions: nodes that are supported by other nodes
        for target_id, premise_ids in supporters.items():
            target_node = graph.nodes.get(target_id)
            if not target_node:
                continue

            # Generate a readable chain name
            chain_hash = hashlib.sha256(
                f"{target_id}{''.join(sorted(premise_ids))}".encode()
            ).hexdigest()[:8]
            chain_name = f"chain_{target_node.agent}_{target_node.round_num}_{chain_hash}"

            chains.append((premise_ids, target_id, chain_name))

        return chains

    async def _verify_chain(
        self,
        graph: ArgumentCartographer,
        premise_ids: list[str],
        conclusion_id: str,
        chain_name: str,
        timeout_seconds: float,
    ) -> VerifiedChain | InvalidChain:
        """
        Verify a single argument chain using formal methods.

        Translates the premises and conclusion into a formal implication
        and attempts to prove it using Lean4 or Z3.
        """
        # Gather claim text
        premise_texts = []
        for pid in premise_ids:
            node = graph.nodes.get(pid)
            if node:
                text = node.full_content or node.summary
                premise_texts.append(text)

        conclusion_node = graph.nodes.get(conclusion_id)
        if not conclusion_node:
            return InvalidChain(
                chain_id=chain_name,
                name=chain_name,
                premise_node_ids=premise_ids,
                conclusion_node_id=conclusion_id,
                reason="Conclusion node not found in graph",
            )

        conclusion_text = conclusion_node.full_content or conclusion_node.summary

        if not premise_texts:
            return InvalidChain(
                chain_id=chain_name,
                name=chain_name,
                premise_node_ids=premise_ids,
                conclusion_node_id=conclusion_id,
                reason="No premise texts could be extracted",
            )

        # Build the formal claim: "premises imply conclusion"
        claim = self._build_implication_claim(premise_texts, conclusion_text)

        # Try to verify using the configured strategy
        proof_result = await self._attempt_verification(claim, timeout_seconds)

        if proof_result and proof_result.status == FormalProofStatus.PROOF_FOUND:
            return VerifiedChain(
                chain_id=chain_name,
                name=chain_name,
                premise_node_ids=premise_ids,
                conclusion_node_id=conclusion_id,
                proof_result=proof_result,
                confidence=proof_result.translation_confidence
                if proof_result.translation_confidence > 0
                else 0.5,
            )
        else:
            reason = "Could not verify that conclusion follows from premises"
            if proof_result:
                if proof_result.error_message:
                    reason = proof_result.error_message
                elif proof_result.status == FormalProofStatus.TRANSLATION_FAILED:
                    reason = "Could not translate argument chain to formal language"
                elif proof_result.status == FormalProofStatus.PROOF_FAILED:
                    reason = "Conclusion does not logically follow from stated premises"
                elif proof_result.status == FormalProofStatus.TIMEOUT:
                    reason = "Verification timed out"

            return InvalidChain(
                chain_id=chain_name,
                name=chain_name,
                premise_node_ids=premise_ids,
                conclusion_node_id=conclusion_id,
                reason=reason,
                proof_result=proof_result,
            )

    def _build_implication_claim(self, premise_texts: list[str], conclusion_text: str) -> str:
        """
        Build a natural language implication from premises to conclusion.

        This is the claim that will be translated to formal language.
        Format: "If [P1] and [P2] and ... then [conclusion]"
        """
        # Clean and truncate texts to avoid overwhelming the translator
        max_premise_len = 200
        max_conclusion_len = 200

        cleaned_premises = []
        for p in premise_texts:
            cleaned = p.strip().replace("\n", " ")
            if len(cleaned) > max_premise_len:
                cleaned = cleaned[:max_premise_len] + "..."
            cleaned_premises.append(cleaned)

        cleaned_conclusion = conclusion_text.strip().replace("\n", " ")
        if len(cleaned_conclusion) > max_conclusion_len:
            cleaned_conclusion = cleaned_conclusion[:max_conclusion_len] + "..."

        if len(cleaned_premises) == 1:
            return f"If {cleaned_premises[0]}, then {cleaned_conclusion}"
        else:
            premises_joined = " and ".join(cleaned_premises)
            return f"If {premises_joined}, then {cleaned_conclusion}"

    async def _attempt_verification(
        self, claim: str, timeout_seconds: float
    ) -> FormalProofResult | None:
        """
        Attempt to verify a claim using the configured strategy.

        Tries Lean4 first (for structural proofs), then falls back to Z3
        for decidable fragments.
        """
        if self._strategy == VerificationStrategy.LEAN_ONLY:
            return await self._try_lean(claim, timeout_seconds)

        elif self._strategy == VerificationStrategy.Z3_ONLY:
            return await self._try_z3(claim, timeout_seconds)

        elif self._strategy == VerificationStrategy.LEAN_WITH_Z3_FALLBACK:
            # Try Lean first
            lean_result = await self._try_lean(claim, timeout_seconds)
            if lean_result and lean_result.status == FormalProofStatus.PROOF_FOUND:
                return lean_result

            # Fall back to Z3
            z3_result = await self._try_z3(claim, timeout_seconds)
            if z3_result and z3_result.status == FormalProofStatus.PROOF_FOUND:
                return z3_result

            # Return the most informative failure
            return lean_result or z3_result

        else:  # AUTO
            # Choose based on whether Z3 can handle it
            if self._z3_backend.is_available and self._z3_backend.can_verify(claim):
                z3_result = await self._try_z3(claim, timeout_seconds)
                if z3_result and z3_result.status == FormalProofStatus.PROOF_FOUND:
                    return z3_result

            # Fall back to Lean
            return await self._try_lean(claim, timeout_seconds)

    async def _try_lean(self, claim: str, timeout_seconds: float) -> FormalProofResult | None:
        """Attempt verification using LeanBackend."""
        if not self._lean_backend.is_available:
            return None

        try:
            translated = await self._lean_backend.translate(claim)
            if translated is None:
                return FormalProofResult(
                    status=FormalProofStatus.TRANSLATION_FAILED,
                    language=self._lean_backend.language,
                    error_message="Failed to translate argument to Lean4",
                    original_claim=claim,
                )
            return await self._lean_backend.prove(translated, timeout_seconds)
        except (RuntimeError, ValueError, OSError) as e:
            logger.warning("Lean verification failed: %s", e)
            return FormalProofResult(
                status=FormalProofStatus.BACKEND_UNAVAILABLE,
                language=self._lean_backend.language,
                error_message=f"Lean error: {e}",
            )

    async def _try_z3(self, claim: str, timeout_seconds: float) -> FormalProofResult | None:
        """Attempt verification using Z3Backend."""
        if not self._z3_backend.is_available:
            return None

        try:
            translated = await self._z3_backend.translate(claim)
            if translated is None:
                return FormalProofResult(
                    status=FormalProofStatus.TRANSLATION_FAILED,
                    language=self._z3_backend.language,
                    error_message="Failed to translate argument to SMT-LIB2",
                    original_claim=claim,
                )
            return await self._z3_backend.prove(translated, timeout_seconds)
        except (RuntimeError, ValueError, OSError) as e:
            logger.warning("Z3 verification failed: %s", e)
            return FormalProofResult(
                status=FormalProofStatus.BACKEND_UNAVAILABLE,
                language=self._z3_backend.language,
                error_message=f"Z3 error: {e}",
            )
