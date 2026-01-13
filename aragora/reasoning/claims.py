"""
Typed Claims and Evidence Kernel.

Provides structured reasoning primitives for debates:
- Claims with typed relationships
- Evidence with source tracking
- Logical inference chains
- Confidence propagation
- Argument graphs

This moves aragora beyond chat orchestration into verifiable reasoning.
"""

import json
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Optional, Any, Callable
from enum import Enum


class ClaimType(Enum):
    """Types of claims in a debate."""

    ASSERTION = "assertion"  # A factual claim
    PROPOSAL = "proposal"  # A suggested action/approach
    OBJECTION = "objection"  # A counter-argument
    CONCESSION = "concession"  # Agreeing with opponent's point
    REBUTTAL = "rebuttal"  # Response to objection
    SYNTHESIS = "synthesis"  # Combining multiple claims
    ASSUMPTION = "assumption"  # Unstated premise
    QUESTION = "question"  # Requesting clarification


class RelationType(Enum):
    """Logical relationships between claims."""

    SUPPORTS = "supports"
    CONTRADICTS = "contradicts"
    REFINES = "refines"
    DEPENDS_ON = "depends_on"
    ANSWERS = "answers"
    SUPERSEDES = "supersedes"
    ELABORATES = "elaborates"
    QUALIFIES = "qualifies"  # Adds conditions/exceptions


class EvidenceType(Enum):
    """Types of evidence."""

    ARGUMENT = "argument"  # Logical reasoning
    DATA = "data"  # Empirical data
    CITATION = "citation"  # Reference to external source
    EXAMPLE = "example"  # Illustrative case
    TOOL_OUTPUT = "tool_output"  # Result from tool execution
    CODE = "code"  # Code snippet
    TEST_RESULT = "test_result"  # Test execution result
    EXPERT_OPINION = "expert_opinion"  # Appeal to authority


@dataclass
class SourceReference:
    """Reference to evidence source."""

    source_type: str  # "agent", "tool", "file", "url", "human"
    identifier: str  # Agent name, file path, URL, etc.
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: dict = field(default_factory=dict)


@dataclass
class TypedEvidence:
    """Structured evidence with source tracking."""

    evidence_id: str
    evidence_type: EvidenceType
    content: str
    source: SourceReference
    strength: float  # 0-1, how strongly this supports the claim
    verified: bool = False  # Whether evidence has been verified
    verification_method: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict:
        d = asdict(self)
        d["evidence_type"] = self.evidence_type.value
        return d


@dataclass
class TypedClaim:
    """A structured claim with type, evidence, and relationships."""

    claim_id: str
    claim_type: ClaimType
    statement: str
    author: str
    confidence: float  # 0-1

    # Evidence
    evidence: list[TypedEvidence] = field(default_factory=list)

    # Logical structure
    premises: list[str] = field(default_factory=list)  # Claim IDs this depends on
    conclusion: Optional[str] = None  # What this claim concludes

    # Status
    status: str = "active"  # "active", "challenged", "refuted", "accepted", "withdrawn"
    challenges: list[str] = field(default_factory=list)  # Claim IDs that challenge this

    # Metadata
    round_num: int = 0
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: dict = field(default_factory=dict)

    @property
    def evidence_strength(self) -> float:
        """Calculate aggregate evidence strength."""
        if not self.evidence:
            return 0.0
        return sum(e.strength for e in self.evidence) / len(self.evidence)

    @property
    def adjusted_confidence(self) -> float:
        """Confidence adjusted by evidence strength and challenges."""
        base = self.confidence * (0.5 + 0.5 * self.evidence_strength)
        challenge_penalty = len(self.challenges) * 0.1
        return max(0.0, min(1.0, base - challenge_penalty))

    def to_dict(self) -> dict:
        d = asdict(self)
        d["claim_type"] = self.claim_type.value
        d["evidence"] = [e.to_dict() for e in self.evidence]
        return d


@dataclass
class ClaimRelation:
    """A relationship between two claims."""

    relation_id: str
    source_claim_id: str
    target_claim_id: str
    relation_type: RelationType
    strength: float = 1.0  # How strong is this relationship
    explanation: Optional[str] = None
    created_by: str = ""
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class ArgumentChain:
    """A chain of reasoning from premises to conclusion."""

    chain_id: str
    name: str
    claims: list[str]  # Ordered claim IDs
    relations: list[str]  # Relation IDs connecting them
    conclusion_claim_id: str
    validity: float = 0.0  # 0-1, how valid is this chain
    soundness: float = 0.0  # 0-1, validity + true premises
    author: str = ""


# ===========================================================================
# Fast Claims Extraction (for real-time streaming visualization)
# ===========================================================================

import re
from functools import lru_cache

from aragora.utils.cache_registry import register_lru_cache

# Patterns for quick claim detection (compiled once for performance)
_CLAIM_PATTERNS = {
    ClaimType.PROPOSAL: re.compile(
        r"\b(should|propose|suggest|recommend|let\'s|we could|consider)\b", re.I
    ),
    ClaimType.OBJECTION: re.compile(
        r"\b(however|but|disagree|object|problem|issue|concern|although|whereas)\b", re.I
    ),
    ClaimType.CONCESSION: re.compile(
        r"\b(agree|accept|true|valid|correct|fair point|you\'re right|indeed)\b", re.I
    ),
    ClaimType.QUESTION: re.compile(
        r"\?|^(what|how|why|when|where|who|which|could you|would you)\b", re.I
    ),
    ClaimType.REBUTTAL: re.compile(
        r"\b(actually|in fact|on the contrary|no,|not quite|that\'s not)\b", re.I
    ),
    ClaimType.SYNTHESIS: re.compile(
        r"\b(therefore|thus|in summary|combining|overall|to conclude)\b", re.I
    ),
}


def fast_extract_claims(text: str, author: str = "unknown") -> list[dict]:
    """
    Fast claim extraction using regex patterns.

    Suitable for real-time streaming visualization where low latency is critical.
    For deeper semantic analysis, use ClaimsKernel.extract_claims_from_message().

    Args:
        text: The text to extract claims from
        author: Name of the agent/author

    Returns:
        List of dicts with: type, text, author, confidence
    """
    if not text or len(text) < 10:
        return []

    claims = []
    # Split into sentences (handle multiple punctuation patterns)
    sentences = re.split(r"(?<=[.!?])\s+", text)

    for sentence in sentences:
        sentence = sentence.strip()
        if len(sentence) < 10:
            continue

        # Detect claim type via pattern matching
        claim_type = ClaimType.ASSERTION  # default
        confidence = 0.3

        for ctype, pattern in _CLAIM_PATTERNS.items():
            if pattern.search(sentence):
                claim_type = ctype
                confidence = 0.5
                break

        # Boost confidence for stronger indicators
        if re.search(r"\b(must|definitely|certainly|clearly|obviously)\b", sentence, re.I):
            confidence = min(0.8, confidence + 0.2)
        elif re.search(r"\b(maybe|perhaps|might|possibly|could be)\b", sentence, re.I):
            confidence = max(0.2, confidence - 0.1)

        claims.append(
            {
                "type": claim_type.value,
                "text": sentence[:200],  # Truncate long sentences
                "author": author,
                "confidence": confidence,
            }
        )

    return claims


# Cached version for repeated calls with same text
@register_lru_cache
@lru_cache(maxsize=1000)
def fast_extract_claims_cached(text: str, author: str = "unknown") -> tuple:
    """
    Cached version of fast_extract_claims.

    Returns tuple instead of list for hashability.
    """
    claims = fast_extract_claims(text, author)
    return tuple(tuple(sorted(c.items())) for c in claims)


class ClaimsKernel:
    """
    The reasoning kernel for structured debate.

    Manages claims, evidence, and relationships to enable:
    - Structured argument construction
    - Logical consistency checking
    - Evidence coverage analysis
    - Confidence propagation
    """

    def __init__(self, debate_id: str):
        self.debate_id = debate_id
        self.claims: dict[str, TypedClaim] = {}
        self.relations: dict[str, ClaimRelation] = {}
        self.chains: dict[str, ArgumentChain] = {}
        self._claim_counter = 0
        self._evidence_counter = 0

    def add_claim(
        self,
        statement: str,
        author: str,
        claim_type: ClaimType = ClaimType.ASSERTION,
        confidence: float = 0.5,
        premises: Optional[list[str]] = None,
        round_num: int = 0,
    ) -> TypedClaim:
        """Add a new claim to the kernel."""
        self._claim_counter += 1
        claim = TypedClaim(
            claim_id=f"{self.debate_id}-c{self._claim_counter:04d}",
            claim_type=claim_type,
            statement=statement,
            author=author,
            confidence=confidence,
            premises=premises or [],
            round_num=round_num,
        )
        self.claims[claim.claim_id] = claim
        return claim

    def add_evidence(
        self,
        claim_id: str,
        content: str,
        evidence_type: EvidenceType,
        source_type: str,
        source_id: str,
        strength: float = 0.5,
    ) -> TypedEvidence:
        """Add evidence to a claim."""
        self._evidence_counter += 1
        evidence = TypedEvidence(
            evidence_id=f"{self.debate_id}-e{self._evidence_counter:04d}",
            evidence_type=evidence_type,
            content=content,
            source=SourceReference(
                source_type=source_type,
                identifier=source_id,
            ),
            strength=strength,
        )

        if claim_id in self.claims:
            self.claims[claim_id].evidence.append(evidence)

        return evidence

    def add_relation(
        self,
        source_claim_id: str,
        target_claim_id: str,
        relation_type: RelationType,
        strength: float = 1.0,
        explanation: Optional[str] = None,
        created_by: str = "",
    ) -> ClaimRelation:
        """Add a relationship between claims."""
        relation = ClaimRelation(
            relation_id=f"{self.debate_id}-r{len(self.relations) + 1:04d}",
            source_claim_id=source_claim_id,
            target_claim_id=target_claim_id,
            relation_type=relation_type,
            strength=strength,
            explanation=explanation,
            created_by=created_by,
        )
        self.relations[relation.relation_id] = relation

        # Update claim status for contradictions
        if relation_type == RelationType.CONTRADICTS:
            if target_claim_id in self.claims:
                self.claims[target_claim_id].challenges.append(source_claim_id)
                self.claims[target_claim_id].status = "challenged"

        return relation

    def challenge_claim(
        self,
        claim_id: str,
        challenger: str,
        objection: str,
        evidence: Optional[str] = None,
    ) -> TypedClaim:
        """Challenge an existing claim with an objection."""
        objection_claim = self.add_claim(
            statement=objection,
            author=challenger,
            claim_type=ClaimType.OBJECTION,
            confidence=0.6,
        )

        self.add_relation(
            source_claim_id=objection_claim.claim_id,
            target_claim_id=claim_id,
            relation_type=RelationType.CONTRADICTS,
            created_by=challenger,
        )

        if evidence:
            self.add_evidence(
                claim_id=objection_claim.claim_id,
                content=evidence,
                evidence_type=EvidenceType.ARGUMENT,
                source_type="agent",
                source_id=challenger,
                strength=0.6,
            )

        return objection_claim

    def synthesize_claims(
        self,
        claim_ids: list[str],
        synthesizer: str,
        synthesis: str,
    ) -> TypedClaim:
        """Create a synthesis claim that combines multiple claims."""
        synthesis_claim = self.add_claim(
            statement=synthesis,
            author=synthesizer,
            claim_type=ClaimType.SYNTHESIS,
            confidence=0.7,
            premises=claim_ids,
        )

        for claim_id in claim_ids:
            self.add_relation(
                source_claim_id=claim_id,
                target_claim_id=synthesis_claim.claim_id,
                relation_type=RelationType.SUPPORTS,
                created_by=synthesizer,
            )

        return synthesis_claim

    def get_claim_graph(self) -> dict:
        """Get claims and relations as a graph structure."""
        nodes = [
            {
                "id": c.claim_id,
                "type": c.claim_type.value,
                "statement": c.statement[:100],
                "author": c.author,
                "confidence": c.adjusted_confidence,
                "status": c.status,
            }
            for c in self.claims.values()
        ]

        edges = [
            {
                "source": r.source_claim_id,
                "target": r.target_claim_id,
                "type": r.relation_type.value,
                "strength": r.strength,
            }
            for r in self.relations.values()
        ]

        return {"nodes": nodes, "edges": edges}

    def get_claims(self) -> list[TypedClaim]:
        """Get all claims."""
        return list(self.claims.values())

    def find_unsupported_claims(self) -> list[TypedClaim]:
        """Find claims with no supporting evidence."""
        return [
            c for c in self.claims.values() if not c.evidence and c.claim_type != ClaimType.QUESTION
        ]

    def find_contradictions(self) -> list[tuple[TypedClaim, TypedClaim]]:
        """Find pairs of contradicting claims."""
        contradictions = []
        for relation in self.relations.values():
            if relation.relation_type == RelationType.CONTRADICTS:
                source = self.claims.get(relation.source_claim_id)
                target = self.claims.get(relation.target_claim_id)
                if source and target:
                    contradictions.append((source, target))
        return contradictions

    def find_unaddressed_objections(self) -> list[TypedClaim]:
        """Find objections that haven't been rebutted."""
        objections = [c for c in self.claims.values() if c.claim_type == ClaimType.OBJECTION]

        unaddressed = []
        for obj in objections:
            # Check if any rebuttal targets this objection
            has_rebuttal = any(
                r.target_claim_id == obj.claim_id and r.relation_type == RelationType.CONTRADICTS
                for r in self.relations.values()
            )
            if not has_rebuttal:
                unaddressed.append(obj)

        return unaddressed

    def calculate_claim_strength(self, claim_id: str) -> float:
        """Calculate overall strength of a claim considering all factors."""
        if claim_id not in self.claims:
            return 0.0

        claim = self.claims[claim_id]

        # Start with adjusted confidence
        strength = claim.adjusted_confidence

        # Add support from other claims
        support_relations = [
            r
            for r in self.relations.values()
            if r.target_claim_id == claim_id and r.relation_type == RelationType.SUPPORTS
        ]
        for rel in support_relations:
            source_claim = self.claims.get(rel.source_claim_id)
            if source_claim:
                strength += 0.1 * source_claim.adjusted_confidence * rel.strength

        # Subtract for unaddressed contradictions
        contradictions = [
            r
            for r in self.relations.values()
            if r.target_claim_id == claim_id and r.relation_type == RelationType.CONTRADICTS
        ]
        for rel in contradictions:
            source_claim = self.claims.get(rel.source_claim_id)
            if source_claim and source_claim.status != "refuted":
                strength -= 0.15 * source_claim.adjusted_confidence * rel.strength

        return max(0.0, min(1.0, strength))

    def get_strongest_claims(self, limit: int = 5) -> list[tuple[TypedClaim, float]]:
        """Get the strongest claims by calculated strength."""
        strengths = [
            (claim, self.calculate_claim_strength(claim.claim_id))
            for claim in self.claims.values()
            if claim.claim_type in (ClaimType.ASSERTION, ClaimType.PROPOSAL, ClaimType.SYNTHESIS)
        ]
        return sorted(strengths, key=lambda x: x[1], reverse=True)[:limit]

    def get_evidence_coverage(self) -> dict:
        """Analyze evidence coverage across claims."""
        total_claims = len(self.claims)
        claims_with_evidence = sum(1 for c in self.claims.values() if c.evidence)

        evidence_by_type: dict[str, int] = {}
        for claim in self.claims.values():
            for ev in claim.evidence:
                ev_type = ev.evidence_type.value
                evidence_by_type[ev_type] = evidence_by_type.get(ev_type, 0) + 1

        return {
            "total_claims": total_claims,
            "claims_with_evidence": claims_with_evidence,
            "coverage_ratio": claims_with_evidence / total_claims if total_claims > 0 else 0,
            "evidence_by_type": evidence_by_type,
            "total_evidence": sum(evidence_by_type.values()),
        }

    def to_dict(self) -> dict:
        """Serialize kernel state."""
        return {
            "debate_id": self.debate_id,
            "claims": {k: v.to_dict() for k, v in self.claims.items()},
            "relations": {k: asdict(v) for k, v in self.relations.items()},
            "chains": {k: asdict(v) for k, v in self.chains.items()},
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialize to JSON."""
        data = self.to_dict()
        # Fix enum serialization in relations
        for rel in data["relations"].values():
            rel["relation_type"] = (
                rel["relation_type"].value
                if isinstance(rel["relation_type"], RelationType)
                else rel["relation_type"]
            )
        return json.dumps(data, indent=indent, default=str)

    async def verify_claim_formally(
        self,
        claim_id: str,
        llm_translator: Optional[Callable] = None,
        timeout_seconds: float = 60.0,
    ) -> Optional[TypedEvidence]:
        """
        Attempt formal verification of a claim using Z3.

        If the claim can be expressed in decidable logic, this attempts
        to prove it using the Z3 SMT solver. If successful, adds verified
        evidence to the claim.

        Args:
            claim_id: ID of the claim to verify
            llm_translator: Optional async callable(prompt, context) -> str
                           for LLM-assisted translation to SMT-LIB2
            timeout_seconds: Maximum time for proof search

        Returns:
            TypedEvidence if verification succeeded, None otherwise
        """
        from aragora.verification.formal import (
            Z3Backend,
            FormalProofStatus,
        )

        if claim_id not in self.claims:
            return None

        claim = self.claims[claim_id]

        # Create Z3 backend with optional LLM translator
        backend = Z3Backend(llm_translator=llm_translator)

        if not backend.is_available:
            return None

        # Check if claim is suitable for formal verification
        claim_type_hint = claim.claim_type.value
        if not backend.can_verify(claim.statement, claim_type_hint):
            return None

        # Attempt translation
        context = f"Debate: {self.debate_id}, Author: {claim.author}"
        formal_statement = await backend.translate(claim.statement, context)

        if formal_statement is None:
            return None

        # Attempt proof
        result = await backend.prove(formal_statement, timeout_seconds)

        if result.status == FormalProofStatus.PROOF_FOUND:
            # Create verified evidence
            self._evidence_counter += 1
            evidence = TypedEvidence(
                evidence_id=f"{self.debate_id}-e{self._evidence_counter:04d}",
                evidence_type=EvidenceType.TOOL_OUTPUT,
                content=f"Formally verified: {result.proof_text}",
                source=SourceReference(
                    source_type="formal_verification",
                    identifier=f"{result.language.value}:{result.proof_hash or 'unknown'}",
                    metadata={
                        "prover_version": result.prover_version,
                        "formal_statement": result.formal_statement,
                        "proof_search_time_ms": result.proof_search_time_ms,
                    },
                ),
                strength=1.0,  # Formal proofs have maximum strength
                verified=True,
                verification_method=f"formal:{result.language.value}",
            )

            claim.evidence.append(evidence)
            return evidence

        elif result.status == FormalProofStatus.PROOF_FAILED:
            # Claim is false - add negative evidence
            self._evidence_counter += 1
            evidence = TypedEvidence(
                evidence_id=f"{self.debate_id}-e{self._evidence_counter:04d}",
                evidence_type=EvidenceType.TOOL_OUTPUT,
                content=f"Formally disproven: {result.proof_text}",
                source=SourceReference(
                    source_type="formal_verification",
                    identifier=f"{result.language.value}:counterexample",
                    metadata={
                        "prover_version": result.prover_version,
                        "formal_statement": result.formal_statement,
                        "counterexample": result.proof_text,
                    },
                ),
                strength=0.0,  # Disproven claim
                verified=True,
                verification_method=f"formal:{result.language.value}:disproven",
            )

            claim.evidence.append(evidence)
            claim.status = "refuted"  # Mark as refuted
            return evidence

        return None

    async def verify_all_claims_formally(
        self,
        llm_translator: Optional[Callable] = None,
        timeout_seconds: float = 30.0,
    ) -> dict[str, str]:
        """
        Attempt formal verification of all claims.

        Returns:
            Dict mapping claim_id -> verification status
        """
        results = {}

        for claim_id in self.claims:
            evidence = await self.verify_claim_formally(claim_id, llm_translator, timeout_seconds)

            if evidence is None:
                results[claim_id] = "not_verifiable"
            elif evidence.strength == 1.0:
                results[claim_id] = "verified"
            else:
                results[claim_id] = "disproven"

        return results

    def generate_summary(self) -> str:
        """Generate a text summary of the argument structure."""
        lines = [
            f"# Argument Summary for Debate {self.debate_id}",
            "",
            f"**Total Claims:** {len(self.claims)}",
            f"**Total Relations:** {len(self.relations)}",
            "",
        ]

        # Coverage
        coverage = self.get_evidence_coverage()
        lines.append(f"**Evidence Coverage:** {coverage['coverage_ratio']:.0%}")
        lines.append("")

        # Strongest claims
        lines.append("## Strongest Claims")
        lines.append("")
        for claim, strength in self.get_strongest_claims(5):
            lines.append(f"- [{strength:.0%}] **{claim.author}**: {claim.statement[:100]}...")
        lines.append("")

        # Unaddressed objections
        objections = self.find_unaddressed_objections()
        if objections:
            lines.append("## Unaddressed Objections")
            lines.append("")
            for obj in objections:
                lines.append(f"- **{obj.author}**: {obj.statement[:100]}...")
            lines.append("")

        # Contradictions
        contradictions = self.find_contradictions()
        if contradictions:
            lines.append("## Active Contradictions")
            lines.append("")
            for c1, c2 in contradictions[:5]:
                lines.append(f"- {c1.author} vs {c2.author}")
            lines.append("")

        return "\n".join(lines)
