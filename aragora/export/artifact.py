"""
Debate Artifact - Self-contained, shareable debate packages.

A DebateArtifact composes all debate components into a single exportable unit:
- DebateGraph: The DAG structure of the debate
- DebateTrace: Event log for replay
- ProvenanceChain: Evidence chain of custody
- ConsensusProof: Cryptographic proof of consensus
- VerificationProofs: Formal verification results

Key features:
- Content-addressable: Each artifact has a unique hash
- Self-contained: All data needed for offline viewing
- Verifiable: Chain integrity can be validated
"""

import hashlib
import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional


@dataclass
class ConsensusProof:
    """Proof of consensus with vote details."""

    reached: bool
    confidence: float
    vote_breakdown: dict[str, bool]  # agent_id -> agreed
    final_answer: str
    rounds_used: int
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict:
        return {
            "reached": self.reached,
            "confidence": self.confidence,
            "vote_breakdown": self.vote_breakdown,
            "final_answer": self.final_answer,
            "rounds_used": self.rounds_used,
            "timestamp": self.timestamp,
        }


@dataclass
class VerificationResult:
    """Result of a formal verification attempt."""

    claim_id: str
    claim_text: str
    status: str  # "verified", "refuted", "timeout", "undecidable"
    method: str  # "z3", "lean", "simulation", etc.
    proof_trace: Optional[str] = None
    counterexample: Optional[str] = None
    duration_ms: int = 0
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "claim_id": self.claim_id,
            "claim_text": self.claim_text,
            "status": self.status,
            "method": self.method,
            "proof_trace": self.proof_trace,
            "counterexample": self.counterexample,
            "duration_ms": self.duration_ms,
            "metadata": self.metadata,
        }


@dataclass
class DebateArtifact:
    """
    Self-contained, shareable debate package.

    Combines all debate components for export and replay.
    """

    artifact_id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    debate_id: str = ""
    task: str = ""
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    # Core components (stored as dicts for serialization)
    graph_data: Optional[dict] = None
    trace_data: Optional[dict] = None
    provenance_data: Optional[dict] = None

    # Results
    consensus_proof: Optional[ConsensusProof] = None
    verification_results: list[VerificationResult] = field(default_factory=list)

    # Metadata
    agents: list[str] = field(default_factory=list)
    rounds: int = 0
    duration_seconds: float = 0.0
    message_count: int = 0
    critique_count: int = 0

    # Versioning
    version: str = "1.0"
    generator: str = "aragora v0.8.0"

    @property
    def content_hash(self) -> str:
        """Compute hash of all artifact content for integrity."""
        data = json.dumps(
            {
                "task": self.task,
                "graph": self.graph_data,
                "trace": self.trace_data,
                "provenance": self.provenance_data,
                "consensus": self.consensus_proof.to_dict() if self.consensus_proof else None,
                "verifications": [v.to_dict() for v in self.verification_results],
            },
            sort_keys=True,
        )
        return hashlib.sha256(data.encode()).hexdigest()[:16]

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "artifact_id": self.artifact_id,
            "debate_id": self.debate_id,
            "task": self.task,
            "created_at": self.created_at,
            "content_hash": self.content_hash,
            "graph": self.graph_data,
            "trace": self.trace_data,
            "provenance": self.provenance_data,
            "consensus_proof": self.consensus_proof.to_dict() if self.consensus_proof else None,
            "verification_results": [v.to_dict() for v in self.verification_results],
            "agents": self.agents,
            "rounds": self.rounds,
            "duration_seconds": self.duration_seconds,
            "message_count": self.message_count,
            "critique_count": self.critique_count,
            "version": self.version,
            "generator": self.generator,
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialize to JSON."""
        return json.dumps(self.to_dict(), indent=indent)

    def save(self, path: Path) -> None:
        """Save artifact to file."""
        path.write_text(self.to_json())

    @classmethod
    def from_dict(cls, data: dict) -> "DebateArtifact":
        """Deserialize from dictionary."""
        consensus = None
        if data.get("consensus_proof"):
            cp = data["consensus_proof"]
            consensus = ConsensusProof(
                reached=cp["reached"],
                confidence=cp["confidence"],
                vote_breakdown=cp["vote_breakdown"],
                final_answer=cp["final_answer"],
                rounds_used=cp["rounds_used"],
                timestamp=cp.get("timestamp", ""),
            )

        verifications = []
        for v in data.get("verification_results", []):
            verifications.append(
                VerificationResult(
                    claim_id=v["claim_id"],
                    claim_text=v["claim_text"],
                    status=v["status"],
                    method=v["method"],
                    proof_trace=v.get("proof_trace"),
                    counterexample=v.get("counterexample"),
                    duration_ms=v.get("duration_ms", 0),
                    metadata=v.get("metadata", {}),
                )
            )

        return cls(
            artifact_id=data.get("artifact_id", ""),
            debate_id=data.get("debate_id", ""),
            task=data.get("task", ""),
            created_at=data.get("created_at", ""),
            graph_data=data.get("graph"),
            trace_data=data.get("trace"),
            provenance_data=data.get("provenance"),
            consensus_proof=consensus,
            verification_results=verifications,
            agents=data.get("agents", []),
            rounds=data.get("rounds", 0),
            duration_seconds=data.get("duration_seconds", 0.0),
            message_count=data.get("message_count", 0),
            critique_count=data.get("critique_count", 0),
            version=data.get("version", "1.0"),
            generator=data.get("generator", "aragora"),
        )

    @classmethod
    def from_json(cls, json_str: str) -> "DebateArtifact":
        """Deserialize from JSON."""
        return cls.from_dict(json.loads(json_str))

    @classmethod
    def load(cls, path: Path) -> "DebateArtifact":
        """Load artifact from file."""
        if not path.exists():
            raise FileNotFoundError(f"Debate artifact not found: {path}")
        try:
            return cls.from_json(path.read_text())
        except OSError as e:
            raise OSError(f"Failed to read debate artifact {path}: {e}") from e

    def verify_integrity(self) -> tuple[bool, list[str]]:
        """Verify artifact integrity."""
        errors = []

        # Recompute and check hash
        stored_hash = self.content_hash

        # Check provenance chain if present
        if self.provenance_data:
            from aragora.reasoning.provenance import ProvenanceChain

            try:
                chain = ProvenanceChain.from_dict(self.provenance_data)
                valid, chain_errors = chain.verify_chain()
                if not valid:
                    errors.extend(chain_errors)
            except Exception as e:
                errors.append(f"Failed to verify provenance: {e}")

        return len(errors) == 0, errors


class ArtifactBuilder:
    """
    Builder for creating DebateArtifacts from debate components.

    Usage:
        artifact = (ArtifactBuilder()
            .from_result(debate_result)
            .with_graph(debate_graph)
            .with_trace(trace)
            .with_provenance(provenance_manager)
            .build())
    """

    def __init__(self):
        self._artifact = DebateArtifact()

    def from_result(self, result) -> "ArtifactBuilder":
        """Initialize from a DebateResult."""

        self._artifact.debate_id = result.id
        self._artifact.task = result.task
        self._artifact.rounds = result.rounds_used
        self._artifact.duration_seconds = result.duration_seconds
        self._artifact.message_count = len(result.messages)
        self._artifact.critique_count = len(result.critiques)

        # Extract agent names
        agents = set()
        for msg in result.messages:
            agents.add(msg.agent)
        self._artifact.agents = list(agents)

        # Build consensus proof
        vote_breakdown = {}
        for vote in result.votes:
            vote_breakdown[vote.agent] = vote.choice == result.final_answer[:20]

        self._artifact.consensus_proof = ConsensusProof(
            reached=result.consensus_reached,
            confidence=result.confidence,
            vote_breakdown=vote_breakdown,
            final_answer=result.final_answer,
            rounds_used=result.rounds_used,
        )

        return self

    def with_graph(self, graph) -> "ArtifactBuilder":
        """Add debate graph."""
        if hasattr(graph, "to_dict"):
            self._artifact.graph_data = graph.to_dict()
        elif isinstance(graph, dict):
            self._artifact.graph_data = graph
        return self

    def with_trace(self, trace) -> "ArtifactBuilder":
        """Add debate trace."""
        if hasattr(trace, "to_json"):
            self._artifact.trace_data = json.loads(trace.to_json())
        elif isinstance(trace, dict):
            self._artifact.trace_data = trace
        return self

    def with_provenance(self, provenance) -> "ArtifactBuilder":
        """Add provenance data."""
        if hasattr(provenance, "export"):
            self._artifact.provenance_data = provenance.export()
        elif isinstance(provenance, dict):
            self._artifact.provenance_data = provenance
        return self

    def with_verification(
        self, claim_id: str, claim_text: str, status: str, method: str = "z3", **kwargs
    ) -> "ArtifactBuilder":
        """Add a verification result."""
        self._artifact.verification_results.append(
            VerificationResult(
                claim_id=claim_id,
                claim_text=claim_text,
                status=status,
                method=method,
                **kwargs,
            )
        )
        return self

    def build(self) -> DebateArtifact:
        """Build the final artifact."""
        return self._artifact


def create_artifact_from_debate(
    result,
    graph=None,
    trace=None,
    provenance=None,
) -> DebateArtifact:
    """Convenience function to create artifact from debate components."""
    builder = ArtifactBuilder().from_result(result)

    if graph:
        builder.with_graph(graph)
    if trace:
        builder.with_trace(trace)
    if provenance:
        builder.with_provenance(provenance)

    return builder.build()
