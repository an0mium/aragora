"""
Evidence Provenance Chain - Cryptographic provenance tracking for debate evidence.

Ensures evidence integrity and traceability through:
- Content-addressable hashing of all evidence
- Chain of custody tracking from source to conclusion
- Merkle tree verification for batch evidence
- Citation linking and dependency graphs
- Tamper detection through hash chains

Key concepts:
- ProvenanceRecord: Immutable record of evidence origin
- ProvenanceChain: Linked chain of evidence transformations
- CitationGraph: Dependencies between claims and evidence
- ProvenanceVerifier: Validates evidence integrity
"""

import hashlib
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional


class SourceType(Enum):
    """Type of evidence source."""

    AGENT_GENERATED = "agent_generated"
    USER_PROVIDED = "user_provided"
    EXTERNAL_API = "external_api"
    WEB_SEARCH = "web_search"
    DOCUMENT = "document"
    CODE_ANALYSIS = "code_analysis"
    DATABASE = "database"
    COMPUTATION = "computation"
    SYNTHESIS = "synthesis"  # Combined from multiple sources
    AUDIO_TRANSCRIPT = "audio_transcript"  # Transcribed audio/video content
    UNKNOWN = "unknown"


class TransformationType(Enum):
    """How evidence was transformed."""

    ORIGINAL = "original"  # First entry
    QUOTED = "quoted"
    PARAPHRASED = "paraphrased"
    SUMMARIZED = "summarized"
    EXTRACTED = "extracted"
    COMPUTED = "computed"
    AGGREGATED = "aggregated"
    VERIFIED = "verified"
    REFUTED = "refuted"
    AMENDED = "amended"


@dataclass
class ProvenanceRecord:
    """Immutable record of evidence origin and transformations."""

    id: str
    content_hash: str  # SHA-256 of content
    source_type: SourceType
    source_id: str  # Identifier of the source (agent, URL, file, etc.)

    # Content
    content: str
    content_type: str = "text"  # text, code, data, etc.

    # Timing
    timestamp: datetime = field(default_factory=datetime.now)

    # Chain linkage
    previous_hash: Optional[str] = None  # Hash of previous record in chain
    parent_ids: list[str] = field(default_factory=list)  # Multi-source synthesis

    # Transformation
    transformation: TransformationType = TransformationType.ORIGINAL
    transformation_note: str = ""

    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0  # Confidence in provenance accuracy
    verified: bool = False
    verifier_id: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.id:
            self.id = str(uuid.uuid4())[:12]
        if not self.content_hash:
            self.content_hash = self._compute_hash()

    def _compute_hash(self) -> str:
        """Compute content hash."""
        data = f"{self.content}:{self.source_type.value}:{self.source_id}"
        return hashlib.sha256(data.encode()).hexdigest()

    def chain_hash(self) -> str:
        """Compute hash including chain linkage."""
        data = f"{self.content_hash}:{self.previous_hash or 'genesis'}:{self.timestamp.isoformat()}"
        return hashlib.sha256(data.encode()).hexdigest()

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "content_hash": self.content_hash,
            "chain_hash": self.chain_hash(),
            "source_type": self.source_type.value,
            "source_id": self.source_id,
            "content": self.content,
            "content_type": self.content_type,
            "timestamp": self.timestamp.isoformat(),
            "previous_hash": self.previous_hash,
            "parent_ids": self.parent_ids,
            "transformation": self.transformation.value,
            "transformation_note": self.transformation_note,
            "confidence": self.confidence,
            "verified": self.verified,
            "verifier_id": self.verifier_id,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ProvenanceRecord":
        return cls(
            id=data["id"],
            content_hash=data["content_hash"],
            source_type=SourceType(data["source_type"]),
            source_id=data["source_id"],
            content=data["content"],
            content_type=data.get("content_type", "text"),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            previous_hash=data.get("previous_hash"),
            parent_ids=data.get("parent_ids", []),
            transformation=TransformationType(data.get("transformation", "original")),
            transformation_note=data.get("transformation_note", ""),
            confidence=data.get("confidence", 1.0),
            verified=data.get("verified", False),
            verifier_id=data.get("verifier_id"),
            metadata=data.get("metadata", {}),
        )


@dataclass
class Citation:
    """A citation linking a claim to evidence."""

    claim_id: str
    evidence_id: str  # ProvenanceRecord id
    relevance: float = 1.0  # How relevant this evidence is to the claim
    support_type: str = "supports"  # supports, contradicts, contextualizes
    citation_text: str = ""  # The specific quoted portion
    metadata: dict[str, Any] = field(default_factory=dict)


class ProvenanceChain:
    """
    Chain of evidence records with cryptographic linking.

    Each record includes the hash of the previous record,
    creating an immutable audit trail.
    """

    def __init__(self, chain_id: Optional[str] = None):
        self.chain_id = chain_id or str(uuid.uuid4())
        self.records: list[ProvenanceRecord] = []
        self.genesis_hash: Optional[str] = None
        self.created_at = datetime.now()

    def add_record(
        self,
        content: str,
        source_type: SourceType,
        source_id: str,
        transformation: TransformationType = TransformationType.ORIGINAL,
        parent_ids: Optional[list[str]] = None,
        **kwargs,
    ) -> ProvenanceRecord:
        """Add a new record to the chain."""

        previous_hash = None
        if self.records:
            previous_hash = self.records[-1].chain_hash()

        record = ProvenanceRecord(
            id=str(uuid.uuid4())[:12],
            content_hash="",  # Will be computed
            source_type=source_type,
            source_id=source_id,
            content=content,
            previous_hash=previous_hash,
            parent_ids=parent_ids or [],
            transformation=transformation,
            **kwargs,
        )

        self.records.append(record)

        if self.genesis_hash is None:
            self.genesis_hash = record.chain_hash()

        return record

    def verify_chain(self) -> tuple[bool, list[str]]:
        """Verify the integrity of the entire chain."""
        errors = []

        if not self.records:
            return True, []

        # Check genesis
        if self.records[0].previous_hash is not None:
            errors.append("Genesis record has non-null previous_hash")

        # Check chain links
        for i in range(1, len(self.records)):
            expected_hash = self.records[i - 1].chain_hash()
            actual_hash = self.records[i].previous_hash

            if expected_hash != actual_hash:
                errors.append(
                    f"Chain break at record {i}: expected {expected_hash[:16]}, "
                    f"got {actual_hash[:16] if actual_hash else 'None'}"
                )

        # Verify content hashes
        for record in self.records:
            computed = record._compute_hash()
            if computed != record.content_hash:
                errors.append(f"Content hash mismatch for record {record.id}")

        return len(errors) == 0, errors

    def get_record(self, record_id: str) -> Optional[ProvenanceRecord]:
        """Get a record by ID."""
        for record in self.records:
            if record.id == record_id:
                return record
        return None

    def get_ancestry(self, record_id: str) -> list[ProvenanceRecord]:
        """Get all ancestors of a record (full provenance path)."""
        record = self.get_record(record_id)
        if not record:
            return []

        ancestry = [record]

        # Follow parent_ids for multi-source synthesis
        for parent_id in record.parent_ids:
            parent_ancestry = self.get_ancestry(parent_id)
            ancestry.extend(parent_ancestry)

        # Follow chain links
        if record.previous_hash:
            for r in reversed(self.records):
                if r.chain_hash() == record.previous_hash:
                    chain_ancestry = self.get_ancestry(r.id)
                    ancestry.extend(chain_ancestry)
                    break

        return ancestry

    def to_dict(self) -> dict:
        return {
            "chain_id": self.chain_id,
            "genesis_hash": self.genesis_hash,
            "created_at": self.created_at.isoformat(),
            "record_count": len(self.records),
            "records": [r.to_dict() for r in self.records],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ProvenanceChain":
        chain = cls(chain_id=data["chain_id"])
        chain.genesis_hash = data.get("genesis_hash")
        chain.created_at = datetime.fromisoformat(data["created_at"])
        chain.records = [ProvenanceRecord.from_dict(r) for r in data.get("records", [])]
        return chain


class CitationGraph:
    """
    Graph of citations linking claims to evidence.

    Tracks which evidence supports which claims and
    enables dependency analysis.
    """

    def __init__(self) -> None:
        self.citations: dict[str, Citation] = {}  # citation_id -> Citation
        self.claim_citations: dict[str, list[str]] = {}  # claim_id -> [citation_ids]
        self.evidence_citations: dict[str, list[str]] = {}  # evidence_id -> [citation_ids]

    def add_citation(
        self,
        claim_id: str,
        evidence_id: str,
        relevance: float = 1.0,
        support_type: str = "supports",
        citation_text: str = "",
    ) -> Citation:
        """Add a citation linking a claim to evidence."""

        citation = Citation(
            claim_id=claim_id,
            evidence_id=evidence_id,
            relevance=relevance,
            support_type=support_type,
            citation_text=citation_text,
        )

        citation_id = f"{claim_id}:{evidence_id}"
        self.citations[citation_id] = citation

        # Index by claim
        if claim_id not in self.claim_citations:
            self.claim_citations[claim_id] = []
        self.claim_citations[claim_id].append(citation_id)

        # Index by evidence
        if evidence_id not in self.evidence_citations:
            self.evidence_citations[evidence_id] = []
        self.evidence_citations[evidence_id].append(citation_id)

        return citation

    def get_claim_evidence(self, claim_id: str) -> list[Citation]:
        """Get all evidence citations for a claim."""
        citation_ids = self.claim_citations.get(claim_id, [])
        return [self.citations[cid] for cid in citation_ids if cid in self.citations]

    def get_evidence_claims(self, evidence_id: str) -> list[Citation]:
        """Get all claims that cite an evidence."""
        citation_ids = self.evidence_citations.get(evidence_id, [])
        return [self.citations[cid] for cid in citation_ids if cid in self.citations]

    def get_supporting_evidence(self, claim_id: str) -> list[Citation]:
        """Get evidence that supports a claim."""
        return [c for c in self.get_claim_evidence(claim_id) if c.support_type == "supports"]

    def get_contradicting_evidence(self, claim_id: str) -> list[Citation]:
        """Get evidence that contradicts a claim."""
        return [c for c in self.get_claim_evidence(claim_id) if c.support_type == "contradicts"]

    def compute_claim_support_score(self, claim_id: str) -> float:
        """Compute net support score for a claim based on evidence."""
        citations = self.get_claim_evidence(claim_id)

        if not citations:
            return 0.0

        support_score = 0.0
        for citation in citations:
            weight = citation.relevance

            if citation.support_type == "supports":
                support_score += weight
            elif citation.support_type == "contradicts":
                support_score -= weight
            # "contextualizes" doesn't affect score

        return support_score / len(citations)

    def find_circular_dependencies(self) -> list[list[str]]:
        """Find circular citation dependencies (evidence citing claims that cite it)."""
        cycles = []

        # Build adjacency list
        adj: dict[str, list[str]] = {}
        for claim_id, citation_ids in self.claim_citations.items():
            for cid in citation_ids:
                citation = self.citations.get(cid)
                if citation:
                    if claim_id not in adj:
                        adj[claim_id] = []
                    adj[claim_id].append(citation.evidence_id)

        # DFS for cycles
        visited = set()
        rec_stack = set()

        def dfs(node: str, path: list[str]) -> Optional[list[str]]:
            visited.add(node)
            rec_stack.add(node)
            path.append(node)

            for neighbor in adj.get(node, []):
                if neighbor in rec_stack:
                    # Found cycle
                    cycle_start = path.index(neighbor)
                    return path[cycle_start:]
                if neighbor not in visited:
                    result = dfs(neighbor, path.copy())
                    if result:
                        return result

            rec_stack.remove(node)
            return None

        for node in adj:
            if node not in visited:
                cycle = dfs(node, [])
                if cycle:
                    cycles.append(cycle)

        return cycles


class MerkleTree:
    """Merkle tree for efficient batch verification of evidence."""

    def __init__(self, records: Optional[list[ProvenanceRecord]] = None):
        self.leaves: list[str] = []
        self.tree: list[list[str]] = []
        self.root: Optional[str] = None

        if records:
            self.build(records)

    def build(self, records: list[ProvenanceRecord]) -> str:
        """Build the Merkle tree from records."""
        if not records:
            self.root = self._hash("")
            return self.root

        # Create leaf hashes
        self.leaves = [r.content_hash for r in records]

        # Pad to power of 2
        while len(self.leaves) & (len(self.leaves) - 1):
            self.leaves.append(self.leaves[-1])

        # Build tree
        self.tree = [self.leaves]
        current_level = self.leaves

        while len(current_level) > 1:
            next_level = []
            for i in range(0, len(current_level), 2):
                left = current_level[i]
                right = current_level[i + 1] if i + 1 < len(current_level) else left
                next_level.append(self._hash(left + right))
            self.tree.append(next_level)
            current_level = next_level

        self.root = self.tree[-1][0]
        return self.root

    def _hash(self, data: str) -> str:
        return hashlib.sha256(data.encode()).hexdigest()

    def get_proof(self, index: int) -> list[tuple[str, bool]]:
        """Get Merkle proof for a leaf at index. Returns [(sibling_hash, is_left), ...]"""
        if index >= len(self.leaves):
            return []

        proof = []
        for level in range(len(self.tree) - 1):
            level_size = len(self.tree[level])
            sibling_index = index ^ 1  # XOR to get sibling

            if sibling_index < level_size:
                sibling_hash = self.tree[level][sibling_index]
                is_left = index % 2 == 1  # Sibling is left if we're right
                proof.append((sibling_hash, is_left))

            index //= 2

        return proof

    def verify_proof(self, leaf_hash: str, proof: list[tuple[str, bool]], root: str) -> bool:
        """Verify a Merkle proof."""
        current = leaf_hash

        for sibling_hash, is_left in proof:
            if is_left:
                current = self._hash(sibling_hash + current)
            else:
                current = self._hash(current + sibling_hash)

        return current == root


class ProvenanceVerifier:
    """Verifies evidence provenance and integrity."""

    def __init__(self, chain: ProvenanceChain, graph: Optional[CitationGraph] = None):
        self.chain = chain
        self.graph = graph or CitationGraph()

    def verify_record(self, record_id: str) -> tuple[bool, list[str]]:
        """Verify a single record's integrity."""
        errors = []

        record = self.chain.get_record(record_id)
        if not record:
            return False, [f"Record {record_id} not found"]

        # Verify content hash
        computed = record._compute_hash()
        if computed != record.content_hash:
            errors.append(
                f"Content hash mismatch: expected {computed[:16]}, got {record.content_hash[:16]}"
            )

        # Verify chain link
        if record.previous_hash:
            found_prev = False
            for r in self.chain.records:
                if r.chain_hash() == record.previous_hash:
                    found_prev = True
                    break
            if not found_prev:
                errors.append(f"Previous hash {record.previous_hash[:16]} not found in chain")

        # Verify parent links
        for parent_id in record.parent_ids:
            if not self.chain.get_record(parent_id):
                errors.append(f"Parent record {parent_id} not found")

        return len(errors) == 0, errors

    def verify_claim_evidence(self, claim_id: str) -> dict[str, Any]:
        """Verify all evidence supporting a claim."""
        citations = self.graph.get_claim_evidence(claim_id)

        # Use typed variables to avoid mypy object type issues
        verified_count = 0
        failed_count = 0
        support_score = 0.0
        error_list: list[str] = []
        evidence_status: dict[str, str] = {}

        for citation in citations:
            record = self.chain.get_record(citation.evidence_id)
            if not record:
                failed_count += 1
                error_list.append(f"Evidence {citation.evidence_id} not found in chain")
                evidence_status[citation.evidence_id] = "not_found"
                continue

            valid, errors = self.verify_record(citation.evidence_id)
            if valid:
                verified_count += 1
                evidence_status[citation.evidence_id] = "verified"
            else:
                failed_count += 1
                error_list.extend(errors)
                evidence_status[citation.evidence_id] = "failed"

        if citations:
            support_score = self.graph.compute_claim_support_score(claim_id)

        return {
            "claim_id": claim_id,
            "citation_count": len(citations),
            "verified_count": verified_count,
            "failed_count": failed_count,
            "support_score": support_score,
            "errors": error_list,
            "evidence_status": evidence_status,
        }

    def generate_provenance_report(self, record_id: str) -> dict[str, Any]:
        """Generate a full provenance report for a record."""
        record = self.chain.get_record(record_id)
        if not record:
            return {"error": f"Record {record_id} not found"}

        ancestry = self.chain.get_ancestry(record_id)

        return {
            "record_id": record_id,
            "content_hash": record.content_hash,
            "chain_hash": record.chain_hash(),
            "source": {
                "type": record.source_type.value,
                "id": record.source_id,
            },
            "transformation_history": [
                {
                    "id": r.id,
                    "transformation": r.transformation.value,
                    "source_type": r.source_type.value,
                    "timestamp": r.timestamp.isoformat(),
                }
                for r in ancestry
            ],
            "ancestry_depth": len(ancestry),
            "verified": record.verified,
            "confidence": record.confidence,
        }


class ProvenanceManager:
    """High-level manager for evidence provenance in debates."""

    def __init__(self, debate_id: Optional[str] = None):
        self.debate_id = debate_id or str(uuid.uuid4())
        self.chain = ProvenanceChain()
        self.graph = CitationGraph()
        self.verifier = ProvenanceVerifier(self.chain, self.graph)

    def record_evidence(
        self,
        content: str,
        source_type: SourceType,
        source_id: str,
        **kwargs,
    ) -> ProvenanceRecord:
        """Record a new piece of evidence."""
        return self.chain.add_record(
            content=content,
            source_type=source_type,
            source_id=source_id,
            **kwargs,
        )

    def cite_evidence(
        self,
        claim_id: str,
        evidence_id: str,
        relevance: float = 1.0,
        support_type: str = "supports",
        citation_text: str = "",
    ) -> Citation:
        """Create a citation from a claim to evidence."""
        return self.graph.add_citation(
            claim_id=claim_id,
            evidence_id=evidence_id,
            relevance=relevance,
            support_type=support_type,
            citation_text=citation_text,
        )

    def synthesize_evidence(
        self,
        parent_ids: list[str],
        synthesized_content: str,
        synthesizer_id: str,
    ) -> ProvenanceRecord:
        """Create a new evidence record from synthesizing multiple sources."""
        return self.chain.add_record(
            content=synthesized_content,
            source_type=SourceType.SYNTHESIS,
            source_id=synthesizer_id,
            transformation=TransformationType.AGGREGATED,
            parent_ids=parent_ids,
        )

    def verify_chain_integrity(self) -> tuple[bool, list[str]]:
        """Verify the entire provenance chain."""
        return self.chain.verify_chain()

    def get_evidence_provenance(self, evidence_id: str) -> dict[str, Any]:
        """Get full provenance report for evidence."""
        return self.verifier.generate_provenance_report(evidence_id)

    def get_claim_support(self, claim_id: str) -> dict[str, Any]:
        """Get verification status of all evidence supporting a claim."""
        return self.verifier.verify_claim_evidence(claim_id)

    def export(self) -> dict:
        """Export provenance data for persistence."""
        return {
            "debate_id": self.debate_id,
            "chain": self.chain.to_dict(),
            "citations": [
                {
                    "claim_id": c.claim_id,
                    "evidence_id": c.evidence_id,
                    "relevance": c.relevance,
                    "support_type": c.support_type,
                    "citation_text": c.citation_text,
                }
                for c in self.graph.citations.values()
            ],
        }

    @classmethod
    def load(cls, data: dict) -> "ProvenanceManager":
        """Load provenance data from export."""
        manager = cls(debate_id=data["debate_id"])
        manager.chain = ProvenanceChain.from_dict(data["chain"])

        for citation_data in data.get("citations", []):
            manager.graph.add_citation(**citation_data)

        manager.verifier = ProvenanceVerifier(manager.chain, manager.graph)
        return manager
