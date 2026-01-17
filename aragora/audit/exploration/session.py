"""Exploration session state and data types.

Tracks the state of an iterative document exploration session, including:
- Current phase (read, question, trace, verify, synthesize)
- Chunks explored and insights gathered
- Questions asked and references traced
- Understanding confidence scores
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional
import uuid


class ExplorationPhase(str, Enum):
    """Current phase of document exploration."""

    READ = "read"  # Reading a document chunk
    QUESTION = "question"  # Generating follow-up questions
    TRACE = "trace"  # Following cross-document references
    VERIFY = "verify"  # Multi-agent verification of findings
    SYNTHESIZE = "synthesize"  # Building cross-document understanding


@dataclass
class Reference:
    """A reference to another location in the document corpus."""

    source_document: str  # Document containing the reference
    source_chunk: str  # Chunk ID where reference was found
    source_text: str  # The referencing text
    target_description: str  # What the reference points to
    target_document: Optional[str] = None  # Resolved target document
    target_chunk: Optional[str] = None  # Resolved target chunk
    resolved: bool = False
    resolution_notes: str = ""


@dataclass
class Question:
    """A follow-up question generated during exploration."""

    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    text: str = ""
    question_type: str = "clarification"  # clarification, evidence, contradiction, etc.
    priority: float = 0.5  # 0-1, higher = more important
    source_chunk: str = ""  # Chunk that prompted this question
    answered: bool = False
    answer: str = ""
    answer_source: str = ""  # Document/chunk where answer was found


@dataclass
class Insight:
    """An insight extracted during exploration."""

    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    title: str = ""
    description: str = ""
    category: str = "general"  # fact, relationship, contradiction, pattern
    confidence: float = 0.5  # 0-1
    evidence_chunks: list[str] = field(default_factory=list)
    evidence_text: str = ""
    verified_by: list[str] = field(default_factory=list)  # Agent names
    disputed_by: list[str] = field(default_factory=list)
    related_insights: list[str] = field(default_factory=list)  # Insight IDs
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "category": self.category,
            "confidence": self.confidence,
            "evidence_chunks": self.evidence_chunks,
            "evidence_text": self.evidence_text,
            "verified_by": self.verified_by,
            "disputed_by": self.disputed_by,
            "related_insights": self.related_insights,
            "tags": self.tags,
        }


@dataclass
class ChunkUnderstanding:
    """Understanding extracted from a single chunk."""

    chunk_id: str
    document_id: str
    summary: str = ""
    key_facts: list[str] = field(default_factory=list)
    entities: list[str] = field(default_factory=list)  # Named entities
    relationships: list[tuple[str, str, str]] = field(
        default_factory=list
    )  # (entity, relation, entity)
    references_found: list[Reference] = field(default_factory=list)
    questions_raised: list[str] = field(default_factory=list)
    confidence: float = 0.5


@dataclass
class SynthesizedUnderstanding:
    """Cross-document synthesized understanding."""

    summary: str = ""
    key_findings: list[Insight] = field(default_factory=list)
    document_relationships: list[tuple[str, str, str]] = field(
        default_factory=list
    )  # (doc, relation, doc)
    contradictions: list[dict[str, Any]] = field(default_factory=list)
    gaps: list[str] = field(default_factory=list)  # Identified knowledge gaps
    confidence: float = 0.5


@dataclass
class ExplorationSession:
    """Tracks the state of an exploration session.

    Pattern from: AuditSession, DebateContext
    """

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    objective: str = ""
    document_ids: list[str] = field(default_factory=list)

    # Phase tracking
    current_phase: ExplorationPhase = ExplorationPhase.READ
    iteration: int = 0
    max_iterations: int = 10

    # Exploration progress
    chunks_explored: list[str] = field(default_factory=list)
    chunks_pending: list[str] = field(default_factory=list)
    questions_asked: list[Question] = field(default_factory=list)
    questions_pending: list[Question] = field(default_factory=list)
    insights: list[Insight] = field(default_factory=list)
    references_traced: list[Reference] = field(default_factory=list)
    references_pending: list[Reference] = field(default_factory=list)

    # Understanding state
    chunk_understandings: dict[str, ChunkUnderstanding] = field(default_factory=dict)
    synthesized: Optional[SynthesizedUnderstanding] = None
    confidence_scores: dict[str, float] = field(default_factory=dict)

    # Timing
    started_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None

    # Convergence tracking
    convergence_history: list[float] = field(default_factory=list)

    @property
    def is_complete(self) -> bool:
        """Check if exploration is complete."""
        return self.completed_at is not None

    @property
    def progress(self) -> float:
        """Calculate exploration progress (0-1)."""
        total = len(self.chunks_explored) + len(self.chunks_pending)
        if total == 0:
            return 0.0
        return len(self.chunks_explored) / total

    @property
    def overall_confidence(self) -> float:
        """Calculate overall understanding confidence."""
        if not self.confidence_scores:
            return 0.0
        return sum(self.confidence_scores.values()) / len(self.confidence_scores)

    def add_insight(self, insight: Insight) -> None:
        """Add an insight to the session."""
        self.insights.append(insight)

    def add_question(self, question: Question) -> None:
        """Add a question to the pending queue."""
        self.questions_pending.append(question)

    def add_reference(self, reference: Reference) -> None:
        """Add a reference to trace."""
        self.references_pending.append(reference)

    def mark_chunk_explored(self, chunk_id: str, understanding: ChunkUnderstanding) -> None:
        """Mark a chunk as explored."""
        if chunk_id in self.chunks_pending:
            self.chunks_pending.remove(chunk_id)
        if chunk_id not in self.chunks_explored:
            self.chunks_explored.append(chunk_id)
        self.chunk_understandings[chunk_id] = understanding

    def next_chunk(self) -> Optional[str]:
        """Get the next chunk to explore."""
        if self.chunks_pending:
            return self.chunks_pending[0]
        return None

    def next_question(self) -> Optional[Question]:
        """Get the next question to answer."""
        unanswered = [q for q in self.questions_pending if not q.answered]
        if unanswered:
            # Sort by priority
            return max(unanswered, key=lambda q: q.priority)
        return None

    def next_reference(self) -> Optional[Reference]:
        """Get the next reference to trace."""
        unresolved = [r for r in self.references_pending if not r.resolved]
        if unresolved:
            return unresolved[0]
        return None

    def to_checkpoint(self) -> dict[str, Any]:
        """Create a checkpoint for session recovery."""
        return {
            "id": self.id,
            "objective": self.objective,
            "document_ids": self.document_ids,
            "current_phase": self.current_phase.value,
            "iteration": self.iteration,
            "chunks_explored": self.chunks_explored,
            "chunks_pending": self.chunks_pending,
            "insights": [i.to_dict() for i in self.insights],
            "confidence_scores": self.confidence_scores,
            "convergence_history": self.convergence_history,
            "started_at": self.started_at.isoformat(),
        }

    @classmethod
    def from_checkpoint(cls, data: dict[str, Any]) -> "ExplorationSession":
        """Restore session from checkpoint."""
        session = cls(
            id=data["id"],
            objective=data["objective"],
            document_ids=data["document_ids"],
            current_phase=ExplorationPhase(data["current_phase"]),
            iteration=data["iteration"],
            chunks_explored=data["chunks_explored"],
            chunks_pending=data["chunks_pending"],
            confidence_scores=data["confidence_scores"],
            convergence_history=data["convergence_history"],
        )
        session.started_at = datetime.fromisoformat(data["started_at"])

        # Restore insights
        for insight_data in data.get("insights", []):
            session.insights.append(
                Insight(
                    id=insight_data["id"],
                    title=insight_data["title"],
                    description=insight_data["description"],
                    category=insight_data["category"],
                    confidence=insight_data["confidence"],
                    evidence_chunks=insight_data["evidence_chunks"],
                    tags=insight_data.get("tags", []),
                )
            )

        return session


@dataclass
class ExplorationResult:
    """Final result of a document exploration."""

    session_id: str
    objective: str
    insights: list[Insight] = field(default_factory=list)
    synthesized_understanding: Optional[SynthesizedUnderstanding] = None
    questions_answered: int = 0
    questions_unanswered: int = 0
    references_resolved: int = 0
    chunks_explored: int = 0
    iterations: int = 0
    duration_seconds: float = 0.0
    final_confidence: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "session_id": self.session_id,
            "objective": self.objective,
            "insights": [i.to_dict() for i in self.insights],
            "synthesized_summary": (
                self.synthesized_understanding.summary if self.synthesized_understanding else ""
            ),
            "questions_answered": self.questions_answered,
            "questions_unanswered": self.questions_unanswered,
            "references_resolved": self.references_resolved,
            "chunks_explored": self.chunks_explored,
            "iterations": self.iterations,
            "duration_seconds": self.duration_seconds,
            "final_confidence": self.final_confidence,
        }
