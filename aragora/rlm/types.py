"""
Type definitions for Recursive Language Models (RLM).

Based on concepts from arXiv:2512.24601.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional


class AbstractionLevel(Enum):
    """
    Levels of context abstraction in hierarchical representation.

    Level 0 is the most detailed (full content), higher levels are
    progressively more compressed summaries.
    """
    FULL = 0        # Original full content
    DETAILED = 1    # Detailed summary (~50% compression)
    SUMMARY = 2     # Key points summary (~80% compression)
    ABSTRACT = 3    # High-level abstract (~95% compression)
    METADATA = 4    # Tags and routing info only


class DecompositionStrategy(Enum):
    """
    Strategies for decomposing long context.

    From the RLM paper, these emerge as common patterns:
    - PEEK: Inspect initial sections to understand structure
    - GREP: Use regex/keyword searches to narrow relevant context
    - PARTITION_MAP: Chunk context and run recursive calls on segments
    - SUMMARIZE: Process subsets and extract summaries
    - HIERARCHICAL: Navigate pre-built abstraction hierarchy
    """
    PEEK = "peek"
    GREP = "grep"
    PARTITION_MAP = "partition_map"
    SUMMARIZE = "summarize"
    HIERARCHICAL = "hierarchical"
    AUTO = "auto"  # Let RLM decide


@dataclass
class RLMConfig:
    """Configuration for RLM processing."""

    # Model configuration
    root_model: str = "claude"  # Model for root LM
    sub_model: str = "gpt-4o-mini"  # Model for sub-LM calls (cheaper)

    # Recursion limits
    max_depth: int = 2  # Maximum recursion depth
    max_sub_calls: int = 10  # Maximum sub-LM calls per level

    # Context management
    target_tokens: int = 4000  # Target context size for each level
    overlap_tokens: int = 200  # Overlap between chunks

    # Compression settings
    compression_ratio: float = 0.3  # Target compression per level
    preserve_structure: bool = True  # Maintain document structure in compression

    # Strategy selection
    default_strategy: DecompositionStrategy = DecompositionStrategy.AUTO

    # Performance
    parallel_sub_calls: bool = True  # Run sub-LM calls in parallel
    cache_compressions: bool = True  # Cache compression results
    cache_ttl_seconds: int = 3600  # Cache TTL

    # Output format
    include_citations: bool = True  # Include source references
    citation_format: str = "[L{level}:{chunk}]"  # Citation format


@dataclass
class AbstractionNode:
    """
    A node in the hierarchical abstraction tree.

    Each node represents content at a specific abstraction level,
    with references to its children (more detailed) and parent (more abstract).
    """
    id: str
    level: AbstractionLevel
    content: str
    token_count: int

    # Hierarchy
    parent_id: Optional[str] = None
    child_ids: list[str] = field(default_factory=list)

    # Source tracking
    source_range: tuple[int, int] = (0, 0)  # (start_char, end_char) in original
    source_chunks: list[str] = field(default_factory=list)  # IDs of source chunks

    # Metadata
    key_topics: list[str] = field(default_factory=list)
    confidence: float = 1.0

    def __post_init__(self) -> None:
        if not self.id:
            import uuid
            self.id = str(uuid.uuid4())[:8]


@dataclass
class RLMContext:
    """
    Hierarchical context representation for RLM processing.

    Stores content at multiple abstraction levels, enabling efficient
    navigation from high-level summaries to detailed content.
    """
    # Original content
    original_content: str
    original_tokens: int

    # Abstraction hierarchy (level -> list of nodes at that level)
    levels: dict[AbstractionLevel, list[AbstractionNode]] = field(default_factory=dict)

    # Quick lookup
    nodes_by_id: dict[str, AbstractionNode] = field(default_factory=dict)

    # Metadata
    source_type: str = "text"  # text, debate, code, document
    created_at: str = ""
    compression_stats: dict[str, Any] = field(default_factory=dict)

    def get_at_level(self, level: AbstractionLevel) -> str:
        """Get concatenated content at a specific abstraction level."""
        if level not in self.levels:
            return self.original_content
        return "\n\n".join(node.content for node in self.levels[level])

    def get_node(self, node_id: str) -> Optional[AbstractionNode]:
        """Get a specific node by ID."""
        return self.nodes_by_id.get(node_id)

    def drill_down(self, node_id: str) -> list[AbstractionNode]:
        """Get more detailed nodes under a given node."""
        node = self.get_node(node_id)
        if not node:
            return []
        return [self.nodes_by_id[cid] for cid in node.child_ids if cid in self.nodes_by_id]

    def total_tokens_at_level(self, level: AbstractionLevel) -> int:
        """Get total token count at a specific level."""
        if level not in self.levels:
            return self.original_tokens
        return sum(node.token_count for node in self.levels[level])


@dataclass
class CompressionResult:
    """Result of hierarchical compression."""

    context: RLMContext

    # Stats
    original_tokens: int
    compressed_tokens: dict[AbstractionLevel, int]
    compression_ratio: dict[AbstractionLevel, float]

    # Performance
    time_seconds: float
    sub_calls_made: int
    cache_hits: int

    # Quality indicators
    estimated_fidelity: float  # 0-1, how much semantic content preserved
    key_topics_extracted: list[str]


@dataclass
class RLMQuery:
    """A query to execute against hierarchical context."""

    query: str

    # Strategy hints
    preferred_strategy: DecompositionStrategy = DecompositionStrategy.AUTO
    start_level: AbstractionLevel = AbstractionLevel.SUMMARY

    # Constraints
    max_tokens_to_examine: int = 10000
    max_recursion_depth: int = 2

    # Output
    require_citations: bool = True
    output_format: str = "text"  # text, json, structured


@dataclass
class RLMResult:
    """Result of an RLM query.

    Supports Prime Intellect's iterative refinement protocol where the LLM
    can signal incomplete answers via ready=False for progressive improvement.

    Tracks which approach was used:
    - TRUE RLM: REPL-based, model writes code to examine context (preferred)
    - COMPRESSION: Pre-summarization fallback (when `rlm` package not installed)
    """

    answer: str

    # Iterative refinement (Prime Intellect alignment)
    ready: bool = True  # Whether answer is complete (False = needs refinement)
    iteration: int = 0  # Current refinement iteration (0 = first attempt)
    refinement_history: list[str] = field(default_factory=list)  # Prior answers

    # Approach tracking (for debugging/telemetry)
    used_true_rlm: bool = False  # TRUE RLM (REPL-based) was used
    used_compression_fallback: bool = False  # Compression fallback was used

    # Provenance
    nodes_examined: list[str] = field(default_factory=list)  # Node IDs that contributed
    levels_traversed: list[AbstractionLevel] = field(default_factory=list)
    citations: list[dict[str, Any]] = field(default_factory=list)  # [{level, chunk, content}]

    # Stats
    tokens_processed: int = 0
    sub_calls_made: int = 0
    time_seconds: float = 0.0

    # Confidence
    confidence: float = 0.0
    uncertainty_sources: list[str] = field(default_factory=list)  # What might be missing


# Type aliases for callbacks
CompressionCallback = Callable[[str, AbstractionLevel], str]
QueryCallback = Callable[[RLMQuery, RLMContext], RLMResult]


class RLMStreamEventType(Enum):
    """Types of streaming events from RLM operations."""

    # Query lifecycle events
    QUERY_START = "query_start"
    QUERY_COMPLETE = "query_complete"

    # Refinement events
    ITERATION_START = "iteration_start"
    ITERATION_COMPLETE = "iteration_complete"
    FEEDBACK_GENERATED = "feedback_generated"

    # Progress events
    LEVEL_ENTERED = "level_entered"
    NODE_EXAMINED = "node_examined"
    SUB_CALL_START = "sub_call_start"
    SUB_CALL_COMPLETE = "sub_call_complete"

    # Content events
    PARTIAL_ANSWER = "partial_answer"
    FINAL_ANSWER = "final_answer"

    # Status events
    CONFIDENCE_UPDATE = "confidence_update"
    ERROR = "error"


@dataclass
class RLMStreamEvent:
    """
    Event emitted during streaming RLM operations.

    Provides granular visibility into RLM query execution for
    progress tracking, debugging, and real-time UI updates.
    """

    event_type: RLMStreamEventType
    timestamp: float = 0.0

    # Query context
    query: str = ""
    iteration: int = 0

    # Progress data
    level: Optional[AbstractionLevel] = None
    node_id: Optional[str] = None

    # Content data
    content: str = ""
    partial_answer: str = ""

    # Stats
    tokens_processed: int = 0
    sub_calls_made: int = 0
    confidence: float = 0.0

    # Error info
    error: Optional[str] = None

    # Full result (only on completion events)
    result: Optional["RLMResult"] = None

    def __post_init__(self) -> None:
        if self.timestamp == 0.0:
            import time
            self.timestamp = time.time()

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data: dict[str, Any] = {
            "event_type": self.event_type.value,
            "timestamp": self.timestamp,
            "query": self.query,
            "iteration": self.iteration,
            "tokens_processed": self.tokens_processed,
            "sub_calls_made": self.sub_calls_made,
            "confidence": self.confidence,
        }

        if self.level:
            data["level"] = self.level.name
        if self.node_id:
            data["node_id"] = self.node_id
        if self.content:
            data["content"] = self.content
        if self.partial_answer:
            data["partial_answer"] = self.partial_answer
        if self.error:
            data["error"] = self.error
        if self.result:
            data["result"] = {
                "answer": self.result.answer,
                "ready": self.result.ready,
                "iteration": self.result.iteration,
                "confidence": self.result.confidence,
            }

        return data
