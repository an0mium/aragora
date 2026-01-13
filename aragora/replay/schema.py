import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

SCHEMA_VERSION = "1.0"


@dataclass
class ReplayEvent:
    """Single event in the debate timeline."""

    event_id: str
    timestamp: float  # Unix timestamp
    offset_ms: int  # Milliseconds since debate start
    event_type: str  # 'turn', 'vote', 'audience_input', 'phase_change', 'system'
    source: str  # Agent ID, 'system', or user ID
    content: str  # Message text or payload
    metadata: Dict[str, Any] = field(default_factory=dict)  # round, reasoning, loop_id, etc.

    def to_jsonl(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

    @classmethod
    def from_jsonl(cls, line: str) -> "ReplayEvent":
        try:
            data = json.loads(line.strip())
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in replay event: {e}") from e
        return cls(**data)


@dataclass
class ReplayMeta:
    """Metadata for a debate recording."""

    schema_version: str = SCHEMA_VERSION
    debate_id: str = ""
    topic: str = ""
    proposal: str = ""
    agents: List[Dict[str, str]] = field(default_factory=list)
    started_at: str = ""
    ended_at: Optional[str] = None
    duration_ms: Optional[int] = None
    status: str = "in_progress"
    final_verdict: Optional[str] = None
    vote_tally: Dict[str, int] = field(default_factory=dict)
    event_count: int = 0
    tags: List[str] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, ensure_ascii=False)

    @classmethod
    def from_json(cls, data: str) -> "ReplayMeta":
        try:
            parsed = json.loads(data)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in replay metadata: {e}") from e
        return cls(**parsed)
