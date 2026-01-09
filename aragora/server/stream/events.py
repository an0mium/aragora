"""
Stream event types and data classes.

Defines the types of events emitted during debates and nomic loop execution,
along with the dataclasses for representing events and audience messages.
"""

import json
import time
from dataclasses import dataclass, field
from enum import Enum


class StreamEventType(Enum):
    """Types of events emitted during debates and nomic loop execution."""
    # Debate events
    DEBATE_START = "debate_start"
    ROUND_START = "round_start"
    AGENT_MESSAGE = "agent_message"
    CRITIQUE = "critique"
    VOTE = "vote"
    CONSENSUS = "consensus"
    DEBATE_END = "debate_end"

    # Token streaming events (for real-time response display)
    TOKEN_START = "token_start"      # Agent begins generating response
    TOKEN_DELTA = "token_delta"      # Incremental token(s) received
    TOKEN_END = "token_end"          # Agent finished generating response

    # Nomic loop events
    CYCLE_START = "cycle_start"
    CYCLE_END = "cycle_end"
    PHASE_START = "phase_start"
    PHASE_END = "phase_end"
    TASK_START = "task_start"
    TASK_COMPLETE = "task_complete"
    TASK_RETRY = "task_retry"
    VERIFICATION_START = "verification_start"
    VERIFICATION_RESULT = "verification_result"
    COMMIT = "commit"
    BACKUP_CREATED = "backup_created"
    BACKUP_RESTORED = "backup_restored"
    ERROR = "error"
    LOG_MESSAGE = "log_message"

    # Multi-loop management events
    LOOP_REGISTER = "loop_register"      # New loop instance started
    LOOP_UNREGISTER = "loop_unregister"  # Loop instance ended
    LOOP_LIST = "loop_list"              # List of active loops (sent on connect)

    # Audience participation events
    USER_VOTE = "user_vote"              # Audience member voted
    USER_SUGGESTION = "user_suggestion"  # Audience member submitted suggestion
    AUDIENCE_SUMMARY = "audience_summary"  # Clustered audience input summary
    AUDIENCE_METRICS = "audience_metrics"  # Vote counts, histograms, conviction distribution
    AUDIENCE_DRAIN = "audience_drain"    # Audience events processed by arena

    # Memory/learning events
    MEMORY_RECALL = "memory_recall"      # Historical context retrieved from memory
    INSIGHT_EXTRACTED = "insight_extracted"  # New insight extracted from debate

    # Ranking/leaderboard events (debate consensus feature)
    MATCH_RECORDED = "match_recorded"    # ELO match recorded, leaderboard updated
    LEADERBOARD_UPDATE = "leaderboard_update"  # Periodic leaderboard snapshot
    GROUNDED_VERDICT = "grounded_verdict"  # Evidence-backed verdict with citations
    MOMENT_DETECTED = "moment_detected"  # Significant narrative moment detected

    # Graph debate events (branching/merging visualization)
    GRAPH_NODE_ADDED = "graph_node_added"  # New node added to debate graph
    GRAPH_BRANCH_CREATED = "graph_branch_created"  # New branch created
    GRAPH_BRANCH_MERGED = "graph_branch_merged"  # Branches merged/synthesized

    # Position tracking events
    FLIP_DETECTED = "flip_detected"      # Agent position reversal detected

    # Human intervention breakpoint events
    BREAKPOINT = "breakpoint"            # Human intervention breakpoint triggered
    BREAKPOINT_RESOLVED = "breakpoint_resolved"  # Breakpoint resolved with guidance

    # Mood/sentiment events (Real-Time Debate Drama)
    MOOD_DETECTED = "mood_detected"      # Agent emotional state analyzed
    MOOD_SHIFT = "mood_shift"            # Significant mood change detected
    DEBATE_ENERGY = "debate_energy"      # Overall debate intensity level

    # Capability probe events (Adversarial Testing)
    PROBE_START = "probe_start"          # Probe session started for agent
    PROBE_RESULT = "probe_result"        # Individual probe result
    PROBE_COMPLETE = "probe_complete"    # All probes complete, report ready

    # Deep Audit events (Intensive Multi-Round Analysis)
    AUDIT_START = "audit_start"          # Deep audit session started
    AUDIT_ROUND = "audit_round"          # Audit round completed (1-6)
    AUDIT_FINDING = "audit_finding"      # Individual finding discovered
    AUDIT_CROSS_EXAM = "audit_cross_exam"  # Cross-examination phase
    AUDIT_VERDICT = "audit_verdict"      # Final audit verdict ready

    # Telemetry events (Cognitive Firewall)
    TELEMETRY_THOUGHT = "telemetry_thought"        # Agent thought process (may be redacted)
    TELEMETRY_CAPABILITY = "telemetry_capability"  # Agent capability verification result
    TELEMETRY_REDACTION = "telemetry_redaction"    # Content was redacted (notification only)
    TELEMETRY_DIAGNOSTIC = "telemetry_diagnostic"  # Internal diagnostic info (dev only)


@dataclass
class StreamEvent:
    """A single event in the debate stream."""
    type: StreamEventType
    data: dict
    timestamp: float = field(default_factory=time.time)
    round: int = 0
    agent: str = ""
    loop_id: str = ""  # For multi-loop tracking
    seq: int = 0  # Global sequence number for ordering
    agent_seq: int = 0  # Per-agent sequence number for token ordering

    def to_dict(self) -> dict:
        result = {
            "type": self.type.value,
            "data": self.data,
            "timestamp": self.timestamp,
            "round": self.round,
            "agent": self.agent,
            "seq": self.seq,
            "agent_seq": self.agent_seq,
        }
        if self.loop_id:
            result["loop_id"] = self.loop_id
        return result

    def to_json(self) -> str:
        return json.dumps(self.to_dict())


@dataclass
class AudienceMessage:
    """A message from an audience member (vote or suggestion)."""
    type: str  # "vote" or "suggestion"
    loop_id: str  # Associated nomic loop
    payload: dict  # Message content (e.g., {"choice": "option1"} for votes)
    timestamp: float = field(default_factory=time.time)
    user_id: str = ""  # Optional user identifier


__all__ = [
    "StreamEventType",
    "StreamEvent",
    "AudienceMessage",
]
