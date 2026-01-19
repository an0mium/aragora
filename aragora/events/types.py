"""
Stream event types and data classes.

Defines the types of events emitted during debates and nomic loop execution,
along with the dataclasses for representing events and audience messages.

This module is part of the shared events layer, accessible to all packages
(CLI, debate, memory, server) without creating circular dependencies.
"""

import json
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from typing import Callable


class StreamEventType(Enum):
    """Types of events emitted during debates and nomic loop execution."""

    # Debate events
    DEBATE_START = "debate_start"
    ROUND_START = "round_start"
    AGENT_MESSAGE = "agent_message"
    CRITIQUE = "critique"
    VOTE = "vote"
    CONSENSUS = "consensus"
    SYNTHESIS = "synthesis"  # Explicit synthesis event for guaranteed delivery
    DEBATE_END = "debate_end"

    # Quick preview events (shown in first 5 seconds of debate initialization)
    QUICK_CLASSIFICATION = "quick_classification"  # Haiku classification of question type/domain
    AGENT_PREVIEW = "agent_preview"  # Agent roles, stances, and brief descriptions
    CONTEXT_PREVIEW = "context_preview"  # Pulse/trending summary, research status

    # Token streaming events (for real-time response display)
    TOKEN_START = "token_start"  # Agent begins generating response
    TOKEN_DELTA = "token_delta"  # Incremental token(s) received
    TOKEN_END = "token_end"  # Agent finished generating response

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
    PHASE_TIMEOUT = "phase_timeout"  # Phase timed out - sent to WebSocket clients
    LOG_MESSAGE = "log_message"

    # Multi-loop management events
    LOOP_REGISTER = "loop_register"  # New loop instance started
    LOOP_UNREGISTER = "loop_unregister"  # Loop instance ended
    LOOP_LIST = "loop_list"  # List of active loops (sent on connect)

    # Audience participation events
    USER_VOTE = "user_vote"  # Audience member voted
    USER_SUGGESTION = "user_suggestion"  # Audience member submitted suggestion
    AUDIENCE_SUMMARY = "audience_summary"  # Clustered audience input summary
    AUDIENCE_METRICS = "audience_metrics"  # Vote counts, histograms, conviction distribution
    AUDIENCE_DRAIN = "audience_drain"  # Audience events processed by arena

    # Memory/learning events
    MEMORY_RECALL = "memory_recall"  # Historical context retrieved from memory
    INSIGHT_EXTRACTED = "insight_extracted"  # New insight extracted from debate

    # Ranking/leaderboard events (debate consensus feature)
    MATCH_RECORDED = "match_recorded"  # ELO match recorded, leaderboard updated
    LEADERBOARD_UPDATE = "leaderboard_update"  # Periodic leaderboard snapshot
    GROUNDED_VERDICT = "grounded_verdict"  # Evidence-backed verdict with citations
    MOMENT_DETECTED = "moment_detected"  # Significant narrative moment detected
    AGENT_ELO_UPDATED = "agent_elo_updated"  # Individual agent ELO change

    # Claim verification events
    CLAIM_VERIFICATION_RESULT = "claim_verification_result"  # Claim verification outcome
    FORMAL_VERIFICATION_RESULT = (
        "formal_verification_result"  # Formal proof verification (Lean4/Z3)
    )

    # Memory tier events
    MEMORY_TIER_PROMOTION = "memory_tier_promotion"  # Memory promoted to faster tier
    MEMORY_TIER_DEMOTION = "memory_tier_demotion"  # Memory demoted to slower tier

    # Graph debate events (branching/merging visualization)
    GRAPH_NODE_ADDED = "graph_node_added"  # New node added to debate graph
    GRAPH_BRANCH_CREATED = "graph_branch_created"  # New branch created
    GRAPH_BRANCH_MERGED = "graph_branch_merged"  # Branches merged/synthesized

    # Position tracking events
    FLIP_DETECTED = "flip_detected"  # Agent position reversal detected

    # Feature integration events (data flow from backends to panels)
    TRAIT_EMERGED = "trait_emerged"  # New agent trait detected by PersonaLaboratory
    RISK_WARNING = "risk_warning"  # Domain risk identified
    EVIDENCE_FOUND = "evidence_found"  # Supporting evidence collected
    CALIBRATION_UPDATE = "calibration_update"  # Confidence calibration updated
    GENESIS_EVOLUTION = "genesis_evolution"  # Agent population evolved
    TRAINING_DATA_EXPORTED = "training_data_exported"  # Training data emitted for Tinker

    # Rhetorical analysis events
    RHETORICAL_OBSERVATION = "rhetorical_observation"  # Rhetorical pattern detected

    # Trickster/hollow consensus events
    HOLLOW_CONSENSUS = "hollow_consensus"  # Hollow consensus detected
    TRICKSTER_INTERVENTION = "trickster_intervention"  # Trickster challenge injected

    # Human intervention breakpoint events
    BREAKPOINT = "breakpoint"  # Human intervention breakpoint triggered
    BREAKPOINT_RESOLVED = "breakpoint_resolved"  # Breakpoint resolved with guidance

    # Progress/heartbeat events (for detecting stalled debates)
    HEARTBEAT = "heartbeat"  # Periodic progress indicator
    AGENT_ERROR = "agent_error"  # Agent encountered an error (but debate continues)
    PHASE_PROGRESS = "phase_progress"  # Progress within a phase (e.g., 3/8 agents complete)

    # Mood/sentiment events (Real-Time Debate Drama)
    MOOD_DETECTED = "mood_detected"  # Agent emotional state analyzed
    MOOD_SHIFT = "mood_shift"  # Significant mood change detected
    DEBATE_ENERGY = "debate_energy"  # Overall debate intensity level

    # Capability probe events (Adversarial Testing)
    PROBE_START = "probe_start"  # Probe session started for agent
    PROBE_RESULT = "probe_result"  # Individual probe result
    PROBE_COMPLETE = "probe_complete"  # All probes complete, report ready

    # Deep Audit events (Intensive Multi-Round Analysis)
    AUDIT_START = "audit_start"  # Deep audit session started
    AUDIT_ROUND = "audit_round"  # Audit round completed (1-6)
    AUDIT_FINDING = "audit_finding"  # Individual finding discovered
    AUDIT_CROSS_EXAM = "audit_cross_exam"  # Cross-examination phase
    AUDIT_VERDICT = "audit_verdict"  # Final audit verdict ready

    # Telemetry events (Cognitive Firewall)
    TELEMETRY_THOUGHT = "telemetry_thought"  # Agent thought process (may be redacted)
    TELEMETRY_CAPABILITY = "telemetry_capability"  # Agent capability verification result
    TELEMETRY_REDACTION = "telemetry_redaction"  # Content was redacted (notification only)
    TELEMETRY_DIAGNOSTIC = "telemetry_diagnostic"  # Internal diagnostic info (dev only)

    # Gauntlet events (Adversarial Validation)
    GAUNTLET_START = "gauntlet_start"  # Gauntlet stress-test started
    GAUNTLET_PHASE = "gauntlet_phase"  # Phase transition (redteam, probe, audit, etc.)
    GAUNTLET_AGENT_ACTIVE = "gauntlet_agent_active"  # Agent became active
    GAUNTLET_ATTACK = "gauntlet_attack"  # Red-team attack executed
    GAUNTLET_FINDING = "gauntlet_finding"  # New finding discovered
    GAUNTLET_PROBE = "gauntlet_probe"  # Capability probe result
    GAUNTLET_VERIFICATION = "gauntlet_verification"  # Formal verification result
    GAUNTLET_RISK = "gauntlet_risk"  # Risk assessment update
    GAUNTLET_PROGRESS = "gauntlet_progress"  # Progress update (percentage, etc.)
    GAUNTLET_VERDICT = "gauntlet_verdict"  # Final verdict determined
    GAUNTLET_COMPLETE = "gauntlet_complete"  # Gauntlet stress-test completed

    # Phase 2: Workflow Builder Events
    WORKFLOW_CREATED = "workflow_created"  # New workflow definition created
    WORKFLOW_UPDATED = "workflow_updated"  # Workflow definition updated
    WORKFLOW_DELETED = "workflow_deleted"  # Workflow definition deleted

    WORKFLOW_START = "workflow_start"  # Workflow execution started
    WORKFLOW_STEP_START = "workflow_step_start"  # Step execution started
    WORKFLOW_STEP_PROGRESS = "workflow_step_progress"  # Step progress update
    WORKFLOW_STEP_COMPLETE = "workflow_step_complete"  # Step execution completed
    WORKFLOW_STEP_FAILED = "workflow_step_failed"  # Step execution failed
    WORKFLOW_STEP_SKIPPED = "workflow_step_skipped"  # Step was skipped

    WORKFLOW_TRANSITION = "workflow_transition"  # Transitioning between steps
    WORKFLOW_CHECKPOINT = "workflow_checkpoint"  # Checkpoint created
    WORKFLOW_RESUMED = "workflow_resumed"  # Workflow resumed from checkpoint

    WORKFLOW_HUMAN_APPROVAL_REQUIRED = "workflow_human_approval_required"  # Waiting for human
    WORKFLOW_HUMAN_APPROVAL_RECEIVED = "workflow_human_approval_received"  # Human responded
    WORKFLOW_HUMAN_APPROVAL_TIMEOUT = "workflow_human_approval_timeout"  # Approval timed out

    WORKFLOW_DEBATE_START = "workflow_debate_start"  # Debate step starting
    WORKFLOW_DEBATE_ROUND = "workflow_debate_round"  # Debate round completed
    WORKFLOW_DEBATE_COMPLETE = "workflow_debate_complete"  # Debate step finished

    WORKFLOW_MEMORY_READ = "workflow_memory_read"  # Knowledge Mound query executed
    WORKFLOW_MEMORY_WRITE = "workflow_memory_write"  # Knowledge stored in Mound

    WORKFLOW_COMPLETE = "workflow_complete"  # Workflow execution completed
    WORKFLOW_FAILED = "workflow_failed"  # Workflow execution failed
    WORKFLOW_TERMINATED = "workflow_terminated"  # Workflow manually terminated

    WORKFLOW_METRICS = "workflow_metrics"  # Workflow execution metrics

    # Voice/Transcription events (Speech-to-Text)
    VOICE_START = "voice_start"  # Voice input session started
    VOICE_CHUNK = "voice_chunk"  # Audio chunk received
    VOICE_TRANSCRIPT = "voice_transcript"  # Real-time transcription segment
    VOICE_END = "voice_end"  # Voice input session ended
    TRANSCRIPTION_QUEUED = "transcription_queued"  # File transcription job queued
    TRANSCRIPTION_STARTED = "transcription_started"  # Transcription processing began
    TRANSCRIPTION_PROGRESS = "transcription_progress"  # Transcription progress update
    TRANSCRIPTION_COMPLETE = "transcription_complete"  # Transcription finished
    TRANSCRIPTION_FAILED = "transcription_failed"  # Transcription error


@dataclass
class StreamEvent:
    """A single event in the debate stream.

    Includes distributed tracing fields for correlation across services,
    consistent with DebateEvent in aragora.debate.event_bus.
    """

    type: StreamEventType
    data: dict
    timestamp: float = field(default_factory=time.time)
    round: int = 0
    agent: str = ""
    loop_id: str = ""  # For multi-loop tracking
    seq: int = 0  # Global sequence number for ordering
    agent_seq: int = 0  # Per-agent sequence number for token ordering
    task_id: str = ""  # Unique task identifier for concurrent outputs from same agent
    # Distributed tracing fields for correlation across services
    correlation_id: str = ""  # Links related events across service boundaries
    trace_id: str = ""  # OpenTelemetry-style trace identifier
    span_id: str = ""  # Current operation span

    def __post_init__(self) -> None:
        """Auto-populate tracing fields from current context if not provided."""
        if not self.correlation_id and not self.trace_id:
            try:
                # Lazy import to avoid circular dependency with server layer
                from aragora.server.middleware.tracing import get_trace_id, get_span_id

                self.trace_id = get_trace_id() or ""
                self.span_id = get_span_id() or ""
                self.correlation_id = self.trace_id  # Use trace_id as correlation_id
            except ImportError:
                pass

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
        if self.task_id:
            result["task_id"] = self.task_id
        # Include tracing fields if present
        if self.correlation_id:
            result["correlation_id"] = self.correlation_id
        if self.trace_id:
            result["trace_id"] = self.trace_id
        if self.span_id:
            result["span_id"] = self.span_id
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


@runtime_checkable
class EventEmitter(Protocol):
    """Abstract interface for event emitters.

    This protocol allows different layers to depend on the emitter
    interface without depending on specific implementations like
    SyncEventEmitter in the server layer.
    """

    def emit(self, event: StreamEvent) -> None:
        """Emit an event."""
        ...

    def set_loop_id(self, loop_id: str) -> None:
        """Set the current loop ID for emitted events."""
        ...


__all__ = [
    "StreamEventType",
    "StreamEvent",
    "AudienceMessage",
    "EventEmitter",
]
