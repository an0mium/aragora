"""
Telemetry events for research integrations.

This module defines event types and constructors for tracking the health
and effectiveness of research integration features.
"""

from dataclasses import dataclass, field
from typing import Optional, Any
from datetime import datetime
from enum import Enum


class TelemetryEventType(Enum):
    """Types of telemetry events for research integrations."""

    # Adaptive Stopping
    STABILITY_CHECK = "stability_check"
    EARLY_TERMINATION = "early_termination"
    STABILITY_GATE_TRIGGERED = "stability_gate_triggered"

    # LaRA Routing
    ROUTING_DECISION = "routing_decision"
    ROUTING_FALLBACK = "routing_fallback"
    ROUTING_OVERRIDE = "routing_override"

    # MUSE
    MUSE_CALCULATION = "muse_calculation"
    MUSE_SUBSET_SELECTED = "muse_subset_selected"

    # ThinkPRM
    PRM_STEP_VERIFIED = "prm_step_verified"
    PRM_ERROR_DETECTED = "prm_error_detected"
    PRM_REVISION_TRIGGERED = "prm_revision_triggered"

    # ASCoT
    ASCOT_FRAGILITY_CALCULATED = "ascot_fragility_calculated"
    ASCOT_CRITICAL_ROUND = "ascot_critical_round"

    # A-HMAD
    ROLE_ASSIGNMENT = "role_assignment"
    DIVERSITY_SCORE = "diversity_score"
    TEAM_COMPOSED = "team_composed"

    # GraphRAG
    GRAPH_EXPANSION = "graph_expansion"
    REASONING_PATH_FOUND = "reasoning_path_found"

    # ClaimCheck
    CLAIM_DECOMPOSED = "claim_decomposed"
    CLAIM_VERIFIED = "claim_verified"

    # General
    FEATURE_FLAG_CHECK = "feature_flag_check"
    INTEGRATION_ERROR = "integration_error"
    BENCHMARK_RUN = "benchmark_run"


@dataclass
class TelemetryEvent:
    """Base telemetry event."""

    event_type: TelemetryEventType
    timestamp: datetime = field(default_factory=datetime.utcnow)
    debate_id: str | None = None
    workspace_id: str | None = None
    round_number: int | None = None
    properties: dict[str, Any] = field(default_factory=dict)
    duration_ms: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert event to dictionary for serialization."""
        return {
            "event_type": self.event_type.value,
            "timestamp": self.timestamp.isoformat(),
            "debate_id": self.debate_id,
            "workspace_id": self.workspace_id,
            "round_number": self.round_number,
            "properties": self.properties,
            "duration_ms": self.duration_ms,
        }


# Event constructors for type safety and convenience


def stability_check_event(
    debate_id: str,
    round_number: int,
    is_stable: bool,
    stability_score: float,
    ks_distance: float,
    muse_gated: bool,
    ascot_gated: bool,
    recommendation: str,
    workspace_id: str | None = None,
) -> TelemetryEvent:
    """Create a stability check telemetry event."""
    return TelemetryEvent(
        event_type=TelemetryEventType.STABILITY_CHECK,
        debate_id=debate_id,
        workspace_id=workspace_id,
        round_number=round_number,
        properties={
            "is_stable": is_stable,
            "stability_score": stability_score,
            "ks_distance": ks_distance,
            "muse_gated": muse_gated,
            "ascot_gated": ascot_gated,
            "recommendation": recommendation,
        },
    )


def early_termination_event(
    debate_id: str,
    round_number: int,
    total_rounds_planned: int,
    rounds_saved: int,
    stability_score: float,
    workspace_id: str | None = None,
) -> TelemetryEvent:
    """Create an early termination telemetry event."""
    return TelemetryEvent(
        event_type=TelemetryEventType.EARLY_TERMINATION,
        debate_id=debate_id,
        workspace_id=workspace_id,
        round_number=round_number,
        properties={
            "total_rounds_planned": total_rounds_planned,
            "rounds_saved": rounds_saved,
            "rounds_saved_pct": rounds_saved / total_rounds_planned * 100
            if total_rounds_planned > 0
            else 0,
            "stability_score": stability_score,
        },
    )


def routing_decision_event(
    workspace_id: str,
    query_hash: str,
    selected_mode: str,
    confidence: float,
    doc_tokens: int,
    query_features: dict[str, Any],
    fallback_mode: str | None,
    duration_ms: float,
) -> TelemetryEvent:
    """Create a routing decision telemetry event."""
    return TelemetryEvent(
        event_type=TelemetryEventType.ROUTING_DECISION,
        workspace_id=workspace_id,
        properties={
            "query_hash": query_hash,
            "selected_mode": selected_mode,
            "confidence": confidence,
            "doc_tokens": doc_tokens,
            "is_factual": query_features.get("is_factual"),
            "is_analytical": query_features.get("is_analytical"),
            "query_length": query_features.get("length_tokens"),
            "fallback_mode": fallback_mode,
        },
        duration_ms=duration_ms,
    )


def muse_calculation_event(
    debate_id: str,
    round_number: int,
    consensus_confidence: float,
    divergence_score: float,
    subset_size: int,
    subset_agents: list[str],
    duration_ms: float,
    workspace_id: str | None = None,
) -> TelemetryEvent:
    """Create a MUSE calculation telemetry event."""
    return TelemetryEvent(
        event_type=TelemetryEventType.MUSE_CALCULATION,
        debate_id=debate_id,
        workspace_id=workspace_id,
        round_number=round_number,
        properties={
            "consensus_confidence": consensus_confidence,
            "divergence_score": divergence_score,
            "subset_size": subset_size,
            "subset_agents": subset_agents,
        },
        duration_ms=duration_ms,
    )


def prm_step_verified_event(
    debate_id: str,
    round_number: int,
    step_id: str,
    agent_id: str,
    verdict: str,
    confidence: float,
    duration_ms: float,
    workspace_id: str | None = None,
) -> TelemetryEvent:
    """Create a PRM step verified telemetry event."""
    return TelemetryEvent(
        event_type=TelemetryEventType.PRM_STEP_VERIFIED,
        debate_id=debate_id,
        workspace_id=workspace_id,
        round_number=round_number,
        properties={
            "step_id": step_id,
            "agent_id": agent_id,
            "verdict": verdict,
            "confidence": confidence,
        },
        duration_ms=duration_ms,
    )


def prm_error_detected_event(
    debate_id: str,
    round_number: int,
    step_id: str,
    agent_id: str,
    verdict: str,
    confidence: float,
    is_late_stage: bool,
    suggested_fix: str | None = None,
    workspace_id: str | None = None,
) -> TelemetryEvent:
    """Create a PRM error detected telemetry event."""
    return TelemetryEvent(
        event_type=TelemetryEventType.PRM_ERROR_DETECTED,
        debate_id=debate_id,
        workspace_id=workspace_id,
        round_number=round_number,
        properties={
            "step_id": step_id,
            "agent_id": agent_id,
            "verdict": verdict,
            "confidence": confidence,
            "is_late_stage": is_late_stage,
            "has_suggested_fix": suggested_fix is not None,
        },
    )


def ascot_fragility_event(
    debate_id: str,
    round_number: int,
    total_rounds: int,
    base_fragility: float,
    dependency_depth: int,
    error_risk: float,
    scrutiny_level: str,
    workspace_id: str | None = None,
) -> TelemetryEvent:
    """Create an ASCoT fragility calculation event."""
    return TelemetryEvent(
        event_type=TelemetryEventType.ASCOT_FRAGILITY_CALCULATED,
        debate_id=debate_id,
        workspace_id=workspace_id,
        round_number=round_number,
        properties={
            "total_rounds": total_rounds,
            "base_fragility": base_fragility,
            "dependency_depth": dependency_depth,
            "error_risk": error_risk,
            "scrutiny_level": scrutiny_level,
            "round_position": round_number / total_rounds if total_rounds > 0 else 0,
        },
    )


def role_assignment_event(
    debate_id: str,
    agent_id: str,
    role: str,
    confidence: float,
    workspace_id: str | None = None,
) -> TelemetryEvent:
    """Create a role assignment telemetry event."""
    return TelemetryEvent(
        event_type=TelemetryEventType.ROLE_ASSIGNMENT,
        debate_id=debate_id,
        workspace_id=workspace_id,
        properties={
            "agent_id": agent_id,
            "role": role,
            "confidence": confidence,
        },
    )


def team_composed_event(
    debate_id: str,
    team_size: int,
    diversity_score: float,
    coverage_score: float,
    topic_alignment: float,
    roles: list[dict[str, Any]],
    workspace_id: str | None = None,
) -> TelemetryEvent:
    """Create a team composition telemetry event."""
    return TelemetryEvent(
        event_type=TelemetryEventType.TEAM_COMPOSED,
        debate_id=debate_id,
        workspace_id=workspace_id,
        properties={
            "team_size": team_size,
            "diversity_score": diversity_score,
            "coverage_score": coverage_score,
            "topic_alignment": topic_alignment,
            "roles": roles,
        },
    )


def graph_expansion_event(
    workspace_id: str,
    query_hash: str,
    seed_count: int,
    expanded_count: int,
    edge_count: int,
    max_hops: int,
    duration_ms: float,
) -> TelemetryEvent:
    """Create a graph expansion telemetry event."""
    return TelemetryEvent(
        event_type=TelemetryEventType.GRAPH_EXPANSION,
        workspace_id=workspace_id,
        properties={
            "query_hash": query_hash,
            "seed_count": seed_count,
            "expanded_count": expanded_count,
            "edge_count": edge_count,
            "max_hops": max_hops,
        },
        duration_ms=duration_ms,
    )


def claim_verified_event(
    debate_id: str,
    claim_id: str,
    claim_type: str,
    verified: bool,
    confidence: float,
    verification_method: str,
    supporting_count: int,
    contradicting_count: int,
    workspace_id: str | None = None,
) -> TelemetryEvent:
    """Create a claim verification telemetry event."""
    return TelemetryEvent(
        event_type=TelemetryEventType.CLAIM_VERIFIED,
        debate_id=debate_id,
        workspace_id=workspace_id,
        properties={
            "claim_id": claim_id,
            "claim_type": claim_type,
            "verified": verified,
            "confidence": confidence,
            "verification_method": verification_method,
            "supporting_count": supporting_count,
            "contradicting_count": contradicting_count,
        },
    )


def integration_error_event(
    integration_name: str,
    error_type: str,
    error_message: str,
    debate_id: str | None = None,
    workspace_id: str | None = None,
) -> TelemetryEvent:
    """Create an integration error telemetry event."""
    return TelemetryEvent(
        event_type=TelemetryEventType.INTEGRATION_ERROR,
        debate_id=debate_id,
        workspace_id=workspace_id,
        properties={
            "integration_name": integration_name,
            "error_type": error_type,
            "error_message": error_message[:500],  # Truncate long messages
        },
    )
