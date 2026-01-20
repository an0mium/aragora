"""
RLM-inspired ready signal pattern for agent self-termination.

This module implements the "ready signal" pattern where agents can
signal when they believe further refinement is unnecessary. This enables
early termination when a quorum of agents reach high-confidence positions.

Example agent output format:
```
[My response here...]

<!-- READY_SIGNAL: {"confidence": 0.85, "ready": true, "reasoning": "Position fully refined"} -->
```
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field


# RLM Ready Signal Configuration
# Minimum confidence threshold to consider an agent "ready"
RLM_READY_CONFIDENCE_THRESHOLD = 0.8
# Fraction of agents that must signal ready to terminate early
RLM_READY_QUORUM = 0.75


@dataclass
class AgentReadinessSignal:
    """
    RLM-inspired "ready signal" from an agent.

    Agents can include structured readiness metadata in their responses:
    - confidence: How confident they are in their position (0.0-1.0)
    - ready: Whether they believe further refinement is unnecessary
    - reasoning: Why they feel ready or not ready

    Example agent output format:
    ```
    [My response here...]

    <!-- READY_SIGNAL: {"confidence": 0.85, "ready": true, "reasoning": "Position fully refined"} -->
    ```
    """

    agent: str
    confidence: float = 0.5
    ready: bool = False
    reasoning: str = ""
    round_num: int = 0

    def is_high_confidence(self) -> bool:
        """Check if agent has high confidence in position."""
        return self.confidence >= RLM_READY_CONFIDENCE_THRESHOLD

    def should_terminate(self) -> bool:
        """Check if this signal indicates termination readiness."""
        return self.ready and self.is_high_confidence()


@dataclass
class CollectiveReadiness:
    """Tracks readiness across all agents using RLM ready signal pattern."""

    signals: dict[str, AgentReadinessSignal] = field(default_factory=dict)
    round_num: int = 0

    @property
    def ready_count(self) -> int:
        """Count of agents signaling ready with high confidence."""
        return sum(1 for s in self.signals.values() if s.should_terminate())

    @property
    def total_count(self) -> int:
        """Total agents with signals."""
        return len(self.signals)

    @property
    def avg_confidence(self) -> float:
        """Average confidence across all agents."""
        if not self.signals:
            return 0.0
        return sum(s.confidence for s in self.signals.values()) / len(self.signals)

    def has_quorum(self) -> bool:
        """Check if enough agents are ready to terminate."""
        if self.total_count == 0:
            return False
        return (self.ready_count / self.total_count) >= RLM_READY_QUORUM

    def update(self, signal: AgentReadinessSignal) -> None:
        """Update with new agent signal."""
        self.signals[signal.agent] = signal
        self.round_num = max(self.round_num, signal.round_num)


def parse_ready_signal(agent_name: str, content: str, round_num: int) -> AgentReadinessSignal:
    """
    Parse RLM ready signal from agent response content.

    Looks for structured metadata in the response:
    - HTML comment format: <!-- READY_SIGNAL: {...} -->
    - JSON block format: ```ready_signal {...} ```
    - Inline markers: [READY: confidence=0.85, ready=true]

    Args:
        agent_name: Name of the agent
        content: Full response content from agent
        round_num: Current round number

    Returns:
        AgentReadinessSignal with parsed values or defaults
    """
    signal = AgentReadinessSignal(agent=agent_name, round_num=round_num)

    if not content:
        return signal

    # Try HTML comment format first (preferred)
    # <!-- READY_SIGNAL: {"confidence": 0.85, "ready": true, "reasoning": "..."} -->
    html_pattern = r'<!--\s*READY_SIGNAL:\s*(\{[^}]+\})\s*-->'
    html_match = re.search(html_pattern, content, re.IGNORECASE)
    if html_match:
        try:
            data = json.loads(html_match.group(1))
            signal.confidence = float(data.get("confidence", 0.5))
            signal.ready = bool(data.get("ready", False))
            signal.reasoning = str(data.get("reasoning", ""))
            return signal
        except (json.JSONDecodeError, ValueError, TypeError):
            pass

    # Try JSON code block format
    # ```ready_signal {"confidence": 0.85, "ready": true} ```
    json_pattern = r'```ready_signal\s*(\{[^}]+\})\s*```'
    json_match = re.search(json_pattern, content, re.IGNORECASE)
    if json_match:
        try:
            data = json.loads(json_match.group(1))
            signal.confidence = float(data.get("confidence", 0.5))
            signal.ready = bool(data.get("ready", False))
            signal.reasoning = str(data.get("reasoning", ""))
            return signal
        except (json.JSONDecodeError, ValueError, TypeError):
            pass

    # Try inline marker format
    # [READY: confidence=0.85, ready=true]
    inline_pattern = r'\[READY:\s*confidence=([0-9.]+),?\s*ready=(true|false)(?:,?\s*reasoning="([^"]*)")?\]'
    inline_match = re.search(inline_pattern, content, re.IGNORECASE)
    if inline_match:
        try:
            signal.confidence = float(inline_match.group(1))
            signal.ready = inline_match.group(2).lower() == "true"
            signal.reasoning = inline_match.group(3) or ""
            return signal
        except (ValueError, TypeError):
            pass

    # Heuristic: Check for explicit "final position" or "ready" statements
    # This allows natural language signaling without structured metadata
    final_markers = [
        r"\bfinal\s+position\b",
        r"\bfully\s+refined\b",
        r"\bno\s+further\s+(changes|refinement|revision)\s+(needed|required|necessary)\b",
        r"\bready\s+to\s+conclude\b",
        r"\bposition\s+is\s+complete\b",
    ]
    for marker in final_markers:
        if re.search(marker, content, re.IGNORECASE):
            # Natural language indicates readiness - set moderate confidence
            signal.confidence = 0.7
            signal.ready = True
            signal.reasoning = "Natural language indicates position finalized"
            break

    return signal
