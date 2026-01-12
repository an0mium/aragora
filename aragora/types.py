"""
Shared type definitions for the Aragora adversarial validation platform (multi-agent debate engine).

This module provides type aliases and NewTypes for common patterns
across the codebase, improving type safety and IDE support.

Usage:
    from aragora.types import AgentName, DebateId, Score

    def get_agent_score(agent: AgentName, debate_id: DebateId) -> Score:
        ...
"""

from __future__ import annotations

from typing import Any, Callable, NewType, TypeAlias, TypeVar

# === Semantic String Types ===
# Use NewType for strings with specific meaning to catch type errors

AgentName = NewType("AgentName", str)
"""Unique identifier for an agent (e.g., 'claude-3-opus', 'gpt-4o')."""

DebateId = NewType("DebateId", str)
"""Unique identifier for a debate (usually a UUID)."""

MatchId = NewType("MatchId", str)
"""Unique identifier for an ELO match record."""

MemoryId = NewType("MemoryId", str)
"""Unique identifier for a memory entry."""

Slug = NewType("Slug", str)
"""URL-friendly identifier for a debate (e.g., 'debate-123abc')."""

SessionId = NewType("SessionId", str)
"""Unique identifier for a debate session."""

# === Numeric Types ===

Score = NewType("Score", float)
"""A score value, typically 0.0 to 1.0."""

EloRating = NewType("EloRating", float)
"""An ELO rating value (default 1500.0)."""

Confidence = NewType("Confidence", float)
"""A confidence value, 0.0 to 1.0."""

Severity = NewType("Severity", float)
"""A severity rating for critiques, 0.0 to 1.0."""

Weight = NewType("Weight", float)
"""A weight value for voting or scoring."""

# === Common Type Aliases ===

MessageList: TypeAlias = "list[Message]"
"""List of Message objects from a debate."""

VoteDict: TypeAlias = dict[AgentName, Score]
"""Mapping of agent names to their scores."""

EloChanges: TypeAlias = dict[AgentName, float]
"""Mapping of agent names to their ELO rating changes."""

Metadata: TypeAlias = dict[str, Any]
"""Generic metadata dictionary."""

JsonDict: TypeAlias = dict[str, Any]
"""A JSON-serializable dictionary."""

# === Callback Types ===

SuccessFn: TypeAlias = Callable[[str], float]
"""Function that scores a response, returning 0.0 to 1.0."""

ProgressCallback: TypeAlias = Callable[[str, float], None]
"""Callback for progress updates: (message, percentage)."""

# === Generic Type Variables ===

T = TypeVar("T")
"""Generic type variable for containers."""

AgentT = TypeVar("AgentT", bound="Agent")
"""Type variable bound to Agent base class."""

# Avoid circular import - import Message only for type checking
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from aragora.core import Agent, Message

# === Constants ===

DEFAULT_ELO_RATING: EloRating = EloRating(1500.0)
"""Default starting ELO rating for new agents."""

DEFAULT_CONFIDENCE: Confidence = Confidence(0.5)
"""Default confidence level when not specified."""
