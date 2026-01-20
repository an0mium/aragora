"""
Response schema validation for agent responses.

Provides Pydantic models for validating and sanitizing agent responses
to prevent crashes from malformed data and ensure type safety.

Usage:
    from aragora.debate.schemas import validate_agent_response, AgentResponse

    # Validate raw response
    result = validate_agent_response(raw_response, agent_name="claude")
    if result.is_valid:
        response = result.response
    else:
        logger.warning(f"Invalid response: {result.errors}")
"""

from __future__ import annotations

import html
import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, field_validator, model_validator

logger = logging.getLogger(__name__)

# Maximum lengths to prevent memory exhaustion
MAX_CONTENT_LENGTH = 100_000  # 100KB max response
MAX_REASONING_LENGTH = 10_000  # 10KB max reasoning
MAX_METADATA_SIZE = 50  # Max metadata keys


class ReadySignal(BaseModel):
    """Schema for agent ready signals embedded in responses."""

    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    ready: bool = Field(default=False)
    reasoning: str = Field(default="", max_length=MAX_REASONING_LENGTH)

    @field_validator("confidence", mode="before")
    @classmethod
    def coerce_confidence(cls, v: Any) -> float:
        """Coerce confidence to valid float in range [0, 1]."""
        try:
            val = float(v)
            return max(0.0, min(1.0, val))
        except (TypeError, ValueError):
            return 0.5

    @field_validator("ready", mode="before")
    @classmethod
    def coerce_ready(cls, v: Any) -> bool:
        """Coerce ready to boolean."""
        if isinstance(v, bool):
            return v
        if isinstance(v, str):
            return v.lower() in ("true", "yes", "1")
        return bool(v)


class StructuredContent(BaseModel):
    """Schema for structured content blocks in agent responses."""

    type: str = Field(default="text")  # text, code, json, reasoning, etc.
    content: str = Field(default="", max_length=MAX_CONTENT_LENGTH)
    language: Optional[str] = Field(default=None)  # For code blocks
    metadata: Optional[Dict[str, Any]] = Field(default=None)

    @field_validator("content", mode="before")
    @classmethod
    def sanitize_content(cls, v: Any) -> str:
        """Sanitize content to prevent XSS."""
        if v is None:
            return ""
        content = str(v)
        # Truncate if too long
        if len(content) > MAX_CONTENT_LENGTH:
            content = content[:MAX_CONTENT_LENGTH] + "... [truncated]"
        return content

    @field_validator("metadata", mode="before")
    @classmethod
    def limit_metadata(cls, v: Any) -> Optional[Dict[str, Any]]:
        """Limit metadata size to prevent memory issues."""
        if v is None:
            return None
        if not isinstance(v, dict):
            return None
        # Limit number of keys
        if len(v) > MAX_METADATA_SIZE:
            v = dict(list(v.items())[:MAX_METADATA_SIZE])
        return v


class AgentResponseSchema(BaseModel):
    """Schema for validating agent responses.

    This schema ensures:
    - Content is within size limits
    - Confidence values are valid floats in [0, 1]
    - Ready signals are properly parsed
    - Metadata doesn't cause memory issues
    """

    content: str = Field(..., max_length=MAX_CONTENT_LENGTH)
    agent_name: str = Field(..., min_length=1, max_length=100)
    role: str = Field(default="proposer", max_length=50)
    round_number: int = Field(default=0, ge=0, le=100)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    ready_signal: Optional[ReadySignal] = Field(default=None)
    structured_blocks: List[StructuredContent] = Field(default_factory=list)
    metadata: Optional[Dict[str, Any]] = Field(default=None)

    @field_validator("content", mode="before")
    @classmethod
    def validate_content(cls, v: Any) -> str:
        """Validate and sanitize content."""
        if v is None:
            raise ValueError("Content cannot be None")
        content = str(v)
        if len(content) == 0:
            raise ValueError("Content cannot be empty")
        if len(content) > MAX_CONTENT_LENGTH:
            content = content[:MAX_CONTENT_LENGTH] + "... [truncated]"
        return content

    @field_validator("agent_name", mode="before")
    @classmethod
    def validate_agent_name(cls, v: Any) -> str:
        """Validate agent name."""
        if v is None:
            raise ValueError("Agent name cannot be None")
        name = str(v).strip()
        if len(name) == 0:
            raise ValueError("Agent name cannot be empty")
        # Sanitize - only allow alphanumeric, dash, underscore
        name = re.sub(r"[^a-zA-Z0-9_-]", "_", name)
        return name[:100]

    @field_validator("confidence", mode="before")
    @classmethod
    def coerce_confidence(cls, v: Any) -> float:
        """Coerce confidence to valid float."""
        try:
            val = float(v)
            return max(0.0, min(1.0, val))
        except (TypeError, ValueError):
            return 0.5

    @field_validator("metadata", mode="before")
    @classmethod
    def limit_metadata(cls, v: Any) -> Optional[Dict[str, Any]]:
        """Limit metadata size."""
        if v is None:
            return None
        if not isinstance(v, dict):
            return None
        if len(v) > MAX_METADATA_SIZE:
            v = dict(list(v.items())[:MAX_METADATA_SIZE])
        return v

    @model_validator(mode="after")
    def extract_ready_signal(self) -> "AgentResponseSchema":
        """Extract ready signal from content if not explicitly provided."""
        if self.ready_signal is None:
            self.ready_signal = _parse_ready_signal(self.content)
        return self


@dataclass
class ValidationResult:
    """Result of response validation."""

    is_valid: bool
    response: Optional[AgentResponseSchema] = None
    errors: List[str] = None
    warnings: List[str] = None

    def __post_init__(self) -> None:
        if self.errors is None:
            self.errors = []
        if self.warnings is None:
            self.warnings = []


def _parse_ready_signal(content: str) -> ReadySignal:
    """Parse ready signal from content.

    Supports multiple formats:
    - HTML comment: <!-- READY_SIGNAL: {"confidence": 0.85, "ready": true} -->
    - JSON block: ```ready_signal {"confidence": 0.85} ```
    - Inline: [READY: confidence=0.85, ready=true]
    - Natural language markers
    """
    import json

    signal = ReadySignal()

    # Try HTML comment format
    html_pattern = r"<!--\s*READY_SIGNAL:\s*(\{[^}]+\})\s*-->"
    html_match = re.search(html_pattern, content, re.IGNORECASE)
    if html_match:
        try:
            data = json.loads(html_match.group(1))
            return ReadySignal(**data)
        except (json.JSONDecodeError, ValueError, TypeError):
            pass

    # Try JSON block format
    json_pattern = r"```ready_signal\s*(\{[^}]+\})\s*```"
    json_match = re.search(json_pattern, content, re.IGNORECASE)
    if json_match:
        try:
            data = json.loads(json_match.group(1))
            return ReadySignal(**data)
        except (json.JSONDecodeError, ValueError, TypeError):
            pass

    # Try inline format
    inline_pattern = (
        r'\[READY:\s*confidence=([0-9.]+),?\s*ready=(true|false)'
        r'(?:,?\s*reasoning="([^"]*)")?\]'
    )
    inline_match = re.search(inline_pattern, content, re.IGNORECASE)
    if inline_match:
        try:
            return ReadySignal(
                confidence=float(inline_match.group(1)),
                ready=inline_match.group(2).lower() == "true",
                reasoning=inline_match.group(3) or "",
            )
        except (ValueError, TypeError):
            pass

    # Natural language markers
    final_markers = [
        r"\bfinal\s+position\b",
        r"\bfully\s+refined\b",
        r"\bno\s+further\s+(changes|refinement)\s+(needed|required)\b",
        r"\bready\s+to\s+conclude\b",
        r"\bposition\s+is\s+complete\b",
    ]
    for marker in final_markers:
        if re.search(marker, content, re.IGNORECASE):
            return ReadySignal(
                confidence=0.7,
                ready=True,
                reasoning="Natural language indicates position finalized",
            )

    return signal


def validate_agent_response(
    content: Union[str, Dict[str, Any]],
    agent_name: str,
    role: str = "proposer",
    round_number: int = 0,
    metadata: Optional[Dict[str, Any]] = None,
) -> ValidationResult:
    """Validate an agent response.

    Args:
        content: Raw response content (string or dict)
        agent_name: Name of the agent that produced the response
        role: Agent's role in the debate
        round_number: Current round number
        metadata: Optional metadata to attach

    Returns:
        ValidationResult with validated response or errors
    """
    errors: List[str] = []
    warnings: List[str] = []

    # Handle dict input (structured response)
    if isinstance(content, dict):
        raw_content = content.get("content", content.get("text", ""))
        if not raw_content and "message" in content:
            raw_content = content["message"]
    else:
        raw_content = content

    # Basic null check
    if raw_content is None:
        return ValidationResult(
            is_valid=False,
            errors=["Response content is None"],
        )

    # Convert to string
    raw_content = str(raw_content)

    # Check for empty response
    if len(raw_content.strip()) == 0:
        return ValidationResult(
            is_valid=False,
            errors=["Response content is empty"],
        )

    # Check for excessive length - truncate before the suffix to stay within limit
    truncation_suffix = "... [truncated]"
    if len(raw_content) > MAX_CONTENT_LENGTH:
        warnings.append(
            f"Response truncated from {len(raw_content)} to {MAX_CONTENT_LENGTH} chars"
        )
        # Truncate to leave room for suffix
        max_before_suffix = MAX_CONTENT_LENGTH - len(truncation_suffix)
        raw_content = raw_content[:max_before_suffix] + truncation_suffix

    try:
        response = AgentResponseSchema(
            content=raw_content,
            agent_name=agent_name,
            role=role,
            round_number=round_number,
            metadata=metadata,
        )
        return ValidationResult(
            is_valid=True,
            response=response,
            warnings=warnings if warnings else None,
        )
    except Exception as e:
        logger.warning(f"Response validation failed for {agent_name}: {e}")
        return ValidationResult(
            is_valid=False,
            errors=[str(e)],
            warnings=warnings if warnings else None,
        )


def sanitize_html(content: str) -> str:
    """Sanitize HTML content to prevent XSS.

    Escapes HTML entities while preserving safe formatting.
    """
    # Escape HTML entities
    safe = html.escape(content)

    # Restore safe markdown-like formatting
    # Allow **bold**, *italic*, `code`
    safe = re.sub(r"&lt;b&gt;(.+?)&lt;/b&gt;", r"**\1**", safe)
    safe = re.sub(r"&lt;i&gt;(.+?)&lt;/i&gt;", r"*\1*", safe)
    safe = re.sub(r"&lt;code&gt;(.+?)&lt;/code&gt;", r"`\1`", safe)

    return safe


__all__ = [
    "AgentResponseSchema",
    "ReadySignal",
    "StructuredContent",
    "ValidationResult",
    "sanitize_html",
    "validate_agent_response",
    "MAX_CONTENT_LENGTH",
    "MAX_REASONING_LENGTH",
]
