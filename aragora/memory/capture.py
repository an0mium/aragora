"""
Tool-level memory capture policies and helpers.

Provides opt-in capture controls for tool/capability usage events so operators
can decide which tools get written into organizational memory.
"""

from __future__ import annotations

import os
import time
from collections import deque
from dataclasses import dataclass, field


def _parse_list_env(name: str) -> list[str]:
    raw = os.environ.get(name, "")
    return [item.strip() for item in raw.split(",") if item.strip()]


def _normalize_name(value: str, case_sensitive: bool) -> str:
    return value if case_sensitive else value.lower()


@dataclass(frozen=True)
class ToolCapturePolicy:
    """Policy controlling which tools/capabilities are captured."""

    allowlist: tuple[str, ...] = ()
    denylist: tuple[str, ...] = ()
    case_sensitive: bool = False

    def should_capture(self, tool_name: str | None) -> bool:
        if not tool_name:
            return False
        normalized = _normalize_name(tool_name, self.case_sensitive)
        if self.allowlist:
            return normalized in self.allowlist
        if self.denylist:
            return normalized not in self.denylist
        return True

    @classmethod
    def from_env(cls) -> ToolCapturePolicy:
        allow = _parse_list_env("ARAGORA_MEMORY_CAPTURE_TOOLS")
        deny = _parse_list_env("ARAGORA_MEMORY_SKIP_TOOLS")
        case_sensitive_raw = os.environ.get("ARAGORA_MEMORY_CAPTURE_CASE_SENSITIVE", "false")
        case_sensitive = case_sensitive_raw.lower() == "true"
        return cls(
            allowlist=tuple(_normalize_name(v, case_sensitive) for v in allow),
            denylist=tuple(_normalize_name(v, case_sensitive) for v in deny),
            case_sensitive=case_sensitive,
        )


@dataclass
class ToolCaptureLimiter:
    """Simple sliding-window limiter for tool capture."""

    max_per_minute: int = 120
    _events: deque[float] = field(default_factory=deque)

    def allow(self) -> bool:
        if self.max_per_minute <= 0:
            return False
        now = time.time()
        cutoff = now - 60.0
        while self._events and self._events[0] < cutoff:
            self._events.popleft()
        if len(self._events) >= self.max_per_minute:
            return False
        self._events.append(now)
        return True


@dataclass
class ToolMemoryCaptureConfig:
    """Configuration for tool event memory capture."""

    enabled: bool = False
    tier: str = "fast"
    importance: float = 0.4
    max_per_minute: int = 120
    max_detail_chars: int = 800

    @classmethod
    def from_env(cls) -> ToolMemoryCaptureConfig:
        enabled = os.environ.get("ARAGORA_MEMORY_CAPTURE_ENABLED", "false").lower() == "true"
        tier = os.environ.get("ARAGORA_MEMORY_CAPTURE_TIER", "fast")
        importance = float(os.environ.get("ARAGORA_MEMORY_CAPTURE_IMPORTANCE", "0.4"))
        max_per_minute = int(os.environ.get("ARAGORA_MEMORY_CAPTURE_MAX_PER_MINUTE", "120"))
        max_detail_chars = int(os.environ.get("ARAGORA_MEMORY_CAPTURE_MAX_DETAIL_CHARS", "800"))
        return cls(
            enabled=enabled,
            tier=tier,
            importance=importance,
            max_per_minute=max_per_minute,
            max_detail_chars=max_detail_chars,
        )


class ToolMemoryCapture:
    """Capture tool usage events into ContinuumMemory (opt-in)."""

    def __init__(
        self,
        policy: ToolCapturePolicy | None = None,
        config: ToolMemoryCaptureConfig | None = None,
    ) -> None:
        self.policy = policy or ToolCapturePolicy.from_env()
        self.config = config or ToolMemoryCaptureConfig.from_env()
        self._limiter = ToolCaptureLimiter(max_per_minute=self.config.max_per_minute)

    def should_capture(self, tool_name: str | None) -> bool:
        if not self.config.enabled:
            return False
        if not self.policy.should_capture(tool_name):
            return False
        if not self._limiter.allow():
            return False
        return True

    def format_content(
        self,
        *,
        tool_name: str,
        agent_name: str,
        debate_id: str | None,
        details: dict | None,
    ) -> str:
        lines = [
            "[Tool Usage]",
            f"Tool: {tool_name}",
            f"Agent: {agent_name}",
        ]
        if debate_id:
            lines.append(f"Debate: {debate_id}")
        if details:
            serialized = str(details)
            if self.config.max_detail_chars > 0 and len(serialized) > self.config.max_detail_chars:
                serialized = serialized[: self.config.max_detail_chars] + "..."
            lines.append(f"Details: {serialized}")
        return "\n".join(lines)
