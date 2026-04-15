"""Canonical roadmap-priority helpers for boss-ready issue governance."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import re

_SECTION_DO_NOW = "do_now"
_SECTION_DELAY = "delay"
_SECTION_AVOID = "avoid"
_SECTION_NAMES = {
    "### Do now": _SECTION_DO_NOW,
    "### Delay": _SECTION_DELAY,
    "### Avoid in this tranche": _SECTION_AVOID,
}
_CODE_RE = re.compile(r"\b([A-Z]+-\d+(?:\.\.\d+)?)\b")
_RANGE_RE = re.compile(r"^(?P<prefix>[A-Z]+)-(?P<start>\d+)\.\.(?P<end>\d+)$")


class RoadmapPriority(str, Enum):
    DO_NOW = "do_now"
    DELAY = "delay"
    AVOID = "avoid"
    UNKNOWN = "unknown"

    @property
    def blocks_boss_ready(self) -> bool:
        return self in {self.DELAY, self.AVOID}


@dataclass(frozen=True)
class RoadmapPriorityMatch:
    priority: RoadmapPriority
    codes: tuple[str, ...]
    blocked_codes: tuple[str, ...]


@dataclass(frozen=True)
class RoadmapPriorityPolicy:
    do_now: frozenset[str]
    delay: frozenset[str]
    avoid: frozenset[str]

    def priority_for_codes(
        self, codes: list[str] | tuple[str, ...] | set[str]
    ) -> RoadmapPriorityMatch:
        ordered = tuple(dict.fromkeys(str(code).strip() for code in codes if str(code).strip()))
        blocked: list[str] = []
        if any(code in self.avoid for code in ordered):
            blocked = [code for code in ordered if code in self.avoid]
            return RoadmapPriorityMatch(RoadmapPriority.AVOID, ordered, tuple(blocked))
        if any(code in self.delay for code in ordered):
            blocked = [code for code in ordered if code in self.delay]
            return RoadmapPriorityMatch(RoadmapPriority.DELAY, ordered, tuple(blocked))
        if any(code in self.do_now for code in ordered):
            return RoadmapPriorityMatch(RoadmapPriority.DO_NOW, ordered, ())
        return RoadmapPriorityMatch(RoadmapPriority.UNKNOWN, ordered, ())

    def priority_for_text(self, *texts: str) -> RoadmapPriorityMatch:
        codes: list[str] = []
        for text in texts:
            codes.extend(extract_roadmap_codes(text))
        return self.priority_for_codes(codes)

    def allows_boss_ready(self, *texts: str) -> bool:
        return not self.priority_for_text(*texts).priority.blocks_boss_ready


def _canonical_path(repo_root: Path | str) -> Path:
    return Path(repo_root) / "docs" / "status" / "NEXT_STEPS_CANONICAL.md"


def load_roadmap_priority_policy(repo_root: Path | str) -> RoadmapPriorityPolicy | None:
    path = _canonical_path(repo_root)
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8")
    buckets: dict[str, set[str]] = {
        _SECTION_DO_NOW: set(),
        _SECTION_DELAY: set(),
        _SECTION_AVOID: set(),
    }
    current_section = ""
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("#"):
            current_section = _SECTION_NAMES.get(line, "")
            continue
        if not current_section or not line.startswith("- "):
            continue
        for token in _CODE_RE.findall(line):
            buckets[current_section].update(expand_roadmap_token(token))
    return RoadmapPriorityPolicy(
        do_now=frozenset(buckets[_SECTION_DO_NOW]),
        delay=frozenset(buckets[_SECTION_DELAY]),
        avoid=frozenset(buckets[_SECTION_AVOID]),
    )


def expand_roadmap_token(token: str) -> tuple[str, ...]:
    normalized = str(token or "").strip()
    if not normalized:
        return ()
    match = _RANGE_RE.match(normalized)
    if not match:
        return (normalized,)
    prefix = match.group("prefix")
    start_text = match.group("start")
    end_text = match.group("end")
    start = int(start_text)
    end = int(end_text)
    width = len(start_text)
    if end < start:
        return (normalized,)
    return tuple(f"{prefix}-{value:0{width}d}" for value in range(start, end + 1))


def extract_roadmap_codes(text: str) -> tuple[str, ...]:
    ordered: list[str] = []
    seen: set[str] = set()
    for raw in _CODE_RE.findall(str(text or "")):
        for code in expand_roadmap_token(raw):
            if code in seen:
                continue
            seen.add(code)
            ordered.append(code)
    return tuple(ordered)


__all__ = [
    "RoadmapPriority",
    "RoadmapPriorityMatch",
    "RoadmapPriorityPolicy",
    "expand_roadmap_token",
    "extract_roadmap_codes",
    "load_roadmap_priority_policy",
]
