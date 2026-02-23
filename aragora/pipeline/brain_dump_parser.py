"""
Brain Dump Natural Language Parser.

Parses unstructured text (bullet lists, numbered lists, paragraphs, prose)
into discrete idea strings compatible with ``IdeaToExecutionPipeline.from_ideas()``.

Usage::

    parser = BrainDumpParser()
    ideas = parser.parse("I think we should build a dashboard. Also need better error handling.")
    # → ["I think we should build a dashboard", "Also need better error handling"]
"""

from __future__ import annotations

import re
import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Pre-compiled patterns for format detection
_BULLET_RE = re.compile(r"^\s*[-*>•]\s+", re.MULTILINE)
_NUMBERED_RE = re.compile(r"^\s*\d+[.)]\s+", re.MULTILINE)
_DOUBLE_NEWLINE_RE = re.compile(r"\n\s*\n")

# Sentence boundary: period/exclamation/question followed by whitespace or end
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

# Urgency signal patterns
_URGENCY_PATTERNS = [
    re.compile(r"\b(urgent|asap|immediately|critical|blocker|showstopper|p0|p1)\b", re.IGNORECASE),
    re.compile(r"\b(must|need to|have to|required|mandatory)\b", re.IGNORECASE),
    re.compile(r"\b(deadline|due date|by (?:end of|eod|eow|monday|tuesday|wednesday|thursday|friday))\b", re.IGNORECASE),
    re.compile(r"!{2,}"),  # Multiple exclamation marks
]

# Entity reference patterns (people, systems, tools, orgs)
_ENTITY_RE = re.compile(
    r"(?:@\w+)"  # @mentions
    r"|(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)"  # Proper nouns (Title Case sequences)
    r"|(?:(?:the|our|their)\s+(?:\w+\s+){0,2}(?:team|system|service|api|database|server|app|platform))",
    re.IGNORECASE,
)

# Theme keyword groups for detection
_THEME_KEYWORDS: dict[str, list[str]] = {
    "performance": ["fast", "slow", "latency", "throughput", "optimize", "cache", "speed", "performance"],
    "security": ["auth", "security", "encrypt", "permission", "access", "vulnerability", "token", "ssl"],
    "ux": ["user experience", "ui", "ux", "interface", "design", "usability", "accessibility", "responsive"],
    "reliability": ["reliable", "uptime", "resilient", "failover", "backup", "redundant", "availability"],
    "scalability": ["scale", "scalable", "horizontal", "vertical", "load", "capacity", "shard"],
    "testing": ["test", "coverage", "qa", "quality", "regression", "integration test", "unit test"],
    "documentation": ["doc", "document", "readme", "guide", "tutorial", "api doc"],
    "infrastructure": ["deploy", "ci", "cd", "pipeline", "docker", "kubernetes", "cloud", "infra"],
    "data": ["data", "database", "migration", "schema", "analytics", "metrics", "storage"],
    "integration": ["integrate", "api", "webhook", "connector", "sync", "import", "export"],
}


@dataclass
class EnrichedBrainDump:
    """Result of enriched brain dump parsing with metadata."""

    ideas: list[str]
    detected_themes: list[str] = field(default_factory=list)
    urgency_signals: list[str] = field(default_factory=list)
    entity_references: list[str] = field(default_factory=list)
    format_detected: str = "prose"
    idea_count: int = 0

    def __post_init__(self) -> None:
        self.idea_count = len(self.ideas)

    def to_dict(self) -> dict:
        return {
            "ideas": self.ideas,
            "detected_themes": self.detected_themes,
            "urgency_signals": self.urgency_signals,
            "entity_references": self.entity_references,
            "format_detected": self.format_detected,
            "idea_count": self.idea_count,
        }


class BrainDumpParser:
    """Parse unstructured brain dump text into discrete idea strings."""

    def parse(self, raw_text: str) -> list[str]:
        """Parse raw text into a list of idea strings.

        Detects the input format (bullets, numbered, paragraphs, prose),
        splits accordingly, merges short fragments, and deduplicates.

        Args:
            raw_text: Unstructured text input (brain dump).

        Returns:
            List of cleaned idea strings ready for ``from_ideas()``.
        """
        if not raw_text or not raw_text.strip():
            return []

        text = raw_text.strip()
        fmt, ideas = self._parse_structural(text)
        # Only merge fragments for prose — structured formats have
        # intentional line breaks that should be preserved.
        if fmt == "prose":
            ideas = self._merge_fragments(ideas)
        ideas = self._deduplicate(ideas)
        return ideas

    def _parse_structural(self, text: str) -> tuple[str, list[str]]:
        """Split text based on detected structure.

        Returns the detected format name and the list of extracted ideas.
        """
        fmt = self._detect_format(text)

        if fmt == "bullets":
            return fmt, self._split_bullets(text)
        elif fmt == "numbered":
            return fmt, self._split_numbered(text)
        elif fmt == "paragraphs":
            return fmt, self._split_paragraphs(text)
        else:
            return fmt, self._split_prose(text)

    def _detect_format(self, text: str) -> str:
        """Detect the format of the input text.

        Returns one of: "bullets", "numbered", "paragraphs", "prose".
        """
        bullet_count = len(_BULLET_RE.findall(text))
        numbered_count = len(_NUMBERED_RE.findall(text))

        if bullet_count >= 2:
            return "bullets"
        if numbered_count >= 2:
            return "numbered"
        if _DOUBLE_NEWLINE_RE.search(text):
            return "paragraphs"
        return "prose"

    def _split_bullets(self, text: str) -> list[str]:
        """Split bullet-list text into ideas."""
        lines = text.split("\n")
        ideas: list[str] = []
        for line in lines:
            cleaned = _BULLET_RE.sub("", line).strip()
            if cleaned:
                ideas.append(cleaned)
        return ideas

    def _split_numbered(self, text: str) -> list[str]:
        """Split numbered-list text into ideas."""
        lines = text.split("\n")
        ideas: list[str] = []
        for line in lines:
            cleaned = _NUMBERED_RE.sub("", line).strip()
            if cleaned:
                ideas.append(cleaned)
        return ideas

    def _split_paragraphs(self, text: str) -> list[str]:
        """Split paragraph-separated text into ideas."""
        paragraphs = _DOUBLE_NEWLINE_RE.split(text)
        ideas: list[str] = []
        for para in paragraphs:
            cleaned = para.strip()
            if cleaned:
                ideas.append(cleaned)
        return ideas

    def _split_prose(self, text: str) -> list[str]:
        """Split continuous prose into sentence-level ideas."""
        sentences = _SENTENCE_SPLIT_RE.split(text)
        ideas: list[str] = []
        for sentence in sentences:
            cleaned = sentence.strip()
            if cleaned:
                ideas.append(cleaned)
        return ideas

    def _merge_fragments(self, ideas: list[str], min_len: int = 20) -> list[str]:
        """Merge short fragments with adjacent ideas.

        Fragments shorter than ``min_len`` are appended to the preceding
        idea (or prepended to the next if first).
        """
        if not ideas:
            return ideas

        merged: list[str] = []
        for idea in ideas:
            if len(idea) < min_len and merged:
                merged[-1] = merged[-1] + ". " + idea
            else:
                merged.append(idea)

        # If first item is still short after merge and there is a next item
        if merged and len(merged[0]) < min_len and len(merged) > 1:
            merged[1] = merged[0] + ". " + merged[1]
            merged.pop(0)

        return merged

    def parse_enriched(self, raw_text: str) -> EnrichedBrainDump:
        """Parse raw text with enriched metadata extraction.

        Returns ideas plus detected themes, urgency signals, and entity
        references — useful for feeding into principles extraction and
        pipeline auto-run.

        Args:
            raw_text: Unstructured text input (brain dump).

        Returns:
            EnrichedBrainDump with ideas and extracted metadata.
        """
        if not raw_text or not raw_text.strip():
            return EnrichedBrainDump(ideas=[])

        text = raw_text.strip()
        fmt, ideas = self._parse_structural(text)
        if fmt == "prose":
            ideas = self._merge_fragments(ideas)
        ideas = self._deduplicate(ideas)

        themes = self._detect_themes(text)
        urgency = self._extract_urgency_signals(text)
        entities = self._extract_entity_references(text)

        return EnrichedBrainDump(
            ideas=ideas,
            detected_themes=themes,
            urgency_signals=urgency,
            entity_references=entities,
            format_detected=fmt,
        )

    def _detect_themes(self, text: str) -> list[str]:
        """Detect thematic keywords in the text."""
        text_lower = text.lower()
        detected: list[str] = []
        for theme, keywords in _THEME_KEYWORDS.items():
            if any(kw in text_lower for kw in keywords):
                detected.append(theme)
        return detected

    def _extract_urgency_signals(self, text: str) -> list[str]:
        """Extract urgency signals from the text."""
        signals: list[str] = []
        for pattern in _URGENCY_PATTERNS:
            matches = pattern.findall(text)
            signals.extend(matches)
        # Deduplicate while preserving order
        seen: set[str] = set()
        unique: list[str] = []
        for s in signals:
            key = s.lower().strip()
            if key and key not in seen:
                seen.add(key)
                unique.append(s.strip())
        return unique

    def _extract_entity_references(self, text: str) -> list[str]:
        """Extract entity references (people, systems, tools)."""
        matches = _ENTITY_RE.findall(text)
        seen: set[str] = set()
        unique: list[str] = []
        for m in matches:
            key = m.strip().lower()
            if key and key not in seen:
                seen.add(key)
                unique.append(m.strip())
        return unique

    def _deduplicate(self, ideas: list[str], threshold: float = 0.8) -> list[str]:
        """Remove near-duplicate ideas using token overlap ratio.

        Two ideas are considered duplicates if their token overlap
        exceeds ``threshold`` (Jaccard similarity on lowercased tokens).
        """
        if not ideas:
            return ideas

        unique: list[str] = []
        seen_token_sets: list[set[str]] = []

        for idea in ideas:
            tokens = set(idea.lower().split())
            if not tokens:
                continue

            is_dup = False
            for seen in seen_token_sets:
                if not seen:
                    continue
                overlap = len(tokens & seen) / len(tokens | seen)
                if overlap >= threshold:
                    is_dup = True
                    break

            if not is_dup:
                unique.append(idea)
                seen_token_sets.append(tokens)

        return unique
