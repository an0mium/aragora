"""
Pattern learning for TestFixer.

Learns from successful and failed fix attempts to improve future
fix generation. Extracts patterns from fixes and provides similarity
matching for applying learned knowledge.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from aragora.nomic.testfixer.orchestrator import FixAttempt
    from aragora.nomic.testfixer.analyzer import FailureAnalysis
    from aragora.nomic.testfixer.proposer import PatchProposal

logger = logging.getLogger(__name__)


@dataclass
class FixPattern:
    """A learned pattern from a successful fix."""

    id: str
    category: str  # FailureCategory value
    error_pattern: str  # Regex or substring to match
    fix_pattern: str  # Description of the fix approach

    # The actual fix diff/code
    fix_diff: str
    fix_file: str

    # Context
    error_type: str
    root_cause: str

    # Statistics
    success_count: int = 1
    failure_count: int = 0
    last_used: str = field(default_factory=lambda: datetime.now().isoformat())
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    # Confidence based on success rate
    @property
    def confidence(self) -> float:
        total = self.success_count + self.failure_count
        if total == 0:
            return 0.5
        return self.success_count / total

    @property
    def is_reliable(self) -> bool:
        """Pattern is reliable if it has good success rate with enough data."""
        return self.confidence >= 0.7 and self.success_count >= 2

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "category": self.category,
            "error_pattern": self.error_pattern,
            "fix_pattern": self.fix_pattern,
            "fix_diff": self.fix_diff,
            "fix_file": self.fix_file,
            "error_type": self.error_type,
            "root_cause": self.root_cause,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "last_used": self.last_used,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FixPattern:
        return cls(
            id=data["id"],
            category=data["category"],
            error_pattern=data["error_pattern"],
            fix_pattern=data["fix_pattern"],
            fix_diff=data.get("fix_diff", ""),
            fix_file=data.get("fix_file", ""),
            error_type=data.get("error_type", ""),
            root_cause=data.get("root_cause", ""),
            success_count=data.get("success_count", 1),
            failure_count=data.get("failure_count", 0),
            last_used=data.get("last_used", datetime.now().isoformat()),
            created_at=data.get("created_at", datetime.now().isoformat()),
        )


@dataclass
class PatternMatch:
    """A match between a failure and a learned pattern."""

    pattern: FixPattern
    similarity: float  # 0-1, how similar the current failure is
    confidence: float  # Combined pattern confidence and similarity

    def __lt__(self, other: PatternMatch) -> bool:
        return self.confidence < other.confidence


class PatternLearner:
    """Learns and applies fix patterns from historical attempts.

    Maintains a database of successful fix patterns and provides
    similarity matching to suggest fixes for new failures.

    Example:
        learner = PatternLearner(Path(".testfixer/patterns.json"))

        # Learn from a successful fix
        learner.learn_from_attempt(attempt)

        # Find similar patterns for a new failure
        matches = learner.find_similar_patterns(analysis)
        if matches:
            best_match = matches[0]
            print(f"Similar fix found: {best_match.pattern.fix_pattern}")
    """

    def __init__(self, store_path: Path | str | None = None):
        """Initialize the pattern learner.

        Args:
            store_path: Path to JSON file for pattern storage.
                       If None, patterns are stored in memory only.
        """
        self.store_path = Path(store_path) if store_path else None
        self.patterns: dict[str, FixPattern] = {}
        self._loaded = False

    def _ensure_loaded(self) -> None:
        """Load patterns from disk if not already loaded."""
        if self._loaded or not self.store_path:
            return

        if self.store_path.exists():
            try:
                with self.store_path.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                    for pattern_data in data.get("patterns", []):
                        pattern = FixPattern.from_dict(pattern_data)
                        self.patterns[pattern.id] = pattern
                logger.info(
                    "pattern_learner.loaded patterns=%d path=%s",
                    len(self.patterns),
                    self.store_path,
                )
            except Exception as e:
                logger.warning(
                    "pattern_learner.load_failed path=%s error=%s",
                    self.store_path,
                    str(e),
                )

        self._loaded = True

    def _save(self) -> None:
        """Persist patterns to disk."""
        if not self.store_path:
            return

        self.store_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "version": 1,
            "updated_at": datetime.now().isoformat(),
            "patterns": [p.to_dict() for p in self.patterns.values()],
        }

        with self.store_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        logger.debug("pattern_learner.saved patterns=%d", len(self.patterns))

    def _generate_pattern_id(self, error_pattern: str, category: str) -> str:
        """Generate a unique ID for a pattern."""
        content = f"{category}:{error_pattern}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def _extract_error_pattern(self, failure_error: str) -> str:
        """Extract a reusable pattern from an error message.

        Removes specific variable names, line numbers, etc. to create
        a pattern that can match similar errors.
        """
        pattern = failure_error

        # Remove line numbers
        pattern = re.sub(r"line \d+", "line N", pattern)

        # Remove specific file paths but keep structure
        pattern = re.sub(r"/[^\s:]+\.py", "/path/to/file.py", pattern)

        # Remove memory addresses
        pattern = re.sub(r"0x[0-9a-fA-F]+", "0x...", pattern)

        # Remove specific object IDs
        pattern = re.sub(r"'<[^>]+>'", "'<object>'", pattern)

        # Keep first 200 chars for matching
        return pattern[:200]

    def _extract_fix_pattern(self, proposal: PatchProposal) -> str:
        """Extract a description of what the fix does."""
        if proposal.description:
            return proposal.description[:500]

        # Generate from diff if no description
        if proposal.patches:
            patch = proposal.patches[0]
            return f"Modified {patch.file_path}"

        return "Unknown fix"

    def _calculate_similarity(
        self,
        pattern: FixPattern,
        analysis: FailureAnalysis,
    ) -> float:
        """Calculate similarity between a pattern and a failure analysis."""
        score = 0.0
        weights_total = 0.0

        # Category match (high weight)
        if pattern.category == analysis.category.value:
            score += 0.4
        weights_total += 0.4

        # Error type match
        if pattern.error_type and pattern.error_type in analysis.failure.error_type:
            score += 0.2
        weights_total += 0.2

        # Error pattern substring match
        error_text = f"{analysis.failure.error_type} {analysis.failure.error_message}"
        pattern_words = set(pattern.error_pattern.lower().split())
        error_words = set(error_text.lower().split())

        if pattern_words and error_words:
            overlap = len(pattern_words & error_words) / len(pattern_words)
            score += 0.3 * overlap
        weights_total += 0.3

        # File similarity (if same type of file)
        if pattern.fix_file and analysis.root_cause_file:
            if pattern.fix_file.split("/")[-1] == analysis.root_cause_file.split("/")[-1]:
                score += 0.1
        weights_total += 0.1

        return score / weights_total if weights_total > 0 else 0.0

    def learn_from_attempt(self, attempt: FixAttempt) -> FixPattern | None:
        """Learn from a fix attempt.

        If successful, extracts a pattern for future use.
        If failed, records the failure to reduce pattern confidence.

        Args:
            attempt: The fix attempt to learn from

        Returns:
            The learned/updated pattern, or None if not learnable
        """
        self._ensure_loaded()

        if not attempt.applied:
            return None

        # Extract pattern identifiers
        error_pattern = self._extract_error_pattern(attempt.failure.error_message)
        category = attempt.analysis.category.value
        pattern_id = self._generate_pattern_id(error_pattern, category)

        if attempt.success:
            # Successful fix - learn or reinforce pattern
            if pattern_id in self.patterns:
                # Reinforce existing pattern
                pattern = self.patterns[pattern_id]
                pattern.success_count += 1
                pattern.last_used = datetime.now().isoformat()
                logger.info(
                    "pattern_learner.reinforced id=%s successes=%d",
                    pattern_id,
                    pattern.success_count,
                )
            else:
                # Create new pattern
                fix_diff = (
                    attempt.proposal.as_diff() if hasattr(attempt.proposal, "as_diff") else ""
                )
                pattern = FixPattern(
                    id=pattern_id,
                    category=category,
                    error_pattern=error_pattern,
                    fix_pattern=self._extract_fix_pattern(attempt.proposal),
                    fix_diff=fix_diff,
                    fix_file=attempt.analysis.root_cause_file,
                    error_type=attempt.failure.error_type,
                    root_cause=attempt.analysis.root_cause,
                )
                self.patterns[pattern_id] = pattern
                logger.info(
                    "pattern_learner.learned id=%s category=%s",
                    pattern_id,
                    category,
                )

            self._save()
            return pattern
        else:
            # Failed fix - record failure
            if pattern_id in self.patterns:
                pattern = self.patterns[pattern_id]
                pattern.failure_count += 1
                pattern.last_used = datetime.now().isoformat()
                logger.info(
                    "pattern_learner.failure_recorded id=%s failures=%d",
                    pattern_id,
                    pattern.failure_count,
                )
                self._save()
                return pattern

            return None

    def find_similar_patterns(
        self,
        analysis: FailureAnalysis,
        min_similarity: float = 0.5,
        max_results: int = 5,
    ) -> list[PatternMatch]:
        """Find patterns similar to the given failure analysis.

        Args:
            analysis: The failure analysis to match against
            min_similarity: Minimum similarity score (0-1)
            max_results: Maximum number of matches to return

        Returns:
            List of PatternMatch objects, sorted by confidence (descending)
        """
        self._ensure_loaded()

        matches = []

        for pattern in self.patterns.values():
            similarity = self._calculate_similarity(pattern, analysis)

            if similarity >= min_similarity:
                # Combined confidence = pattern reliability * similarity
                confidence = pattern.confidence * similarity

                matches.append(
                    PatternMatch(
                        pattern=pattern,
                        similarity=similarity,
                        confidence=confidence,
                    )
                )

        # Sort by confidence descending
        matches.sort(key=lambda m: m.confidence, reverse=True)

        logger.debug(
            "pattern_learner.search category=%s matches=%d",
            analysis.category.value,
            len(matches),
        )

        return matches[:max_results]

    def get_reliable_patterns(
        self,
        category: str | None = None,
    ) -> list[FixPattern]:
        """Get all reliable patterns, optionally filtered by category.

        Args:
            category: Optional FailureCategory value to filter by

        Returns:
            List of reliable patterns
        """
        self._ensure_loaded()

        patterns = [p for p in self.patterns.values() if p.is_reliable]

        if category:
            patterns = [p for p in patterns if p.category == category]

        return sorted(patterns, key=lambda p: p.confidence, reverse=True)

    def suggest_heuristic(self, analysis: FailureAnalysis) -> str | None:
        """Suggest a fix approach based on learned patterns.

        Args:
            analysis: The failure analysis

        Returns:
            Suggested fix approach, or None if no good match
        """
        matches = self.find_similar_patterns(analysis, min_similarity=0.6)

        if not matches:
            return None

        best = matches[0]

        if best.confidence < 0.5:
            return None

        return (
            f"Based on similar past fix (confidence: {best.confidence:.0%}):\n"
            f"{best.pattern.fix_pattern}\n\n"
            f"Original error pattern: {best.pattern.error_pattern[:100]}"
        )

    def export_patterns(self) -> list[dict[str, Any]]:
        """Export all patterns as dictionaries.

        Returns:
            List of pattern dictionaries
        """
        self._ensure_loaded()
        return [p.to_dict() for p in self.patterns.values()]

    def import_patterns(self, patterns: list[dict[str, Any]]) -> int:
        """Import patterns from dictionaries.

        Args:
            patterns: List of pattern dictionaries

        Returns:
            Number of patterns imported
        """
        self._ensure_loaded()

        imported = 0
        for data in patterns:
            try:
                pattern = FixPattern.from_dict(data)
                if pattern.id not in self.patterns:
                    self.patterns[pattern.id] = pattern
                    imported += 1
            except Exception as e:
                logger.warning("pattern_learner.import_failed error=%s", str(e))

        if imported > 0:
            self._save()
            logger.info("pattern_learner.imported count=%d", imported)

        return imported

    def get_statistics(self) -> dict[str, Any]:
        """Get statistics about learned patterns.

        Returns:
            Dictionary with pattern statistics
        """
        self._ensure_loaded()

        total = len(self.patterns)
        reliable = len([p for p in self.patterns.values() if p.is_reliable])

        by_category: dict[str, int] = {}
        total_successes = 0
        total_failures = 0

        for pattern in self.patterns.values():
            by_category[pattern.category] = by_category.get(pattern.category, 0) + 1
            total_successes += pattern.success_count
            total_failures += pattern.failure_count

        return {
            "total_patterns": total,
            "reliable_patterns": reliable,
            "by_category": by_category,
            "total_successes": total_successes,
            "total_failures": total_failures,
            "overall_success_rate": (
                total_successes / (total_successes + total_failures)
                if (total_successes + total_failures) > 0
                else 0.0
            ),
        }
