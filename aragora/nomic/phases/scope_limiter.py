"""
Scope limiter for nomic loop implementation.

Evaluates design complexity and rejects designs that are too ambitious
for implementation, suggesting simplifications instead.

This addresses the implementation bottleneck where debates produce good
ideas but implementations fail due to scope mismatch.
"""

import logging
import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class ScopeEvaluation:
    """Result of scope evaluation."""

    is_implementable: bool = True
    complexity_score: float = 0.0  # 0-1, lower is simpler
    file_count: int = 0
    risk_factors: List[str] = field(default_factory=list)
    suggested_simplifications: List[str] = field(default_factory=list)
    reason: str = ""

    @property
    def is_too_complex(self) -> bool:
        """Check if design is too complex to implement."""
        return self.complexity_score > 0.7 or self.file_count > 5

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "is_implementable": self.is_implementable,
            "is_too_complex": self.is_too_complex,
            "complexity_score": self.complexity_score,
            "file_count": self.file_count,
            "risk_factors": self.risk_factors,
            "suggested_simplifications": self.suggested_simplifications,
            "reason": self.reason,
        }


# Patterns that indicate high complexity
COMPLEXITY_PATTERNS = [
    (r"refactor\s+(?:the\s+)?entire", 0.3, "Full refactor mentioned"),
    (r"new\s+(?:module|package|system)", 0.2, "New module/system"),
    (r"(?:database|schema)\s+migration", 0.25, "Database migration"),
    (r"(?:breaking|api)\s+changes?", 0.2, "Breaking changes"),
    (r"multiple\s+(?:files|modules|components)", 0.15, "Multiple components"),
    (r"integration\s+with\s+\w+\s+(?:api|service)", 0.15, "External integration"),
    (r"(?:authentication|authorization)\s+(?:system|flow)", 0.2, "Auth system"),
    (r"(?:async|concurrent|parallel)\s+(?:processing|execution)", 0.1, "Concurrency"),
    (r"(?:real-?time|streaming)\s+(?:updates?|data)", 0.1, "Real-time features"),
]

# Patterns that indicate simplicity
SIMPLICITY_PATTERNS = [
    (r"add\s+(?:a\s+)?(?:simple\s+)?(?:function|method|helper)", -0.1, "Simple function"),
    (r"fix\s+(?:a\s+)?(?:bug|typo|issue)", -0.15, "Bug fix"),
    (r"update\s+(?:config|settings|documentation)", -0.1, "Config update"),
    (r"add\s+(?:logging|comments|docstrings)", -0.1, "Documentation"),
    (r"single\s+file\s+(?:change|modification)", -0.15, "Single file"),
]


class ScopeLimiter:
    """
    Evaluates design complexity and suggests simplifications.

    The goal is to prevent the nomic loop from attempting designs
    that are too complex to implement in a single cycle, which leads
    to failed implementations and wasted cycles.
    """

    def __init__(
        self,
        max_complexity: float = 0.7,
        max_files: int = 5,
        protected_files: Optional[List[str]] = None,
    ):
        """
        Initialize scope limiter.

        Args:
            max_complexity: Maximum complexity score (0-1)
            max_files: Maximum files that can be modified
            protected_files: Files that cannot be modified
        """
        self.max_complexity = max_complexity
        self.max_files = max_files
        self.protected_files = protected_files or [
            "CLAUDE.md",
            "core.py",
            "aragora/__init__.py",
            ".env",
            "scripts/nomic_loop.py",
        ]

    def evaluate(self, design: str) -> ScopeEvaluation:
        """
        Evaluate a design for implementability.

        Args:
            design: Design text from design phase

        Returns:
            ScopeEvaluation with complexity analysis
        """
        evaluation = ScopeEvaluation()
        design_lower = design.lower()

        # Calculate complexity score from patterns
        complexity = 0.0
        for pattern, weight, reason in COMPLEXITY_PATTERNS:
            if re.search(pattern, design_lower, re.IGNORECASE):
                complexity += weight
                evaluation.risk_factors.append(reason)

        for pattern, weight, reason in SIMPLICITY_PATTERNS:
            if re.search(pattern, design_lower, re.IGNORECASE):
                complexity += weight  # weight is negative for simplicity

        evaluation.complexity_score = max(0.0, min(1.0, complexity))

        # Count affected files
        file_matches = re.findall(
            r'(?:create|modify|update|change|edit|add\s+to)\s+[`"]?([a-zA-Z0-9_/]+\.py)[`"]?',
            design,
            re.IGNORECASE,
        )
        # Also check for file paths mentioned
        file_paths = re.findall(r"[a-zA-Z_][a-zA-Z0-9_/]+\.py", design)
        all_files = set(file_matches + file_paths)
        evaluation.file_count = len(all_files)

        # Check for protected files
        for f in all_files:
            for protected in self.protected_files:
                if protected in f:
                    evaluation.risk_factors.append(
                        f"Attempts to modify protected file: {protected}"
                    )
                    evaluation.is_implementable = False

        # Determine implementability
        if evaluation.complexity_score > self.max_complexity:
            evaluation.is_implementable = False
            evaluation.reason = f"Complexity score {evaluation.complexity_score:.2f} exceeds limit {self.max_complexity}"
            evaluation.suggested_simplifications = self._suggest_simplifications(design, evaluation)
        elif evaluation.file_count > self.max_files:
            evaluation.is_implementable = False
            evaluation.reason = f"Affects {evaluation.file_count} files, limit is {self.max_files}"
            evaluation.suggested_simplifications = [
                f"Break into {evaluation.file_count // self.max_files + 1} smaller changes",
                "Focus on one module/component per cycle",
                "Defer tests/documentation to follow-up cycle",
            ]
        else:
            evaluation.is_implementable = True
            evaluation.reason = "Design is within scope limits"

        return evaluation

    def _suggest_simplifications(self, design: str, evaluation: ScopeEvaluation) -> List[str]:
        """Generate simplification suggestions based on risk factors."""
        suggestions = []

        for risk in evaluation.risk_factors:
            if "refactor" in risk.lower():
                suggestions.append("Focus on adding new code instead of refactoring existing code")
            elif "new module" in risk.lower() or "new system" in risk.lower():
                suggestions.append("Start with a minimal prototype, not a full module")
            elif "migration" in risk.lower():
                suggestions.append("Avoid schema changes; add optional fields if needed")
            elif "breaking" in risk.lower():
                suggestions.append("Maintain backward compatibility; add new APIs alongside old")
            elif "integration" in risk.lower():
                suggestions.append("Use mock/stub for external services in first iteration")
            elif "auth" in risk.lower():
                suggestions.append("Implement auth in a follow-up cycle")

        if not suggestions:
            suggestions = [
                "Reduce scope to single-file change",
                "Focus on the minimal viable implementation",
                "Defer complex features to future cycles",
            ]

        return suggestions[:3]  # Return top 3 suggestions

    def simplify_for_implementation(
        self, design: str, evaluation: ScopeEvaluation
    ) -> Tuple[str, str]:
        """
        Generate a simplified version of the design.

        Args:
            design: Original design
            evaluation: Scope evaluation result

        Returns:
            Tuple of (simplified_design, simplification_note)
        """
        if evaluation.is_implementable:
            return design, "Design is within scope"

        # Generate a scope-limited version
        note = (
            f"Original design complexity: {evaluation.complexity_score:.2f} "
            f"(limit: {self.max_complexity})\n"
            f"Files affected: {evaluation.file_count} (limit: {self.max_files})\n"
            f"\nSimplifications applied:\n"
        )

        for i, suggestion in enumerate(evaluation.suggested_simplifications, 1):
            note += f"  {i}. {suggestion}\n"

        # Create a simplified prompt that can be prepended to design
        simplified = (
            "## SCOPE LIMITATION APPLIED\n\n"
            f"The original design was too complex (score: {evaluation.complexity_score:.2f}). "
            "Please implement only the MINIMAL version:\n\n"
            "REQUIRED CONSTRAINTS:\n"
            f"- Modify at most {self.max_files} files\n"
            "- No breaking changes\n"
            "- No new modules (extend existing ones)\n"
            "- Skip tests in this cycle (add in follow-up)\n"
            "\n---\n\n"
            f"{design}"
        )

        return simplified, note


def check_design_scope(design: str, max_files: int = 5) -> ScopeEvaluation:
    """
    Convenience function to check design scope.

    Args:
        design: Design text to evaluate
        max_files: Maximum allowed files

    Returns:
        ScopeEvaluation result
    """
    limiter = ScopeLimiter(max_files=max_files)
    return limiter.evaluate(design)


__all__ = ["ScopeLimiter", "ScopeEvaluation", "check_design_scope"]
