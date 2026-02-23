"""Risk Scorer for Self-Improvement Pipeline.

Evaluates proposed goals and changes to determine risk level before
execution. Used by SelfImprovePipeline when safe_mode is enabled to
gate execution based on risk thresholds.

Risk Categories:
    LOW      - Documentation, tests, comments (score < 0.3)
    MEDIUM   - New features, refactors, non-critical code (0.3 <= score < 0.5)
    HIGH     - Security, auth, data handling, core engine (0.5 <= score < 0.8)
    CRITICAL - Infrastructure, deployment, protected files (score >= 0.8)

Recommendations:
    AUTO   - Execute in worktree, auto-commit (score < threshold)
    REVIEW - Execute in worktree, stage but don't commit (threshold <= score < 0.8)
    BLOCK  - Skip execution, log warning (score >= 0.8)

Usage:
    scorer = RiskScorer()
    assessment = scorer.score_goal("Add unit tests for auth module")
    if assessment.recommendation == "auto":
        # Safe to execute automatically
        ...
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)

# Protected files that always push risk to CRITICAL
PROTECTED_FILES = [
    "CLAUDE.md",
    "aragora/__init__.py",
    ".env",
    ".env.local",
    "scripts/nomic_loop.py",
    "aragora/debate/orchestrator.py",
    "aragora/core.py",
    "aragora/core/__init__.py",
    "aragora/core_types.py",
]

# Keywords indicating high-risk domains
HIGH_RISK_KEYWORDS = [
    "security", "auth", "authentication", "authorization",
    "encrypt", "decrypt", "credential", "password", "token",
    "secret", "key_rotation", "ssrf", "injection", "xss",
    "rbac", "permission", "oidc", "saml", "mfa", "totp",
    "database", "migration", "schema", "postgres", "redis",
    "delete", "drop", "truncate", "purge",
    "billing", "payment", "cost", "metering",
    "privacy", "gdpr", "anonymization", "consent", "pii",
]

# Keywords indicating critical infrastructure risk
CRITICAL_RISK_KEYWORDS = [
    "deploy", "deployment", "infrastructure", "docker",
    "kubernetes", "k8s", "helm", "terraform",
    "ci/cd", "pipeline", "github actions", "workflow",
    "production", "prod", "release",
    "backup", "disaster recovery", "failover",
    "nomic_loop", "self_improve", "autonomous",
]

# Keywords indicating low-risk domains
LOW_RISK_KEYWORDS = [
    "documentation", "readme", "docstring", "comment",
    "test", "spec", "fixture", "mock",
    "typo", "formatting", "whitespace", "lint",
    "type hint", "annotation", "typehint",
    "changelog", "license",
]

# Paths that indicate low-risk changes
LOW_RISK_PATHS = [
    "docs/", "tests/", "test_", "README",
    "CHANGELOG", "LICENSE", ".md",
    "examples/", "scripts/demo_",
]

# Paths that indicate high-risk changes
HIGH_RISK_PATHS = [
    "aragora/security/", "aragora/auth/", "aragora/rbac/",
    "aragora/privacy/", "aragora/billing/",
    "aragora/storage/", "aragora/tenancy/",
    "aragora/debate/orchestrator.py",
    "aragora/debate/consensus.py",
]

# Paths that indicate critical-risk changes
CRITICAL_RISK_PATHS = [
    ".env", ".github/workflows/",
    "docker/", "Dockerfile",
    "scripts/nomic_loop.py",
    "aragora/nomic/self_improve.py",
    "aragora/nomic/autonomous_orchestrator.py",
    "aragora/ops/",
    "aragora/backup/",
]


class RiskCategory(str, Enum):
    """Risk level categories for proposed changes."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class RiskFactor:
    """A single factor contributing to the overall risk score."""

    name: str
    weight: float  # How much this factor contributed (0.0-1.0)
    detail: str  # Human-readable explanation


@dataclass
class RiskScore:
    """Risk assessment result for a proposed goal or change.

    Attributes:
        score: Numeric risk score from 0.0 (safe) to 1.0 (dangerous).
        category: Risk category (LOW, MEDIUM, HIGH, CRITICAL).
        factors: List of factors that contributed to the score.
        recommendation: Execution recommendation (auto, review, block).
        goal: The original goal that was scored.
    """

    score: float
    category: RiskCategory
    factors: list[RiskFactor] = field(default_factory=list)
    recommendation: str = "review"  # "auto" | "review" | "block"
    goal: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dictionary."""
        return {
            "score": self.score,
            "category": self.category.value,
            "factors": [
                {"name": f.name, "weight": f.weight, "detail": f.detail}
                for f in self.factors
            ],
            "recommendation": self.recommendation,
            "goal": self.goal,
        }


class RiskScorer:
    """Evaluates proposed changes for risk before execution.

    Analyzes goal text, file scope, and other signals to produce a
    risk score that determines whether changes should be auto-executed,
    staged for review, or blocked.

    Args:
        threshold: Score below which changes are auto-approved (default 0.5).
        block_threshold: Score at or above which changes are blocked (default 0.8).
        protected_files: Override list of protected files.
    """

    def __init__(
        self,
        threshold: float = 0.5,
        block_threshold: float = 0.8,
        protected_files: list[str] | None = None,
    ) -> None:
        self.threshold = threshold
        self.block_threshold = block_threshold
        self.protected_files = protected_files or list(PROTECTED_FILES)

    def score_goal(
        self,
        goal: str,
        file_scope: list[str] | None = None,
        estimated_files_changed: int | None = None,
        has_tests: bool | None = None,
        complexity_score: int | None = None,
    ) -> RiskScore:
        """Score a goal for risk level.

        Args:
            goal: The goal description text.
            file_scope: List of file paths the goal will touch.
            estimated_files_changed: Estimated number of files that will be changed.
            has_tests: Whether tests exist or are included for the change.
            complexity_score: Complexity estimate from TaskDecomposer (0-10).

        Returns:
            RiskScore with score, category, factors, and recommendation.
        """
        factors: list[RiskFactor] = []
        file_scope = file_scope or []
        goal_lower = goal.lower()

        # --- Factor 1: Keyword analysis on goal text ---
        keyword_score = self._score_keywords(goal_lower)
        if keyword_score > 0:
            factors.append(RiskFactor(
                name="keyword_analysis",
                weight=keyword_score,
                detail=self._keyword_detail(goal_lower),
            ))

        # --- Factor 2: File scope analysis ---
        file_score = self._score_file_scope(file_scope)
        if file_score > 0:
            factors.append(RiskFactor(
                name="file_scope",
                weight=file_score,
                detail=f"File scope risk from {len(file_scope)} file(s)",
            ))

        # --- Factor 3: Protected file detection ---
        protected_hits = self._detect_protected_files(goal_lower, file_scope)
        if protected_hits:
            factors.append(RiskFactor(
                name="protected_files",
                weight=1.0,
                detail=f"Touches protected files: {', '.join(protected_hits)}",
            ))

        # --- Factor 4: Scale of change ---
        scale_score = self._score_scale(
            file_scope, estimated_files_changed, complexity_score
        )
        if scale_score > 0:
            factors.append(RiskFactor(
                name="change_scale",
                weight=scale_score,
                detail=self._scale_detail(
                    file_scope, estimated_files_changed, complexity_score
                ),
            ))

        # --- Factor 5: Test coverage ---
        test_score = self._score_test_coverage(has_tests, goal_lower, file_scope)
        if test_score > 0:
            factors.append(RiskFactor(
                name="test_coverage",
                weight=test_score,
                detail="Changes lack associated tests" if not has_tests else "Test coverage factor",
            ))

        # --- Factor 6: Test-only change dampening ---
        # If all files in scope are test/doc files, dampen the overall risk
        all_low_risk = file_scope and all(
            any(lp.lower() in fp.lower() for lp in LOW_RISK_PATHS)
            for fp in file_scope
        )
        if all_low_risk and not protected_hits:
            factors = [
                RiskFactor(
                    name=f.name,
                    weight=f.weight * 0.4,  # Dampen all factors
                    detail=f.detail,
                )
                if f.name == "keyword_analysis"
                else f
                for f in factors
            ]

        # --- Compute weighted aggregate ---
        raw_score = self._aggregate_score(factors, protected_hits)

        # Clamp to [0, 1]
        score = max(0.0, min(1.0, raw_score))

        # Determine category
        category = self._categorize(score)

        # Determine recommendation
        recommendation = self._recommend(score)

        result = RiskScore(
            score=round(score, 3),
            category=category,
            factors=factors,
            recommendation=recommendation,
            goal=goal,
        )

        logger.info(
            "risk_scored goal=%s score=%.3f category=%s recommendation=%s factors=%d",
            goal[:60],
            score,
            category.value,
            recommendation,
            len(factors),
        )

        return result

    def score_subtask(self, subtask: Any) -> RiskScore:
        """Score a SubTask object for risk.

        Convenience wrapper that extracts fields from a SubTask dataclass.
        """
        title = getattr(subtask, "title", "") or ""
        description = getattr(subtask, "description", "") or ""
        file_scope = getattr(subtask, "file_scope", []) or []
        complexity = getattr(subtask, "estimated_complexity", None)

        # Convert complexity string to numeric if needed
        complexity_score = None
        if isinstance(complexity, str):
            complexity_map = {"low": 2, "medium": 5, "high": 8, "critical": 10}
            complexity_score = complexity_map.get(complexity.lower())
        elif isinstance(complexity, (int, float)):
            complexity_score = int(complexity)

        goal_text = f"{title}: {description}" if title else description
        return self.score_goal(
            goal=goal_text,
            file_scope=file_scope,
            complexity_score=complexity_score,
        )

    # ---- Scoring helpers ----

    def _score_keywords(self, goal_lower: str) -> float:
        """Score based on keyword matches in the goal text."""
        # Check for low-risk indicators first (reduce score)
        low_matches = sum(1 for kw in LOW_RISK_KEYWORDS if kw in goal_lower)
        high_matches = sum(1 for kw in HIGH_RISK_KEYWORDS if kw in goal_lower)
        critical_matches = sum(1 for kw in CRITICAL_RISK_KEYWORDS if kw in goal_lower)

        if critical_matches > 0:
            return min(1.0, 0.7 + critical_matches * 0.1)
        if high_matches > 0:
            return min(1.0, 0.4 + high_matches * 0.1)
        if low_matches > 0:
            return max(0.0, 0.1 - low_matches * 0.03)
        return 0.2  # Neutral baseline

    def _keyword_detail(self, goal_lower: str) -> str:
        """Generate a human-readable detail for keyword analysis."""
        matched: list[str] = []
        for kw in CRITICAL_RISK_KEYWORDS:
            if kw in goal_lower:
                matched.append(f"CRITICAL:{kw}")
        for kw in HIGH_RISK_KEYWORDS:
            if kw in goal_lower:
                matched.append(f"HIGH:{kw}")
        for kw in LOW_RISK_KEYWORDS:
            if kw in goal_lower:
                matched.append(f"LOW:{kw}")
        if matched:
            return f"Keywords matched: {', '.join(matched[:5])}"
        return "No specific risk keywords detected"

    def _score_file_scope(self, file_scope: list[str]) -> float:
        """Score based on the files that will be modified."""
        if not file_scope:
            return 0.15  # Unknown scope adds mild risk

        max_file_risk = 0.0
        for fp in file_scope:
            fp_lower = fp.lower()
            for path in CRITICAL_RISK_PATHS:
                if path.lower() in fp_lower:
                    max_file_risk = max(max_file_risk, 0.9)
            for path in HIGH_RISK_PATHS:
                if path.lower() in fp_lower:
                    max_file_risk = max(max_file_risk, 0.6)
            for path in LOW_RISK_PATHS:
                if path.lower() in fp_lower:
                    max_file_risk = max(max_file_risk, 0.05)

        return max_file_risk if max_file_risk > 0 else 0.15

    def _detect_protected_files(
        self, goal_lower: str, file_scope: list[str]
    ) -> list[str]:
        """Detect if any protected files are referenced."""
        hits: list[str] = []
        for pf in self.protected_files:
            pf_lower = pf.lower()
            pf_name = pf.split("/")[-1].lower()

            # Check file_scope
            for fp in file_scope:
                fp_lower_path = fp.lower()
                if pf_lower in fp_lower_path or fp_lower_path.endswith(pf_name):
                    if pf not in hits:
                        hits.append(pf)

            # Check goal text for explicit mentions
            if pf_name in goal_lower and pf not in hits:
                hits.append(pf)

        return hits

    def _score_scale(
        self,
        file_scope: list[str],
        estimated_files: int | None,
        complexity: int | None,
    ) -> float:
        """Score based on the scale and complexity of the change."""
        score = 0.0

        # File count factor
        file_count = estimated_files or len(file_scope)
        if file_count > 20:
            score = max(score, 0.7)
        elif file_count > 10:
            score = max(score, 0.5)
        elif file_count > 5:
            score = max(score, 0.3)
        elif file_count > 0:
            score = max(score, 0.1)

        # Complexity factor
        if complexity is not None:
            if complexity >= 8:
                score = max(score, 0.6)
            elif complexity >= 5:
                score = max(score, 0.3)
            elif complexity >= 3:
                score = max(score, 0.15)

        return score

    def _scale_detail(
        self,
        file_scope: list[str],
        estimated_files: int | None,
        complexity: int | None,
    ) -> str:
        """Generate a human-readable detail for scale analysis."""
        parts: list[str] = []
        file_count = estimated_files or len(file_scope)
        if file_count > 0:
            parts.append(f"{file_count} file(s)")
        if complexity is not None:
            parts.append(f"complexity={complexity}/10")
        return f"Scale: {', '.join(parts)}" if parts else "Scale: minimal"

    def _score_test_coverage(
        self, has_tests: bool | None, goal_lower: str, file_scope: list[str]
    ) -> float:
        """Score based on whether changes include tests."""
        # If tests are explicitly included, reduce risk
        if has_tests is True:
            return 0.0

        # If the goal is about writing tests, low risk
        if any(kw in goal_lower for kw in ["add test", "write test", "test coverage"]):
            return 0.0

        # If file_scope includes test files, low risk
        if any("test" in fp.lower() for fp in file_scope):
            return 0.0

        # Changes without tests add risk
        if has_tests is False:
            return 0.3

        # Unknown test status
        return 0.1

    def _aggregate_score(
        self, factors: list[RiskFactor], protected_hits: list[str]
    ) -> float:
        """Aggregate factor weights into a final score.

        Protected file hits immediately push score to CRITICAL range.
        Critical-risk file paths (>= 0.9) also push to critical range.
        Otherwise, uses max-dominated blend.
        """
        if protected_hits:
            # Protected files always push to critical
            return 0.9

        if not factors:
            return 0.1  # No signals = low risk

        # If any factor is in the critical range (file_scope hitting
        # CRITICAL_RISK_PATHS), preserve that signal
        file_scope_factor = next(
            (f for f in factors if f.name == "file_scope"), None
        )
        if file_scope_factor and file_scope_factor.weight >= 0.85:
            return max(0.8, file_scope_factor.weight)

        # Use a combination of max factor and weighted average
        max_weight = max(f.weight for f in factors)
        avg_weight = sum(f.weight for f in factors) / len(factors)

        # Blend: 75% max factor, 25% average (max strongly dominates)
        return 0.75 * max_weight + 0.25 * avg_weight

    def _categorize(self, score: float) -> RiskCategory:
        """Map a numeric score to a risk category."""
        if score >= 0.8:
            return RiskCategory.CRITICAL
        if score >= 0.5:
            return RiskCategory.HIGH
        if score >= 0.3:
            return RiskCategory.MEDIUM
        return RiskCategory.LOW

    def _recommend(self, score: float) -> str:
        """Determine execution recommendation based on score and thresholds."""
        if score >= self.block_threshold:
            return "block"
        if score >= self.threshold:
            return "review"
        return "auto"
