"""
Domain detection for task routing.

Provides keyword-based domain detection from natural language task descriptions.
"""

import re
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from aragora.routing.selection import TaskRequirements

# Domain detection keywords
DOMAIN_KEYWORDS: dict[str, list[str]] = {
    "security": [
        "security",
        "auth",
        "authentication",
        "authorization",
        "encrypt",
        "vulnerability",
        "attack",
        "xss",
        "sql injection",
        "csrf",
        "token",
        "password",
        "credential",
        "permission",
        "access control",
        "firewall",
        "sanitize",
        "validate input",
        "owasp",
        "rate limit",
        "brute force",
    ],
    "performance": [
        "performance",
        "optimize",
        "speed",
        "latency",
        "cache",
        "caching",
        "memory",
        "cpu",
        "throughput",
        "bottleneck",
        "profil",
        "benchmark",
        "slow",
        "fast",
        "efficient",
        "scale",
        "scaling",
        "load",
        "concurrent",
    ],
    "architecture": [
        "architecture",
        "design",
        "pattern",
        "refactor",
        "structure",
        "modular",
        "decouple",
        "interface",
        "abstract",
        "dependency",
        "solid",
        "dry",
        "single responsibility",
        "microservice",
        "monolith",
    ],
    "testing": [
        "test",
        "testing",
        "unittest",
        "pytest",
        "mock",
        "coverage",
        "integration test",
        "e2e",
        "end-to-end",
        "tdd",
        "bdd",
        "fixture",
        "assertion",
        "spec",
        "verify",
        "validate",
    ],
    "api": [
        "api",
        "endpoint",
        "rest",
        "graphql",
        "grpc",
        "http",
        "request",
        "response",
        "route",
        "handler",
        "middleware",
        "cors",
        "versioning",
        "openapi",
        "swagger",
        "webhook",
    ],
    "database": [
        "database",
        "db",
        "sql",
        "query",
        "schema",
        "migration",
        "index",
        "transaction",
        "orm",
        "postgresql",
        "mysql",
        "sqlite",
        "mongodb",
        "redis",
        "cache",
        "nosql",
        "join",
        "foreign key",
    ],
    "frontend": [
        "frontend",
        "ui",
        "ux",
        "react",
        "vue",
        "angular",
        "css",
        "html",
        "component",
        "render",
        "state",
        "redux",
        "hook",
        "responsive",
        "accessibility",
        "a11y",
        "animation",
    ],
    "devops": [
        "deploy",
        "deployment",
        "ci",
        "cd",
        "docker",
        "kubernetes",
        "k8s",
        "pipeline",
        "github actions",
        "terraform",
        "aws",
        "cloud",
        "infra",
        "monitoring",
        "logging",
        "observability",
    ],
    "debugging": [
        "debug",
        "bug",
        "fix",
        "error",
        "exception",
        "traceback",
        "crash",
        "issue",
        "problem",
        "broken",
        "fail",
        "not working",
        "investigate",
    ],
    "documentation": [
        "document",
        "readme",
        "docstring",
        "comment",
        "explain",
        "tutorial",
        "guide",
        "specification",
        "api doc",
    ],
    "ethics": [
        "ethics",
        "ethical",
        "fairness",
        "bias",
        "privacy",
        "consent",
        "responsible",
        "governance",
        "compliance",
        "gdpr",
        "moral",
        "transparency",
        "accountability",
        "harm",
        "safety",
        "alignment",
    ],
    "philosophy": [
        "philosophy",
        "philosophical",
        "epistemology",
        "epistemological",
        "ontology",
        "ontological",
        "logic",
        "logical",
        "reasoning",
        "argument",
        "premise",
        "conclusion",
        "fallacy",
        "dialectic",
        "metaphysics",
        "metaphysical",
        "theory of",
        "concept",
        "definition",
        "truth claim",
        "knowledge",
        "belief",
        "justify",
        "justification",
        "foundational",
        "first principles",
    ],
    "data_analysis": [
        "data",
        "analysis",
        "dataset",
        "statistics",
        "statistical",
        "pandas",
        "numpy",
        "visualization",
        "chart",
        "plot",
        "correlation",
        "regression",
        "machine learning",
        "ml",
        "prediction",
        "model",
        "feature",
        "training",
        "jupyter",
        "notebook",
        "csv",
        "json",
        "etl",
        "pipeline",
    ],
    "general": [
        "implement",
        "create",
        "build",
        "add",
        "update",
        "change",
        "modify",
        "code",
        "function",
        "class",
        "method",
        "module",
        "library",
        "package",
    ],
}


class DomainDetector:
    """
    Detects task domain from natural language description.

    Uses keyword matching with weighted scoring to identify
    the primary and secondary domains for a task.
    """

    def __init__(self, custom_keywords: Optional[dict[str, list[str]]] = None):
        """Initialize with optional custom domain keywords."""
        self.keywords = DOMAIN_KEYWORDS.copy()
        if custom_keywords:
            for domain, words in custom_keywords.items():
                if domain in self.keywords:
                    self.keywords[domain].extend(words)
                else:
                    self.keywords[domain] = words

    def detect(self, task_text: str, top_n: int = 3) -> list[tuple[str, float]]:
        """
        Detect domains from task text.

        Args:
            task_text: The task description
            top_n: Number of top domains to return

        Returns:
            List of (domain, confidence) tuples, sorted by confidence
        """
        text_lower = task_text.lower()
        scores: dict[str, float] = {}

        for domain, keywords in self.keywords.items():
            score = 0.0
            for keyword in keywords:
                # Count occurrences with word boundaries
                pattern = r"\b" + re.escape(keyword) + r"\b"
                matches = len(re.findall(pattern, text_lower))
                if matches > 0:
                    # Longer keywords are more specific, weight them higher
                    weight = 1.0 + len(keyword.split()) * 0.5
                    score += matches * weight

            if score > 0:
                scores[domain] = score

        # Normalize scores
        if scores:
            max_score = max(scores.values())
            scores = {d: s / max_score for d, s in scores.items()}

        # Sort by score descending
        sorted_domains = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        # Default to "general" if no domains detected
        if not sorted_domains:
            return [("general", 0.5)]

        return sorted_domains[:top_n]

    def get_primary_domain(self, task_text: str) -> str:
        """Get the primary domain for a task."""
        domains = self.detect(task_text, top_n=1)
        return domains[0][0] if domains else "general"

    def get_task_requirements(
        self,
        task_text: str,
        task_id: Optional[str] = None,
    ) -> "TaskRequirements":
        """
        Create TaskRequirements from task text with auto-detected domains.

        Args:
            task_text: The task description
            task_id: Optional task identifier

        Returns:
            TaskRequirements with detected domains
        """
        # Import here to avoid circular imports
        from aragora.routing.selection import TaskRequirements

        domains = self.detect(task_text, top_n=3)

        primary = domains[0][0] if domains else "general"
        secondary = [d for d, _ in domains[1:] if _ > 0.3]  # Only include confident secondary

        # Detect required traits from keywords
        traits = []
        text_lower = task_text.lower()
        if any(w in text_lower for w in ["critical", "important", "careful"]):
            traits.append("thorough")
        if any(w in text_lower for w in ["fast", "quick", "asap"]):
            traits.append("fast")
        if any(w in text_lower for w in ["security", "secure", "safe"]):
            traits.append("security")
        if any(w in text_lower for w in ["creative", "novel", "innovative"]):
            traits.append("creative")

        return TaskRequirements(
            task_id=task_id or f"task-{hash(task_text) % 10000:04d}",
            description=task_text[:500],
            primary_domain=primary,
            secondary_domains=secondary,
            required_traits=traits,
        )


__all__ = [
    "DOMAIN_KEYWORDS",
    "DomainDetector",
]
