"""Domain classification helpers for Arena debates.

Extracted from orchestrator.py to reduce its size. These functions handle
task domain detection for vertical-specific optimizations.
"""

from __future__ import annotations

from functools import lru_cache

from aragora.utils.cache_registry import register_lru_cache


@register_lru_cache
@lru_cache(maxsize=1024)
def compute_domain_from_task(task_lower: str) -> str:
    """Compute domain from lowercased task string.

    Module-level cached helper to avoid O(n) string matching
    for repeated task strings across debate instances.

    Args:
        task_lower: Lowercased task description string

    Returns:
        Domain name: security, performance, testing, architecture,
        debugging, api, database, frontend, or general
    """
    if any(w in task_lower for w in ("security", "hack", "vulnerability", "auth", "encrypt")):
        return "security"
    if any(w in task_lower for w in ("performance", "speed", "optimize", "cache", "latency")):
        return "performance"
    if any(w in task_lower for w in ("test", "testing", "coverage", "regression")):
        return "testing"
    if any(w in task_lower for w in ("design", "architecture", "pattern", "structure")):
        return "architecture"
    if any(w in task_lower for w in ("bug", "error", "fix", "crash", "exception")):
        return "debugging"
    if any(w in task_lower for w in ("api", "endpoint", "rest", "graphql")):
        return "api"
    if any(w in task_lower for w in ("database", "sql", "query", "schema")):
        return "database"
    if any(w in task_lower for w in ("ui", "frontend", "react", "css", "layout")):
        return "frontend"
    return "general"


# Backward compatibility alias
_compute_domain_from_task = compute_domain_from_task


__all__ = ["compute_domain_from_task", "_compute_domain_from_task"]
