"""
Protected file checksum verification for nomic loop safety.

Ensures critical infrastructure files are not accidentally or
maliciously modified during the self-improvement cycle.
"""

import hashlib
from pathlib import Path


# Files that must NEVER be deleted or broken
PROTECTED_FILES = [
    # Core nomic loop infrastructure
    "scripts/nomic_loop.py",  # The nomic loop itself - CRITICAL
    "scripts/run_nomic_with_stream.py",  # Streaming wrapper - protects --auto flag
    # Core aragora modules
    "aragora/__init__.py",  # Core package initialization
    "aragora/core/__init__.py",  # Core package surface
    "aragora/core_types.py",  # Core types and abstractions
    "aragora/debate/orchestrator.py",  # Debate infrastructure
    "aragora/agents/__init__.py",  # Agent system
    "aragora/implement/__init__.py",  # Implementation system
    # Valuable features added by nomic loop
    "aragora/agents/cli_agents.py",  # CLI agent harnesses (KiloCode, Claude, Codex, Grok)
    "aragora/server/stream.py",  # Streaming, AudienceInbox, TokenBucket
    "aragora/memory/store.py",  # CritiqueStore, AgentReputation
    "aragora/debate/embeddings.py",  # DebateEmbeddingsDatabase for historical search
    # Live dashboard (web interface)
    "aragora/live/src/components/AgentPanel.tsx",  # Agent activity panel with colors
    "aragora/live/src/components/UserParticipation.tsx",  # User participation UI
    "aragora/live/src/app/page.tsx",  # Main dashboard page
    "aragora/live/tailwind.config.js",  # Tailwind config with agent colors
]

# Global cache for protected file checksums (computed at startup)
_PROTECTED_FILE_CHECKSUMS: dict[str, str] = {}


def compute_file_checksum(filepath: Path) -> str:
    """Compute SHA-256 checksum of a file."""
    if not filepath.exists():
        return ""
    with open(filepath, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()[:16]  # Short hash for logging


def init_protected_checksums(base_path: Path) -> dict[str, str]:
    """Initialize checksums for all protected files at startup."""
    global _PROTECTED_FILE_CHECKSUMS
    for rel_path in PROTECTED_FILES:
        full_path = base_path / rel_path
        if full_path.exists():
            _PROTECTED_FILE_CHECKSUMS[rel_path] = compute_file_checksum(full_path)
    return _PROTECTED_FILE_CHECKSUMS


def verify_protected_files_unchanged(base_path: Path) -> tuple[bool, list[str]]:
    """
    Verify that protected files haven't been unexpectedly modified.

    Security measure: Detects accidental or malicious modifications to
    critical infrastructure between phases.

    Returns:
        Tuple of (all_ok, list of modified files)
    """
    modified = []
    for rel_path, expected_hash in _PROTECTED_FILE_CHECKSUMS.items():
        full_path = base_path / rel_path
        current_hash = compute_file_checksum(full_path)
        if current_hash != expected_hash:
            modified.append(rel_path)
    return len(modified) == 0, modified


def get_protected_checksums() -> dict[str, str]:
    """Get the current protected file checksums."""
    return _PROTECTED_FILE_CHECKSUMS.copy()


SAFETY_PREAMBLE = """
=== CRITICAL SAFETY RULES ===
You are modifying a self-improving system. These rules are NON-NEGOTIABLE:

1. NEVER DELETE OR BREAK:
   - scripts/nomic_loop.py (the loop itself)
   - aragora/__init__.py (core package)
   - aragora/core.py (core types)
   - aragora/debate/orchestrator.py (debate infrastructure)
   - Any file that enables the nomic loop to function

2. ANABOLISM OVER CATABOLISM:
   - ADD features, don't remove working ones
   - EXTEND functionality, don't simplify it away
   - Only remove code that is BROKEN or HARMFUL
   - When in doubt, keep existing functionality

3. PRESERVE CORE CAPABILITIES:
   - Multi-agent debate must keep working
   - File logging must keep working
   - Git integration must keep working
   - All existing API contracts must be maintained

4. DEFENSIVE CODING:
   - New features should not break existing ones
   - Add tests for new functionality
   - Maintain backward compatibility

5. TECHNICAL DEBT - REDUCE SAFELY:
   - Reducing technical debt is GOOD when it's safe
   - Safe refactoring: improve code without changing behavior
   - UNSAFE: removing functionality, breaking APIs, deleting imports
   - SAFE: renaming for clarity, extracting functions, improving types
   - Test that refactored code works identically to original
   - If unsure whether a change is safe, DON'T MAKE IT

6. AGENT PROMPTS ARE SACRED:
   - NEVER modify agent system prompts in the codebase
   - Agent prompts define the personalities and safety constraints
   - Changes to agent prompts require UNANIMOUS consent from all agents
   - If ANY doubt exists about modifying prompts, DO NOT MODIFY THEM
   - This includes prompts in: nomic_loop.py, agents/*.py, any prompt templates
   - The only exception: fixing an obvious typo or syntax error
===========================
"""
