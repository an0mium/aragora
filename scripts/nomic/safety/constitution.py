"""
Cryptographically signed Constitution for nomic loop safety.

The Constitution defines immutable rules that the nomic loop cannot modify,
preventing wireheading and ensuring safety invariants are maintained.

Key management:
- Private key: ARAGORA_CONSTITUTION_KEY environment variable (base64 Ed25519)
- Public key: Embedded in this module for verification
- Signing: Use scripts/sign_constitution.py offline tool

The Constitution protects against:
1. Agents modifying judging criteria to be trivially satisfied
2. Deletion of safety mechanisms (backup, restore, checksum)
3. Single-agent takeover via consensus manipulation
4. Gradual drift of safety rules through prompt modification
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, Optional

logger = logging.getLogger(__name__)

# Default public key - replace with actual key after generation
# This is a placeholder that will be overwritten by sign_constitution.py
DEFAULT_PUBLIC_KEY = b""

# Constitution file location
DEFAULT_CONSTITUTION_PATH = Path(".nomic/constitution.json")


@dataclass
class ConstitutionRule:
    """A single rule in the Constitution."""

    id: str  # Unique identifier (e.g., "CORE-001")
    category: Literal["immutable", "amendable", "advisory"]
    rule: str  # The actual rule text
    rationale: str  # Why this rule exists

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "category": self.category,
            "rule": self.rule,
            "rationale": self.rationale,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ConstitutionRule:
        return cls(
            id=data["id"],
            category=data["category"],
            rule=data["rule"],
            rationale=data["rationale"],
        )


@dataclass
class Constitution:
    """The complete Constitution document."""

    version: str
    rules: list[ConstitutionRule]
    protected_files: list[str]
    protected_functions: dict[str, list[str]]  # file -> function names
    amendment_threshold: float  # Fraction of agents needed to amend (0.0-1.0)
    signature: str = ""  # Ed25519 signature (base64)
    signed_at: str = ""  # ISO timestamp

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "rules": [r.to_dict() for r in self.rules],
            "protected_files": self.protected_files,
            "protected_functions": self.protected_functions,
            "amendment_threshold": self.amendment_threshold,
            "signature": self.signature,
            "signed_at": self.signed_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Constitution:
        return cls(
            version=data["version"],
            rules=[ConstitutionRule.from_dict(r) for r in data["rules"]],
            protected_files=data["protected_files"],
            protected_functions=data["protected_functions"],
            amendment_threshold=data["amendment_threshold"],
            signature=data.get("signature", ""),
            signed_at=data.get("signed_at", ""),
        )

    def get_signable_content(self) -> bytes:
        """Get the content that should be signed (excludes signature fields)."""
        content = {
            "version": self.version,
            "rules": [r.to_dict() for r in self.rules],
            "protected_files": self.protected_files,
            "protected_functions": self.protected_functions,
            "amendment_threshold": self.amendment_threshold,
        }
        # Use sorted keys and no whitespace for deterministic serialization
        return json.dumps(content, sort_keys=True, separators=(",", ":")).encode("utf-8")

    def get_immutable_rules(self) -> list[ConstitutionRule]:
        """Get only immutable rules (cannot be amended)."""
        return [r for r in self.rules if r.category == "immutable"]

    def get_amendable_rules(self) -> list[ConstitutionRule]:
        """Get amendable rules (can be changed with sufficient consensus)."""
        return [r for r in self.rules if r.category == "amendable"]


@dataclass
class ConstitutionViolation:
    """Details of a Constitution violation."""

    rule_id: str
    rule_text: str
    violation_type: Literal["file_modification", "function_modification", "rule_violation"]
    details: str
    severity: Literal["critical", "high", "medium"]


class ConstitutionVerifier:
    """Verifies Constitution integrity and checks for violations.

    Usage:
        verifier = ConstitutionVerifier()
        if not verifier.verify_signature():
            abort_cycle("Constitution tampered")

        allowed, reason = verifier.check_file_modification_allowed("path/to/file.py", diff)
        if not allowed:
            rollback_file("path/to/file.py")
    """

    def __init__(
        self,
        constitution_path: Path = DEFAULT_CONSTITUTION_PATH,
        public_key: Optional[bytes] = None,
    ):
        self.path = Path(constitution_path)
        self.public_key = public_key or self._load_public_key()
        self._constitution: Optional[Constitution] = None
        self._load_constitution()

    def _load_public_key(self) -> bytes:
        """Load public key from environment or default."""
        key_b64 = os.environ.get("ARAGORA_CONSTITUTION_PUBLIC_KEY", "")
        if key_b64:
            return base64.b64decode(key_b64)
        return DEFAULT_PUBLIC_KEY

    def _load_constitution(self) -> None:
        """Load Constitution from file."""
        if not self.path.exists():
            logger.warning(f"Constitution not found at {self.path}")
            self._constitution = None
            return

        try:
            with open(self.path) as f:
                data = json.load(f)
            self._constitution = Constitution.from_dict(data)
        except Exception as e:
            logger.error(f"Failed to load Constitution: {e}")
            self._constitution = None

    @property
    def constitution(self) -> Optional[Constitution]:
        """Get loaded Constitution."""
        return self._constitution

    def is_available(self) -> bool:
        """Check if Constitution is loaded and valid."""
        return self._constitution is not None

    def verify_signature(self) -> bool:
        """Verify Constitution hasn't been tampered with.

        Returns True if:
        - No public key configured (signature verification disabled)
        - Signature is valid

        Returns False if:
        - Constitution not loaded
        - Signature verification fails
        """
        if not self._constitution:
            logger.error("Cannot verify signature: Constitution not loaded")
            return False

        if not self.public_key:
            logger.info("Constitution signature verification disabled (no public key)")
            return True  # No key = verification disabled

        if not self._constitution.signature:
            logger.warning("Constitution has no signature")
            return False

        try:
            from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

            public_key = Ed25519PublicKey.from_public_bytes(self.public_key)
            signature = base64.b64decode(self._constitution.signature)
            content = self._constitution.get_signable_content()

            public_key.verify(signature, content)
            logger.info(f"Constitution signature valid (version {self._constitution.version})")
            return True

        except ImportError:
            logger.warning("cryptography library not installed - signature verification disabled")
            return True
        except Exception as e:
            logger.error(f"Constitution signature verification failed: {e}")
            return False

    def check_phase_allowed(
        self,
        phase: str,
        proposed_changes: list[str],
    ) -> tuple[bool, str]:
        """Check if proposed changes in a phase violate Constitution rules.

        Args:
            phase: The nomic phase (e.g., "implement", "verify")
            proposed_changes: List of file paths to be modified

        Returns:
            (allowed, reason) - True if allowed, False with reason if blocked
        """
        if not self._constitution:
            return True, ""  # No constitution = allow all

        violations: list[str] = []

        # Check protected files
        for file_path in proposed_changes:
            normalized = self._normalize_path(file_path)
            if normalized in self._constitution.protected_files:
                violations.append(f"Cannot modify protected file: {normalized}")

        if violations:
            return False, "; ".join(violations)

        return True, ""

    def check_file_modification_allowed(
        self,
        file_path: str,
        diff: str,
    ) -> tuple[bool, str]:
        """Check if a specific file modification respects Constitution rules.

        Args:
            file_path: Path to the modified file
            diff: The diff content showing changes

        Returns:
            (allowed, reason) - True if allowed, False with reason if blocked
        """
        if not self._constitution:
            return True, ""  # No constitution = allow all

        normalized = self._normalize_path(file_path)

        # Check if file is completely protected
        if normalized in self._constitution.protected_files:
            return False, f"File {normalized} is protected by Constitution"

        # Check if specific functions are protected
        if normalized in self._constitution.protected_functions:
            protected_funcs = self._constitution.protected_functions[normalized]
            for func_name in protected_funcs:
                if self._diff_modifies_function(diff, func_name):
                    return (
                        False,
                        f"Function {func_name} in {normalized} is protected by Constitution",
                    )

        return True, ""

    def _normalize_path(self, path: str) -> str:
        """Normalize file path for comparison."""
        # Remove leading ./ or absolute path prefix
        path = str(path)
        if path.startswith("./"):
            path = path[2:]
        if path.startswith("/"):
            # Try to extract relative path from repo root
            repo_markers = ["aragora/", "scripts/", "tests/", ".nomic/"]
            for marker in repo_markers:
                if marker in path:
                    idx = path.index(marker)
                    path = path[idx:]
                    break
        return path

    def _diff_modifies_function(self, diff: str, func_name: str) -> bool:
        """Check if a diff modifies a specific function.

        Uses simple heuristics - looks for function definition being changed.
        """
        # Look for function definition patterns in changed lines
        patterns = [
            rf"^[-+]\s*def {func_name}\s*\(",  # Python function
            rf"^[-+]\s*async def {func_name}\s*\(",  # Python async function
            rf"^[-+].*\bfunction {func_name}\s*\(",  # JavaScript function
        ]

        for line in diff.split("\n"):
            for pattern in patterns:
                if re.match(pattern, line):
                    return True

        return False

    def get_violations(
        self,
        file_path: str,
        content: str,
    ) -> list[ConstitutionViolation]:
        """Check content for any Constitution rule violations.

        Args:
            file_path: Path to the file
            content: Full file content

        Returns:
            List of violations found
        """
        if not self._constitution:
            return []

        violations: list[ConstitutionViolation] = []

        # Check for rule-specific violations
        for rule in self._constitution.rules:
            violation = self._check_rule_violation(rule, file_path, content)
            if violation:
                violations.append(violation)

        return violations

    def _check_rule_violation(
        self,
        rule: ConstitutionRule,
        file_path: str,
        content: str,
    ) -> Optional[ConstitutionViolation]:
        """Check if content violates a specific rule."""
        rule_id = rule.id

        # CORE-001: Never delete backup/restore mechanism
        if rule_id == "CORE-001" and "backups.py" in file_path:
            if "def restore_backup" not in content or "def create_backup" not in content:
                return ConstitutionViolation(
                    rule_id=rule_id,
                    rule_text=rule.rule,
                    violation_type="function_modification",
                    details="Backup/restore functions appear to be removed",
                    severity="critical",
                )

        # CORE-002: Never modify Constitution verification
        if rule_id == "CORE-002" and "constitution.py" in file_path:
            if "class ConstitutionVerifier" not in content:
                return ConstitutionViolation(
                    rule_id=rule_id,
                    rule_text=rule.rule,
                    violation_type="function_modification",
                    details="Constitution verifier appears to be removed",
                    severity="critical",
                )

        # EVAL-001: Judging must include objective metrics
        if rule_id == "EVAL-001" and "nomic_loop.py" in file_path:
            # Check that test-based verification exists
            if "phase_verify" in content.lower():
                if "test" not in content.lower() and "pytest" not in content.lower():
                    return ConstitutionViolation(
                        rule_id=rule_id,
                        rule_text=rule.rule,
                        violation_type="rule_violation",
                        details="Verification phase appears to lack test-based evaluation",
                        severity="high",
                    )

        return None


def create_default_constitution() -> Constitution:
    """Create a default Constitution with core safety rules."""
    return Constitution(
        version="1.0.0",
        rules=[
            ConstitutionRule(
                id="CORE-001",
                category="immutable",
                rule="Never delete or disable the backup/restore mechanism",
                rationale="Rollback is the last line of defense against catastrophic changes",
            ),
            ConstitutionRule(
                id="CORE-002",
                category="immutable",
                rule="Never modify the Constitution verification code",
                rationale="Prevents self-disabling of safety checks",
            ),
            ConstitutionRule(
                id="CORE-003",
                category="immutable",
                rule="All code changes must pass existing tests before commit",
                rationale="Prevents breaking changes from being deployed",
            ),
            ConstitutionRule(
                id="EVAL-001",
                category="immutable",
                rule="Judging criteria must include objective metrics (test pass/fail, syntax validity)",
                rationale="Prevents wireheading via subjective-only evaluation",
            ),
            ConstitutionRule(
                id="AGENT-001",
                category="amendable",
                rule="Minimum 3 agents required for consensus",
                rationale="Prevents single-agent takeover",
            ),
            ConstitutionRule(
                id="AGENT-002",
                category="amendable",
                rule="No single agent can have veto power over consensus",
                rationale="Ensures democratic decision-making",
            ),
            ConstitutionRule(
                id="SAFETY-001",
                category="immutable",
                rule="Changes to protected files require human approval",
                rationale="Maintains human oversight of critical infrastructure",
            ),
        ],
        protected_files=[
            "scripts/nomic/safety/constitution.py",
            "scripts/nomic/safety/checksums.py",
            "scripts/nomic/safety/backups.py",
            ".nomic/constitution.json",
        ],
        protected_functions={
            "scripts/nomic_loop.py": [
                "verify_constitution",
                "restore_backup",
                "_rollback_to_backup",
            ],
            "scripts/nomic/safety/checksums.py": ["verify_protected_files_unchanged"],
            "scripts/nomic/safety/backups.py": ["restore_backup", "create_backup"],
        },
        amendment_threshold=0.75,
    )


def save_constitution(constitution: Constitution, path: Path = DEFAULT_CONSTITUTION_PATH) -> None:
    """Save Constitution to file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(constitution.to_dict(), f, indent=2)
    logger.info(f"Constitution saved to {path}")
