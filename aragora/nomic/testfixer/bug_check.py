"""Post-fix static analysis validation for TestFixer.

Runs BugDetector on patched files before and after a fix to ensure
the fix does not introduce new high-severity bugs.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class BugCheckResult:
    """Result of a post-fix bug check."""

    new_bugs: list[Any] = field(default_factory=list)
    resolved_bugs: list[Any] = field(default_factory=list)
    passes: bool = True
    summary: str = ""


class PostFixBugChecker:
    """Check patched files for newly introduced bugs.

    Scans each patched file before and after the fix using BugDetector,
    then diffs the bug lists. The check passes if no new CRITICAL or
    HIGH severity bugs are introduced.
    """

    def __init__(
        self,
        repo_path: Path,
        detector: Any | None = None,
    ) -> None:
        self._repo_path = Path(repo_path)
        self._detector = detector
        if self._detector is None:
            try:
                from aragora.audit.bug_detector import BugDetector

                self._detector = BugDetector(
                    include_low_severity=False,
                    include_info=False,
                    include_smells=False,
                )
            except Exception as exc:
                logger.warning("bug_check.detector_unavailable error=%s", exc)

    def check_patches(self, proposal: Any) -> BugCheckResult:
        """Check a PatchProposal for newly introduced bugs.

        Compares bugs in each patched file before and after the fix.
        A patch passes if it introduces no new CRITICAL or HIGH bugs.

        Args:
            proposal: PatchProposal with patches to check.

        Returns:
            BugCheckResult with new/resolved bugs and pass status.
        """
        if self._detector is None:
            return BugCheckResult(
                passes=True,
                summary="Bug detector unavailable, skipping check",
            )

        all_new: list[Any] = []
        all_resolved: list[Any] = []

        for patch in proposal.patches:
            file_path = self._repo_path / patch.file_path
            if not file_path.exists():
                continue

            try:
                after_bugs = self._detector.detect_in_file(str(file_path))
            except Exception as exc:
                logger.warning(
                    "bug_check.scan_error file=%s error=%s",
                    patch.file_path,
                    exc,
                )
                continue

            # Compute before-bugs from original content
            before_bugs = self._scan_content(patch.original_content, patch.file_path)

            before_ids = {self._bug_key(b) for b in before_bugs}
            after_ids = {self._bug_key(b) for b in after_bugs}

            new_bugs = [b for b in after_bugs if self._bug_key(b) not in before_ids]
            resolved = [b for b in before_bugs if self._bug_key(b) not in after_ids]

            all_new.extend(new_bugs)
            all_resolved.extend(resolved)

        # Check severity of new bugs
        high_severity_new = [
            b for b in all_new if getattr(b, "severity", None)
            in _HIGH_SEVERITIES
        ]

        passes = len(high_severity_new) == 0
        summary_parts = [
            f"{len(all_new)} new bug(s)",
            f"{len(all_resolved)} resolved bug(s)",
        ]
        if high_severity_new:
            summary_parts.append(
                f"{len(high_severity_new)} high/critical severity"
            )

        return BugCheckResult(
            new_bugs=all_new,
            resolved_bugs=all_resolved,
            passes=passes,
            summary=", ".join(summary_parts),
        )

    def _scan_content(self, content: str, relative_path: str) -> list[Any]:
        """Scan file content for bugs by writing to temp and scanning."""
        import tempfile

        if not content or self._detector is None:
            return []

        try:
            suffix = Path(relative_path).suffix or ".py"
            with tempfile.NamedTemporaryFile(
                mode="w",
                suffix=suffix,
                delete=False,
                encoding="utf-8",
            ) as tmp:
                tmp.write(content)
                tmp.flush()
                return self._detector.detect_in_file(tmp.name)
        except Exception as exc:
            logger.warning(
                "bug_check.scan_content_error file=%s error=%s",
                relative_path,
                exc,
            )
            return []
        finally:
            try:
                Path(tmp.name).unlink(missing_ok=True)
            except OSError:
                pass

    @staticmethod
    def _bug_key(bug: Any) -> str:
        """Create a stable key for a bug to enable before/after diffing."""
        return (
            f"{getattr(bug, 'category', '')}:"
            f"{getattr(bug, 'line_number', '')}:"
            f"{getattr(bug, 'pattern_name', '')}:"
            f"{getattr(bug, 'title', '')}"
        )


# Severity values that cause a check to fail
try:
    from aragora.audit.bug_detector import BugSeverity

    _HIGH_SEVERITIES = {BugSeverity.CRITICAL, BugSeverity.HIGH}
except ImportError:
    _HIGH_SEVERITIES = {"critical", "high"}
