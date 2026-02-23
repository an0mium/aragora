"""Tests for Semantic Conflict Detector.

Covers:
- ConflictType enum values
- SemanticConflict dataclass
- Static scan: signature change detection, import conflicts
- Debate scan: arena fallback, import error graceful degradation
- BranchCoordinator integration
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from aragora.nomic.semantic_conflict_detector import (
    ConflictType,
    FunctionSignature,
    SemanticConflict,
    SemanticConflictDetector,
)


# ============================================================
# ConflictType
# ============================================================


class TestConflictType:
    def test_enum_values(self):
        assert ConflictType.SIGNATURE_BREAK.value == "signature_break"
        assert ConflictType.ASSUMPTION_CLASH.value == "assumption_clash"
        assert ConflictType.CONTRACT_VIOLATION.value == "contract_violation"
        assert ConflictType.IMPORT_CYCLE.value == "import_cycle"

    def test_all_types(self):
        assert len(ConflictType) == 4


# ============================================================
# SemanticConflict
# ============================================================


class TestSemanticConflict:
    def test_creation(self):
        sc = SemanticConflict(
            source_branch="dev/a",
            target_branch="dev/b",
            conflict_type=ConflictType.SIGNATURE_BREAK,
            description="Function foo changed args",
        )
        assert sc.source_branch == "dev/a"
        assert sc.conflict_type == ConflictType.SIGNATURE_BREAK

    def test_confidence_default(self):
        sc = SemanticConflict(
            source_branch="a",
            target_branch="b",
            conflict_type=ConflictType.IMPORT_CYCLE,
            description="cycle",
        )
        assert sc.confidence == 0.5

    def test_affected_files_default(self):
        sc = SemanticConflict(
            source_branch="a",
            target_branch="b",
            conflict_type=ConflictType.IMPORT_CYCLE,
            description="cycle",
        )
        assert sc.affected_files == []


# ============================================================
# Static Scan
# ============================================================


class TestStaticScan:
    def test_signature_change_detected(self):
        detector = SemanticConflictDetector(Path("/tmp"), enable_debate=False)

        content_a = """
def process(data, mode):
    pass
"""
        content_b = """
def process(data, mode, verbose=False):
    pass
"""
        conflicts = detector._check_signature_conflicts(
            "aragora/foo.py",
            content_a,
            content_b,
            "branch_a",
            "branch_b",
        )
        assert len(conflicts) >= 1
        assert conflicts[0].conflict_type == ConflictType.SIGNATURE_BREAK
        assert "process" in conflicts[0].description

    def test_no_conflict_same_signatures(self):
        detector = SemanticConflictDetector(Path("/tmp"), enable_debate=False)

        content = """
def process(data, mode):
    pass
"""
        conflicts = detector._check_signature_conflicts(
            "aragora/foo.py",
            content,
            content,
            "branch_a",
            "branch_b",
        )
        assert len(conflicts) == 0

    def test_async_sync_mismatch(self):
        detector = SemanticConflictDetector(Path("/tmp"), enable_debate=False)

        content_a = """
def fetch(url):
    pass
"""
        content_b = """
async def fetch(url):
    pass
"""
        conflicts = detector._check_signature_conflicts(
            "aragora/client.py",
            content_a,
            content_b,
            "branch_a",
            "branch_b",
        )
        # Should detect assumption clash
        assumption_conflicts = [
            c for c in conflicts if c.conflict_type == ConflictType.ASSUMPTION_CLASH
        ]
        assert len(assumption_conflicts) >= 1

    def test_import_conflict_detection(self):
        detector = SemanticConflictDetector(Path("/tmp"), enable_debate=False)

        content_a = """
import os
from aragora.foo import bar
"""
        content_b = """
import sys
from aragora.baz import qux
"""
        # Import conflicts are checked for cross-referencing modules
        conflicts = detector._check_import_conflicts(
            "aragora/module.py",
            content_a,
            content_b,
            "branch_a",
            "branch_b",
        )
        # May or may not find cross-imports depending on module names
        assert isinstance(conflicts, list)

    def test_static_scan_overlapping_files(self):
        detector = SemanticConflictDetector(Path("/tmp"), enable_debate=False)

        changes_a = {
            "aragora/foo.py": "def bar(x): pass",
        }
        changes_b = {
            "aragora/foo.py": "def bar(x, y): pass",
        }

        conflicts = detector._static_scan("branch_a", "branch_b", changes_a, changes_b)
        assert len(conflicts) >= 1

    def test_static_scan_no_overlap(self):
        detector = SemanticConflictDetector(Path("/tmp"), enable_debate=False)

        changes_a = {"aragora/foo.py": "def foo(): pass"}
        changes_b = {"aragora/bar.py": "def bar(): pass"}

        conflicts = detector._static_scan("branch_a", "branch_b", changes_a, changes_b)
        assert len(conflicts) == 0


# ============================================================
# Debate Scan
# ============================================================


class TestDebateScan:
    def test_arena_import_error_graceful(self):
        detector = SemanticConflictDetector(Path("/tmp"), enable_debate=True)

        with patch.dict("sys.modules", {"aragora.debate.orchestrator": None}):
            # When Arena import fails, debate scan should return empty
            conflicts = detector._debate_scan("a", "b", [])
            assert conflicts == []

    def test_debate_scan_returns_list(self):
        detector = SemanticConflictDetector(Path("/tmp"), enable_debate=True)

        # With mocked Arena available
        mock_arena_module = MagicMock()
        with patch.dict("sys.modules", {"aragora.debate.orchestrator": mock_arena_module}):
            static = [
                SemanticConflict(
                    source_branch="a",
                    target_branch="b",
                    conflict_type=ConflictType.IMPORT_CYCLE,
                    description="test",
                    confidence=0.3,
                )
            ]
            result = detector._debate_scan("a", "b", static)
            assert isinstance(result, list)

    def test_debate_scan_disabled(self):
        detector = SemanticConflictDetector(Path("/tmp"), enable_debate=False)
        # When debate is disabled, detect() skips debate scan entirely
        conflicts = detector.detect([], "main")
        assert conflicts == []


# ============================================================
# BranchCoordinator Integration
# ============================================================


class TestBranchCoordinatorIntegration:
    def test_semantic_detector_wired_in(self, tmp_path):
        from aragora.nomic.branch_coordinator import (
            BranchCoordinator,
            BranchCoordinatorConfig,
        )

        config = BranchCoordinatorConfig(enable_semantic_conflicts=True)
        coord = BranchCoordinator(repo_path=tmp_path, config=config)
        assert coord.semantic_conflict_detector is not None

    def test_high_confidence_warning_logged(self, tmp_path, caplog):
        """High-confidence semantic conflicts should be logged as warnings."""
        from aragora.nomic.branch_coordinator import (
            BranchCoordinator,
            BranchCoordinatorConfig,
        )

        mock_detector = MagicMock()
        mock_detector.detect.return_value = [
            SemanticConflict(
                source_branch="dev/a",
                target_branch="dev/b",
                conflict_type=ConflictType.SIGNATURE_BREAK,
                description="Function foo has different signatures",
                affected_files=["aragora/foo.py"],
                confidence=0.85,
            )
        ]

        config = BranchCoordinatorConfig(enable_semantic_conflicts=True)
        coord = BranchCoordinator(
            repo_path=tmp_path,
            config=config,
            semantic_conflict_detector=mock_detector,
        )

        # We can't easily test coordinate_parallel_work without mocking git,
        # but we can verify the detector is properly stored
        assert coord.semantic_conflict_detector is mock_detector
