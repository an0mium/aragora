"""Tests for SICA self-improving code assistant."""

import pytest
import asyncio
import tempfile
from pathlib import Path
from aragora.nomic.sica_improver import (
    SICAImprover,
    SICAConfig,
    ImprovementType,
    ImprovementOpportunity,
    ImprovementPatch,
    ValidationResult,
    PatchApprovalStatus,
    create_sica_improver,
)


class TestSICAImprover:
    """Test suite for SICAImprover."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        # Create temp repo
        self.temp_dir = tempfile.mkdtemp()
        self.repo_path = Path(self.temp_dir)

        # Create sample Python file with issues
        sample_code = '''"""Sample module with issues."""

def process_data(data):
    result = ""
    for item in data:
        result += str(item)  # String concat in loop
    return result

def load_file(path):
    f = open(path)  # Missing error handling
    content = f.read()
    f.close()
    return content

def complex_function(a, b, c):
    try:
        if a > 0:
            if b > 0:
                if c > 0:
                    return a + b + c
    except:  # Bare except
        pass
    return 0
'''
        (self.repo_path / "sample.py").write_text(sample_code)

    def teardown_method(self) -> None:
        """Clean up test fixtures."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_init_default_config(self) -> None:
        """Test initialization with default config."""
        improver = SICAImprover(self.repo_path)
        assert improver.config.require_human_approval is True
        assert improver.config.run_tests is True

    def test_init_custom_config(self) -> None:
        """Test initialization with custom config."""
        config = SICAConfig(
            require_human_approval=False,
            min_confidence=0.8,
        )
        improver = SICAImprover(self.repo_path, config)
        assert improver.config.require_human_approval is False
        assert improver.config.min_confidence == 0.8

    @pytest.mark.asyncio
    async def test_find_reliability_opportunities(self) -> None:
        """Test finding reliability improvement opportunities."""
        improver = SICAImprover(self.repo_path)

        opportunities = await improver.find_opportunities(types=[ImprovementType.RELIABILITY])

        # Should find bare except and missing error handling
        assert len(opportunities) >= 1
        types = [o.improvement_type for o in opportunities]
        assert all(t == ImprovementType.RELIABILITY for t in types)

    @pytest.mark.asyncio
    async def test_find_testability_opportunities(self) -> None:
        """Test finding testability improvement opportunities."""
        improver = SICAImprover(self.repo_path)

        opportunities = await improver.find_opportunities(types=[ImprovementType.TESTABILITY])

        # Should find missing docstrings
        assert len(opportunities) >= 1

    @pytest.mark.asyncio
    async def test_find_opportunities_respects_filters(self) -> None:
        """Test that opportunity finding respects confidence/priority filters."""
        config = SICAConfig(
            min_confidence=0.99,  # Very high threshold
            min_priority=0.99,
        )
        improver = SICAImprover(self.repo_path, config)

        opportunities = await improver.find_opportunities()

        # Should find very few or no opportunities at this threshold
        assert len(opportunities) <= 1

    @pytest.mark.asyncio
    async def test_generate_patch_bare_except(self) -> None:
        """Test patch generation for bare except."""
        improver = SICAImprover(self.repo_path)

        # Create opportunity for bare except
        opp = ImprovementOpportunity(
            id="test-1",
            file_path="sample.py",
            line_start=21,
            line_end=21,
            improvement_type=ImprovementType.RELIABILITY,
            description="Bare except clause",
            priority=0.7,
            confidence=0.9,
            estimated_effort="low",
        )

        patch = await improver.generate_patch(opp)

        assert patch is not None
        assert "except Exception:" in patch.patched_content

    @pytest.mark.asyncio
    async def test_validate_patch_lint_failure(self) -> None:
        """Test patch validation with lint failure."""
        config = SICAConfig(
            run_tests=False,
            run_typecheck=False,
            run_lint=True,
            lint_command="python -c 'import sys; sys.exit(1)'",  # Always fail
        )
        improver = SICAImprover(self.repo_path, config)

        patch = ImprovementPatch(
            id="test-patch",
            opportunity_id="test-opp",
            file_path="sample.py",
            original_content="x = 1",
            patched_content="x = 2",
            diff="",
            description="Test patch",
        )

        result = await improver.validate_patch(patch)

        assert result is False
        assert patch.validation_result == ValidationResult.FAILED_LINT

    @pytest.mark.asyncio
    async def test_request_approval_auto_approve(self) -> None:
        """Test auto-approval for high-confidence patches."""
        config = SICAConfig(
            require_human_approval=False,
            auto_approve_threshold=0.5,
        )
        improver = SICAImprover(self.repo_path, config)

        patch = ImprovementPatch(
            id="test-patch",
            opportunity_id="test-opp",
            file_path="sample.py",
            original_content="x = 1",
            patched_content="x = 2",
            diff="",
            description="Test",
            validation_result=ValidationResult.PASSED,
        )

        approved = await improver.request_approval(patch)

        assert approved is True
        assert patch.approval_status in (
            PatchApprovalStatus.APPROVED,
            PatchApprovalStatus.AUTO_APPROVED,
        )

    @pytest.mark.asyncio
    async def test_request_approval_callback(self) -> None:
        """Test approval with callback."""
        approval_called = False

        async def approval_callback(patch: ImprovementPatch) -> bool:
            nonlocal approval_called
            approval_called = True
            return True

        config = SICAConfig(
            require_human_approval=True,
            approval_callback=approval_callback,
        )
        improver = SICAImprover(self.repo_path, config)

        patch = ImprovementPatch(
            id="test-patch",
            opportunity_id="test-opp",
            file_path="sample.py",
            original_content="x = 1",
            patched_content="x = 2",
            diff="",
            description="Test",
            validation_result=ValidationResult.PASSED,
        )

        approved = await improver.request_approval(patch)

        assert approval_called
        assert approved is True

    @pytest.mark.asyncio
    async def test_apply_and_rollback_patch(self) -> None:
        """Test applying and rolling back a patch."""
        config = SICAConfig(
            backup_before_apply=True,
            run_tests=False,
        )
        improver = SICAImprover(self.repo_path, config)

        original = (self.repo_path / "sample.py").read_text()

        patch = ImprovementPatch(
            id="test-patch",
            opportunity_id="test-opp",
            file_path="sample.py",
            original_content=original,
            patched_content=original.replace("except:", "except Exception:"),
            diff="",
            description="Fix bare except",
        )

        # Apply
        applied = await improver.apply_patch(patch)
        assert applied is True
        assert patch.applied is True

        # Verify file changed
        current = (self.repo_path / "sample.py").read_text()
        assert "except Exception:" in current

        # Rollback
        rolled_back = await improver.rollback_patch(patch)
        assert rolled_back is True
        assert patch.applied is False

        # Verify file restored
        restored = (self.repo_path / "sample.py").read_text()
        assert "except:" in restored
        assert original == restored

    def test_get_metrics_empty(self) -> None:
        """Test metrics when no cycles run."""
        improver = SICAImprover(self.repo_path)
        metrics = improver.get_metrics()

        assert metrics["total_cycles"] == 0
        assert metrics["success_rate"] == 0.0

    def test_reset(self) -> None:
        """Test reset clears state."""
        improver = SICAImprover(self.repo_path)
        improver._backups["test"] = None
        improver._applied_patches.append(None)

        improver.reset()

        assert len(improver._backups) == 0
        assert len(improver._applied_patches) == 0


class TestImprovementOpportunity:
    """Test ImprovementOpportunity dataclass."""

    def test_to_dict(self) -> None:
        """Test conversion to dictionary."""
        opp = ImprovementOpportunity(
            id="test-1",
            file_path="test.py",
            line_start=10,
            line_end=15,
            improvement_type=ImprovementType.RELIABILITY,
            description="Test issue",
            priority=0.7,
            confidence=0.8,
            estimated_effort="medium",
        )

        d = opp.to_dict()

        assert d["id"] == "test-1"
        assert d["line_range"] == "10-15"
        assert d["type"] == "reliability"


class TestImprovementPatch:
    """Test ImprovementPatch dataclass."""

    def test_to_dict(self) -> None:
        """Test conversion to dictionary."""
        patch = ImprovementPatch(
            id="patch-1",
            opportunity_id="opp-1",
            file_path="test.py",
            original_content="x = 1",
            patched_content="x = 2",
            diff="-x = 1\n+x = 2",
            description="Change value",
            validation_result=ValidationResult.PASSED,
            approval_status=PatchApprovalStatus.APPROVED,
        )

        d = patch.to_dict()

        assert d["id"] == "patch-1"
        assert d["validation"] == "passed"
        assert d["approval"] == "approved"


class TestCreateSICAImprover:
    """Test the factory function."""

    def test_creates_with_defaults(self, tmp_path: Path) -> None:
        """Test factory creates improver with defaults."""
        improver = create_sica_improver(tmp_path)
        assert isinstance(improver, SICAImprover)
        assert improver.config.require_human_approval is True

    def test_creates_with_custom_types(self, tmp_path: Path) -> None:
        """Test factory accepts custom improvement types."""
        improver = create_sica_improver(
            tmp_path,
            improvement_types=["performance", "security"],
            require_human_approval=False,
        )
        assert ImprovementType.PERFORMANCE in improver.config.improvement_types
        assert ImprovementType.SECURITY in improver.config.improvement_types
