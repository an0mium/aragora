"""
Tests for aragora.skills.loader module.

Covers:
- SkillLoadError exception
- SkillLoader class
- DeclarativeSkill class
- load_skills convenience function
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest

from aragora.skills.base import (
    Skill,
    SkillCapability,
    SkillContext,
    SkillManifest,
    SkillResult,
    SkillStatus,
)
from aragora.skills.loader import (
    DeclarativeSkill,
    SkillLoadError,
    SkillLoader,
    load_skills,
)
from aragora.skills.registry import SkillRegistry, reset_skill_registry


# =============================================================================
# Test Fixtures
# =============================================================================


class TestSkillForLoading(Skill):
    """A test skill that can be discovered by the loader."""

    @property
    def manifest(self) -> SkillManifest:
        return SkillManifest(
            name="test_loader_skill",
            version="1.0.0",
            capabilities=[SkillCapability.WEB_SEARCH],
            input_schema={"query": {"type": "string"}},
        )

    async def execute(
        self,
        input_data: Dict[str, Any],
        context: SkillContext,
    ) -> SkillResult:
        return SkillResult.create_success({"loaded": True})


# Module-level skill instance for discovery
TEST_SKILL_INSTANCE = TestSkillForLoading()

# SKILLS list for discovery
SKILLS = [TestSkillForLoading()]


def register_skills():
    """Function that returns skills for discovery."""
    return [TestSkillForLoading()]


@pytest.fixture
def registry() -> SkillRegistry:
    """Create a fresh registry for testing."""
    return SkillRegistry()


@pytest.fixture
def loader(registry: SkillRegistry) -> SkillLoader:
    """Create a loader with fresh registry."""
    return SkillLoader(registry=registry, auto_register=True)


@pytest.fixture
def temp_skill_file():
    """Create a temporary Python file with a skill class."""
    content = """
from aragora.skills.base import Skill, SkillManifest, SkillResult, SkillCapability, SkillContext
from typing import Any, Dict

class TempFileSkill(Skill):
    @property
    def manifest(self) -> SkillManifest:
        return SkillManifest(
            name="temp_file_skill",
            version="1.0.0",
            capabilities=[SkillCapability.READ_LOCAL],
            input_schema={"path": {"type": "string"}},
        )

    async def execute(
        self,
        input_data: Dict[str, Any],
        context: SkillContext,
    ) -> SkillResult:
        return SkillResult.create_success({"from_file": True})
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(content)
        f.flush()
        yield Path(f.name)

    # Cleanup
    Path(f.name).unlink(missing_ok=True)


@pytest.fixture
def temp_skill_directory():
    """Create a temporary directory with skill files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)

        # Create a skill file
        skill_file = tmppath / "my_skill.py"
        skill_file.write_text("""
from aragora.skills.base import Skill, SkillManifest, SkillResult, SkillCapability, SkillContext
from typing import Any, Dict

class DirectorySkill(Skill):
    @property
    def manifest(self) -> SkillManifest:
        return SkillManifest(
            name="directory_skill",
            version="1.0.0",
            capabilities=[],
            input_schema={},
        )

    async def execute(self, input_data: Dict[str, Any], context: SkillContext) -> SkillResult:
        return SkillResult.create_success({})
""")

        # Create another skill file
        another_skill = tmppath / "another_skill.py"
        another_skill.write_text("""
from aragora.skills.base import Skill, SkillManifest, SkillResult, SkillCapability, SkillContext
from typing import Any, Dict

class AnotherSkill(Skill):
    @property
    def manifest(self) -> SkillManifest:
        return SkillManifest(
            name="another_skill",
            version="1.0.0",
            capabilities=[],
            input_schema={},
        )

    async def execute(self, input_data: Dict[str, Any], context: SkillContext) -> SkillResult:
        return SkillResult.create_success({})
""")

        # Create a private file that should be skipped
        private_file = tmppath / "_private_skill.py"
        private_file.write_text("""
# This should be skipped
""")

        yield tmppath


@pytest.fixture
def temp_manifest_json():
    """Create a temporary JSON manifest file."""
    manifest_data = {
        "name": "json_manifest_skill",
        "version": "1.0.0",
        "capabilities": ["web_search"],
        "input_schema": {"query": {"type": "string"}},
        "description": "Skill from JSON manifest",
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(manifest_data, f)
        f.flush()
        yield Path(f.name)

    Path(f.name).unlink(missing_ok=True)


# =============================================================================
# SkillLoadError Tests
# =============================================================================


class TestSkillLoadError:
    """Tests for SkillLoadError exception."""

    def test_exception_message(self):
        """Test exception contains message."""
        error = SkillLoadError("Failed to load skill")
        assert str(error) == "Failed to load skill"

    def test_exception_inheritance(self):
        """Test exception inherits from Exception."""
        error = SkillLoadError("Test")
        assert isinstance(error, Exception)


# =============================================================================
# SkillLoader Module Loading Tests
# =============================================================================


class TestSkillLoaderModuleLoading:
    """Tests for SkillLoader.load_module method."""

    def test_load_module_success(self, loader: SkillLoader):
        """Test loading skills from a module."""
        # Load this test module itself
        skills = loader.load_module("tests.skills.test_loader")

        assert len(skills) > 0
        # Should find TestSkillForLoading class
        names = [s.manifest.name for s in skills]
        assert "test_loader_skill" in names

    def test_load_module_already_loaded(self, loader: SkillLoader):
        """Test loading same module twice returns empty."""
        loader.load_module("tests.skills.test_loader")
        skills = loader.load_module("tests.skills.test_loader")

        assert len(skills) == 0

    def test_load_module_nonexistent(self, loader: SkillLoader):
        """Test loading nonexistent module raises error."""
        with pytest.raises(SkillLoadError, match="Failed to import"):
            loader.load_module("nonexistent.module.path")

    def test_load_module_with_register_false(self, registry: SkillRegistry):
        """Test loading without auto-registration."""
        loader = SkillLoader(registry=registry, auto_register=False)
        skills = loader.load_module("tests.skills.test_loader")

        assert len(skills) > 0
        # Should not be registered
        assert not registry.has_skill("test_loader_skill")

    def test_load_module_with_explicit_register(self, loader: SkillLoader):
        """Test explicit register parameter."""
        loader._auto_register = False
        skills = loader.load_module("tests.skills.test_loader", register=True)

        assert len(skills) > 0
        # Should be registered due to explicit parameter
        assert loader._registry.has_skill("test_loader_skill")


# =============================================================================
# SkillLoader File Loading Tests
# =============================================================================


class TestSkillLoaderFileLoading:
    """Tests for SkillLoader.load_file method."""

    def test_load_file_success(self, loader: SkillLoader, temp_skill_file: Path):
        """Test loading skills from a file."""
        skills = loader.load_file(temp_skill_file)

        assert len(skills) == 1
        assert skills[0].manifest.name == "temp_file_skill"

    def test_load_file_nonexistent(self, loader: SkillLoader):
        """Test loading nonexistent file raises error."""
        with pytest.raises(SkillLoadError, match="File not found"):
            loader.load_file("/nonexistent/path/skill.py")

    def test_load_file_not_python(self, loader: SkillLoader):
        """Test loading non-Python file raises error."""
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            f.write(b"Not Python")
            f.flush()
            try:
                with pytest.raises(SkillLoadError, match="Not a Python file"):
                    loader.load_file(f.name)
            finally:
                Path(f.name).unlink()

    def test_load_file_with_syntax_error(self, loader: SkillLoader):
        """Test loading file with syntax error raises error."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("def invalid syntax(:\n")
            f.flush()
            try:
                with pytest.raises(SkillLoadError, match="Failed to load"):
                    loader.load_file(f.name)
            finally:
                Path(f.name).unlink()


# =============================================================================
# SkillLoader Directory Loading Tests
# =============================================================================


class TestSkillLoaderDirectoryLoading:
    """Tests for SkillLoader.load_directory method."""

    def test_load_directory_success(self, loader: SkillLoader, temp_skill_directory: Path):
        """Test loading skills from a directory."""
        skills = loader.load_directory(temp_skill_directory)

        assert len(skills) == 2
        names = [s.manifest.name for s in skills]
        assert "directory_skill" in names
        assert "another_skill" in names

    def test_load_directory_skips_private(self, loader: SkillLoader, temp_skill_directory: Path):
        """Test that private files (starting with _) are skipped."""
        skills = loader.load_directory(temp_skill_directory)

        # Should not include _private_skill.py
        names = [s.manifest.name for s in skills]
        assert "_private" not in str(names)

    def test_load_directory_nonexistent(self, loader: SkillLoader):
        """Test loading nonexistent directory raises error."""
        with pytest.raises(SkillLoadError, match="Directory not found"):
            loader.load_directory("/nonexistent/directory")

    def test_load_directory_not_a_directory(self, loader: SkillLoader, temp_skill_file: Path):
        """Test loading a file as directory raises error."""
        with pytest.raises(SkillLoadError, match="Not a directory"):
            loader.load_directory(temp_skill_file)

    def test_load_directory_recursive(self, loader: SkillLoader):
        """Test recursive directory loading."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)

            # Create a subdirectory with a skill
            subdir = tmppath / "subdir"
            subdir.mkdir()
            (subdir / "nested_skill.py").write_text("""
from aragora.skills.base import Skill, SkillManifest, SkillResult, SkillContext
from typing import Any, Dict

class NestedSkill(Skill):
    @property
    def manifest(self) -> SkillManifest:
        return SkillManifest(name="nested_skill", version="1.0.0", capabilities=[], input_schema={})

    async def execute(self, input_data: Dict[str, Any], context: SkillContext) -> SkillResult:
        return SkillResult.create_success({})
""")

            # Non-recursive should find 0
            skills = loader.load_directory(tmppath, recursive=False)
            assert len(skills) == 0

            # Recursive should find 1
            loader._loaded_modules.clear()  # Reset loaded tracking
            skills = loader.load_directory(tmppath, recursive=True)
            assert len(skills) == 1
            assert skills[0].manifest.name == "nested_skill"


# =============================================================================
# SkillLoader Built-in Loading Tests
# =============================================================================


class TestSkillLoaderBuiltinLoading:
    """Tests for SkillLoader.load_builtin_skills method."""

    def test_load_builtin_skills(self, loader: SkillLoader):
        """Test loading built-in skills."""
        skills = loader.load_builtin_skills()

        # Should load at least web_search if available
        # May be empty if builtin modules don't exist
        assert isinstance(skills, list)

    def test_load_builtin_skills_handles_missing(self, loader: SkillLoader):
        """Test that missing builtin modules are handled gracefully."""
        # Should not raise even if some modules are missing
        skills = loader.load_builtin_skills()
        assert isinstance(skills, list)


# =============================================================================
# SkillLoader Manifest Loading Tests
# =============================================================================


class TestSkillLoaderManifestLoading:
    """Tests for SkillLoader.load_from_manifest method."""

    def test_load_json_manifest(self, loader: SkillLoader, temp_manifest_json: Path):
        """Test loading skill from JSON manifest."""
        skill = loader.load_from_manifest(temp_manifest_json)

        assert skill is not None
        assert skill.manifest.name == "json_manifest_skill"
        assert skill.manifest.description == "Skill from JSON manifest"

    def test_load_yaml_manifest(self, loader: SkillLoader):
        """Test loading skill from YAML manifest."""
        pytest.importorskip("yaml")

        manifest_data = """
name: yaml_manifest_skill
version: "1.0.0"
capabilities:
  - external_api
input_schema:
  url:
    type: string
description: Skill from YAML manifest
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(manifest_data)
            f.flush()
            try:
                skill = loader.load_from_manifest(f.name)

                assert skill is not None
                assert skill.manifest.name == "yaml_manifest_skill"
            finally:
                Path(f.name).unlink()

    def test_load_manifest_nonexistent(self, loader: SkillLoader):
        """Test loading nonexistent manifest raises error."""
        with pytest.raises(SkillLoadError, match="Manifest not found"):
            loader.load_from_manifest("/nonexistent/manifest.json")

    def test_load_manifest_unsupported_format(self, loader: SkillLoader):
        """Test loading unsupported format raises error."""
        with tempfile.NamedTemporaryFile(suffix=".xml", delete=False) as f:
            f.write(b"<manifest></manifest>")
            f.flush()
            try:
                with pytest.raises(SkillLoadError, match="Unsupported manifest format"):
                    loader.load_from_manifest(f.name)
            finally:
                Path(f.name).unlink()

    def test_load_manifest_invalid_json(self, loader: SkillLoader):
        """Test loading invalid JSON raises error."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("{ invalid json }")
            f.flush()
            try:
                with pytest.raises(SkillLoadError, match="Failed to load manifest"):
                    loader.load_from_manifest(f.name)
            finally:
                Path(f.name).unlink()


# =============================================================================
# SkillLoader Skill Extraction Tests
# =============================================================================


class TestSkillLoaderExtraction:
    """Tests for skill extraction from modules."""

    def test_extract_from_register_skills_function(self, loader: SkillLoader):
        """Test extraction via register_skills function."""
        # Create a mock module with register_skills function
        mock_module = MagicMock()
        mock_module.register_skills = lambda: [TestSkillForLoading()]

        skills = loader._extract_skills_from_module(mock_module)

        assert len(skills) >= 1
        names = [s.manifest.name for s in skills]
        assert "test_loader_skill" in names

    def test_extract_from_skills_list(self, loader: SkillLoader):
        """Test extraction via SKILLS constant."""
        mock_module = MagicMock(spec=[])
        mock_module.SKILLS = [TestSkillForLoading()]

        skills = loader._extract_skills_from_module(mock_module)

        assert len(skills) >= 1

    def test_extract_from_skill_class(self, loader: SkillLoader):
        """Test extraction by finding Skill subclasses."""
        mock_module = MagicMock(spec=[])
        mock_module.TestSkillForLoading = TestSkillForLoading

        # Make dir() return the class name
        def mock_dir(obj):
            return ["TestSkillForLoading"]

        with patch("builtins.dir", mock_dir):
            skills = loader._extract_skills_from_module(mock_module)

        assert len(skills) >= 1

    def test_extract_from_skill_instance(self, loader: SkillLoader):
        """Test extraction of existing skill instances."""
        mock_module = MagicMock(spec=[])
        skill_instance = TestSkillForLoading()
        mock_module.my_skill = skill_instance

        def mock_dir(obj):
            return ["my_skill"]

        with patch("builtins.dir", mock_dir):
            skills = loader._extract_skills_from_module(mock_module)

        assert len(skills) >= 1

    def test_extract_skips_abstract_classes(self, loader: SkillLoader):
        """Test that abstract classes are skipped."""
        # The base Skill class should not be instantiated
        mock_module = MagicMock(spec=[])
        mock_module.Skill = Skill

        skills = loader._extract_skills_from_module(mock_module)

        # Should not include the abstract Skill class
        for s in skills:
            assert s.__class__ != Skill


# =============================================================================
# DeclarativeSkill Tests
# =============================================================================


class TestDeclarativeSkill:
    """Tests for DeclarativeSkill class."""

    @pytest.fixture
    def manifest(self) -> SkillManifest:
        return SkillManifest(
            name="declarative_skill",
            version="1.0.0",
            capabilities=[SkillCapability.EXTERNAL_API],
            input_schema={"url": {"type": "string"}},
        )

    def test_manifest_property(self, manifest: SkillManifest):
        """Test manifest property returns the manifest."""
        skill = DeclarativeSkill(manifest)
        assert skill.manifest is manifest
        assert skill.manifest.name == "declarative_skill"

    @pytest.mark.asyncio
    async def test_execute_without_executor(self, manifest: SkillManifest):
        """Test execute returns NOT_IMPLEMENTED without executor."""
        skill = DeclarativeSkill(manifest)
        context = SkillContext()

        result = await skill.execute({"url": "test"}, context)

        assert result.status == SkillStatus.NOT_IMPLEMENTED
        assert "no executor" in result.error_message.lower()

    @pytest.mark.asyncio
    async def test_execute_with_executor(self, manifest: SkillManifest):
        """Test execute calls provided executor."""

        async def mock_executor(input_data, context):
            return SkillResult.create_success({"executed": input_data})

        skill = DeclarativeSkill(manifest, executor=mock_executor)
        context = SkillContext()

        result = await skill.execute({"url": "test"}, context)

        assert result.success is True
        assert result.data["executed"]["url"] == "test"


# =============================================================================
# load_skills Convenience Function Tests
# =============================================================================


class TestLoadSkillsFunction:
    """Tests for load_skills convenience function."""

    def test_load_skills_from_module(self):
        """Test loading skills from module path."""
        reset_skill_registry()
        skills = load_skills("tests.skills.test_loader")

        assert len(skills) > 0

    def test_load_skills_from_file(self, temp_skill_file: Path):
        """Test loading skills from file path."""
        reset_skill_registry()
        skills = load_skills(str(temp_skill_file))

        assert len(skills) == 1

    def test_load_skills_from_directory(self, temp_skill_directory: Path):
        """Test loading skills from directory path."""
        reset_skill_registry()
        skills = load_skills(str(temp_skill_directory))

        assert len(skills) == 2

    def test_load_skills_multiple_sources(self, temp_skill_file: Path, temp_skill_directory: Path):
        """Test loading from multiple sources."""
        reset_skill_registry()
        skills = load_skills(
            str(temp_skill_file),
            str(temp_skill_directory),
        )

        # 1 from file + 2 from directory
        assert len(skills) == 3

    def test_load_skills_with_custom_registry(self, temp_skill_file: Path):
        """Test loading with custom registry."""
        registry = SkillRegistry()
        skills = load_skills(str(temp_skill_file), registry=registry)

        assert len(skills) == 1
        assert registry.has_skill("temp_file_skill")
