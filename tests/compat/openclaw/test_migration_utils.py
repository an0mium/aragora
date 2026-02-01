"""Tests for OpenClaw Migration Utilities."""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent
from unittest.mock import MagicMock

import pytest

from aragora.compat.openclaw.migration_utils import (
    MigrationContext,
    MigrationMode,
    MigrationState,
    export_to_openclaw,
    import_openclaw_skills,
)
from aragora.compat.openclaw.skill_converter import OpenClawBridgeSkill
from aragora.compat.openclaw.skill_parser import (
    OpenClawSkillFrontmatter,
    ParsedOpenClawSkill,
)
from aragora.skills.base import Skill, SkillCapability, SkillManifest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_parsed(
    name: str = "test-skill",
    description: str = "A test skill",
    requires: list[str] | None = None,
    tags: list[str] | None = None,
) -> ParsedOpenClawSkill:
    """Create a ParsedOpenClawSkill for testing."""
    return ParsedOpenClawSkill(
        frontmatter=OpenClawSkillFrontmatter(
            name=name,
            description=description,
            requires=requires or [],
            tags=tags or [],
        ),
        instructions=f"# {name}\n\nInstructions for {name}.",
    )


def _make_native_skill(
    name: str = "test_skill",
    description: str = "A native skill",
) -> MagicMock:
    """Create a mock native Aragora Skill."""
    mock = MagicMock(spec=Skill)
    mock.manifest = SkillManifest(
        name=name,
        version="1.0.0",
        description=description,
        capabilities=[SkillCapability.WEB_SEARCH],
        input_schema={},
        tags=["native"],
        max_execution_time_seconds=30.0,
    )
    return mock


SKILL_A_MD = dedent("""\
    ---
    name: alpha-skill
    description: Alpha skill
    metadata:
      openclaw:
        requires:
          - browser
        timeout: 60
    tags:
      - alpha
    ---

    # Alpha Skill

    Alpha instructions.
""")


SKILL_B_MD = dedent("""\
    ---
    name: beta-skill
    description: Beta skill
    metadata:
      openclaw:
        requires:
          - shell
        timeout: 120
    tags:
      - beta
    ---

    # Beta Skill

    Beta instructions.
""")


# ---------------------------------------------------------------------------
# Tests: MigrationMode
# ---------------------------------------------------------------------------


class TestMigrationMode:
    """Test MigrationMode enum values."""

    def test_enum_values(self) -> None:
        assert MigrationMode.OPENCLAW_ONLY == "openclaw_only"
        assert MigrationMode.ARAGORA_ONLY == "aragora_only"
        assert MigrationMode.PREFER_ARAGORA == "prefer_aragora"
        assert MigrationMode.PARALLEL == "parallel"

    def test_enum_is_string(self) -> None:
        assert isinstance(MigrationMode.OPENCLAW_ONLY, str)


# ---------------------------------------------------------------------------
# Tests: MigrationState
# ---------------------------------------------------------------------------


class TestMigrationState:
    """Test MigrationState dataclass."""

    def test_defaults(self) -> None:
        """MigrationState should have sensible defaults."""
        state = MigrationState()

        assert state.mode == MigrationMode.PREFER_ARAGORA
        assert state.imported_skills == {}
        assert state.native_overrides == {}
        assert state.migration_log == []
        assert state.total_skills == 0
        assert state.migrated_count == 0

    def test_migration_percentage_no_skills(self) -> None:
        """With no imported skills, migration percentage should be 100%."""
        state = MigrationState()

        assert state.migration_percentage == 100.0

    def test_migration_percentage_partial(self) -> None:
        """migration_percentage should reflect proportion of native overrides."""
        bridge_a = MagicMock(spec=OpenClawBridgeSkill)
        bridge_b = MagicMock(spec=OpenClawBridgeSkill)
        bridge_c = MagicMock(spec=OpenClawBridgeSkill)
        bridge_d = MagicMock(spec=OpenClawBridgeSkill)
        native = _make_native_skill()

        state = MigrationState(
            imported_skills={
                "skill_a": bridge_a,
                "skill_b": bridge_b,
                "skill_c": bridge_c,
                "skill_d": bridge_d,
            },
            native_overrides={
                "skill_a": native,
            },
        )

        assert state.total_skills == 4
        assert state.migrated_count == 1
        assert state.migration_percentage == 25.0

    def test_migration_percentage_fully_migrated(self) -> None:
        """When all imported skills have native overrides, percentage should be 100%."""
        bridge_a = MagicMock(spec=OpenClawBridgeSkill)
        bridge_b = MagicMock(spec=OpenClawBridgeSkill)
        native_a = _make_native_skill(name="skill_a")
        native_b = _make_native_skill(name="skill_b")

        state = MigrationState(
            imported_skills={"skill_a": bridge_a, "skill_b": bridge_b},
            native_overrides={"skill_a": native_a, "skill_b": native_b},
        )

        assert state.migration_percentage == 100.0


# ---------------------------------------------------------------------------
# Tests: MigrationContext.resolve_skill()
# ---------------------------------------------------------------------------


class TestResolveSkill:
    """Test MigrationContext.resolve_skill() routing logic."""

    def _make_context_with_skills(
        self,
        mode: MigrationMode,
    ) -> tuple[MigrationContext, OpenClawBridgeSkill, MagicMock]:
        """Create a context with one imported skill and one native override."""
        from aragora.compat.openclaw.skill_converter import OpenClawSkillConverter

        parsed = _make_parsed(name="shared-skill")
        bridge = OpenClawSkillConverter.convert(parsed)

        native = _make_native_skill(name="shared_skill")

        state = MigrationState(
            mode=mode,
            imported_skills={"shared-skill": bridge},
            native_overrides={"shared-skill": native},
        )
        context = MigrationContext(state)
        return context, bridge, native

    def test_openclaw_only_returns_imported_skill(self) -> None:
        """In OPENCLAW_ONLY mode, resolve_skill returns the imported bridge skill."""
        context, bridge, _native = self._make_context_with_skills(MigrationMode.OPENCLAW_ONLY)

        result = context.resolve_skill("shared-skill")

        assert result is bridge

    def test_openclaw_only_returns_none_for_unknown(self) -> None:
        """In OPENCLAW_ONLY mode, unknown skill returns None."""
        context, _, _ = self._make_context_with_skills(MigrationMode.OPENCLAW_ONLY)

        result = context.resolve_skill("nonexistent")

        assert result is None

    def test_aragora_only_returns_native_skill(self) -> None:
        """In ARAGORA_ONLY mode, resolve_skill returns the native override."""
        context, _bridge, native = self._make_context_with_skills(MigrationMode.ARAGORA_ONLY)

        result = context.resolve_skill("shared-skill")

        assert result is native

    def test_aragora_only_returns_none_for_imported_only(self) -> None:
        """In ARAGORA_ONLY mode, a skill that only exists as imported returns None."""
        from aragora.compat.openclaw.skill_converter import OpenClawSkillConverter

        parsed = _make_parsed(name="imported-only")
        bridge = OpenClawSkillConverter.convert(parsed)

        state = MigrationState(
            mode=MigrationMode.ARAGORA_ONLY,
            imported_skills={"imported-only": bridge},
        )
        context = MigrationContext(state)

        result = context.resolve_skill("imported-only")

        assert result is None

    def test_prefer_aragora_returns_native_when_available(self) -> None:
        """In PREFER_ARAGORA mode, native override should be preferred."""
        context, _bridge, native = self._make_context_with_skills(MigrationMode.PREFER_ARAGORA)

        result = context.resolve_skill("shared-skill")

        assert result is native

    def test_prefer_aragora_falls_back_to_imported(self) -> None:
        """In PREFER_ARAGORA mode, falls back to imported when no native exists."""
        from aragora.compat.openclaw.skill_converter import OpenClawSkillConverter

        parsed = _make_parsed(name="bridge-only")
        bridge = OpenClawSkillConverter.convert(parsed)

        state = MigrationState(
            mode=MigrationMode.PREFER_ARAGORA,
            imported_skills={"bridge-only": bridge},
        )
        context = MigrationContext(state)

        result = context.resolve_skill("bridge-only")

        assert result is bridge

    def test_parallel_returns_native_when_available(self) -> None:
        """In PARALLEL mode, native override should be preferred."""
        context, _bridge, native = self._make_context_with_skills(MigrationMode.PARALLEL)

        result = context.resolve_skill("shared-skill")

        assert result is native

    def test_parallel_falls_back_to_imported(self) -> None:
        """In PARALLEL mode, falls back to imported when no native exists."""
        from aragora.compat.openclaw.skill_converter import OpenClawSkillConverter

        parsed = _make_parsed(name="bridge-only")
        bridge = OpenClawSkillConverter.convert(parsed)

        state = MigrationState(
            mode=MigrationMode.PARALLEL,
            imported_skills={"bridge-only": bridge},
        )
        context = MigrationContext(state)

        result = context.resolve_skill("bridge-only")

        assert result is bridge

    def test_resolve_returns_none_when_not_found(self) -> None:
        """resolve_skill should return None when the skill is not registered."""
        context = MigrationContext()

        result = context.resolve_skill("nonexistent-skill")

        assert result is None

    def test_default_context_has_prefer_aragora_mode(self) -> None:
        """MigrationContext with no args should use PREFER_ARAGORA mode."""
        context = MigrationContext()

        assert context.state.mode == MigrationMode.PREFER_ARAGORA


# ---------------------------------------------------------------------------
# Tests: MigrationContext.register_native()
# ---------------------------------------------------------------------------


class TestRegisterNative:
    """Test MigrationContext.register_native()."""

    def test_register_native_adds_to_overrides(self) -> None:
        """register_native should add the skill to native_overrides."""
        context = MigrationContext()
        native = _make_native_skill(name="my_skill")

        context.register_native("my_skill", native)

        assert "my_skill" in context.state.native_overrides
        assert context.state.native_overrides["my_skill"] is native

    def test_register_native_adds_to_log(self) -> None:
        """register_native should append an entry to the migration log."""
        context = MigrationContext()
        native = _make_native_skill(name="my_skill")

        context.register_native("my_skill", native)

        assert len(context.state.migration_log) == 1
        log_entry = context.state.migration_log[0]
        assert log_entry["action"] == "register_native"
        assert log_entry["skill_name"] == "my_skill"

    def test_register_native_updates_migration_count(self) -> None:
        """After registering a native override, migrated_count should increase."""
        from aragora.compat.openclaw.skill_converter import OpenClawSkillConverter

        parsed = _make_parsed(name="tracked-skill")
        bridge = OpenClawSkillConverter.convert(parsed)

        state = MigrationState(
            imported_skills={"tracked-skill": bridge},
        )
        context = MigrationContext(state)

        assert context.state.migrated_count == 0

        native = _make_native_skill(name="tracked_skill")
        context.register_native("tracked-skill", native)

        assert context.state.migrated_count == 1
        assert context.state.migration_percentage == 100.0


# ---------------------------------------------------------------------------
# Tests: MigrationContext.import_skill()
# ---------------------------------------------------------------------------


class TestImportSkill:
    """Test MigrationContext.import_skill()."""

    def test_import_skill_creates_and_registers_bridge(self) -> None:
        """import_skill should convert and register the bridge skill."""
        context = MigrationContext()
        parsed = _make_parsed(name="new-skill", requires=["browser"])

        bridge = context.import_skill(parsed)

        assert isinstance(bridge, OpenClawBridgeSkill)
        assert "new-skill" in context.state.imported_skills
        assert context.state.imported_skills["new-skill"] is bridge

    def test_import_skill_adds_to_log(self) -> None:
        """import_skill should append an entry to the migration log."""
        context = MigrationContext()
        parsed = _make_parsed(name="logged-skill", requires=["shell"])

        context.import_skill(parsed)

        assert len(context.state.migration_log) == 1
        log_entry = context.state.migration_log[0]
        assert log_entry["action"] == "import"
        assert log_entry["skill_name"] == "logged-skill"
        assert log_entry["requires"] == ["shell"]

    def test_import_skill_updates_total_skills(self) -> None:
        """After importing, total_skills should increase."""
        context = MigrationContext()

        assert context.state.total_skills == 0

        context.import_skill(_make_parsed(name="skill-1"))
        assert context.state.total_skills == 1

        context.import_skill(_make_parsed(name="skill-2"))
        assert context.state.total_skills == 2


# ---------------------------------------------------------------------------
# Tests: import_openclaw_skills()
# ---------------------------------------------------------------------------


class TestImportOpenclawSkills:
    """Test import_openclaw_skills() batch import."""

    def test_import_from_directory(self, tmp_path: Path) -> None:
        """import_openclaw_skills should find and convert all SKILL.md files."""
        alpha_dir = tmp_path / "skills" / "alpha"
        alpha_dir.mkdir(parents=True)
        (alpha_dir / "SKILL.md").write_text(SKILL_A_MD, encoding="utf-8")

        beta_dir = tmp_path / "skills" / "beta"
        beta_dir.mkdir(parents=True)
        (beta_dir / "SKILL.md").write_text(SKILL_B_MD, encoding="utf-8")

        bridges = import_openclaw_skills(tmp_path)

        assert len(bridges) == 2
        for bridge in bridges:
            assert isinstance(bridge, OpenClawBridgeSkill)

    def test_import_with_context_registers_skills(self, tmp_path: Path) -> None:
        """When a context is provided, imported skills should be registered."""
        alpha_dir = tmp_path / "skills" / "alpha"
        alpha_dir.mkdir(parents=True)
        (alpha_dir / "SKILL.md").write_text(SKILL_A_MD, encoding="utf-8")

        beta_dir = tmp_path / "skills" / "beta"
        beta_dir.mkdir(parents=True)
        (beta_dir / "SKILL.md").write_text(SKILL_B_MD, encoding="utf-8")

        context = MigrationContext()
        bridges = import_openclaw_skills(tmp_path, context=context)

        assert len(bridges) == 2
        assert context.state.total_skills == 2
        assert len(context.state.migration_log) == 2

        # Both log entries should be imports
        for entry in context.state.migration_log:
            assert entry["action"] == "import"

    def test_import_empty_directory(self, tmp_path: Path) -> None:
        """import_openclaw_skills on an empty directory should return empty list."""
        bridges = import_openclaw_skills(tmp_path)

        assert bridges == []

    def test_import_without_context(self, tmp_path: Path) -> None:
        """import_openclaw_skills without context should still convert skills."""
        alpha_dir = tmp_path / "alpha"
        alpha_dir.mkdir()
        (alpha_dir / "SKILL.md").write_text(SKILL_A_MD, encoding="utf-8")

        bridges = import_openclaw_skills(tmp_path)

        assert len(bridges) == 1
        assert isinstance(bridges[0], OpenClawBridgeSkill)


# ---------------------------------------------------------------------------
# Tests: export_to_openclaw()
# ---------------------------------------------------------------------------


class TestExportToOpenclaw:
    """Test export_to_openclaw() file output."""

    def test_export_creates_skill_md_file(self, tmp_path: Path) -> None:
        """export_to_openclaw should create a SKILL.md file in a skill directory."""
        mock_skill = _make_native_skill(name="exported_skill", description="An exported skill")

        result_path = export_to_openclaw(mock_skill, tmp_path)

        assert result_path.exists()
        assert result_path.name == "SKILL.md"
        content = result_path.read_text(encoding="utf-8")
        assert "exported_skill" in content
        assert "An exported skill" in content

    def test_export_creates_skill_directory(self, tmp_path: Path) -> None:
        """export_to_openclaw should create a subdirectory named after the skill."""
        mock_skill = _make_native_skill(name="my_tool")

        result_path = export_to_openclaw(mock_skill, tmp_path)

        skill_dir = tmp_path / "my_tool"
        assert skill_dir.is_dir()
        assert result_path == skill_dir / "SKILL.md"

    def test_export_returns_path_to_skill_md(self, tmp_path: Path) -> None:
        """The returned path should point to the SKILL.md file."""
        mock_skill = _make_native_skill(name="path_test")

        result_path = export_to_openclaw(mock_skill, tmp_path)

        assert isinstance(result_path, Path)
        assert result_path.suffix == ".md"
        assert result_path.name == "SKILL.md"

    def test_export_file_has_frontmatter(self, tmp_path: Path) -> None:
        """The exported file should contain YAML frontmatter markers."""
        mock_skill = _make_native_skill(name="fm_test", description="Frontmatter test")

        result_path = export_to_openclaw(mock_skill, tmp_path)
        content = result_path.read_text(encoding="utf-8")

        assert content.startswith("---\n")
        assert content.count("---") >= 2
        assert "name: fm_test" in content
        assert "description: Frontmatter test" in content

    def test_export_overwrites_existing(self, tmp_path: Path) -> None:
        """Exporting twice should overwrite the existing SKILL.md."""
        mock_skill = _make_native_skill(name="overwrite_test", description="First version")
        export_to_openclaw(mock_skill, tmp_path)

        mock_skill_v2 = _make_native_skill(name="overwrite_test", description="Second version")
        result_path = export_to_openclaw(mock_skill_v2, tmp_path)

        content = result_path.read_text(encoding="utf-8")
        assert "Second version" in content
