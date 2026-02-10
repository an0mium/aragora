"""Tests for OpenClaw Skill Converter."""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import pytest

from aragora.compat.openclaw.skill_converter import (
    OpenClawBridgeSkill,
    OpenClawSkillConverter,
)
from aragora.compat.openclaw.skill_parser import (
    OpenClawSkillFrontmatter,
    ParsedOpenClawSkill,
)
from aragora.skills.base import Skill, SkillCapability, SkillManifest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_parsed(
    name: str = "web-researcher",
    description: str = "Search the web",
    version: str = "2.0.0",
    requires: list[str] | None = None,
    timeout: int = 300,
    tags: list[str] | None = None,
    instructions: str = "# Web Researcher\n\nYou search the web...",
) -> ParsedOpenClawSkill:
    """Create a ParsedOpenClawSkill for testing."""
    return ParsedOpenClawSkill(
        frontmatter=OpenClawSkillFrontmatter(
            name=name,
            description=description,
            version=version,
            requires=requires or [],
            timeout=timeout,
            tags=tags or [],
        ),
        instructions=instructions,
    )


FULL_SKILL_MD = dedent("""\
    ---
    name: code-analyzer
    description: Analyze source code for patterns
    version: "1.2.0"
    metadata:
      openclaw:
        requires:
          - file_read
          - code_execution
        timeout: 120
    tags:
      - analysis
      - code
    ---

    # Code Analyzer

    You are a code analysis assistant that finds patterns and issues.
""")


# ---------------------------------------------------------------------------
# Tests: convert()
# ---------------------------------------------------------------------------


class TestConvert:
    """Test OpenClawSkillConverter.convert()."""

    def test_convert_creates_bridge_skill_with_correct_manifest(self) -> None:
        """convert() should produce an OpenClawBridgeSkill with a correct manifest."""
        parsed = _make_parsed(
            name="web-researcher",
            description="Search the web",
            version="2.0.0",
            requires=["browser", "file_write"],
            timeout=300,
            tags=["research"],
        )

        bridge = OpenClawSkillConverter.convert(parsed)

        assert isinstance(bridge, OpenClawBridgeSkill)
        assert isinstance(bridge, Skill)

        manifest = bridge.manifest
        assert isinstance(manifest, SkillManifest)
        assert manifest.name == "web_researcher"
        assert manifest.version == "2.0.0"
        assert manifest.description == "Search the web"
        assert manifest.max_execution_time_seconds == 300.0

    def test_convert_sanitizes_name_hyphens(self) -> None:
        """Hyphens in the name should be replaced with underscores."""
        parsed = _make_parsed(name="my-cool-skill")
        bridge = OpenClawSkillConverter.convert(parsed)

        assert bridge.manifest.name == "my_cool_skill"

    def test_convert_sanitizes_name_spaces(self) -> None:
        """Spaces in the name should be replaced with underscores."""
        parsed = _make_parsed(name="my cool skill")
        bridge = OpenClawSkillConverter.convert(parsed)

        assert bridge.manifest.name == "my_cool_skill"

    def test_convert_sanitizes_name_uppercase(self) -> None:
        """Name should be lowercased."""
        parsed = _make_parsed(name="My-COOL-Skill")
        bridge = OpenClawSkillConverter.convert(parsed)

        assert bridge.manifest.name == "my_cool_skill"

    def test_convert_maps_capabilities_correctly(self) -> None:
        """OpenClaw requires should be mapped to Aragora SkillCapability values."""
        parsed = _make_parsed(requires=["browser", "file_write", "shell"])
        bridge = OpenClawSkillConverter.convert(parsed)

        capabilities = bridge.manifest.capabilities
        assert SkillCapability.WEB_FETCH in capabilities
        assert SkillCapability.WRITE_LOCAL in capabilities
        assert SkillCapability.SHELL_EXECUTION in capabilities

    def test_convert_uses_default_capabilities_when_none_specified(self) -> None:
        """When no requires are specified, defaults to READ_LOCAL only (no shell)."""
        parsed = _make_parsed(requires=[])
        bridge = OpenClawSkillConverter.convert(parsed, skip_scan=True)

        capabilities = bridge.manifest.capabilities
        assert SkillCapability.READ_LOCAL in capabilities
        assert SkillCapability.SHELL_EXECUTION not in capabilities
        assert len(capabilities) == 1

    def test_convert_fallback_description(self) -> None:
        """When description is empty, a fallback description is generated."""
        parsed = _make_parsed(name="helper-tool", description="")
        bridge = OpenClawSkillConverter.convert(parsed)

        assert "helper-tool" in bridge.manifest.description
        assert "OpenClaw skill" in bridge.manifest.description

    def test_convert_fallback_name(self) -> None:
        """When name is empty, it defaults to openclaw_skill."""
        parsed = _make_parsed(name="")
        bridge = OpenClawSkillConverter.convert(parsed)

        assert bridge.manifest.name == "openclaw_skill"


# ---------------------------------------------------------------------------
# Tests: convert_file()
# ---------------------------------------------------------------------------


class TestConvertFile:
    """Test OpenClawSkillConverter.convert_file()."""

    def test_convert_file_parses_and_converts(self, tmp_path: Path) -> None:
        """convert_file() should parse a SKILL.md file and return a bridge skill."""
        skill_file = tmp_path / "SKILL.md"
        skill_file.write_text(FULL_SKILL_MD, encoding="utf-8")

        bridge = OpenClawSkillConverter.convert_file(skill_file)

        assert isinstance(bridge, OpenClawBridgeSkill)
        assert bridge.manifest.name == "code_analyzer"
        assert bridge.manifest.version == "1.2.0"
        assert bridge.manifest.description == "Analyze source code for patterns"
        assert SkillCapability.READ_LOCAL in bridge.manifest.capabilities
        assert SkillCapability.CODE_EXECUTION in bridge.manifest.capabilities


# ---------------------------------------------------------------------------
# Tests: convert_directory()
# ---------------------------------------------------------------------------


class TestConvertDirectory:
    """Test OpenClawSkillConverter.convert_directory()."""

    def test_convert_directory_converts_multiple_skills(self, tmp_path: Path) -> None:
        """convert_directory() should find and convert all SKILL.md files."""
        skill_a_dir = tmp_path / "skills" / "alpha"
        skill_a_dir.mkdir(parents=True)
        (skill_a_dir / "SKILL.md").write_text(
            dedent("""\
                ---
                name: alpha-skill
                description: Alpha
                ---

                Alpha instructions.
            """),
            encoding="utf-8",
        )

        skill_b_dir = tmp_path / "skills" / "beta"
        skill_b_dir.mkdir(parents=True)
        (skill_b_dir / "SKILL.md").write_text(
            dedent("""\
                ---
                name: beta-skill
                description: Beta
                ---

                Beta instructions.
            """),
            encoding="utf-8",
        )

        bridges = OpenClawSkillConverter.convert_directory(tmp_path)

        assert len(bridges) == 2
        names = {b.manifest.name for b in bridges}
        assert names == {"alpha_skill", "beta_skill"}
        for bridge in bridges:
            assert isinstance(bridge, OpenClawBridgeSkill)


# ---------------------------------------------------------------------------
# Tests: to_skill_md()
# ---------------------------------------------------------------------------


class TestToSkillMd:
    """Test OpenClawSkillConverter.to_skill_md()."""

    def test_to_skill_md_generates_valid_format(self) -> None:
        """to_skill_md() should produce a SKILL.md string with frontmatter."""
        parsed = _make_parsed(
            name="web-researcher",
            description="Search the web",
            version="2.0.0",
            requires=["browser"],
            tags=["research"],
        )
        bridge = OpenClawSkillConverter.convert(parsed)

        md = OpenClawSkillConverter.to_skill_md(bridge)

        assert md.startswith("---\n")
        assert "name: web_researcher" in md
        assert "description: Search the web" in md
        assert "version: 2.0.0" in md
        assert md.count("---") >= 2
        # Instructions section
        assert "# web_researcher" in md


# ---------------------------------------------------------------------------
# Tests: BridgeSkill properties
# ---------------------------------------------------------------------------


class TestBridgeSkillProperties:
    """Test OpenClawBridgeSkill instance properties."""

    def test_instructions_returns_raw_instructions(self) -> None:
        """The instructions property should return the raw instruction text."""
        raw = "# My Skill\n\nDo things step by step."
        parsed = _make_parsed(instructions=raw)
        bridge = OpenClawSkillConverter.convert(parsed)

        assert bridge.instructions == raw

    def test_manifest_tags_include_openclaw_and_bridge(self) -> None:
        """Tags should always start with 'openclaw' and 'bridge'."""
        parsed = _make_parsed(tags=["research", "web"])
        bridge = OpenClawSkillConverter.convert(parsed)

        tags = bridge.manifest.tags
        assert "openclaw" in tags
        assert "bridge" in tags
        assert tags[0] == "openclaw"
        assert tags[1] == "bridge"

    def test_manifest_tags_include_frontmatter_tags(self) -> None:
        """Frontmatter tags should be appended after the bridge tags."""
        parsed = _make_parsed(tags=["custom", "extra"])
        bridge = OpenClawSkillConverter.convert(parsed)

        tags = bridge.manifest.tags
        assert "custom" in tags
        assert "extra" in tags
        assert tags == ["openclaw", "bridge", "custom", "extra"]

    def test_manifest_debate_compatible(self) -> None:
        """Bridge skills should be marked as debate compatible."""
        parsed = _make_parsed()
        bridge = OpenClawSkillConverter.convert(parsed)

        assert bridge.manifest.debate_compatible is True
