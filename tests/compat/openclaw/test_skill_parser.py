"""Tests for OpenClaw SKILL.md parser."""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import pytest

from aragora.compat.openclaw.skill_parser import (
    OpenClawSkillFrontmatter,
    OpenClawSkillParser,
    ParsedOpenClawSkill,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


FULL_SKILL_MD = dedent("""\
    ---
    name: web-researcher
    description: Search the web and summarize findings
    version: "2.1.0"
    author: test-author
    metadata:
      openclaw:
        requires:
          - browser
          - file_write
        timeout: 300
    tags:
      - research
      - web
    ---

    # Web Researcher

    You are a web research assistant that finds and summarizes information.
""")


MINIMAL_SKILL_MD = dedent("""\
    ---
    name: minimal-skill
    description: A minimal skill
    ---

    Do something simple.
""")


NO_FRONTMATTER_MD = dedent("""\
    # Plain Markdown Skill

    This skill has no frontmatter at all.
    Just raw instructions.
""")


EMPTY_FRONTMATTER_MD = dedent("""\
    ---
    ---

    Instructions after empty frontmatter.
""")


# ---------------------------------------------------------------------------
# Tests: parse() with content
# ---------------------------------------------------------------------------


class TestParseContent:
    """Test OpenClawSkillParser.parse with string content."""

    def test_parse_full_content(self) -> None:
        """Parse content with complete frontmatter and instructions."""
        skill = OpenClawSkillParser.parse(FULL_SKILL_MD)

        assert isinstance(skill, ParsedOpenClawSkill)
        assert isinstance(skill.frontmatter, OpenClawSkillFrontmatter)
        assert "Web Researcher" in skill.instructions
        assert "web research assistant" in skill.instructions

    def test_parse_without_frontmatter(self) -> None:
        """Content without frontmatter should produce defaults and full instructions."""
        skill = OpenClawSkillParser.parse(NO_FRONTMATTER_MD)

        assert skill.frontmatter.name == ""
        assert skill.frontmatter.description == ""
        assert skill.frontmatter.requires == []
        assert "Plain Markdown Skill" in skill.instructions
        assert "no frontmatter" in skill.instructions

    def test_parse_empty_frontmatter(self) -> None:
        """Empty frontmatter block should produce defaults."""
        skill = OpenClawSkillParser.parse(EMPTY_FRONTMATTER_MD)

        assert skill.frontmatter.name == ""
        assert skill.frontmatter.description == ""
        assert skill.frontmatter.version == "1.0.0"
        assert skill.frontmatter.requires == []
        assert skill.frontmatter.tags == []
        assert "Instructions after empty frontmatter" in skill.instructions

    def test_source_path_stored(self) -> None:
        """source_path should be stored when passed to parse()."""
        fake_path = Path("/tmp/fake/SKILL.md")
        skill = OpenClawSkillParser.parse(MINIMAL_SKILL_MD, source_path=fake_path)

        assert skill.source_path == fake_path


# ---------------------------------------------------------------------------
# Tests: frontmatter fields
# ---------------------------------------------------------------------------


class TestFrontmatterParsing:
    """Test individual frontmatter field extraction."""

    def test_name_field(self) -> None:
        skill = OpenClawSkillParser.parse(FULL_SKILL_MD)
        assert skill.frontmatter.name == "web-researcher"

    def test_description_field(self) -> None:
        skill = OpenClawSkillParser.parse(FULL_SKILL_MD)
        assert skill.frontmatter.description == "Search the web and summarize findings"

    def test_version_field(self) -> None:
        skill = OpenClawSkillParser.parse(FULL_SKILL_MD)
        assert skill.frontmatter.version == "2.1.0"

    def test_author_field(self) -> None:
        skill = OpenClawSkillParser.parse(FULL_SKILL_MD)
        assert skill.frontmatter.author == "test-author"

    def test_requires_field(self) -> None:
        skill = OpenClawSkillParser.parse(FULL_SKILL_MD)
        assert skill.frontmatter.requires == ["browser", "file_write"]

    def test_timeout_field(self) -> None:
        skill = OpenClawSkillParser.parse(FULL_SKILL_MD)
        assert skill.frontmatter.timeout == 300

    def test_tags_field(self) -> None:
        skill = OpenClawSkillParser.parse(FULL_SKILL_MD)
        assert skill.frontmatter.tags == ["research", "web"]

    def test_metadata_field(self) -> None:
        skill = OpenClawSkillParser.parse(FULL_SKILL_MD)
        assert isinstance(skill.frontmatter.metadata, dict)
        assert "openclaw" in skill.frontmatter.metadata

    def test_default_version_when_missing(self) -> None:
        skill = OpenClawSkillParser.parse(MINIMAL_SKILL_MD)
        assert skill.frontmatter.version == "1.0.0"

    def test_default_timeout_when_missing(self) -> None:
        skill = OpenClawSkillParser.parse(MINIMAL_SKILL_MD)
        assert skill.frontmatter.timeout == 300


# ---------------------------------------------------------------------------
# Tests: properties on ParsedOpenClawSkill
# ---------------------------------------------------------------------------


class TestSkillProperties:
    """Test convenience properties on ParsedOpenClawSkill."""

    def test_name_property(self) -> None:
        skill = OpenClawSkillParser.parse(FULL_SKILL_MD)
        assert skill.name == "web-researcher"
        assert skill.name == skill.frontmatter.name

    def test_description_property(self) -> None:
        skill = OpenClawSkillParser.parse(FULL_SKILL_MD)
        assert skill.description == "Search the web and summarize findings"
        assert skill.description == skill.frontmatter.description

    def test_requires_property(self) -> None:
        skill = OpenClawSkillParser.parse(FULL_SKILL_MD)
        assert skill.requires == ["browser", "file_write"]
        assert skill.requires is skill.frontmatter.requires


# ---------------------------------------------------------------------------
# Tests: parse_file
# ---------------------------------------------------------------------------


class TestParseFile:
    """Test OpenClawSkillParser.parse_file from disk."""

    def test_parse_file_from_disk(self, tmp_path: Path) -> None:
        """Successfully read and parse a SKILL.md written to disk."""
        skill_file = tmp_path / "SKILL.md"
        skill_file.write_text(FULL_SKILL_MD, encoding="utf-8")

        skill = OpenClawSkillParser.parse_file(skill_file)

        assert skill.name == "web-researcher"
        assert skill.source_path == skill_file
        assert "Web Researcher" in skill.instructions

    def test_parse_file_not_found(self) -> None:
        """Raise FileNotFoundError for a non-existent file."""
        with pytest.raises(FileNotFoundError, match="Skill file not found"):
            OpenClawSkillParser.parse_file("/nonexistent/path/SKILL.md")


# ---------------------------------------------------------------------------
# Tests: parse_directory
# ---------------------------------------------------------------------------


class TestParseDirectory:
    """Test OpenClawSkillParser.parse_directory for recursive discovery."""

    def test_parse_directory_with_multiple_skills(self, tmp_path: Path) -> None:
        """Find and parse multiple SKILL.md files in nested directories."""
        # Create two skill directories
        skill_a = tmp_path / "skills" / "alpha"
        skill_a.mkdir(parents=True)
        (skill_a / "SKILL.md").write_text(
            dedent("""\
                ---
                name: alpha-skill
                description: Alpha
                ---

                Alpha instructions.
            """),
            encoding="utf-8",
        )

        skill_b = tmp_path / "skills" / "beta"
        skill_b.mkdir(parents=True)
        (skill_b / "SKILL.md").write_text(
            dedent("""\
                ---
                name: beta-skill
                description: Beta
                ---

                Beta instructions.
            """),
            encoding="utf-8",
        )

        skills = OpenClawSkillParser.parse_directory(tmp_path)

        assert len(skills) == 2
        names = {s.name for s in skills}
        assert names == {"alpha-skill", "beta-skill"}

    def test_parse_empty_directory(self, tmp_path: Path) -> None:
        """Return empty list when no SKILL.md files exist."""
        skills = OpenClawSkillParser.parse_directory(tmp_path)
        assert skills == []

    def test_parse_nonexistent_directory(self) -> None:
        """Return empty list when directory does not exist."""
        skills = OpenClawSkillParser.parse_directory("/nonexistent/dir")
        assert skills == []


# ---------------------------------------------------------------------------
# Tests: supporting file discovery
# ---------------------------------------------------------------------------


class TestSupportingFiles:
    """Test discovery of supporting files alongside SKILL.md."""

    def test_supporting_files_discovered(self, tmp_path: Path) -> None:
        """Supporting files in the same directory are listed."""
        skill_file = tmp_path / "SKILL.md"
        skill_file.write_text(MINIMAL_SKILL_MD, encoding="utf-8")

        # Create supporting files alongside the skill
        (tmp_path / "helpers.py").write_text("# helper code", encoding="utf-8")
        (tmp_path / "config.json").write_text("{}", encoding="utf-8")

        skill = OpenClawSkillParser.parse_file(skill_file)

        supporting_names = {f.name for f in skill.supporting_files}
        assert "helpers.py" in supporting_names
        assert "config.json" in supporting_names
        # SKILL.md itself should not appear
        assert "SKILL.md" not in supporting_names

    def test_no_supporting_files(self, tmp_path: Path) -> None:
        """No supporting files when directory only contains SKILL.md."""
        skill_file = tmp_path / "SKILL.md"
        skill_file.write_text(MINIMAL_SKILL_MD, encoding="utf-8")

        skill = OpenClawSkillParser.parse_file(skill_file)
        assert skill.supporting_files == []

    def test_pyc_files_excluded(self, tmp_path: Path) -> None:
        """Compiled Python files are excluded from supporting files."""
        skill_file = tmp_path / "SKILL.md"
        skill_file.write_text(MINIMAL_SKILL_MD, encoding="utf-8")

        (tmp_path / "module.pyc").write_bytes(b"\x00")
        (tmp_path / "real_file.txt").write_text("content", encoding="utf-8")

        skill = OpenClawSkillParser.parse_file(skill_file)

        supporting_names = {f.name for f in skill.supporting_files}
        assert "module.pyc" not in supporting_names
        assert "real_file.txt" in supporting_names
