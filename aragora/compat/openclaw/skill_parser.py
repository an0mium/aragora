"""
OpenClaw SKILL.md Parser.

Parses OpenClaw's SKILL.md format which consists of:
- YAML frontmatter (name, description, metadata)
- Markdown body (instructions for the agent)
- Supporting file discovery

Example SKILL.md:
    ---
    name: web-researcher
    description: Search the web and summarize findings
    metadata:
      openclaw:
        requires:
          - browser
          - file_write
        timeout: 300
    ---

    # Web Researcher

    You are a web research assistant...
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Regex to match YAML frontmatter delimited by ---
_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)


@dataclass
class OpenClawSkillFrontmatter:
    """Parsed YAML frontmatter from SKILL.md."""

    name: str = ""
    description: str = ""
    version: str = "1.0.0"
    author: str = ""
    requires: list[str] = field(default_factory=list)
    timeout: int = 300
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ParsedOpenClawSkill:
    """A fully parsed OpenClaw skill."""

    frontmatter: OpenClawSkillFrontmatter
    instructions: str = ""
    source_path: Path | None = None
    supporting_files: list[Path] = field(default_factory=list)

    @property
    def name(self) -> str:
        return self.frontmatter.name

    @property
    def description(self) -> str:
        return self.frontmatter.description

    @property
    def requires(self) -> list[str]:
        return self.frontmatter.requires


class OpenClawSkillParser:
    """Parser for OpenClaw SKILL.md files."""

    @staticmethod
    def parse(content: str, source_path: Path | None = None) -> ParsedOpenClawSkill:
        """
        Parse a SKILL.md string.

        Args:
            content: The SKILL.md content to parse.
            source_path: Optional path to the source file.

        Returns:
            ParsedOpenClawSkill with frontmatter and instructions.
        """
        frontmatter = OpenClawSkillFrontmatter()
        instructions = content

        match = _FRONTMATTER_RE.match(content)
        if match:
            frontmatter = OpenClawSkillParser._parse_frontmatter(match.group(1))
            instructions = content[match.end() :].strip()

        # Discover supporting files
        supporting_files: list[Path] = []
        if source_path and source_path.is_file():
            skill_dir = source_path.parent
            supporting_files = OpenClawSkillParser._discover_supporting_files(skill_dir)

        return ParsedOpenClawSkill(
            frontmatter=frontmatter,
            instructions=instructions,
            source_path=source_path,
            supporting_files=supporting_files,
        )

    @staticmethod
    def parse_file(path: str | Path) -> ParsedOpenClawSkill:
        """
        Parse a SKILL.md file from disk.

        Args:
            path: Path to the SKILL.md file.

        Returns:
            ParsedOpenClawSkill with frontmatter and instructions.

        Raises:
            FileNotFoundError: If the file doesn't exist.
        """
        filepath = Path(path)
        if not filepath.exists():
            raise FileNotFoundError(f"Skill file not found: {filepath}")

        content = filepath.read_text(encoding="utf-8")
        return OpenClawSkillParser.parse(content, source_path=filepath)

    @staticmethod
    def parse_directory(directory: str | Path) -> list[ParsedOpenClawSkill]:
        """
        Parse all SKILL.md files in a directory.

        Args:
            directory: Path to search for SKILL.md files.

        Returns:
            List of parsed skills found in the directory tree.
        """
        dirpath = Path(directory)
        if not dirpath.is_dir():
            return []

        skills: list[ParsedOpenClawSkill] = []
        for skill_file in dirpath.rglob("SKILL.md"):
            try:
                skill = OpenClawSkillParser.parse_file(skill_file)
                skills.append(skill)
            except Exception as e:
                logger.warning(f"Failed to parse {skill_file}: {e}")

        return skills

    @staticmethod
    def _parse_frontmatter(yaml_str: str) -> OpenClawSkillFrontmatter:
        """Parse YAML frontmatter into typed dataclass."""
        try:
            import yaml

            data = yaml.safe_load(yaml_str) or {}
        except ImportError:
            data = OpenClawSkillParser._simple_yaml_parse(yaml_str)
        except Exception as exc:
            logger.debug("Failed to parse YAML frontmatter: %s", exc)
            data = {}

        if not isinstance(data, dict):
            data = {}

        # Extract openclaw-specific metadata
        metadata = data.get("metadata", {})
        openclaw_meta = metadata.get("openclaw", {}) if isinstance(metadata, dict) else {}

        requires = openclaw_meta.get("requires", []) if isinstance(openclaw_meta, dict) else []
        timeout = openclaw_meta.get("timeout", 300) if isinstance(openclaw_meta, dict) else 300

        return OpenClawSkillFrontmatter(
            name=str(data.get("name", "")),
            description=str(data.get("description", "")),
            version=str(data.get("version", "1.0.0")),
            author=str(data.get("author", "")),
            requires=requires if isinstance(requires, list) else [],
            timeout=int(timeout) if isinstance(timeout, (int, float)) else 300,
            tags=data.get("tags", []) if isinstance(data.get("tags"), list) else [],
            metadata=metadata if isinstance(metadata, dict) else {},
        )

    @staticmethod
    def _simple_yaml_parse(yaml_str: str) -> dict[str, Any]:
        """Simple YAML-like key: value parser (fallback when PyYAML unavailable)."""
        result: dict[str, Any] = {}
        for line in yaml_str.strip().split("\n"):
            line = line.strip()
            if ":" in line and not line.startswith("#"):
                key, _, value = line.partition(":")
                key = key.strip()
                value = value.strip()
                if value:
                    # Try to parse as number or boolean
                    if value.lower() in ("true", "yes"):
                        result[key] = True
                    elif value.lower() in ("false", "no"):
                        result[key] = False
                    elif value.isdigit():
                        result[key] = int(value)
                    else:
                        result[key] = value
        return result

    @staticmethod
    def _discover_supporting_files(directory: Path) -> list[Path]:
        """Find supporting files in a skill directory."""
        supporting: list[Path] = []
        skip_names = {"SKILL.md", "__pycache__", ".git"}
        skip_extensions = {".pyc", ".pyo"}

        if not directory.is_dir():
            return supporting

        for item in directory.iterdir():
            if item.name in skip_names:
                continue
            if item.suffix in skip_extensions:
                continue
            if item.is_file():
                supporting.append(item)

        return sorted(supporting)
