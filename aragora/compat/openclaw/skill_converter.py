"""
OpenClaw Skill Converter.

Converts parsed OpenClaw skills (SKILL.md) into Aragora Skill instances.

Conversion flow:
    SKILL.md → OpenClawSkillParser → ParsedOpenClawSkill → OpenClawSkillConverter → Skill

The converter creates a bridge skill that wraps the OpenClaw instructions
and routes execution through the OpenClaw proxy.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from aragora.skills.base import (
    Skill,
    SkillCapability,
    SkillContext,
    SkillManifest,
    SkillResult,
)

from .capability_mapper import CapabilityMapper
from .skill_parser import OpenClawSkillParser, ParsedOpenClawSkill
from .skill_scanner import DangerousSkillError, SkillScanner, Verdict

logger = logging.getLogger(__name__)


class OpenClawBridgeSkill(Skill):
    """
    A Skill that wraps OpenClaw instructions and routes through the proxy.

    This bridge skill is created by the converter and acts as an adapter
    between the Aragora skill system and OpenClaw's execution model.
    """

    def __init__(
        self,
        parsed_skill: ParsedOpenClawSkill,
        capabilities: list[SkillCapability],
    ):
        self._parsed = parsed_skill
        self._capabilities = capabilities

    @property
    def manifest(self) -> SkillManifest:
        name = self._parsed.name or "openclaw_skill"
        # Sanitize name for Aragora (lowercase, underscores)
        safe_name = name.lower().replace("-", "_").replace(" ", "_")

        return SkillManifest(
            name=safe_name,
            version=self._parsed.frontmatter.version,
            description=self._parsed.description or f"OpenClaw skill: {name}",
            capabilities=self._capabilities,
            input_schema={
                "action": {
                    "type": "string",
                    "description": "Action to execute",
                    "required": True,
                },
                "params": {
                    "type": "object",
                    "description": "Action parameters",
                },
            },
            tags=["openclaw", "bridge"] + self._parsed.frontmatter.tags,
            debate_compatible=True,
            max_execution_time_seconds=float(self._parsed.frontmatter.timeout),
        )

    async def execute(
        self,
        input_data: dict[str, Any],
        context: SkillContext,
    ) -> SkillResult:
        """Execute the OpenClaw skill via the proxy."""
        try:
            from aragora.skills.builtin.openclaw_skill import OpenClawSkill

            # Delegate to the OpenClaw skill with the instructions as context
            proxy_skill = OpenClawSkill()
            result = await proxy_skill.execute(input_data, context)

            # Add bridge metadata
            if result.data:
                result.data["bridge_skill"] = self._parsed.name
                result.data["instructions_length"] = len(self._parsed.instructions)

            return result

        except (ImportError, RuntimeError, ValueError, TypeError, OSError) as e:
            logger.exception(f"Bridge skill execution failed: {e}")
            return SkillResult.create_failure(
                f"OpenClaw bridge execution failed: {e}",
                error_code="bridge_error",
            )

    @property
    def instructions(self) -> str:
        """Get the raw OpenClaw instructions."""
        return self._parsed.instructions


class OpenClawSkillConverter:
    """Converts OpenClaw skills to Aragora skills."""

    @staticmethod
    def convert(
        parsed_skill: ParsedOpenClawSkill,
        *,
        skip_scan: bool = False,
    ) -> OpenClawBridgeSkill:
        """
        Convert a parsed OpenClaw skill to an Aragora bridge skill.

        The skill is scanned for malicious patterns before conversion.
        DANGEROUS skills raise :class:`DangerousSkillError`.
        SUSPICIOUS skills are converted but a warning is attached to the
        bridge skill metadata.

        Args:
            parsed_skill: The parsed SKILL.md data.
            skip_scan: If True, bypass the malware scan (for tests only).

        Returns:
            An OpenClawBridgeSkill that can be registered in Aragora.

        Raises:
            DangerousSkillError: If the skill is classified as DANGEROUS.
        """
        # --- Security scan ---------------------------------------------------
        scan_warning: str | None = None
        if not skip_scan:
            scanner = SkillScanner()
            scan_result = scanner.scan(parsed_skill)

            if scan_result.verdict == Verdict.DANGEROUS:
                logger.warning(
                    "Rejecting dangerous OpenClaw skill %r: risk_score=%d",
                    parsed_skill.name,
                    scan_result.risk_score,
                )
                raise DangerousSkillError(scan_result)

            if scan_result.verdict == Verdict.SUSPICIOUS:
                scan_warning = (
                    f"Skill flagged as SUSPICIOUS (risk_score={scan_result.risk_score}). "
                    f"Findings: {'; '.join(f.description for f in scan_result.findings[:5])}"
                )
                logger.warning(
                    "OpenClaw skill %r is suspicious: %s",
                    parsed_skill.name,
                    scan_warning,
                )

        # --- Capability mapping -----------------------------------------------
        capabilities = CapabilityMapper.to_aragora(parsed_skill.requires)

        # If no capabilities specified, default to safe read-only access.
        # SECURITY: Never grant SHELL_EXECUTION by default -- a malicious
        # SKILL.md that omits `requires:` should not get shell access.
        if not capabilities:
            capabilities = [SkillCapability.READ_LOCAL]

        bridge = OpenClawBridgeSkill(
            parsed_skill=parsed_skill,
            capabilities=capabilities,
        )

        # Attach scan warning to bridge metadata so callers can inspect it
        if scan_warning:
            bridge._scan_warning = scan_warning  # type: ignore[attr-defined]

        return bridge

    @staticmethod
    def convert_file(
        path: str | Path,
        *,
        skip_scan: bool = False,
    ) -> OpenClawBridgeSkill:
        """
        Parse and convert a SKILL.md file.

        Args:
            path: Path to the SKILL.md file.
            skip_scan: If True, bypass the malware scan.

        Returns:
            An OpenClawBridgeSkill.

        Raises:
            DangerousSkillError: If the skill is classified as DANGEROUS.
        """
        parsed = OpenClawSkillParser.parse_file(path)
        return OpenClawSkillConverter.convert(parsed, skip_scan=skip_scan)

    @staticmethod
    def convert_directory(
        directory: str | Path,
        *,
        skip_scan: bool = False,
    ) -> list[OpenClawBridgeSkill]:
        """
        Parse and convert all SKILL.md files in a directory.

        Dangerous skills are logged and skipped rather than raising.

        Args:
            directory: Path to search for SKILL.md files.
            skip_scan: If True, bypass the malware scan.

        Returns:
            List of converted bridge skills (dangerous ones excluded).
        """
        parsed_skills = OpenClawSkillParser.parse_directory(directory)
        results: list[OpenClawBridgeSkill] = []
        for s in parsed_skills:
            try:
                results.append(OpenClawSkillConverter.convert(s, skip_scan=skip_scan))
            except DangerousSkillError as exc:
                logger.warning(
                    "Skipping dangerous skill %r in directory scan: %s",
                    s.name,
                    exc,
                )
        return results

    @staticmethod
    def to_skill_md(skill: Skill) -> str:
        """
        Export an Aragora skill to SKILL.md format.

        Args:
            skill: An Aragora Skill instance.

        Returns:
            SKILL.md formatted string.
        """
        manifest = skill.manifest
        openclaw_caps = CapabilityMapper.to_openclaw(manifest.capabilities)

        lines = [
            "---",
            f"name: {manifest.name}",
            f"description: {manifest.description}",
            f"version: {manifest.version}",
        ]

        if openclaw_caps:
            lines.append("metadata:")
            lines.append("  openclaw:")
            lines.append("    requires:")
            for cap in openclaw_caps:
                lines.append(f"      - {cap}")
            if manifest.max_execution_time_seconds:
                lines.append(f"    timeout: {int(manifest.max_execution_time_seconds)}")

        if manifest.tags:
            lines.append("tags:")
            for tag in manifest.tags:
                lines.append(f"  - {tag}")

        lines.append("---")
        lines.append("")
        lines.append(f"# {manifest.name}")
        lines.append("")
        lines.append(manifest.description)
        lines.append("")

        return "\n".join(lines)
