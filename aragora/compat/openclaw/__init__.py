"""
OpenClaw Compatibility Layer.

Provides migration utilities for converting between OpenClaw and Aragora formats:
- SKILL.md parsing and conversion
- Capability mapping between OpenClaw and Aragora
- Computer-use action bridging
- Migration workflows
"""

from .capability_mapper import CapabilityMapper
from .skill_parser import OpenClawSkillParser, ParsedOpenClawSkill
from .skill_converter import OpenClawSkillConverter
from .skill_scanner import DangerousSkillError, ScanResult, SkillScanner, Verdict

__all__ = [
    "CapabilityMapper",
    "DangerousSkillError",
    "OpenClawSkillConverter",
    "OpenClawSkillParser",
    "ParsedOpenClawSkill",
    "ScanResult",
    "SkillScanner",
    "Verdict",
]
