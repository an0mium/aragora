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

__all__ = [
    "CapabilityMapper",
    "OpenClawSkillConverter",
    "OpenClawSkillParser",
    "ParsedOpenClawSkill",
]
