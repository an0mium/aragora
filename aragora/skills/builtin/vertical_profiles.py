"""Vertical Profile Skills — one-step domain configuration.

Installable skills that configure the entire debate pipeline for a
specific industry vertical. Each profile packages:
- Evaluation weight profiles (from evaluation/llm_judge.py)
- Domain-specific rubrics
- Compliance artifact generation
- Knowledge Mound adapter configuration
- Agent selection preferences

Usage:
    from aragora.skills.builtin.vertical_profiles import HealthcareHIPAASkill

    skill = HealthcareHIPAASkill()
    result = await skill.execute(
        {"action": "configure"},
        context=SkillContext(...)
    )
    # Arena is now configured for healthcare HIPAA decisions
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from aragora.skills.base import (
    Skill,
    SkillCapability,
    SkillContext,
    SkillManifest,
    SkillResult,
    SkillStatus,
)

logger = logging.getLogger(__name__)


@dataclass
class VerticalConfig:
    """Configuration bundle for a vertical profile."""

    name: str
    display_name: str
    weight_profile: str  # Key in WEIGHT_PROFILES from llm_judge.py
    description: str
    compliance_frameworks: list[str] = field(default_factory=list)
    required_dimensions: list[str] = field(default_factory=list)
    recommended_agents: list[str] = field(default_factory=list)
    arena_overrides: dict[str, Any] = field(default_factory=dict)
    km_adapters: list[str] = field(default_factory=list)
    preset_base: str = "enterprise"

    def to_arena_config(self) -> dict[str, Any]:
        """Convert to ArenaConfig kwargs."""
        from aragora.debate.presets import get_preset

        config = get_preset(self.preset_base)
        config.update(self.arena_overrides)
        config["_vertical_profile"] = self.name
        config["_weight_profile"] = self.weight_profile
        return config


class VerticalProfileSkill(Skill):
    """Base class for vertical profile skills."""

    def __init__(self, config: VerticalConfig):
        self._config = config

    @property
    def manifest(self) -> SkillManifest:
        return SkillManifest(
            name=f"vertical_{self._config.name}",
            version="1.0.0",
            description=self._config.description,
            capabilities=[SkillCapability.READ_DATABASE],
            input_schema={
                "action": {
                    "type": "string",
                    "enum": ["configure", "info", "validate"],
                    "description": "Action to perform",
                },
            },
            output_schema={"type": "object"},
            tags=["vertical", "configuration", self._config.name],
        )

    async def execute(
        self, input_data: dict[str, Any], context: SkillContext | None = None
    ) -> SkillResult:
        action = input_data.get("action", "info")

        if action == "configure":
            return await self._configure(context)
        elif action == "validate":
            return await self._validate(context)
        else:
            return self._info()

    def _info(self) -> SkillResult:
        """Return profile information."""
        return SkillResult(status=SkillStatus.SUCCESS, data={
            "name": self._config.name,
            "display_name": self._config.display_name,
            "description": self._config.description,
            "weight_profile": self._config.weight_profile,
            "compliance_frameworks": self._config.compliance_frameworks,
            "required_dimensions": self._config.required_dimensions,
            "recommended_agents": self._config.recommended_agents,
        })

    async def _configure(self, context: SkillContext | None = None) -> SkillResult:
        """Apply vertical configuration to the arena."""
        try:
            config = self._config.to_arena_config()
            return SkillResult(status=SkillStatus.SUCCESS, data={
                "applied": True,
                "profile": self._config.name,
                "arena_config": config,
                "compliance": self._config.compliance_frameworks,
                "message": f"Applied {self._config.display_name} vertical profile",
            })
        except (ImportError, ValueError, TypeError) as e:
            return SkillResult(status=SkillStatus.FAILURE, error_message=str(e))

    async def _validate(self, context: SkillContext | None = None) -> SkillResult:
        """Validate that the current configuration meets vertical requirements."""
        checks = []

        # Check weight profile exists
        try:
            from aragora.evaluation.llm_judge import WEIGHT_PROFILES

            if self._config.weight_profile in WEIGHT_PROFILES:
                checks.append({"check": "weight_profile", "status": "pass"})
            else:
                checks.append({"check": "weight_profile", "status": "fail",
                               "detail": f"Profile '{self._config.weight_profile}' not found"})
        except ImportError:
            checks.append({"check": "weight_profile", "status": "skip",
                           "detail": "LLM judge not available"})

        # Check compliance artifacts
        if self._config.compliance_frameworks:
            try:
                from aragora.compliance.eu_ai_act import ComplianceArtifactGenerator
                checks.append({"check": "compliance_generator", "status": "pass"})
            except ImportError:
                checks.append({"check": "compliance_generator", "status": "skip",
                               "detail": "Compliance module not available"})

        all_pass = all(c["status"] != "fail" for c in checks)
        return SkillResult(status=SkillStatus.SUCCESS, data={
            "valid": all_pass,
            "checks": checks,
            "profile": self._config.name,
        })


# --- Concrete vertical skills ---

class HealthcareHIPAASkill(VerticalProfileSkill):
    """Healthcare HIPAA compliance vertical profile."""

    def __init__(self):
        super().__init__(VerticalConfig(
            name="healthcare_hipaa",
            display_name="Healthcare (HIPAA)",
            weight_profile="healthcare_hipaa",
            description="Configure debates for HIPAA-compliant healthcare decisions with "
                       "emphasis on patient safety, regulatory compliance, and clinical evidence.",
            compliance_frameworks=["hipaa", "hitech"],
            required_dimensions=["safety", "accuracy", "evidence"],
            recommended_agents=["claude", "gpt4"],
            arena_overrides={
                "enable_compliance_artifacts": True,
                "enable_provenance": True,
                "enable_receipt_generation": True,
                "enable_receipt_auto_sign": True,
            },
            km_adapters=["compliance"],
        ))


class HealthcareClinicalSkill(VerticalProfileSkill):
    """Healthcare clinical decision-making vertical profile."""

    def __init__(self):
        super().__init__(VerticalConfig(
            name="healthcare_clinical",
            display_name="Healthcare (Clinical)",
            weight_profile="healthcare_clinical",
            description="Configure debates for clinical decisions with emphasis on "
                       "evidence-based medicine and patient outcomes.",
            compliance_frameworks=["hipaa"],
            required_dimensions=["accuracy", "evidence", "reasoning"],
            recommended_agents=["claude", "gpt4"],
            arena_overrides={
                "enable_provenance": True,
                "enable_knowledge_extraction": True,
            },
            km_adapters=["compliance", "debate"],
        ))


class FinancialAuditSkill(VerticalProfileSkill):
    """Financial audit vertical profile."""

    def __init__(self):
        super().__init__(VerticalConfig(
            name="financial_audit",
            display_name="Financial (Audit)",
            weight_profile="financial_audit",
            description="Configure debates for financial audit decisions with emphasis on "
                       "accuracy, completeness, and SOX compliance.",
            compliance_frameworks=["sox", "gaap"],
            required_dimensions=["accuracy", "completeness", "reasoning"],
            recommended_agents=["claude", "gpt4"],
            arena_overrides={
                "enable_compliance_artifacts": True,
                "enable_receipt_generation": True,
                "enable_receipt_auto_sign": True,
                "enable_provenance": True,
            },
            km_adapters=["compliance"],
        ))


class FinancialRiskSkill(VerticalProfileSkill):
    """Financial risk assessment vertical profile."""

    def __init__(self):
        super().__init__(VerticalConfig(
            name="financial_risk",
            display_name="Financial (Risk)",
            weight_profile="financial_risk",
            description="Configure debates for financial risk assessment with emphasis on "
                       "quantitative reasoning and scenario analysis.",
            compliance_frameworks=["sox"],
            required_dimensions=["accuracy", "reasoning", "evidence"],
            recommended_agents=["claude", "gpt4", "gemini"],
            arena_overrides={
                "enable_provenance": True,
                "enable_knowledge_extraction": True,
                "enable_trickster": True,
            },
            km_adapters=["compliance", "debate"],
        ))


class LegalContractSkill(VerticalProfileSkill):
    """Legal contract analysis vertical profile."""

    def __init__(self):
        super().__init__(VerticalConfig(
            name="legal_contract",
            display_name="Legal (Contract)",
            weight_profile="legal_contract",
            description="Configure debates for legal contract analysis with emphasis on "
                       "completeness, accuracy, and precedent analysis.",
            compliance_frameworks=[],
            required_dimensions=["accuracy", "completeness", "reasoning"],
            recommended_agents=["claude", "gpt4"],
            arena_overrides={
                "enable_provenance": True,
                "enable_knowledge_extraction": True,
            },
            km_adapters=["debate"],
        ))


class LegalDueDiligenceSkill(VerticalProfileSkill):
    """Legal due diligence vertical profile."""

    def __init__(self):
        super().__init__(VerticalConfig(
            name="legal_due_diligence",
            display_name="Legal (Due Diligence)",
            weight_profile="legal_due_diligence",
            description="Configure debates for legal due diligence with emphasis on "
                       "thorough risk identification and evidence gathering.",
            compliance_frameworks=[],
            required_dimensions=["completeness", "evidence", "accuracy"],
            recommended_agents=["claude", "gpt4"],
            arena_overrides={
                "enable_provenance": True,
                "enable_trickster": True,
                "enable_knowledge_extraction": True,
            },
            km_adapters=["debate", "compliance"],
        ))


class ComplianceSOXSkill(VerticalProfileSkill):
    """SOX compliance vertical profile."""

    def __init__(self):
        super().__init__(VerticalConfig(
            name="compliance_sox",
            display_name="Compliance (SOX)",
            weight_profile="compliance_sox",
            description="Configure debates for SOX compliance decisions with emphasis on "
                       "internal controls, audit trails, and regulatory requirements.",
            compliance_frameworks=["sox", "coso"],
            required_dimensions=["accuracy", "completeness", "safety"],
            recommended_agents=["claude", "gpt4"],
            arena_overrides={
                "enable_compliance_artifacts": True,
                "enable_receipt_generation": True,
                "enable_receipt_auto_sign": True,
                "enable_provenance": True,
                "enable_bead_tracking": True,
            },
            km_adapters=["compliance"],
        ))


# Registry of all vertical profile skills
VERTICAL_SKILLS = {
    "healthcare_hipaa": HealthcareHIPAASkill,
    "healthcare_clinical": HealthcareClinicalSkill,
    "financial_audit": FinancialAuditSkill,
    "financial_risk": FinancialRiskSkill,
    "legal_contract": LegalContractSkill,
    "legal_due_diligence": LegalDueDiligenceSkill,
    "compliance_sox": ComplianceSOXSkill,
}


def get_vertical_skill(name: str) -> VerticalProfileSkill:
    """Get a vertical profile skill by name."""
    cls = VERTICAL_SKILLS.get(name)
    if cls is None:
        available = ", ".join(sorted(VERTICAL_SKILLS.keys()))
        raise ValueError(f"Unknown vertical profile: {name}. Available: {available}")
    return cls()


def list_vertical_profiles() -> list[dict[str, Any]]:
    """List all available vertical profiles."""
    profiles = []
    for name, cls in sorted(VERTICAL_SKILLS.items()):
        skill = cls()
        profiles.append({
            "name": name,
            "display_name": skill._config.display_name,
            "description": skill._config.description,
            "compliance": skill._config.compliance_frameworks,
        })
    return profiles
