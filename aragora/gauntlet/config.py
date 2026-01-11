"""
Gauntlet Configuration.

Defines the configuration for a Gauntlet validation run.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class AttackCategory(Enum):
    """Categories of adversarial attacks to run."""

    # Security attacks
    SECURITY = "security"
    INJECTION = "injection"
    PRIVILEGE_ESCALATION = "privilege_escalation"

    # Compliance attacks
    COMPLIANCE = "compliance"
    GDPR = "gdpr"
    HIPAA = "hipaa"
    AI_ACT = "ai_act"

    # Architecture attacks
    ARCHITECTURE = "architecture"
    SCALABILITY = "scalability"
    PERFORMANCE = "performance"

    # Logic attacks
    LOGIC = "logic"
    EDGE_CASES = "edge_cases"
    ASSUMPTIONS = "assumptions"
    COUNTEREXAMPLES = "counterexamples"

    # Operational attacks
    OPERATIONAL = "operational"
    DEPENDENCY_FAILURE = "dependency_failure"
    RACE_CONDITIONS = "race_conditions"


class ProbeCategory(Enum):
    """Categories of capability probes to run."""

    CONTRADICTION = "contradiction"
    HALLUCINATION = "hallucination"
    SYCOPHANCY = "sycophancy"
    PERSISTENCE = "persistence"
    CALIBRATION = "calibration"
    REASONING_DEPTH = "reasoning_depth"
    EDGE_CASE = "edge_case"
    INSTRUCTION_INJECTION = "instruction_injection"
    CAPABILITY_EXAGGERATION = "capability_exaggeration"


class Verdict(Enum):
    """Gauntlet verdict outcomes."""

    PASS = "pass"
    CONDITIONAL = "conditional"
    FAIL = "fail"


@dataclass
class GauntletConfig:
    """
    Configuration for a Gauntlet validation run.

    Controls which attacks, probes, and scenarios to run,
    as well as thresholds for pass/fail verdicts.
    """

    # Attack configuration
    attack_categories: list[AttackCategory] = field(
        default_factory=lambda: [
            AttackCategory.SECURITY,
            AttackCategory.LOGIC,
            AttackCategory.ARCHITECTURE,
        ]
    )
    attack_rounds: int = 2
    attacks_per_category: int = 3

    # Probe configuration
    probe_categories: list[ProbeCategory] = field(
        default_factory=lambda: [
            ProbeCategory.CONTRADICTION,
            ProbeCategory.HALLUCINATION,
            ProbeCategory.SYCOPHANCY,
        ]
    )
    probes_per_category: int = 2

    # Scenario configuration
    run_scenario_matrix: bool = True
    scenario_preset: Optional[str] = "comprehensive"  # scale, time_horizon, risk, comprehensive
    max_parallel_scenarios: int = 3

    # Agent configuration
    agents: list[str] = field(
        default_factory=lambda: ["anthropic-api", "openai-api"]
    )

    # Verdict thresholds
    critical_threshold: int = 0  # Max critical issues for PASS
    high_threshold: int = 2  # Max high issues for PASS
    vulnerability_rate_threshold: float = 0.2  # Max vulnerability rate for PASS
    consensus_threshold: float = 0.7  # Min consensus confidence for PASS
    robustness_threshold: float = 0.6  # Min robustness score for PASS

    # Output configuration
    output_dir: Optional[str] = None
    output_formats: list[str] = field(
        default_factory=lambda: ["json", "md"]
    )

    # Timeouts (seconds)
    attack_timeout: int = 60
    probe_timeout: int = 30
    scenario_timeout: int = 120

    def __post_init__(self):
        """Validate configuration."""
        if not self.agents:
            raise ValueError("At least one agent required")

        if self.attack_rounds < 1:
            raise ValueError("attack_rounds must be >= 1")

        if not (0 <= self.vulnerability_rate_threshold <= 1):
            raise ValueError("vulnerability_rate_threshold must be 0-1")

        if not (0 <= self.consensus_threshold <= 1):
            raise ValueError("consensus_threshold must be 0-1")

        if not (0 <= self.robustness_threshold <= 1):
            raise ValueError("robustness_threshold must be 0-1")

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "attack_categories": [c.value for c in self.attack_categories],
            "attack_rounds": self.attack_rounds,
            "attacks_per_category": self.attacks_per_category,
            "probe_categories": [c.value for c in self.probe_categories],
            "probes_per_category": self.probes_per_category,
            "run_scenario_matrix": self.run_scenario_matrix,
            "scenario_preset": self.scenario_preset,
            "max_parallel_scenarios": self.max_parallel_scenarios,
            "agents": self.agents,
            "critical_threshold": self.critical_threshold,
            "high_threshold": self.high_threshold,
            "vulnerability_rate_threshold": self.vulnerability_rate_threshold,
            "consensus_threshold": self.consensus_threshold,
            "robustness_threshold": self.robustness_threshold,
            "output_dir": self.output_dir,
            "output_formats": self.output_formats,
            "attack_timeout": self.attack_timeout,
            "probe_timeout": self.probe_timeout,
            "scenario_timeout": self.scenario_timeout,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "GauntletConfig":
        """Create from dictionary."""
        # Convert string values back to enums
        if "attack_categories" in data:
            data["attack_categories"] = [
                AttackCategory(c) if isinstance(c, str) else c
                for c in data["attack_categories"]
            ]
        if "probe_categories" in data:
            data["probe_categories"] = [
                ProbeCategory(c) if isinstance(c, str) else c
                for c in data["probe_categories"]
            ]
        return cls(**data)

    @classmethod
    def security_focused(cls) -> "GauntletConfig":
        """Create security-focused configuration."""
        return cls(
            attack_categories=[
                AttackCategory.SECURITY,
                AttackCategory.INJECTION,
                AttackCategory.PRIVILEGE_ESCALATION,
            ],
            probe_categories=[
                ProbeCategory.INSTRUCTION_INJECTION,
                ProbeCategory.EDGE_CASE,
            ],
            attack_rounds=3,
            critical_threshold=0,
            high_threshold=0,
        )

    @classmethod
    def compliance_focused(cls) -> "GauntletConfig":
        """Create compliance-focused configuration."""
        return cls(
            attack_categories=[
                AttackCategory.COMPLIANCE,
                AttackCategory.GDPR,
                AttackCategory.HIPAA,
                AttackCategory.AI_ACT,
            ],
            probe_categories=[
                ProbeCategory.HALLUCINATION,
                ProbeCategory.CALIBRATION,
            ],
            attack_rounds=2,
            run_scenario_matrix=False,  # Focus on compliance, not scenarios
        )

    @classmethod
    def quick(cls) -> "GauntletConfig":
        """Create quick validation configuration."""
        return cls(
            attack_categories=[AttackCategory.SECURITY, AttackCategory.LOGIC],
            probe_categories=[ProbeCategory.CONTRADICTION],
            attack_rounds=1,
            attacks_per_category=2,
            probes_per_category=1,
            run_scenario_matrix=False,
        )
