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
    ADVERSARIAL_INPUT = "adversarial_input"

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
    LOGICAL_FALLACY = "logical_fallacy"
    EDGE_CASES = "edge_cases"
    EDGE_CASE = "edge_case"
    ASSUMPTIONS = "assumptions"
    UNSTATED_ASSUMPTION = "unstated_assumption"
    COUNTEREXAMPLES = "counterexamples"
    COUNTEREXAMPLE = "counterexample"

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


class GauntletPhase(Enum):
    """Phases of the Gauntlet validation pipeline."""

    NOT_STARTED = "not_started"
    RISK_ASSESSMENT = "risk_assessment"
    SCENARIO_ANALYSIS = "scenario_analysis"
    ADVERSARIAL_PROBING = "adversarial_probing"
    FORMAL_VERIFICATION = "formal_verification"
    DEEP_AUDIT = "deep_audit"
    SYNTHESIS = "synthesis"
    COMPLETE = "complete"
    FAILED = "failed"


class GauntletSeverity(Enum):
    """Severity levels for Gauntlet findings."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


@dataclass
class PassFailCriteria:
    """Criteria for determining pass/fail verdict."""

    max_critical_findings: int = 0
    max_high_findings: int = 2
    min_robustness_score: float = 0.7
    min_verification_coverage: float = 0.0
    require_consensus: bool = False

    @classmethod
    def strict(cls) -> "PassFailCriteria":
        """Strict criteria - no critical or high findings allowed."""
        return cls(
            max_critical_findings=0,
            max_high_findings=0,
            min_robustness_score=0.85,
            min_verification_coverage=0.5,
            require_consensus=True,
        )

    @classmethod
    def lenient(cls) -> "PassFailCriteria":
        """Lenient criteria - allows some findings."""
        return cls(
            max_critical_findings=1,
            max_high_findings=5,
            min_robustness_score=0.5,
            min_verification_coverage=0.0,
            require_consensus=False,
        )


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


@dataclass
class GauntletFinding:
    """A finding from the Gauntlet validation process."""

    id: str = field(default_factory=lambda: f"finding-{id(object())}")
    severity: GauntletSeverity = GauntletSeverity.MEDIUM
    category: str = ""
    title: str = ""
    description: str = ""
    source_phase: GauntletPhase = GauntletPhase.NOT_STARTED

    # Optional details
    recommendations: list[str] = field(default_factory=list)
    attack_type: Optional[AttackCategory] = None
    exploitability: float = 0.5
    impact: float = 0.5
    risk_score: float = 0.5

    # Verification
    is_verified: bool = False
    verification_method: Optional[str] = None

    # Metadata
    metadata: dict = field(default_factory=dict)


@dataclass
class PhaseResult:
    """Result from a single Gauntlet phase."""

    phase: GauntletPhase
    status: str  # "completed", "skipped", "failed"
    duration_ms: int = 0
    findings: list[GauntletFinding] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)
    error: Optional[str] = None


@dataclass
class GauntletResult:
    """Result of a complete Gauntlet validation run."""

    id: str = field(default_factory=lambda: f"gauntlet-{id(object())}")
    config: Optional[GauntletConfig] = None
    input_text: str = ""
    agents_used: list[str] = field(default_factory=list)

    # Current status
    current_phase: GauntletPhase = GauntletPhase.NOT_STARTED

    # Results
    phase_results: list[PhaseResult] = field(default_factory=list)
    findings: list[GauntletFinding] = field(default_factory=list)

    # Scores
    risk_score: float = 0.0
    robustness_score: float = 0.5
    confidence: float = 0.5

    # Phase-specific metrics
    scenarios_tested: int = 0
    probes_executed: int = 0
    verified_claims: int = 0
    total_claims: int = 0
    consensus_reached: bool = False
    agent_votes: dict = field(default_factory=dict)

    # Verdict
    passed: bool = False
    verdict_summary: str = ""

    # Timing
    total_duration_ms: int = 0

    @property
    def severity_counts(self) -> dict:
        """Count findings by severity."""
        counts = {s.value: 0 for s in GauntletSeverity}
        for f in self.findings:
            counts[f.severity.value] = counts.get(f.severity.value, 0) + 1
        return counts

    @property
    def critical_findings(self) -> list:
        """Get critical severity findings."""
        return [f for f in self.findings if f.severity == GauntletSeverity.CRITICAL]

    def evaluate_pass_fail(self) -> None:
        """Evaluate pass/fail based on criteria."""
        criteria = PassFailCriteria()

        critical_count = len(self.critical_findings)
        high_count = len([f for f in self.findings if f.severity == GauntletSeverity.HIGH])

        if critical_count > criteria.max_critical_findings:
            self.passed = False
            self.verdict_summary = f"FAIL: {critical_count} critical findings"
        elif high_count > criteria.max_high_findings:
            self.passed = False
            self.verdict_summary = f"FAIL: {high_count} high findings"
        elif self.robustness_score < criteria.min_robustness_score:
            self.passed = False
            self.verdict_summary = f"FAIL: Robustness {self.robustness_score:.0%}"
        else:
            self.passed = True
            self.verdict_summary = f"PASS: {len(self.findings)} findings"

    def to_dict(self) -> dict:
        """Convert result to dictionary."""
        return {
            "id": self.id,
            "passed": self.passed,
            "verdict_summary": self.verdict_summary,
            "risk_score": self.risk_score,
            "robustness_score": self.robustness_score,
            "severity_counts": self.severity_counts,
            "findings_count": len(self.findings),
            "total_duration_ms": self.total_duration_ms,
        }

    def to_receipt(self):
        """Convert to DecisionReceipt."""
        from .receipt import DecisionReceipt
        return DecisionReceipt.from_gauntlet_result(self)
