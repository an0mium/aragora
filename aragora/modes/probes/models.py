"""
Data models for capability probing.

Defines probe types, results, and vulnerability reports used across
the probing system.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional


class ProbeType(Enum):
    """Types of capability probes."""

    CONTRADICTION = "contradiction"  # Try to get agent to contradict itself
    HALLUCINATION = "hallucination"  # Check for made-up facts
    SYCOPHANCY = "sycophancy"  # Check if agent just agrees
    PERSISTENCE = "persistence"  # Check if agent gives up too easily
    CONFIDENCE_CALIBRATION = "confidence_calibration"  # Check confidence accuracy
    REASONING_DEPTH = "reasoning_depth"  # Probe logical reasoning
    EDGE_CASE = "edge_case"  # Find boundary failures
    INSTRUCTION_INJECTION = "instruction_injection"  # Test for prompt injection
    CAPABILITY_EXAGGERATION = "capability_exaggeration"  # Test for overclaiming


class VulnerabilitySeverity(Enum):
    """Severity levels for discovered vulnerabilities."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ProbeResult:
    """Result of a single probe."""

    probe_id: str
    probe_type: ProbeType
    target_agent: str

    # Probe details
    probe_prompt: str
    agent_response: str

    # Analysis
    vulnerability_found: bool
    vulnerability_description: str = ""
    severity: VulnerabilitySeverity = VulnerabilitySeverity.LOW

    # Evidence
    evidence: str = ""
    contradiction_with: Optional[str] = None  # Previous statement if contradicting

    # Timing
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    response_time_ms: float = 0.0

    def to_dict(self) -> dict:
        return {
            "probe_id": self.probe_id,
            "probe_type": self.probe_type.value,
            "target_agent": self.target_agent,
            "probe_prompt": self.probe_prompt[:500],
            "agent_response": self.agent_response[:500],
            "vulnerability_found": self.vulnerability_found,
            "vulnerability_description": self.vulnerability_description,
            "severity": self.severity.value,
            "evidence": self.evidence,
            "created_at": self.created_at,
        }


@dataclass
class VulnerabilityReport:
    """Comprehensive report of agent vulnerabilities."""

    report_id: str
    target_agent: str
    probes_run: int
    vulnerabilities_found: int

    # Breakdown by type
    by_type: dict[str, list[ProbeResult]] = field(default_factory=dict)

    # Summary stats
    vulnerability_rate: float = 0.0
    critical_count: int = 0
    high_count: int = 0
    medium_count: int = 0
    low_count: int = 0

    # Recommendations
    recommendations: list[str] = field(default_factory=list)

    # ELO impact
    elo_penalty: float = 0.0

    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict:
        return {
            "report_id": self.report_id,
            "target_agent": self.target_agent,
            "probes_run": self.probes_run,
            "vulnerabilities_found": self.vulnerabilities_found,
            "vulnerability_rate": self.vulnerability_rate,
            "breakdown": {
                "critical": self.critical_count,
                "high": self.high_count,
                "medium": self.medium_count,
                "low": self.low_count,
            },
            "by_type": {k: [p.to_dict() for p in v] for k, v in self.by_type.items()},
            "recommendations": self.recommendations,
            "elo_penalty": self.elo_penalty,
            "created_at": self.created_at,
        }
