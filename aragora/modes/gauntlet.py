"""
Gauntlet Mode - Adversarial Validation Engine.

DEPRECATED: This module is deprecated. Use aragora.gauntlet instead.

The canonical Gauntlet implementation is now in the aragora.gauntlet package,
which provides a more feature-complete API with:
- GauntletRunner for simple 3-phase execution
- GauntletOrchestrator for full 5-phase execution
- Templates for common use cases
- Receipt generation for compliance
- Persona-based regulatory testing

Migration:
    # Old (deprecated)
    from aragora.modes.gauntlet import GauntletOrchestrator

    # New (recommended)
    from aragora.gauntlet import GauntletOrchestrator
"""

from __future__ import annotations

import warnings

warnings.warn(
    "aragora.modes.gauntlet is deprecated. Use aragora.gauntlet instead. "
    "See docs/GAUNTLET_ARCHITECTURE.md for migration guide.",
    DeprecationWarning,
    stacklevel=2,
)

import asyncio
import hashlib
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional, Type

from aragora.core import Agent
from aragora.debate.consensus import (
    DissentRecord,
    UnresolvedTension,
)
from aragora.debate.risk_assessor import RiskAssessment, RiskAssessor, RiskLevel

# Import shared types from aragora.gauntlet.types (canonical source)
from aragora.gauntlet.types import InputType, Verdict
from aragora.modes.deep_audit import (
    DeepAuditConfig,
    DeepAuditOrchestrator,
    DeepAuditVerdict,
)
from aragora.modes.prober import (
    CapabilityProber,
    ProbeType,
    VulnerabilityReport,
    VulnerabilitySeverity,
)
from aragora.modes.redteam import (
    AttackType,
    RedTeamMode,
    RedTeamResult,
)
from aragora.verification.formal import (
    FormalProofStatus,
    get_formal_verification_manager,
)

# Import personas for compliance-aware stress testing
try:
    from aragora.gauntlet.personas import PersonaAttack, RegulatoryPersona, get_persona

    PERSONAS_AVAILABLE = True
except ImportError:
    PERSONAS_AVAILABLE = False
    RegulatoryPersona: Optional[Type[Any]] = None
    PersonaAttack: Optional[Type[Any]] = None

logger = logging.getLogger(__name__)


@dataclass
class GauntletProgress:
    """Progress update during Gauntlet execution."""

    phase: str  # Current phase name
    phase_number: int  # 1-indexed phase number
    total_phases: int  # Total number of phases
    percent: float  # 0-100
    message: str  # Human-readable status
    findings_so_far: int = 0  # Findings discovered so far
    current_task: Optional[str] = None  # Current sub-task


# Type for progress callback
ProgressCallback = Callable[[GauntletProgress], None]

# InputType and Verdict are imported from aragora.gauntlet.types (canonical source)
# This ensures consistency across the gauntlet system


@dataclass
class Finding:
    """A finding from the Gauntlet process."""

    finding_id: str
    category: str  # "attack", "probe", "audit", "verification", "risk"
    severity: float  # 0-1
    title: str
    description: str
    evidence: str = ""
    mitigation: Optional[str] = None
    source: str = ""  # Which component found this
    verified: bool = False  # Was this formally verified?
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    @property
    def severity_level(self) -> str:
        """Human-readable severity level."""
        if self.severity >= 0.9:
            return "CRITICAL"
        elif self.severity >= 0.7:
            return "HIGH"
        elif self.severity >= 0.4:
            return "MEDIUM"
        return "LOW"


@dataclass
class VerifiedClaim:
    """A claim that was formally verified."""

    claim: str
    verified: bool
    verification_method: str  # "z3", "lean", "manual"
    proof_hash: Optional[str] = None
    verification_time_ms: float = 0.0


@dataclass
class GauntletConfig:
    """Configuration for Gauntlet stress-testing."""

    # Input configuration
    input_type: InputType = InputType.SPEC
    input_content: str = ""
    input_path: Optional[Path] = None

    # Which attack types to run (None = all)
    attack_types: Optional[list[AttackType]] = None

    # Which probe types to run (None = all)
    probe_types: Optional[list[ProbeType]] = None

    # Thresholds
    severity_threshold: float = 0.5  # Findings below this are filtered
    risk_threshold: float = 0.7  # Above this triggers warning

    # Timing
    max_duration_seconds: int = 600  # 10 minute max
    verification_timeout_seconds: float = 60.0

    # Parallelism
    parallel_attacks: int = 5
    parallel_probes: int = 3

    # Feature toggles
    enable_redteam: bool = True
    enable_probing: bool = True
    enable_deep_audit: bool = True
    enable_verification: bool = True
    enable_risk_assessment: bool = True

    # Deep audit rounds (fewer for speed, more for thoroughness)
    deep_audit_rounds: int = 4

    # Regulatory persona for compliance-aware stress testing
    # Can be a RegulatoryPersona instance or string name ("gdpr", "hipaa", "ai_act", "security")
    persona: Optional[Any] = None  # RegulatoryPersona or str

    def __post_init__(self):
        # Load content from path if provided
        if self.input_path and not self.input_content:
            self.input_content = self.input_path.read_text()

        # Resolve persona string to instance
        if self.persona and isinstance(self.persona, str) and PERSONAS_AVAILABLE:
            self.persona = get_persona(self.persona)


@dataclass
class GauntletResult:
    """Complete result of a Gauntlet stress-test."""

    # Identifiers (required)
    gauntlet_id: str
    input_type: InputType
    input_summary: str  # First 500 chars of input

    # Verdict (required)
    verdict: Verdict
    confidence: float  # 0-1

    # Scores (required)
    risk_score: float  # 0-1, aggregate risk
    robustness_score: float  # 0-1, how well input held up
    coverage_score: float  # 0-1, how thoroughly tested

    # Optional fields with defaults
    input_hash: str = ""  # SHA-256 of full input content

    # Findings by severity
    critical_findings: list[Finding] = field(default_factory=list)
    high_findings: list[Finding] = field(default_factory=list)
    medium_findings: list[Finding] = field(default_factory=list)
    low_findings: list[Finding] = field(default_factory=list)

    # Consensus & dissent
    consensus_reached: bool = False
    dissenting_views: list[DissentRecord] = field(default_factory=list)
    unresolved_tensions: list[UnresolvedTension] = field(default_factory=list)

    # Verification
    verified_claims: list[VerifiedClaim] = field(default_factory=list)
    unverified_claims: list[str] = field(default_factory=list)
    verification_coverage: float = 0.0  # % of claims that were verified

    # Risk assessment
    risk_assessments: list[RiskAssessment] = field(default_factory=list)

    # Sub-results (for drill-down)
    redteam_result: Optional[RedTeamResult] = None
    probe_report: Optional[VulnerabilityReport] = None
    audit_verdict: Optional[DeepAuditVerdict] = None

    # Metadata
    agents_involved: list[str] = field(default_factory=list)
    duration_seconds: float = 0.0
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    @property
    def all_findings(self) -> list[Finding]:
        """All findings sorted by severity."""
        return (
            self.critical_findings + self.high_findings + self.medium_findings + self.low_findings
        )

    @property
    def total_findings(self) -> int:
        """Total number of findings."""
        return len(self.all_findings)

    @property
    def severity_counts(self) -> dict[str, int]:
        """Count findings by severity level."""
        return {
            "critical": len(self.critical_findings),
            "high": len(self.high_findings),
            "medium": len(self.medium_findings),
            "low": len(self.low_findings),
        }

    @property
    def checksum(self) -> str:
        """Generate integrity checksum for the result."""
        content = f"{self.gauntlet_id}:{self.verdict.value}:{self.confidence}:{self.total_findings}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def to_dict(self) -> dict:
        """Convert result to a JSON-serializable dictionary."""
        findings = []
        for finding in self.all_findings:
            findings.append(
                {
                    "id": finding.finding_id,
                    "category": finding.category,
                    "severity": finding.severity,
                    "severity_level": finding.severity_level,
                    "title": finding.title,
                    "description": finding.description,
                    "evidence": finding.evidence,
                    "mitigation": finding.mitigation,
                    "source": finding.source,
                    "verified": finding.verified,
                    "timestamp": finding.timestamp,
                }
            )

        return {
            "gauntlet_id": self.gauntlet_id,
            "input_type": self.input_type.value,
            "input_summary": self.input_summary,
            "input_hash": self.input_hash,
            "verdict": self.verdict.value,
            "confidence": self.confidence,
            "risk_score": self.risk_score,
            "robustness_score": self.robustness_score,
            "coverage_score": self.coverage_score,
            "verification_coverage": self.verification_coverage,
            "severity_counts": self.severity_counts,
            "findings": findings,
            "consensus_reached": self.consensus_reached,
            "agents_involved": self.agents_involved,
            "duration_seconds": self.duration_seconds,
            "created_at": self.created_at,
        }

    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            "=" * 60,
            "GAUNTLET STRESS-TEST RESULT",
            "=" * 60,
            "",
            f"ID: {self.gauntlet_id}",
            f"Input Type: {self.input_type.value}",
            "",
            f"VERDICT: {self.verdict.value.upper()}",
            f"Confidence: {self.confidence:.0%}",
            "",
            "--- Scores ---",
            f"Risk Score: {self.risk_score:.0%}",
            f"Robustness Score: {self.robustness_score:.0%}",
            f"Coverage Score: {self.coverage_score:.0%}",
            f"Verification Coverage: {self.verification_coverage:.0%}",
            "",
            "--- Findings ---",
            f"Critical: {len(self.critical_findings)}",
            f"High: {len(self.high_findings)}",
            f"Medium: {len(self.medium_findings)}",
            f"Low: {len(self.low_findings)}",
            "",
        ]

        if self.critical_findings:
            lines.append("CRITICAL ISSUES:")
            for f in self.critical_findings[:5]:
                lines.append(f"  - {f.title}")

        if self.dissenting_views:
            lines.append("")
            lines.append(f"Dissenting Views: {len(self.dissenting_views)}")

        if self.unresolved_tensions:
            lines.append(f"Unresolved Tensions: {len(self.unresolved_tensions)}")

        lines.append("")
        lines.append(f"Duration: {self.duration_seconds:.1f}s")
        lines.append(f"Agents: {', '.join(self.agents_involved)}")
        lines.append(f"Checksum: {self.checksum}")

        return "\n".join(lines)


class GauntletOrchestrator:
    """
    Orchestrates comprehensive adversarial stress-testing.

    Combines multiple validation techniques:
    1. Red-team attacks for adversarial probing
    2. Capability probing for agent reliability
    3. Deep audit for intensive analysis
    4. Formal verification for provable claims
    5. Risk assessment for domain hazards

    Usage:
        orchestrator = GauntletOrchestrator(agents)
        result = await orchestrator.run(config)
        print(result.summary())
    """

    def __init__(
        self,
        agents: list[Agent],
        run_agent_fn: Optional[Callable] = None,
        on_progress: Optional[ProgressCallback] = None,
    ):
        """
        Initialize Gauntlet orchestrator.

        Args:
            agents: Agents to participate in stress-testing
            run_agent_fn: Optional function to run agents (async callable)
            on_progress: Optional callback for progress updates
        """
        self.agents = agents
        self.run_agent_fn = run_agent_fn or self._default_run_agent
        self.on_progress = on_progress

        # Initialize sub-components
        self.redteam_mode = RedTeamMode()
        self.prober = CapabilityProber()
        self.risk_assessor = RiskAssessor()
        self.verification_manager = get_formal_verification_manager()

        # Tracking
        self._finding_counter = 0
        self._start_time: Optional[datetime] = None
        self._findings_count = 0

    def _emit_progress(
        self,
        phase: str,
        phase_number: int,
        total_phases: int,
        percent: float,
        message: str,
        current_task: Optional[str] = None,
    ) -> None:
        """Emit progress update if callback is configured."""
        if self.on_progress:
            progress = GauntletProgress(
                phase=phase,
                phase_number=phase_number,
                total_phases=total_phases,
                percent=percent,
                message=message,
                findings_so_far=self._findings_count,
                current_task=current_task,
            )
            self.on_progress(progress)

    async def _default_run_agent(self, agent: Agent, prompt: str) -> str:
        """Default agent runner using agent.generate()."""
        from aragora.server.stream.arena_hooks import streaming_task_context

        task_id = f"{agent.name}:gauntlet"
        with streaming_task_context(task_id):
            return await agent.generate(prompt, [])

    async def run(self, config: GauntletConfig) -> GauntletResult:
        """
        Run a complete Gauntlet stress-test.

        Args:
            config: Configuration for the stress-test

        Returns:
            GauntletResult with verdict and findings
        """
        self._start_time = datetime.now()
        self._findings_count = 0
        gauntlet_id = f"gauntlet-{datetime.now().strftime('%Y%m%d%H%M%S')}-{uuid.uuid4().hex[:6]}"

        logger.info("=" * 60)
        logger.info(f"GAUNTLET STRESS-TEST: {gauntlet_id}")
        logger.info(f"Input Type: {config.input_type.value}")
        logger.info(f"Agents: {', '.join(a.name for a in self.agents)}")
        logger.info("=" * 60)

        # Emit initial progress
        self._emit_progress("Initialization", 1, 3, 0, "Starting Gauntlet stress-test...")

        all_findings: list[Finding] = []
        dissenting_views: list[DissentRecord] = []
        unresolved_tensions: list[UnresolvedTension] = []
        verified_claims: list[VerifiedClaim] = []
        unverified_claims: list[str] = []

        # Initialize sub-results
        redteam_result: Optional[RedTeamResult] = None
        probe_report: Optional[VulnerabilityReport] = None
        audit_verdict: Optional[DeepAuditVerdict] = None
        risk_assessments: list[RiskAssessment] = []

        # 1. Risk Assessment (fast, run first)
        if config.enable_risk_assessment:
            logger.info("--- Phase 1: Risk Assessment ---")
            self._emit_progress(
                "Risk Assessment",
                1,
                3,
                10,
                "Analyzing domain-specific risks...",
                current_task="risk_assessment",
            )
            risk_assessments = self.risk_assessor.assess_topic(config.input_content[:2000])
            for ra in risk_assessments:
                all_findings.append(
                    Finding(
                        finding_id=self._next_finding_id(),
                        category="risk",
                        severity=self._risk_level_to_severity(ra.level),
                        title=f"Domain Risk: {ra.category}",
                        description=ra.description,
                        mitigation=", ".join(ra.mitigations),
                        source="RiskAssessor",
                    )
                )
            self._findings_count = len(all_findings)
            self._emit_progress(
                "Risk Assessment",
                1,
                3,
                20,
                f"Risk assessment complete: {len(risk_assessments)} risks identified",
            )

        # 2. Run parallel stress tests
        tasks = []

        if config.enable_redteam and self.agents:
            tasks.append(("redteam", self._run_redteam(config)))

        if config.enable_probing and self.agents:
            tasks.append(("probing", self._run_probing(config)))

        if config.enable_deep_audit and self.agents:
            tasks.append(("deep_audit", self._run_deep_audit(config)))

        if config.enable_verification:
            tasks.append(("verification", self._run_verification(config)))

        # Persona-specific compliance attacks
        if config.persona and PERSONAS_AVAILABLE and self.agents:
            tasks.append(("persona", self._run_persona_attacks(config)))

        # Execute with timeout
        logger.info("--- Phase 2: Parallel Stress Tests ---")
        task_names = [t[0] for t in tasks]
        self._emit_progress(
            "Stress Testing",
            2,
            3,
            25,
            f"Running {len(tasks)} parallel stress tests: {', '.join(task_names)}",
            current_task=task_names[0] if task_names else None,
        )

        if tasks:
            try:
                results = await asyncio.wait_for(
                    asyncio.gather(*[t[1] for t in tasks], return_exceptions=True),
                    timeout=config.max_duration_seconds,
                )

                # Process results
                completed_tasks = 0
                for (task_name, _), result in zip(tasks, results):
                    completed_tasks += 1
                    progress_pct = 25 + (completed_tasks / len(tasks)) * 50  # 25-75%

                    if isinstance(result, Exception):
                        logger.warning(f"{task_name} failed: {result}")
                        self._emit_progress(
                            "Stress Testing",
                            2,
                            3,
                            progress_pct,
                            f"{task_name} failed: {str(result)[:50]}",
                            current_task=task_name,
                        )
                        continue

                    if task_name == "redteam" and result:
                        redteam_result = result
                        new_findings = self._redteam_to_findings(result)
                        all_findings.extend(new_findings)
                        self._findings_count = len(all_findings)
                        self._emit_progress(
                            "Stress Testing",
                            2,
                            3,
                            progress_pct,
                            f"Red-team complete: {result.total_attacks} attacks, {len(new_findings)} findings",
                            current_task="redteam",
                        )

                    elif task_name == "probing" and result:
                        probe_report = result
                        new_findings = self._probe_to_findings(result)
                        all_findings.extend(new_findings)
                        self._findings_count = len(all_findings)
                        self._emit_progress(
                            "Stress Testing",
                            2,
                            3,
                            progress_pct,
                            f"Probing complete: {result.vulnerabilities_found} vulnerabilities",
                            current_task="probing",
                        )

                    elif task_name == "deep_audit" and result:
                        audit_verdict = result
                        findings, dissents, tensions = self._audit_to_findings(result)
                        all_findings.extend(findings)
                        dissenting_views.extend(dissents)
                        unresolved_tensions.extend(tensions)
                        self._findings_count = len(all_findings)
                        self._emit_progress(
                            "Stress Testing",
                            2,
                            3,
                            progress_pct,
                            f"Deep audit complete: {len(findings)} findings, {len(audit_verdict.unanimous_issues)} unanimous",
                            current_task="deep_audit",
                        )

                    elif task_name == "verification" and result:
                        verified, unverified = result
                        verified_claims.extend(verified)
                        unverified_claims.extend(unverified)
                        self._emit_progress(
                            "Stress Testing",
                            2,
                            3,
                            progress_pct,
                            f"Verification complete: {len(verified)} verified, {len(unverified)} unverified",
                            current_task="verification",
                        )

                    elif task_name == "persona" and result:
                        # Persona findings are already Finding objects
                        all_findings.extend(result)
                        self._findings_count = len(all_findings)
                        self._emit_progress(
                            "Stress Testing",
                            2,
                            3,
                            progress_pct,
                            f"Persona attacks complete: {len(result)} findings",
                            current_task="persona",
                        )

            except asyncio.TimeoutError:
                logger.warning(f"Gauntlet timed out after {config.max_duration_seconds}s")
                self._emit_progress(
                    "Stress Testing", 2, 3, 75, f"Timeout after {config.max_duration_seconds}s"
                )

        # 3. Aggregate and score
        logger.info("--- Phase 3: Aggregation ---")
        self._emit_progress("Aggregation", 3, 3, 80, "Filtering and scoring findings...")

        # Filter findings by threshold
        all_findings = [f for f in all_findings if f.severity >= config.severity_threshold]
        self._findings_count = len(all_findings)

        # Sort into severity buckets
        critical = [f for f in all_findings if f.severity >= 0.9]
        high = [f for f in all_findings if 0.7 <= f.severity < 0.9]
        medium = [f for f in all_findings if 0.4 <= f.severity < 0.7]
        low = [f for f in all_findings if f.severity < 0.4]

        self._emit_progress(
            "Aggregation",
            3,
            3,
            85,
            f"Categorized: {len(critical)} critical, {len(high)} high, {len(medium)} medium",
        )

        # Calculate aggregate scores
        risk_score = self._calculate_risk_score(all_findings, risk_assessments)
        robustness_score = redteam_result.robustness_score if redteam_result else 1.0
        coverage_score = self._calculate_coverage_score(redteam_result, probe_report, audit_verdict)
        verification_coverage = (
            len(verified_claims) / (len(verified_claims) + len(unverified_claims))
            if (verified_claims or unverified_claims)
            else 0.0
        )

        self._emit_progress(
            "Aggregation",
            3,
            3,
            90,
            f"Scores: risk={risk_score:.0%}, robustness={robustness_score:.0%}",
        )

        # Determine verdict
        verdict, confidence = self._determine_verdict(
            critical, high, medium, risk_score, robustness_score, dissenting_views
        )
        self._emit_progress(
            "Aggregation",
            3,
            3,
            95,
            f"Verdict: {verdict.value.upper()} ({confidence:.0%} confidence)",
        )

        # Build result
        duration = (datetime.now() - self._start_time).total_seconds()
        input_hash = hashlib.sha256(config.input_content.encode()).hexdigest()

        result = GauntletResult(
            gauntlet_id=gauntlet_id,
            input_type=config.input_type,
            input_summary=config.input_content[:500],
            input_hash=input_hash,
            verdict=verdict,
            confidence=confidence,
            risk_score=risk_score,
            robustness_score=robustness_score,
            coverage_score=coverage_score,
            critical_findings=critical,
            high_findings=high,
            medium_findings=medium,
            low_findings=low,
            consensus_reached=audit_verdict.confidence > 0.7 if audit_verdict else False,
            dissenting_views=dissenting_views,
            unresolved_tensions=unresolved_tensions,
            verified_claims=verified_claims,
            unverified_claims=unverified_claims,
            verification_coverage=verification_coverage,
            risk_assessments=risk_assessments,
            redteam_result=redteam_result,
            probe_report=probe_report,
            audit_verdict=audit_verdict,
            agents_involved=[a.name for a in self.agents],
            duration_seconds=duration,
        )

        logger.info(f"\n{result.summary()}")

        self._emit_progress(
            "Complete",
            3,
            3,
            100,
            f"Gauntlet complete: {result.total_findings} findings, {verdict.value.upper()}",
        )

        return result

    def _next_finding_id(self) -> str:
        """Generate unique finding ID."""
        self._finding_counter += 1
        return f"finding-{self._finding_counter:04d}"

    def _risk_level_to_severity(self, level: RiskLevel) -> float:
        """Convert RiskLevel to severity float."""
        mapping = {
            RiskLevel.LOW: 0.25,
            RiskLevel.MEDIUM: 0.5,
            RiskLevel.HIGH: 0.75,
            RiskLevel.CRITICAL: 0.95,
        }
        return mapping.get(level, 0.5)

    async def _run_redteam(self, config: GauntletConfig) -> Optional[RedTeamResult]:
        """Run red-team adversarial testing."""
        logger.info("Running red-team attacks...")

        if not self.agents:
            return None

        try:
            # Use extra agent as proposer for defense if available
            red_team_agents = self.agents[: config.parallel_attacks]
            proposer_agent = None
            if len(self.agents) > config.parallel_attacks:
                proposer_agent = self.agents[config.parallel_attacks]
                logger.info(f"Using {getattr(proposer_agent, 'name', 'agent')} as defender")

            result = await self.redteam_mode.run_redteam(
                target_proposal=config.input_content,
                proposer="input_author",
                red_team_agents=red_team_agents,
                run_agent_fn=self.run_agent_fn,
                max_rounds=3,
                proposer_agent=proposer_agent,
            )
            logger.info(
                f"Red-team: {result.total_attacks} attacks, robustness={result.robustness_score:.0%}"
            )
            return result
        except Exception as e:
            logger.warning(f"Red-team failed: {e}")
            return None

    async def _run_persona_attacks(self, config: GauntletConfig) -> list[Finding]:
        """Run persona-specific compliance attacks."""
        if not config.persona or not PERSONAS_AVAILABLE:
            return []

        if not self.agents:
            return []

        logger.info(f"Running persona attacks: {config.persona.name}...")

        findings: list[Finding] = []
        persona = config.persona

        # Run each persona attack using available agents
        attack_agents = self.agents[: config.parallel_attacks]

        for attack in persona.attack_prompts:
            try:
                # Generate attack prompt
                attack_prompt = persona.get_attack_prompt(
                    config.input_content[:5000],
                    attack,  # Limit context
                )

                # Run attack with first available agent
                agent = attack_agents[0] if attack_agents else self.agents[0]
                response = await self.run_agent_fn(agent, attack_prompt)

                # Parse response for findings
                parsed_findings = self._parse_persona_response(
                    response, attack, persona, agent.name
                )
                findings.extend(parsed_findings)

            except Exception as e:
                logger.debug(f"Persona attack {attack.id} failed: {e}")

        logger.info(
            f"Persona attacks: {len(findings)} findings from {len(persona.attack_prompts)} attacks"
        )
        return findings

    def _parse_persona_response(
        self,
        response: str,
        attack: Any,  # PersonaAttack
        persona: Any,  # RegulatoryPersona
        agent_name: str,
    ) -> list[Finding]:
        """Parse persona attack response into findings."""
        findings = []

        # Simple heuristic parsing - look for severity indicators
        response_lower = response.lower()

        # Check if response indicates findings
        has_critical = "critical" in response_lower and (
            "finding" in response_lower
            or "violation" in response_lower
            or "issue" in response_lower
        )
        has_high = "high" in response_lower and (
            "finding" in response_lower or "risk" in response_lower or "severity" in response_lower
        )
        has_medium = "medium" in response_lower and (
            "finding" in response_lower or "risk" in response_lower or "severity" in response_lower
        )

        # If response contains compliance findings, create a finding
        compliance_indicators = [
            "violation",
            "non-compliant",
            "missing",
            "inadequate",
            "failure",
            "gap",
            "risk",
            "concern",
            "issue",
        ]

        if any(ind in response_lower for ind in compliance_indicators):
            # Determine severity from response
            if has_critical:
                severity = 0.95
            elif has_high:
                severity = 0.75
            elif has_medium:
                severity = 0.50
            else:
                severity = 0.35

            # Apply persona severity weight
            if attack.category in persona.severity_weights:
                severity = min(1.0, severity * persona.severity_weights[attack.category])

            # Extract first paragraph as summary
            paragraphs = [p.strip() for p in response.split("\n\n") if p.strip()]
            summary = paragraphs[0] if paragraphs else response[:300]

            findings.append(
                Finding(
                    finding_id=self._next_finding_id(),
                    category=f"persona/{attack.category}",
                    severity=severity,
                    title=f"{persona.regulation}: {attack.name}",
                    description=summary[:500],
                    evidence=response[:1000],
                    mitigation=None,  # Would need more sophisticated parsing
                    source=f"Persona/{persona.name}/{agent_name}",
                )
            )

        return findings

    async def _run_probing(self, config: GauntletConfig) -> Optional[VulnerabilityReport]:
        """Run capability probing on agents."""
        logger.info("Running capability probes...")

        if not self.agents:
            return None

        try:
            # Probe the first agent (usually the most capable)
            target_agent = self.agents[0]
            report = await self.prober.probe_agent(
                target_agent=target_agent,
                run_agent_fn=self.run_agent_fn,
                probe_types=config.probe_types,
                probes_per_type=2,  # Reduced for speed
            )
            logger.info(
                f"Probing: {report.vulnerabilities_found}/{report.probes_run} vulnerabilities"
            )
            return report
        except Exception as e:
            logger.warning(f"Probing failed: {e}")
            return None

    async def _run_deep_audit(self, config: GauntletConfig) -> Optional[DeepAuditVerdict]:
        """Run deep audit analysis."""
        logger.info("Running deep audit...")

        if not self.agents:
            return None

        try:
            audit_config = DeepAuditConfig(
                rounds=config.deep_audit_rounds,
                enable_research=False,  # Faster without web research
                risk_threshold=config.risk_threshold,
            )
            orchestrator = DeepAuditOrchestrator(self.agents, audit_config)
            verdict = await orchestrator.run(
                task=f"Analyze and critique this {config.input_type.value}:\n\n{config.input_content[:5000]}",
                context="This is a stress-test to find weaknesses and blind spots.",
            )
            logger.info(
                f"Deep audit: confidence={verdict.confidence:.0%}, {len(verdict.findings)} findings"
            )
            return verdict
        except Exception as e:
            logger.warning(f"Deep audit failed: {e}")
            return None

    async def _run_verification(
        self, config: GauntletConfig
    ) -> tuple[list[VerifiedClaim], list[str]]:
        """Run formal verification on extractable claims."""
        logger.info("Running formal verification...")

        verified: list[VerifiedClaim] = []
        unverified: list[str] = []

        # Extract potential claims from input (simple heuristic)
        claims = self._extract_verifiable_claims(config.input_content)

        for claim in claims[:10]:  # Limit to 10 claims
            try:
                result = await self.verification_manager.attempt_formal_verification(
                    claim=claim,
                    timeout_seconds=config.verification_timeout_seconds,
                )

                if result.status == FormalProofStatus.PROOF_FOUND:
                    verified.append(
                        VerifiedClaim(
                            claim=claim,
                            verified=True,
                            verification_method=result.language.value,
                            proof_hash=result.proof_hash,
                            verification_time_ms=result.proof_search_time_ms,
                        )
                    )
                elif result.status == FormalProofStatus.PROOF_FAILED:
                    verified.append(
                        VerifiedClaim(
                            claim=claim,
                            verified=False,
                            verification_method=result.language.value,
                            verification_time_ms=result.proof_search_time_ms,
                        )
                    )
                else:
                    unverified.append(claim)

            except Exception as e:
                logger.debug(f"Verification failed for claim: {e}")
                unverified.append(claim)

        logger.info(f"Verification: {len(verified)} verified, {len(unverified)} unverified")
        return verified, unverified

    def _extract_verifiable_claims(self, content: str) -> list[str]:
        """Extract claims that might be formally verifiable."""
        import re

        claims = []

        # Look for mathematical/logical statements
        patterns = [
            r"(?:if|when)\s+[^.]+then\s+[^.]+",  # If-then statements
            r"for all\s+[^.]+",  # Universal quantifiers
            r"there exists\s+[^.]+",  # Existential quantifiers
            r"[^.]*(?:must|shall|always|never)[^.]+",  # Modal claims
            r"[^.]*(?:implies|entails|guarantees)[^.]+",  # Logical implications
        ]

        for pattern in patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            claims.extend(matches[:5])  # Limit per pattern

        return list(set(claims))[:10]

    def _redteam_to_findings(self, result: RedTeamResult) -> list[Finding]:
        """Convert red-team result to findings."""
        findings = []

        for attack in result.critical_issues:
            findings.append(
                Finding(
                    finding_id=self._next_finding_id(),
                    category="attack",
                    severity=attack.severity,
                    title=f"Attack: {attack.attack_type.value}",
                    description=attack.attack_description,
                    evidence=attack.evidence,
                    mitigation=attack.mitigation,
                    source=f"RedTeam/{attack.attacker}",
                )
            )

        return findings

    def _probe_to_findings(self, report: VulnerabilityReport) -> list[Finding]:
        """Convert probe report to findings."""
        findings = []

        severity_map = {
            VulnerabilitySeverity.CRITICAL: 0.95,
            VulnerabilitySeverity.HIGH: 0.75,
            VulnerabilitySeverity.MEDIUM: 0.5,
            VulnerabilitySeverity.LOW: 0.25,
        }

        for probe_type, results in report.by_type.items():
            for probe_result in results:
                if probe_result.vulnerability_found:
                    findings.append(
                        Finding(
                            finding_id=self._next_finding_id(),
                            category="probe",
                            severity=severity_map.get(probe_result.severity, 0.5),
                            title=f"Probe: {probe_type}",
                            description=probe_result.vulnerability_description
                            or "Vulnerability detected",
                            evidence=probe_result.evidence or "",
                            source=f"Prober/{report.target_agent}",
                        )
                    )

        return findings

    def _audit_to_findings(
        self, verdict: DeepAuditVerdict
    ) -> tuple[list[Finding], list[DissentRecord], list[UnresolvedTension]]:
        """Convert deep audit verdict to findings and dissent records."""
        findings = []
        dissents = []
        tensions = []

        # Convert audit findings
        for af in verdict.findings:
            findings.append(
                Finding(
                    finding_id=self._next_finding_id(),
                    category="audit",
                    severity=af.severity,
                    title=f"Audit: {af.category}",
                    description=af.summary,
                    evidence=af.details,
                    source="DeepAudit",
                )
            )

        # Convert unanimous issues to high-severity findings
        for issue in verdict.unanimous_issues:
            findings.append(
                Finding(
                    finding_id=self._next_finding_id(),
                    category="audit",
                    severity=0.85,  # Unanimous = high severity
                    title="Unanimous Issue",
                    description=issue,
                    source="DeepAudit/Unanimous",
                )
            )

        # Convert split opinions to dissent records
        for opinion in verdict.split_opinions:
            dissents.append(
                DissentRecord(
                    agent="multiple",
                    claim_id="",
                    dissent_type="partial",
                    reasons=[opinion],
                    severity=0.5,
                )
            )

        # Convert risk areas to tensions
        for risk in verdict.risk_areas:
            tensions.append(
                UnresolvedTension(
                    tension_id=f"tension-{uuid.uuid4().hex[:6]}",
                    description=risk,
                    agents_involved=[],
                    options=[],
                    impact="Identified during deep audit",
                )
            )

        return findings, dissents, tensions

    def _calculate_risk_score(
        self,
        findings: list[Finding],
        risk_assessments: list[RiskAssessment],
    ) -> float:
        """Calculate aggregate risk score."""
        if not findings and not risk_assessments:
            return 0.0

        # Weight by severity
        finding_risk = sum(f.severity**2 for f in findings)  # Square to emphasize high severity
        finding_max = len(findings) if findings else 1

        # Factor in domain risks
        domain_risk = sum(
            self._risk_level_to_severity(ra.level) * ra.confidence for ra in risk_assessments
        )
        domain_max = len(risk_assessments) if risk_assessments else 1

        # Combine
        combined = (finding_risk / finding_max + domain_risk / domain_max) / 2
        return min(1.0, combined)

    def _calculate_coverage_score(
        self,
        redteam: Optional[RedTeamResult],
        probe: Optional[VulnerabilityReport],
        audit: Optional[DeepAuditVerdict],
    ) -> float:
        """Calculate test coverage score."""
        scores = []

        if redteam:
            scores.append(redteam.coverage_score)

        if probe:
            # Coverage based on probe types tested
            scores.append(min(1.0, probe.probes_run / 20))

        if audit:
            # Coverage based on audit completion
            scores.append(audit.confidence)

        return sum(scores) / len(scores) if scores else 0.0

    def _determine_verdict(
        self,
        critical: list[Finding],
        high: list[Finding],
        medium: list[Finding],
        risk_score: float,
        robustness_score: float,
        dissents: list[DissentRecord],
    ) -> tuple[Verdict, float]:
        """Determine final verdict and confidence."""

        # Automatic rejection conditions
        if len(critical) >= 2:
            return Verdict.REJECTED, 0.9

        if len(critical) >= 1 and len(high) >= 3:
            return Verdict.REJECTED, 0.85

        if risk_score > 0.8:
            return Verdict.REJECTED, 0.8

        # Needs review conditions
        if len(critical) == 1:
            return Verdict.NEEDS_REVIEW, 0.7

        if len(high) >= 3:
            return Verdict.NEEDS_REVIEW, 0.65

        if len(dissents) >= 3:
            return Verdict.NEEDS_REVIEW, 0.6

        if risk_score > 0.6:
            return Verdict.NEEDS_REVIEW, 0.6

        # Approved with conditions
        if len(high) >= 1 or len(medium) >= 3:
            confidence = robustness_score * (1 - risk_score * 0.3)
            return Verdict.APPROVED_WITH_CONDITIONS, confidence

        # Clean approval
        confidence = robustness_score * (1 - risk_score * 0.2)
        return Verdict.APPROVED, min(0.95, confidence)


# Convenience function for quick stress-testing
async def run_gauntlet(
    input_content: str,
    agents: list[Agent],
    input_type: InputType = InputType.SPEC,
    **config_kwargs,
) -> GauntletResult:
    """
    Run a Gauntlet stress-test.

    Args:
        input_content: Content to stress-test
        agents: Agents to participate
        input_type: Type of input
        **config_kwargs: Additional GauntletConfig options

    Returns:
        GauntletResult with verdict and findings

    Example:
        result = await run_gauntlet(
            spec_document,
            agents=[claude, gpt4, gemini],
            input_type=InputType.SPEC,
        )
        print(result.verdict)  # APPROVED, NEEDS_REVIEW, or REJECTED
    """
    config = GauntletConfig(
        input_type=input_type,
        input_content=input_content,
        **config_kwargs,
    )
    orchestrator = GauntletOrchestrator(agents)
    return await orchestrator.run(config)


# Pre-configured Gauntlet profiles
QUICK_GAUNTLET = GauntletConfig(
    deep_audit_rounds=2,
    parallel_attacks=2,
    enable_verification=False,
    max_duration_seconds=120,
)

THOROUGH_GAUNTLET = GauntletConfig(
    deep_audit_rounds=6,
    parallel_attacks=5,
    parallel_probes=5,
    enable_verification=True,
    max_duration_seconds=900,  # 15 min
)

CODE_REVIEW_GAUNTLET = GauntletConfig(
    input_type=InputType.CODE,
    attack_types=[
        AttackType.SECURITY,
        AttackType.EDGE_CASE,
        AttackType.RACE_CONDITION,
        AttackType.RESOURCE_EXHAUSTION,
    ],
    deep_audit_rounds=4,
    enable_verification=True,
)

POLICY_GAUNTLET = GauntletConfig(
    input_type=InputType.POLICY,
    attack_types=[
        AttackType.LOGICAL_FALLACY,
        AttackType.UNSTATED_ASSUMPTION,
        AttackType.EDGE_CASE,
        AttackType.COUNTEREXAMPLE,
    ],
    deep_audit_rounds=5,
    severity_threshold=0.3,  # More sensitive
)


# Compliance-focused gauntlet profiles
def get_compliance_gauntlet(persona_name: str = "gdpr") -> GauntletConfig:
    """
    Get a compliance-focused Gauntlet config with a regulatory persona.

    Args:
        persona_name: Name of persona ("gdpr", "hipaa", "ai_act", "security")

    Returns:
        GauntletConfig with persona configured

    Example:
        config = get_compliance_gauntlet("gdpr")
        result = await run_gauntlet(spec, agents, config=config)
    """
    return GauntletConfig(
        input_type=InputType.POLICY,
        attack_types=[
            AttackType.LOGICAL_FALLACY,
            AttackType.UNSTATED_ASSUMPTION,
            AttackType.EDGE_CASE,
            AttackType.COUNTEREXAMPLE,
        ],
        deep_audit_rounds=4,
        severity_threshold=0.3,
        persona=persona_name,  # Will be resolved in __post_init__
    )


GDPR_GAUNTLET = get_compliance_gauntlet("gdpr") if PERSONAS_AVAILABLE else None
HIPAA_GAUNTLET = get_compliance_gauntlet("hipaa") if PERSONAS_AVAILABLE else None
AI_ACT_GAUNTLET = get_compliance_gauntlet("ai_act") if PERSONAS_AVAILABLE else None
SECURITY_GAUNTLET = get_compliance_gauntlet("security") if PERSONAS_AVAILABLE else None
SOX_GAUNTLET = get_compliance_gauntlet("sox") if PERSONAS_AVAILABLE else None
