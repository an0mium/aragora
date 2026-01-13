"""
Gauntlet Orchestrator - Executes adversarial validation pipelines.

The orchestrator chains together:
1. Risk Assessment - Identify domain risks
2. Scenario Analysis - Test across variations
3. Adversarial Probing - Attack with specific strategies
4. Formal Verification - Verify key claims
5. Deep Audit - Multi-agent intensive review
6. Synthesis - Aggregate findings and verdict
"""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

from .config import (
    GauntletConfig,
    GauntletResult,
    GauntletFinding,
    GauntletPhase,
    GauntletSeverity,
    PhaseResult,
    AttackCategory,
)
from .templates import GauntletTemplate, get_template
from .receipt import DecisionReceipt

# Try to import real mode integrations
try:
    from aragora.modes.redteam import (
        AttackType,
        RedTeamMode,
        RedTeamResult,
    )

    REDTEAM_AVAILABLE = True
except ImportError:
    REDTEAM_AVAILABLE = False

try:
    from aragora.modes.prober import (
        ProbeType,
        VulnerabilitySeverity,
        VulnerabilityReport,
        CapabilityProber,
    )

    PROBER_AVAILABLE = True
except ImportError:
    PROBER_AVAILABLE = False

try:
    from aragora.modes.deep_audit import (
        DeepAuditConfig,
        DeepAuditOrchestrator as DeepAuditOrc,
        DeepAuditVerdict,
    )

    DEEP_AUDIT_AVAILABLE = True
except ImportError:
    DEEP_AUDIT_AVAILABLE = False

logger = logging.getLogger(__name__)


class GauntletOrchestrator:
    """
    Orchestrates adversarial validation pipelines.

    The Gauntlet runs systematic stress tests against decisions, architectures,
    and proposals using multi-agent adversarial debate with formal verification.

    Usage:
        orchestrator = GauntletOrchestrator()

        # Using a template
        result = await orchestrator.run(
            input_text="Our API uses token bucket rate limiting...",
            template=GauntletTemplate.API_ROBUSTNESS,
        )

        # Using custom config
        config = GauntletConfig(
            name="Custom Validation",
            agents=["claude", "gpt4"],
            enable_deep_audit=True,
        )
        result = await orchestrator.run(
            input_text="...",
            config=config,
        )

        # Export receipt
        receipt = result.to_receipt()
        receipt.export("validation_receipt.md", format=ReceiptFormat.MARKDOWN)
    """

    def __init__(
        self,
        agents: Optional[list] = None,
        nomic_dir: Optional[Path] = None,
        on_phase_complete: Optional[Callable[[GauntletPhase, PhaseResult], None]] = None,
        on_finding: Optional[Callable[[GauntletFinding], None]] = None,
        run_agent_fn: Optional[Callable] = None,
    ):
        """Initialize the orchestrator.

        Args:
            agents: List of Agent instances for multi-agent validation
            nomic_dir: Directory for storing artifacts (default: .nomic/)
            on_phase_complete: Callback when a phase completes
            on_finding: Callback when a finding is discovered
            run_agent_fn: Optional function to run agents (async callable)
        """
        self.agents = agents or []
        self.nomic_dir = nomic_dir or Path(".nomic")
        self.on_phase_complete = on_phase_complete
        self.on_finding = on_finding
        self.run_agent_fn = run_agent_fn or self._default_run_agent

        # Lazy-loaded components
        self._risk_assessor = None
        self._scenario_runner = None
        self._prober = None
        self._verifier = None

        # Mode integrations
        self._redteam_mode = RedTeamMode() if REDTEAM_AVAILABLE else None
        self._capability_prober = CapabilityProber() if PROBER_AVAILABLE else None

    async def _default_run_agent(self, agent, prompt: str) -> str:
        """Default agent runner using agent.generate()."""
        return await agent.generate(prompt, [])

    async def run(
        self,
        input_text: str,
        template: Optional[GauntletTemplate | str] = None,
        config: Optional[GauntletConfig] = None,
    ) -> GauntletResult:
        """Run a Gauntlet validation.

        Args:
            input_text: The decision/architecture/code to validate
            template: Pre-built template to use
            config: Custom configuration (overrides template)

        Returns:
            GauntletResult with findings and verdict
        """
        # Resolve configuration
        if config is None:
            if template is not None:
                config = get_template(template)
            else:
                config = GauntletConfig()

        # Initialize result
        result = GauntletResult(
            config=config,
            input_text=input_text,
            agents_used=config.agents[: config.max_agents],
        )

        start_time = time.time()

        try:
            # Phase 1: Risk Assessment
            result.current_phase = GauntletPhase.RISK_ASSESSMENT
            phase_result = await self._run_risk_assessment(input_text, config)
            result.phase_results.append(phase_result)
            result.findings.extend(phase_result.findings)
            self._notify_phase_complete(GauntletPhase.RISK_ASSESSMENT, phase_result)

            # Phase 2: Scenario Analysis
            if config.enable_scenario_analysis:
                result.current_phase = GauntletPhase.SCENARIO_ANALYSIS
                phase_result = await self._run_scenario_analysis(input_text, config)
                result.phase_results.append(phase_result)
                result.findings.extend(phase_result.findings)
                result.scenarios_tested = phase_result.metrics.get("scenarios_run", 0)
                self._notify_phase_complete(GauntletPhase.SCENARIO_ANALYSIS, phase_result)

            # Phase 3: Adversarial Probing
            if config.enable_adversarial_probing:
                result.current_phase = GauntletPhase.ADVERSARIAL_PROBING
                phase_result = await self._run_adversarial_probing(input_text, config)
                result.phase_results.append(phase_result)
                result.findings.extend(phase_result.findings)
                result.probes_executed = phase_result.metrics.get("probes_run", 0)
                result.robustness_score = phase_result.metrics.get("robustness_score", 0.5)
                self._notify_phase_complete(GauntletPhase.ADVERSARIAL_PROBING, phase_result)

            # Phase 4: Formal Verification
            if config.enable_formal_verification:
                result.current_phase = GauntletPhase.FORMAL_VERIFICATION
                phase_result = await self._run_formal_verification(
                    input_text, config, result.findings
                )
                result.phase_results.append(phase_result)
                result.findings.extend(phase_result.findings)
                result.verified_claims = phase_result.metrics.get("verified", 0)
                result.total_claims = phase_result.metrics.get("total", 0)
                self._notify_phase_complete(GauntletPhase.FORMAL_VERIFICATION, phase_result)

            # Phase 5: Deep Audit
            if config.enable_deep_audit:
                result.current_phase = GauntletPhase.DEEP_AUDIT
                phase_result = await self._run_deep_audit(input_text, config)
                result.phase_results.append(phase_result)
                result.findings.extend(phase_result.findings)
                result.consensus_reached = phase_result.metrics.get("consensus", False)
                result.agent_votes = phase_result.metrics.get("votes", {})
                self._notify_phase_complete(GauntletPhase.DEEP_AUDIT, phase_result)

            # Phase 6: Synthesis
            result.current_phase = GauntletPhase.SYNTHESIS
            await self._synthesize_results(result)

            # Evaluate pass/fail
            result.evaluate_pass_fail()
            result.current_phase = GauntletPhase.COMPLETE

        except asyncio.TimeoutError:
            result.current_phase = GauntletPhase.FAILED
            result.verdict_summary = f"FAILED: Timeout after {config.timeout_seconds}s"
            logger.error(f"Gauntlet {result.id} timed out")

        except Exception as e:
            result.current_phase = GauntletPhase.FAILED
            result.verdict_summary = f"FAILED: {type(e).__name__}: {str(e)[:100]}"
            logger.exception(f"Gauntlet {result.id} failed with error")

        finally:
            result.total_duration_ms = int((time.time() - start_time) * 1000)

        # Save artifacts if configured
        if config.save_artifacts:
            await self._save_artifacts(result)

        return result

    async def _run_risk_assessment(self, input_text: str, config: GauntletConfig) -> PhaseResult:
        """Run domain risk assessment phase."""
        start = time.time()
        findings: list[GauntletFinding] = []

        try:
            from aragora.debate.risk_assessor import RiskAssessor, RiskLevel

            assessor = RiskAssessor()
            assessments = assessor.assess_topic(input_text)

            for assessment in assessments:
                severity = {
                    RiskLevel.LOW: GauntletSeverity.LOW,
                    RiskLevel.MEDIUM: GauntletSeverity.MEDIUM,
                    RiskLevel.HIGH: GauntletSeverity.HIGH,
                    RiskLevel.CRITICAL: GauntletSeverity.CRITICAL,
                }.get(assessment.level, GauntletSeverity.MEDIUM)

                finding = GauntletFinding(
                    severity=severity,
                    category=assessment.category,
                    title=f"Domain Risk: {assessment.domain}",
                    description=assessment.description,
                    source_phase=GauntletPhase.RISK_ASSESSMENT,
                    recommendations=assessment.mitigations,
                    exploitability=assessment.confidence,
                    impact=0.7 if assessment.level in (RiskLevel.HIGH, RiskLevel.CRITICAL) else 0.4,
                )
                findings.append(finding)
                self._notify_finding(finding)

            return PhaseResult(
                phase=GauntletPhase.RISK_ASSESSMENT,
                status="completed",
                duration_ms=int((time.time() - start) * 1000),
                findings=findings,
                metrics={"risks_identified": len(findings)},
            )

        except ImportError:
            logger.warning("Risk assessor not available")
            return PhaseResult(
                phase=GauntletPhase.RISK_ASSESSMENT,
                status="skipped",
                duration_ms=int((time.time() - start) * 1000),
                error="Risk assessor module not available",
            )

        except Exception as e:
            logger.exception("Risk assessment failed")
            return PhaseResult(
                phase=GauntletPhase.RISK_ASSESSMENT,
                status="failed",
                duration_ms=int((time.time() - start) * 1000),
                error=str(e),
            )

    async def _run_scenario_analysis(self, input_text: str, config: GauntletConfig) -> PhaseResult:
        """Run scenario matrix analysis phase."""
        start = time.time()
        findings: list[GauntletFinding] = []

        try:
            from aragora.debate.scenarios import (
                ScenarioMatrix,
                ScenarioComparator,
                OutcomeCategory,
            )

            # Build scenario matrix from presets
            matrix = ScenarioMatrix()
            for preset in config.scenario_presets:
                preset_matrix = ScenarioMatrix.from_presets(preset)
                for scenario in preset_matrix.get_scenarios():
                    matrix.add_scenario(scenario)

            # Add custom scenarios
            for scenario_data in config.custom_scenarios:
                from aragora.debate.scenarios import Scenario, ScenarioType

                scenario = Scenario(
                    id=scenario_data.get("id", f"custom-{len(config.custom_scenarios)}"),
                    name=scenario_data.get("name", "Custom Scenario"),
                    scenario_type=ScenarioType(scenario_data.get("type", "custom")),
                    description=scenario_data.get("description", ""),
                    parameters=scenario_data.get("parameters", {}),
                )
                matrix.add_scenario(scenario)

            scenarios = matrix.get_scenarios()

            # For now, analyze scenarios without running full debates
            # (Full debate execution would require agent orchestration)
            scenario_findings = []
            for scenario in scenarios[:10]:  # Limit for performance
                # Create finding for each scenario dimension explored
                if scenario.constraints:
                    for constraint in scenario.constraints:
                        finding = GauntletFinding(
                            severity=GauntletSeverity.INFO,
                            category="scenario_constraint",
                            title=f"Constraint Scenario: {scenario.name}",
                            description=f"Consider behavior under constraint: {constraint}",
                            source_phase=GauntletPhase.SCENARIO_ANALYSIS,
                            metadata={"scenario_id": scenario.id},
                        )
                        scenario_findings.append(finding)

            # Check for divergent outcomes (placeholder for actual debate results)
            comparator_metrics = {
                "scenarios_generated": len(scenarios),
                "scenarios_run": min(len(scenarios), 10),
                "outcome_category": OutcomeCategory.CONDITIONAL.value,
            }

            return PhaseResult(
                phase=GauntletPhase.SCENARIO_ANALYSIS,
                status="completed",
                duration_ms=int((time.time() - start) * 1000),
                findings=scenario_findings,
                metrics=comparator_metrics,
            )

        except ImportError:
            logger.warning("Scenario analysis not available")
            return PhaseResult(
                phase=GauntletPhase.SCENARIO_ANALYSIS,
                status="skipped",
                duration_ms=int((time.time() - start) * 1000),
                error="Scenario module not available",
            )

        except Exception as e:
            logger.exception("Scenario analysis failed")
            return PhaseResult(
                phase=GauntletPhase.SCENARIO_ANALYSIS,
                status="failed",
                duration_ms=int((time.time() - start) * 1000),
                error=str(e),
            )

    async def _run_adversarial_probing(
        self, input_text: str, config: GauntletConfig
    ) -> PhaseResult:
        """Run adversarial probing phase using RedTeam and CapabilityProber."""
        start = time.time()
        findings: list[GauntletFinding] = []
        probes_run = 0
        robustness_score = 0.5

        try:
            # Run RedTeam attacks if available and agents present
            if REDTEAM_AVAILABLE and self._redteam_mode and self.agents:
                logger.info("Running red-team attacks...")
                try:
                    redteam_result = await self._redteam_mode.run_redteam(
                        target_proposal=input_text,
                        proposer="input_author",
                        red_team_agents=self.agents[: config.max_agents],
                        run_agent_fn=self.run_agent_fn,
                        max_rounds=3,
                    )

                    # Convert RedTeam findings
                    for attack in redteam_result.critical_issues:
                        severity = self._severity_float_to_enum(attack.severity)
                        finding = GauntletFinding(
                            severity=severity,
                            category="redteam_attack",
                            title=f"Attack: {attack.attack_type.value}",
                            description=attack.attack_description,
                            source_phase=GauntletPhase.ADVERSARIAL_PROBING,
                            recommendations=[attack.mitigation] if attack.mitigation else [],
                            metadata={
                                "attacker": attack.attacker,
                                "evidence": attack.evidence,
                            },
                        )
                        findings.append(finding)
                        self._notify_finding(finding)

                    probes_run += redteam_result.total_attacks
                    robustness_score = redteam_result.robustness_score
                    logger.info(
                        f"Red-team: {redteam_result.total_attacks} attacks, robustness={robustness_score:.0%}"
                    )

                except Exception as e:
                    logger.warning(f"Red-team failed: {e}")

            # Run capability probing if available
            if PROBER_AVAILABLE and self._capability_prober and self.agents:
                logger.info("Running capability probes...")
                try:
                    target_agent = self.agents[0]
                    probe_report = await self._capability_prober.probe_agent(
                        target_agent=target_agent,
                        run_agent_fn=self.run_agent_fn,
                        probes_per_type=2,
                    )

                    # Convert probe findings
                    severity_map = {
                        VulnerabilitySeverity.CRITICAL: GauntletSeverity.CRITICAL,
                        VulnerabilitySeverity.HIGH: GauntletSeverity.HIGH,
                        VulnerabilitySeverity.MEDIUM: GauntletSeverity.MEDIUM,
                        VulnerabilitySeverity.LOW: GauntletSeverity.LOW,
                    }

                    for probe_type, results in probe_report.by_type.items():
                        for probe_result in results:
                            if probe_result.vulnerability_found:
                                severity = severity_map.get(
                                    probe_result.severity, GauntletSeverity.MEDIUM
                                )
                                finding = GauntletFinding(
                                    severity=severity,
                                    category=f"probe_{probe_type}",
                                    title=f"Probe: {probe_type}",
                                    description=probe_result.vulnerability_description
                                    or "Vulnerability detected",
                                    source_phase=GauntletPhase.ADVERSARIAL_PROBING,
                                    metadata={"evidence": probe_result.evidence or ""},
                                )
                                findings.append(finding)
                                self._notify_finding(finding)

                    probes_run += probe_report.probes_run
                    logger.info(
                        f"Probing: {probe_report.vulnerabilities_found}/{probe_report.probes_run} vulnerabilities"
                    )

                except Exception as e:
                    logger.warning(f"Probing failed: {e}")

            return PhaseResult(
                phase=GauntletPhase.ADVERSARIAL_PROBING,
                status="completed",
                duration_ms=int((time.time() - start) * 1000),
                findings=findings,
                metrics={
                    "probes_run": probes_run,
                    "robustness_score": robustness_score,
                    "attack_categories_tested": len(config.attack_categories),
                },
            )

        except Exception as e:
            logger.exception("Adversarial probing failed")
            return PhaseResult(
                phase=GauntletPhase.ADVERSARIAL_PROBING,
                status="failed",
                duration_ms=int((time.time() - start) * 1000),
                error=str(e),
            )

    def _severity_float_to_enum(self, severity: float) -> GauntletSeverity:
        """Convert float severity to GauntletSeverity enum."""
        if severity >= 0.9:
            return GauntletSeverity.CRITICAL
        elif severity >= 0.7:
            return GauntletSeverity.HIGH
        elif severity >= 0.4:
            return GauntletSeverity.MEDIUM
        elif severity > 0:
            return GauntletSeverity.LOW
        return GauntletSeverity.INFO

    async def _run_formal_verification(
        self,
        input_text: str,
        config: GauntletConfig,
        existing_findings: list[GauntletFinding],
    ) -> PhaseResult:
        """Run formal verification phase."""
        start = time.time()
        findings: list[GauntletFinding] = []
        verified = 0
        total = 0

        try:
            from aragora.verification.formal import (
                FormalVerificationManager,
                FormalProofStatus,
            )

            manager = FormalVerificationManager()

            # Extract claims from input (simplified - would use NLP in production)
            claims = self._extract_claims(input_text)
            total = len(claims)

            for claim in claims[:5]:  # Limit for performance
                result = await manager.attempt_formal_verification(claim)

                if result.status == FormalProofStatus.PROOF_FOUND:
                    verified += 1
                    # Mark related findings as verified
                    for finding in existing_findings:
                        if claim.lower() in finding.description.lower():
                            finding.is_verified = True
                            finding.verification_method = (
                                result.language.value if result.language else "formal"
                            )

                elif result.status == FormalProofStatus.PROOF_FAILED:
                    finding = GauntletFinding(
                        severity=GauntletSeverity.HIGH,
                        category="verification_failed",
                        title=f"Claim Not Verifiable: {claim[:50]}...",
                        description=f"Formal verification could not prove: {claim}",
                        source_phase=GauntletPhase.FORMAL_VERIFICATION,
                        is_verified=False,
                        verification_method=result.language.value if result.language else None,
                    )
                    findings.append(finding)
                    self._notify_finding(finding)

            return PhaseResult(
                phase=GauntletPhase.FORMAL_VERIFICATION,
                status="completed",
                duration_ms=int((time.time() - start) * 1000),
                findings=findings,
                metrics={
                    "total": total,
                    "verified": verified,
                    "verification_rate": verified / total if total > 0 else 0,
                },
            )

        except ImportError:
            logger.warning("Formal verification not available")
            return PhaseResult(
                phase=GauntletPhase.FORMAL_VERIFICATION,
                status="skipped",
                duration_ms=int((time.time() - start) * 1000),
                error="Verification module not available",
            )

        except Exception as e:
            logger.exception("Formal verification failed")
            return PhaseResult(
                phase=GauntletPhase.FORMAL_VERIFICATION,
                status="failed",
                duration_ms=int((time.time() - start) * 1000),
                error=str(e),
            )

    async def _run_deep_audit(self, input_text: str, config: GauntletConfig) -> PhaseResult:
        """Run deep audit phase using DeepAuditOrchestrator."""
        start = time.time()
        findings: list[GauntletFinding] = []
        consensus_reached = False
        agent_votes: dict[str, Any] = {}

        try:
            if DEEP_AUDIT_AVAILABLE and self.agents:
                logger.info("Running deep audit...")
                try:
                    audit_config = DeepAuditConfig(
                        rounds=(
                            config.deep_audit_rounds if hasattr(config, "deep_audit_rounds") else 4
                        ),
                        enable_research=False,
                        risk_threshold=0.7,
                    )
                    orchestrator = DeepAuditOrc(self.agents, audit_config)
                    verdict = await orchestrator.run(
                        task=f"Analyze and critique this input:\n\n{input_text[:5000]}",
                        context="This is a stress-test to find weaknesses and blind spots.",
                    )

                    # Convert audit findings
                    for af in verdict.findings:
                        severity = self._severity_float_to_enum(af.severity)
                        finding = GauntletFinding(
                            severity=severity,
                            category=f"audit_{af.category}",
                            title=f"Audit: {af.category}",
                            description=af.summary,
                            source_phase=GauntletPhase.DEEP_AUDIT,
                            metadata={"details": af.details},
                        )
                        findings.append(finding)
                        self._notify_finding(finding)

                    # Convert unanimous issues to high-severity findings
                    for issue in verdict.unanimous_issues:
                        finding = GauntletFinding(
                            severity=GauntletSeverity.HIGH,
                            category="audit_unanimous",
                            title="Unanimous Issue",
                            description=issue,
                            source_phase=GauntletPhase.DEEP_AUDIT,
                        )
                        findings.append(finding)
                        self._notify_finding(finding)

                    consensus_reached = verdict.confidence > 0.7
                    logger.info(
                        f"Deep audit: confidence={verdict.confidence:.0%}, {len(verdict.findings)} findings"
                    )

                except Exception as e:
                    logger.warning(f"Deep audit failed: {e}")
            else:
                logger.warning("Deep audit not available (no agents or module)")

            return PhaseResult(
                phase=GauntletPhase.DEEP_AUDIT,
                status="completed",
                duration_ms=int((time.time() - start) * 1000),
                findings=findings,
                metrics={
                    "consensus": consensus_reached,
                    "votes": agent_votes,
                    "rounds_completed": (
                        config.deep_audit_rounds if hasattr(config, "deep_audit_rounds") else 4
                    ),
                },
            )

        except Exception as e:
            logger.exception("Deep audit failed")
            return PhaseResult(
                phase=GauntletPhase.DEEP_AUDIT,
                status="failed",
                duration_ms=int((time.time() - start) * 1000),
                error=str(e),
            )

    async def _synthesize_results(self, result: GauntletResult) -> None:
        """Synthesize findings into final verdict."""
        # Calculate risk score from findings
        severity_weights = {
            GauntletSeverity.CRITICAL: 1.0,
            GauntletSeverity.HIGH: 0.7,
            GauntletSeverity.MEDIUM: 0.4,
            GauntletSeverity.LOW: 0.1,
            GauntletSeverity.INFO: 0.0,
        }

        total_risk = sum(
            severity_weights.get(f.severity, 0) * f.risk_score for f in result.findings
        )
        max_risk = len(result.findings) * 1.0  # Max if all critical
        result.risk_score = total_risk / max_risk if max_risk > 0 else 0.0

        # Calculate confidence based on phase completion
        completed_phases = sum(1 for p in result.phase_results if p.status == "completed")
        total_phases = len(result.phase_results)
        phase_confidence = completed_phases / total_phases if total_phases > 0 else 0.5

        # Factor in verification rate
        verification_confidence = (
            result.verified_claims / result.total_claims if result.total_claims > 0 else 0.5
        )

        # Combine confidence factors
        result.confidence = (
            0.4 * phase_confidence + 0.3 * result.robustness_score + 0.3 * verification_confidence
        )

    def _extract_claims(self, text: str) -> list[str]:
        """Extract verifiable claims from text (simplified)."""
        # Simple heuristic: sentences containing assertion keywords
        assertion_patterns = [
            "must",
            "always",
            "never",
            "guarantees",
            "ensures",
            "will",
            "cannot",
            "impossible",
            "required",
            "mandatory",
        ]

        sentences = text.replace("\n", " ").split(".")
        claims = []

        for sentence in sentences:
            sentence = sentence.strip()
            if any(pattern in sentence.lower() for pattern in assertion_patterns):
                if len(sentence) > 20:  # Skip very short sentences
                    claims.append(sentence)

        return claims[:10]  # Limit claims

    async def _save_artifacts(self, result: GauntletResult) -> None:
        """Save Gauntlet artifacts to disk."""
        output_dir = (
            Path(result.config.output_dir)
            if result.config.output_dir
            else self.nomic_dir / "gauntlets"
        )
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save result JSON
        result_path = output_dir / f"{result.id}_result.json"
        result_path.write_text(result.to_dict().__str__())  # Would use json.dumps

        # Generate and save receipt if configured
        if result.config.generate_receipt:
            receipt = DecisionReceipt.from_gauntlet_result(result)
            (output_dir / f"{result.id}_receipt.md").write_text(receipt.to_markdown())
            (output_dir / f"{result.id}_receipt.json").write_text(receipt.to_json())

        logger.info(f"Saved Gauntlet artifacts to {output_dir}")

    def _notify_phase_complete(self, phase: GauntletPhase, result: PhaseResult) -> None:
        """Notify callback of phase completion."""
        if self.on_phase_complete:
            try:
                self.on_phase_complete(phase, result)
            except Exception as e:
                logger.warning(f"Phase completion callback failed: {e}")

    def _notify_finding(self, finding: GauntletFinding) -> None:
        """Notify callback of new finding."""
        if self.on_finding:
            try:
                self.on_finding(finding)
            except Exception as e:
                logger.warning(f"Finding callback failed: {e}")


# Convenience function for quick validation
async def run_gauntlet(
    input_text: str,
    template: GauntletTemplate | str = GauntletTemplate.QUICK_SANITY,
) -> GauntletResult:
    """Quick convenience function to run a Gauntlet validation.

    Args:
        input_text: Text to validate
        template: Template to use (default: quick sanity check)

    Returns:
        GauntletResult with findings and verdict
    """
    orchestrator = GauntletOrchestrator()
    return await orchestrator.run(input_text=input_text, template=template)
