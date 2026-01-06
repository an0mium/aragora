"""
Auditing and security analysis endpoint handlers.

Endpoints:
- POST /api/debates/capability-probe - Run capability probes on an agent
- POST /api/debates/deep-audit - Run deep audit on a task
- POST /api/debates/:id/red-team - Run red team analysis on a debate
"""

import asyncio
import json
import logging
import re
import time
import uuid
from datetime import datetime
from typing import Optional

from .base import (
    BaseHandler,
    HandlerResult,
    json_response,
    error_response,
    SAFE_ID_PATTERN,
)

logger = logging.getLogger(__name__)

# Lazy import flags
PROBER_AVAILABLE = False
REDTEAM_AVAILABLE = False
DEBATE_AVAILABLE = False
CapabilityProber = None
RedTeamMode = None
create_agent = None

try:
    from aragora.modes.prober import CapabilityProber as _CP
    CapabilityProber = _CP
    PROBER_AVAILABLE = True
except ImportError:
    pass

try:
    from aragora.modes.redteam import RedTeamMode as _RT
    RedTeamMode = _RT
    REDTEAM_AVAILABLE = True
except ImportError:
    pass

try:
    from aragora.debate import create_agent as _ca
    create_agent = _ca
    DEBATE_AVAILABLE = True
except ImportError:
    pass

from aragora.server.error_utils import safe_error_message as _safe_error_message


class AuditingHandler(BaseHandler):
    """Handler for security auditing and capability probing endpoints."""

    ROUTES = [
        "/api/debates/capability-probe",
        "/api/debates/deep-audit",
    ]

    def can_handle(self, path: str) -> bool:
        """Check if this handler can process the given path."""
        if path in self.ROUTES:
            return True
        # Handle /api/debates/:id/red-team pattern
        if path.startswith("/api/debates/") and path.endswith("/red-team"):
            return True
        return False

    def handle(self, path: str, query_params: dict, handler) -> Optional[HandlerResult]:
        """Route auditing requests to appropriate methods.

        Note: These endpoints require POST with request body, which handler provides.
        """
        if path == "/api/debates/capability-probe":
            return self._run_capability_probe(handler)

        if path == "/api/debates/deep-audit":
            return self._run_deep_audit(handler)

        if path.startswith("/api/debates/") and path.endswith("/red-team"):
            # Extract debate_id from path
            parts = path.split("/")
            if len(parts) >= 4:
                debate_id = parts[3]
                if not re.match(SAFE_ID_PATTERN, debate_id):
                    return error_response("Invalid debate_id format", 400)
                return self._run_red_team_analysis(debate_id, handler)

        return None

    def _run_capability_probe(self, handler) -> HandlerResult:
        """Run capability probes on an agent to find vulnerabilities.

        POST body:
            agent_name: Name of agent to probe (required)
            probe_types: List of probe types (optional)
            probes_per_type: Number of probes per type (default: 3, max: 10)
            model_type: Agent model type (optional, default: anthropic-api)
        """
        if not PROBER_AVAILABLE:
            return error_response("Capability prober not available", 503)

        if not DEBATE_AVAILABLE or create_agent is None:
            return error_response("Agent system not available", 503)

        try:
            # Read request body from handler
            data = self._read_json_body(handler)
            if data is None:
                return error_response("Invalid JSON body", 400)

            agent_name = data.get('agent_name', '').strip()
            if not agent_name:
                return error_response("Missing required field: agent_name", 400)

            if not re.match(SAFE_ID_PATTERN, agent_name):
                return error_response("Invalid agent_name format", 400)

            probe_type_strs = data.get('probe_types', [
                'contradiction', 'hallucination', 'sycophancy', 'persistence'
            ])
            try:
                probes_per_type = min(int(data.get('probes_per_type', 3)), 10)
            except (ValueError, TypeError):
                return error_response("probes_per_type must be an integer", 400)
            model_type = data.get('model_type', 'anthropic-api')

            from aragora.modes.prober import ProbeType, CapabilityProber

            # Convert string probe types to enum
            probe_types = []
            for pt_str in probe_type_strs:
                try:
                    probe_types.append(ProbeType(pt_str))
                except ValueError:
                    pass

            if not probe_types:
                return error_response("No valid probe types specified", 400)

            # Create agent for probing
            try:
                agent = create_agent(model_type, name=agent_name, role="proposer")
            except Exception as e:
                return error_response(f"Failed to create agent: {str(e)}", 400)

            # Create prober
            elo_system = self.ctx.get("elo_system")
            prober = CapabilityProber(
                elo_system=elo_system,
                elo_penalty_multiplier=5.0
            )

            report_id = f"probe-report-{uuid.uuid4().hex[:8]}"

            # Define run_agent_fn callback for prober
            async def run_agent_fn(target_agent, prompt: str) -> str:
                try:
                    if asyncio.iscoroutinefunction(target_agent.generate):
                        return await target_agent.generate(prompt)
                    else:
                        return target_agent.generate(prompt)
                except Exception as e:
                    return f"[Agent Error: {str(e)}]"

            # Run probes asynchronously
            async def run_probes():
                return await prober.probe_agent(
                    target_agent=agent,
                    run_agent_fn=run_agent_fn,
                    probe_types=probe_types,
                    probes_per_type=probes_per_type,
                )

            # Execute in event loop
            # Use asyncio.run() for proper event loop lifecycle management
            try:
                report = asyncio.run(run_probes())
            except Exception as e:
                return error_response(f"Probe execution failed: {str(e)}", 500)

            # Transform results
            by_type_transformed = {}
            for probe_type_key, results in report.by_type.items():
                transformed_results = []
                for r in results:
                    result_dict = r.to_dict() if hasattr(r, 'to_dict') else r
                    passed = not result_dict.get('vulnerability_found', False)
                    transformed_results.append({
                        "probe_id": result_dict.get('probe_id', ''),
                        "type": result_dict.get('probe_type', probe_type_key),
                        "passed": passed,
                        "severity": str(result_dict.get('severity', '')).lower() if result_dict.get('severity') else None,
                        "description": result_dict.get('vulnerability_description', ''),
                        "details": result_dict.get('evidence', ''),
                        "response_time_ms": result_dict.get('response_time_ms', 0),
                    })
                by_type_transformed[probe_type_key] = transformed_results

            # Record in ELO system
            if elo_system and report.probes_run > 0:
                robustness_score = 1.0 - report.vulnerability_rate
                try:
                    elo_system.record_redteam_result(
                        agent_name=agent_name,
                        robustness_score=robustness_score,
                        successful_attacks=report.vulnerabilities_found,
                        total_attacks=report.probes_run,
                        critical_vulnerabilities=report.critical_count,
                        session_id=report_id
                    )
                except Exception as e:
                    logger.warning(f"Failed to record ELO result for capability probe: {e}")

            # Save results
            nomic_dir = self.ctx.get("nomic_dir")
            if nomic_dir:
                try:
                    probes_dir = nomic_dir / "probes" / agent_name
                    probes_dir.mkdir(parents=True, exist_ok=True)
                    date_str = datetime.now().strftime("%Y-%m-%d")
                    probe_file = probes_dir / f"{date_str}_{report.report_id}.json"
                    probe_file.write_text(json.dumps(report.to_dict(), indent=2, default=str))
                except Exception as e:
                    logger.error(f"Failed to save probe report to {nomic_dir}: {e}")

            # Calculate summary
            passed_count = report.probes_run - report.vulnerabilities_found
            pass_rate = passed_count / report.probes_run if report.probes_run > 0 else 1.0

            return json_response({
                "report_id": report.report_id,
                "target_agent": agent_name,
                "probes_run": report.probes_run,
                "vulnerabilities_found": report.vulnerabilities_found,
                "vulnerability_rate": round(report.vulnerability_rate, 3),
                "elo_penalty": round(report.elo_penalty, 1),
                "by_type": by_type_transformed,
                "summary": {
                    "total": report.probes_run,
                    "passed": passed_count,
                    "failed": report.vulnerabilities_found,
                    "pass_rate": round(pass_rate, 3),
                    "critical": report.critical_count,
                    "high": report.high_count,
                    "medium": report.medium_count,
                    "low": report.low_count,
                },
                "recommendations": report.recommendations,
                "created_at": report.created_at,
            })

        except Exception as e:
            return error_response(_safe_error_message(e, "capability_probe"), 500)

    def _run_deep_audit(self, handler) -> HandlerResult:
        """Run a deep audit (Heavy3-inspired intensive multi-round debate protocol).

        POST body:
            task: The question/decision to audit (required)
            context: Additional context/documents (optional)
            agent_names: List of agent names (optional)
            model_type: Agent model type (optional, default: anthropic-api)
            config: Optional configuration object
        """
        try:
            from aragora.modes.deep_audit import (
                DeepAuditOrchestrator,
                DeepAuditConfig,
                STRATEGY_AUDIT,
                CONTRACT_AUDIT,
                CODE_ARCHITECTURE_AUDIT,
            )
        except ImportError:
            return error_response("Deep audit module not available", 503)

        if not DEBATE_AVAILABLE or create_agent is None:
            return error_response("Agent system not available", 503)

        try:
            data = self._read_json_body(handler)
            if data is None:
                return error_response("Invalid JSON body", 400)

            task = data.get('task', '').strip()
            if not task:
                return error_response("Missing required field: task", 400)

            context = data.get('context', '')
            agent_names = data.get('agent_names', [])
            model_type = data.get('model_type', 'anthropic-api')
            config_data = data.get('config', {})

            # Use pre-configured audit type if specified
            audit_type = config_data.get('audit_type', '')
            if audit_type == 'strategy':
                config = STRATEGY_AUDIT
            elif audit_type == 'contract':
                config = CONTRACT_AUDIT
            elif audit_type == 'code_architecture':
                config = CODE_ARCHITECTURE_AUDIT
            else:
                try:
                    rounds = min(int(config_data.get('rounds', 6)), 10)
                    cross_examination_depth = min(int(config_data.get('cross_examination_depth', 3)), 10)
                    risk_threshold = float(config_data.get('risk_threshold', 0.7))
                except (ValueError, TypeError) as e:
                    return error_response(f"Invalid config parameter: {e}", 400)

                config = DeepAuditConfig(
                    rounds=rounds,
                    enable_research=config_data.get('enable_research', True),
                    cross_examination_depth=cross_examination_depth,
                    risk_threshold=risk_threshold,
                )

            # Create agents
            if not agent_names:
                agent_names = ['Claude-Analyst', 'Claude-Skeptic', 'Claude-Synthesizer']

            agents = []
            for name in agent_names[:5]:
                if not re.match(SAFE_ID_PATTERN, name):
                    continue
                try:
                    agent = create_agent(model_type, name=name, role="proposer")
                    agents.append(agent)
                except Exception as e:
                    logger.debug(f"Failed to create audit agent {name}: {e}")

            if len(agents) < 2:
                return error_response("Need at least 2 agents for deep audit", 400)

            audit_id = f"audit-{uuid.uuid4().hex[:8]}"
            start_time = time.time()

            # Run audit
            orchestrator = DeepAuditOrchestrator(agents, config)

            async def run_audit():
                return await orchestrator.run(task, context)

            # Use asyncio.run() for proper event loop lifecycle management
            try:
                verdict = asyncio.run(run_audit())
            except Exception as e:
                return error_response(f"Deep audit execution failed: {str(e)}", 500)

            end_time = time.time()
            duration_ms = (end_time - start_time) * 1000

            # Calculate ELO adjustments
            elo_adjustments = {}
            elo_system = self.ctx.get("elo_system")
            if elo_system:
                for finding in verdict.findings:
                    for agent_name in finding.agents_agree:
                        elo_adjustments[agent_name] = elo_adjustments.get(agent_name, 0) + 2
                    for agent_name in finding.agents_disagree:
                        elo_adjustments[agent_name] = elo_adjustments.get(agent_name, 0) - 1

            # Save results
            nomic_dir = self.ctx.get("nomic_dir")
            if nomic_dir:
                try:
                    audits_dir = nomic_dir / "audits"
                    audits_dir.mkdir(parents=True, exist_ok=True)
                    date_str = datetime.now().strftime("%Y-%m-%d")
                    audit_file = audits_dir / f"{date_str}_{audit_id}.json"
                    audit_file.write_text(json.dumps({
                        "audit_id": audit_id,
                        "task": task,
                        "context": context[:1000],
                        "agents": [a.name for a in agents],
                        "recommendation": verdict.recommendation,
                        "confidence": verdict.confidence,
                        "unanimous_issues": verdict.unanimous_issues,
                        "split_opinions": verdict.split_opinions,
                        "risk_areas": verdict.risk_areas,
                        "findings": [
                            {
                                "category": f.category,
                                "summary": f.summary,
                                "details": f.details,
                                "agents_agree": f.agents_agree,
                                "agents_disagree": f.agents_disagree,
                                "confidence": f.confidence,
                                "severity": f.severity,
                                "citations": f.citations,
                            }
                            for f in verdict.findings
                        ],
                        "config": {
                            "rounds": config.rounds,
                            "enable_research": config.enable_research,
                            "cross_examination_depth": config.cross_examination_depth,
                            "risk_threshold": config.risk_threshold,
                        },
                        "duration_ms": duration_ms,
                        "elo_adjustments": elo_adjustments,
                        "created_at": datetime.now().isoformat(),
                    }, indent=2, default=str))
                except Exception as e:
                    logger.error(f"Failed to save deep audit report to {nomic_dir}: {e}")

            return json_response({
                "audit_id": audit_id,
                "task": task,
                "recommendation": verdict.recommendation,
                "confidence": verdict.confidence,
                "unanimous_issues": verdict.unanimous_issues,
                "split_opinions": verdict.split_opinions,
                "risk_areas": verdict.risk_areas,
                "findings": [
                    {
                        "category": f.category,
                        "summary": f.summary,
                        "details": f.details[:500],
                        "agents_agree": f.agents_agree,
                        "agents_disagree": f.agents_disagree,
                        "confidence": f.confidence,
                        "severity": f.severity,
                    }
                    for f in verdict.findings
                ],
                "cross_examination_notes": verdict.cross_examination_notes[:2000],
                "citations": verdict.citations[:20],
                "rounds_completed": config.rounds,
                "duration_ms": round(duration_ms, 1),
                "agents": [a.name for a in agents],
                "elo_adjustments": elo_adjustments,
                "summary": {
                    "unanimous_count": len(verdict.unanimous_issues),
                    "split_count": len(verdict.split_opinions),
                    "risk_count": len(verdict.risk_areas),
                    "findings_count": len(verdict.findings),
                    "high_severity_count": sum(1 for f in verdict.findings if f.severity >= 0.7),
                }
            })

        except Exception as e:
            return error_response(_safe_error_message(e, "deep_audit"), 500)

    def _analyze_proposal_for_redteam(
        self, proposal: str, attack_types: list, debate_data: dict
    ) -> list:
        """Analyze a proposal for potential vulnerabilities."""
        try:
            from aragora.modes.redteam import AttackType
        except ImportError:
            return []

        findings = []
        proposal_lower = proposal.lower() if proposal else ""

        vulnerability_patterns = {
            'logical_fallacy': {
                'keywords': ['always', 'never', 'all', 'none', 'obviously', 'clearly'],
                'description': 'Absolute language suggests potential logical fallacy',
                'base_severity': 0.4,
            },
            'edge_case': {
                'keywords': ['usually', 'most', 'typical', 'normal', 'standard'],
                'description': 'Generalization may miss edge cases',
                'base_severity': 0.5,
            },
            'unstated_assumption': {
                'keywords': ['should', 'must', 'need', 'require'],
                'description': 'Prescriptive language may hide unstated assumptions',
                'base_severity': 0.45,
            },
            'counterexample': {
                'keywords': ['best', 'optimal', 'superior', 'only'],
                'description': 'Strong claims may be vulnerable to counterexamples',
                'base_severity': 0.55,
            },
            'scalability': {
                'keywords': ['scale', 'growth', 'expand', 'distributed'],
                'description': 'Scalability claims require validation',
                'base_severity': 0.5,
            },
            'security': {
                'keywords': ['secure', 'safe', 'protected', 'auth', 'encrypt'],
                'description': 'Security claims need rigorous testing',
                'base_severity': 0.6,
            },
        }

        for attack_type in attack_types:
            try:
                AttackType(attack_type)
            except ValueError:
                continue

            pattern = vulnerability_patterns.get(attack_type, {})
            keywords = pattern.get('keywords', [])
            base_severity = pattern.get('base_severity', 0.5)

            matches = sum(1 for kw in keywords if kw in proposal_lower)
            severity = min(0.9, base_severity + (matches * 0.1))

            if matches > 0:
                findings.append({
                    "attack_type": attack_type,
                    "description": pattern.get('description', f"Potential {attack_type} issue"),
                    "severity": round(severity, 2),
                    "exploitability": round(severity * 0.8, 2),
                    "keyword_matches": matches,
                    "requires_manual_review": severity > 0.6,
                })
            else:
                findings.append({
                    "attack_type": attack_type,
                    "description": f"No obvious {attack_type.replace('_', ' ')} patterns detected",
                    "severity": round(base_severity * 0.5, 2),
                    "exploitability": round(base_severity * 0.3, 2),
                    "keyword_matches": 0,
                    "requires_manual_review": False,
                })

        return findings

    def _run_red_team_analysis(self, debate_id: str, handler) -> HandlerResult:
        """Run adversarial red-team analysis on a debate.

        POST body:
            attack_types: List of attack types (optional)
            max_rounds: Maximum attack/defend rounds (default: 3, max: 5)
            focus_proposal: Optional specific proposal to analyze
        """
        if not REDTEAM_AVAILABLE:
            return error_response("Red team mode not available", 503)

        try:
            data = self._read_json_body(handler)
            if data is None:
                data = {}

            storage = self.ctx.get("storage")
            if not storage:
                return error_response("Storage not configured", 500)

            debate_data = storage.get_by_slug(debate_id) or storage.get_by_id(debate_id)
            if not debate_data:
                return error_response("Debate not found", 404)

            from aragora.modes.redteam import AttackType

            attack_type_names = data.get('attack_types', [
                'logical_fallacy', 'edge_case', 'unstated_assumption',
                'counterexample', 'scalability', 'security'
            ])
            max_rounds = min(int(data.get('max_rounds', 3)), 5)

            focus_proposal = data.get('focus_proposal') or (
                debate_data.get('consensus_answer') or
                debate_data.get('final_answer') or
                debate_data.get('task', '')
            )

            session_id = f"redteam-{datetime.now().strftime('%Y%m%d%H%M%S')}-{uuid.uuid4().hex[:6]}"

            # Analyze proposal for potential weaknesses
            findings = self._analyze_proposal_for_redteam(
                focus_proposal, attack_type_names, debate_data
            )

            # Calculate robustness based on finding severity
            avg_severity = sum(f.get('severity', 0.5) for f in findings) / max(len(findings), 1)
            robustness_score = max(0.0, 1.0 - avg_severity)

            return json_response({
                "session_id": session_id,
                "debate_id": debate_id,
                "target_proposal": focus_proposal[:500] if focus_proposal else "",
                "attack_types": attack_type_names,
                "max_rounds": max_rounds,
                "findings": findings,
                "robustness_score": round(robustness_score, 2),
                "status": "analysis_complete",
                "created_at": datetime.now().isoformat(),
            })

        except Exception as e:
            return error_response(_safe_error_message(e, "red_team_analysis"), 500)

    # _read_json_body moved to BaseHandler.read_json_body
    def _read_json_body(self, handler) -> Optional[dict]:
        """Read and parse JSON body - delegates to base class."""
        return self.read_json_body(handler)
