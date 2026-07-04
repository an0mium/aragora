"""
Security Debate Runner.

Runs multi-agent debates for remediation recommendations on critical
security findings. Relocated from aragora.events.security_events (P4a E7a)
because it is the domain-coupled half of that module: it imports
aragora.debate.orchestrator.Arena, aragora.debate.protocol, and
aragora.agents.api_agents.{anthropic,openai}.

Self-registers trigger_security_debate as the callback that
aragora.events.security_events.SecurityEventEmitter invokes for its
auto-debate path (see register_security_debate_runner), so the domain-free
events module never imports aragora.debate or aragora.agents directly.
Registration is a plain module-level side effect, so any composition root
that imports this module first makes the runner available process-wide.
Two composition roots import it explicitly rather than relying solely on
incidental transitive imports: aragora.debate.orchestrator (Arena, for any
Arena-backed process) and aragora.analysis.codebase.sast.scanner (SAST
auto-scan findings, which can run in isolation from the Arena-backed server
stack).
"""

from __future__ import annotations

import json
import logging
import uuid
from typing import Any

from aragora.events.security_events import (
    SecurityEvent,
    _register_default_security_debate_runner,
    _store_security_debate_result,
)

logger = logging.getLogger(__name__)


def build_security_debate_question(event: SecurityEvent) -> str:
    """
    Build a debate question from security findings.

    Args:
        event: Security event with findings

    Returns:
        Formatted debate question
    """
    findings = event.findings[:5]  # Limit to top 5 findings

    if not findings:
        return f"Analyze and recommend remediation for security findings in {event.repository or 'the codebase'}."

    # Group by type
    vulns = [f for f in findings if f.finding_type == "vulnerability"]
    secrets = [f for f in findings if f.finding_type == "secret"]

    question_parts = []

    if vulns:
        vuln_summary = ", ".join(f"{v.cve_id or v.title} in {v.package_name}" for v in vulns[:3])
        question_parts.append(f"vulnerabilities ({vuln_summary})")

    if secrets:
        secret_types = set(s.metadata.get("secret_type", "unknown") for s in secrets)
        question_parts.append(f"exposed secrets ({', '.join(secret_types)})")

    findings_str = " and ".join(question_parts)

    return (
        f"Analyze the following critical security findings and provide remediation recommendations:\n\n"
        f"Repository: {event.repository or 'Unknown'}\n"
        f"Findings: {findings_str}\n\n"
        f"Details:\n"
        + "\n".join(
            f"- {f.severity.value.upper()}: {f.title} - {f.description[:200]}" for f in findings
        )
        + "\n\n"
        "What is the recommended prioritized remediation plan, considering:\n"
        "1. Immediate mitigations (quick wins)\n"
        "2. Root cause fixes\n"
        "3. Preventive measures for future\n"
        "4. Impact on existing functionality"
    )


async def trigger_security_debate(
    event: SecurityEvent,
    confidence_threshold: float = 0.7,
    agents: list[Any] | None = None,
    timeout_seconds: int = 300,
) -> str | None:
    """
    Trigger a multi-agent debate for security remediation.

    Args:
        event: Security event with findings
        confidence_threshold: Minimum consensus confidence
        agents: Optional list of agents (uses defaults if None)
        timeout_seconds: Maximum debate duration

    Returns:
        Debate ID if triggered, None if failed
    """
    try:
        from aragora.core import Environment, DebateResult
        from aragora.debate.protocol import DebateProtocol
        from aragora.debate.orchestrator import Arena

        # Build debate question
        question = build_security_debate_question(event)
        event.debate_question = question

        # Create environment
        env = Environment(
            task=question,
            context=json.dumps(
                {
                    "security_event_id": event.id,
                    "repository": event.repository,
                    "scan_id": event.scan_id,
                    "findings": [f.to_dict() for f in event.findings],
                }
            ),
        )

        # Create protocol for security debates
        protocol = DebateProtocol(
            rounds=3,
            consensus="majority",
            convergence_detection=True,
            convergence_threshold=0.85,
            timeout_seconds=timeout_seconds,
        )

        # Get default agents if none provided
        if agents is None:
            agents = await _get_security_debate_agents()

        if not agents:
            logger.warning("No agents available for security debate")
            return None

        # Run debate
        arena = Arena(
            environment=env,
            agents=agents,
            protocol=protocol,
            org_id=event.workspace_id or "default",
        )

        logger.info("[Security] Starting debate for %s findings", len(event.findings))

        result: DebateResult = await arena.run()
        debate_id = getattr(result, "debate_id", "") or f"security_debate_{uuid.uuid4().hex[:12]}"
        event.debate_requested = True
        event.debate_id = debate_id

        logger.info(
            f"[Security] Debate {debate_id} completed: "
            f"consensus={result.consensus_reached}, confidence={result.confidence:.2f}"
        )

        # Store result for later retrieval
        await _store_security_debate_result(debate_id, event, result)

        return debate_id

    except ImportError as e:
        logger.warning("Arena not available for security debate: %s", e)
        return None
    except (RuntimeError, ValueError, TypeError, OSError) as e:
        logger.exception("Failed to run security debate: %s", e)
        return None


async def _get_security_debate_agents() -> list[Any]:
    """Get agents suitable for security debates."""
    agents = await _get_agent_factory_security_agents()
    if agents:
        return agents

    try:
        from aragora.agents.api_agents.anthropic import AnthropicAPIAgent as AnthropicAgent
        from aragora.agents.api_agents.openai import OpenAIAPIAgent as OpenAIAgent

        agents = []

        try:
            agents.append(
                AnthropicAgent(
                    name="claude-security",
                    model="claude-opus-4-8",
                )
            )
        except (ValueError, RuntimeError) as e:
            logger.debug("Could not create Anthropic security agent: %s", e)

        try:
            agents.append(
                OpenAIAgent(
                    name="gpt-security",
                    model="gpt-4o",
                )
            )
        except (ValueError, RuntimeError) as e:
            logger.debug("Could not create OpenAI security agent: %s", e)

        return agents
    except ImportError:
        logger.debug("Could not import agent modules for security debate")
        return []


async def _get_agent_factory_security_agents() -> list[Any]:
    """Use a deployment-provided agent factory when available."""
    try:
        from aragora.agents.factory import get_available_agents
    except ImportError:
        logger.debug("Security agent factory pool not available")
        return []

    try:
        agents = await get_available_agents(
            capabilities=["security", "code_analysis"],
            min_count=2,
            max_count=4,
        )
    except (RuntimeError, ValueError, TypeError, OSError) as exc:
        logger.debug("Security agent factory pool failed: %s", exc)
        return []

    return list(agents or [])


_register_default_security_debate_runner(trigger_security_debate)


__all__ = [
    "trigger_security_debate",
    "build_security_debate_question",
]
