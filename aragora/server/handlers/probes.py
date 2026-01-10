"""
Capability probing endpoint handlers.

Endpoints:
- POST /api/probes/capability - Run capability probes on an agent to find vulnerabilities
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

from aragora.utils.optional_imports import try_import
from aragora.server.validation import validate_agent_name
from aragora.server.http_utils import run_async
from aragora.debate.sanitization import OutputSanitizer
from .base import (
    BaseHandler,
    HandlerResult,
    json_response,
    error_response,
    handle_errors,
    invalidate_leaderboard_cache,
)
from aragora.server.middleware.rate_limit import rate_limit

logger = logging.getLogger(__name__)

# Optional imports for prober
_prober_imports, PROBER_AVAILABLE = try_import(
    "aragora.modes.prober",
    "ProbeType", "CapabilityProber"
)
ProbeType = _prober_imports.get("ProbeType")
CapabilityProber = _prober_imports.get("CapabilityProber")

# Optional imports for agent creation
_agent_imports, AGENT_AVAILABLE = try_import(
    "aragora.agents.base",
    "create_agent"
)
create_agent = _agent_imports.get("create_agent")


def _safe_int(value, default: int = 0) -> int:
    """Safely convert value to int, returning default on failure."""
    try:
        return int(value)
    except (ValueError, TypeError):
        return default


class ProbesHandler(BaseHandler):
    """Handler for capability probing endpoints."""

    ROUTES = [
        "/api/probes/capability",
        "/api/probes/run",  # Legacy route
    ]

    def can_handle(self, path: str) -> bool:
        """Check if this handler can process the given path."""
        return path in self.ROUTES

    def handle(self, path: str, query_params: dict, handler=None) -> Optional[HandlerResult]:
        """Route GET requests - none for this handler."""
        return None

    def handle_post(self, path: str, query_params: dict, handler) -> Optional[HandlerResult]:
        """Route POST requests to appropriate methods."""
        if path in ("/api/probes/capability", "/api/probes/run"):
            return self._run_capability_probe(handler)
        return None

    @rate_limit(requests_per_minute=10, burst=3, limiter_name="capability_probe")
    @handle_errors("capability probe")
    def _run_capability_probe(self, handler) -> HandlerResult:
        """Run capability probes on an agent to find vulnerabilities.

        POST body:
            agent_name: Name of agent to probe (required)
            probe_types: List of probe types (optional, default: all)
                Options: contradiction, hallucination, sycophancy, persistence,
                         confidence_calibration, reasoning_depth, edge_case
            probes_per_type: Number of probes per type (default: 3, max: 10)
            model_type: Agent model type (optional, default: anthropic-api)

        Returns:
            report_id: Unique report ID
            target_agent: Name of probed agent
            probes_run: Total probes executed
            vulnerabilities_found: Count of vulnerabilities detected
            vulnerability_rate: Fraction of probes that found vulnerabilities
            elo_penalty: ELO penalty applied
            by_type: Results grouped by probe type with passed/failed status
            summary: Quick stats for UI display
        """
        if not PROBER_AVAILABLE:
            return json_response({
                "error": "Capability prober not available",
                "hint": "Prober module failed to import"
            }, status=503)

        if not AGENT_AVAILABLE or create_agent is None:
            return json_response({
                "error": "Agent system not available",
                "hint": "Debate module or create_agent failed to import"
            }, status=503)

        # Read request body
        body = self.read_json_body(handler)
        if body is None:
            return error_response("Invalid JSON body or body too large", status=400)

        agent_name = body.get('agent_name', '').strip()
        if not agent_name:
            return error_response("Missing required field: agent_name", status=400)

        # Validate agent name
        is_valid, err = validate_agent_name(agent_name)
        if not is_valid:
            return error_response(err, status=400)

        probe_type_strs = body.get('probe_types', [
            'contradiction', 'hallucination', 'sycophancy', 'persistence'
        ])
        probes_per_type = min(_safe_int(body.get('probes_per_type', 3), 3), 10)
        model_type = body.get('model_type', 'anthropic-api')

        # Convert string probe types to enum
        probe_types = []
        invalid_types = []
        for pt_str in probe_type_strs:
            try:
                probe_types.append(ProbeType(pt_str))
            except ValueError:
                invalid_types.append(pt_str)

        if not probe_types:
            valid_options = [pt.value for pt in ProbeType] if ProbeType else []
            return error_response(
                f"No valid probe types specified. Invalid: {invalid_types}. "
                f"Valid options: {valid_options}",
                status=400
            )

        # Log warning for any skipped invalid types
        if invalid_types:
            logger.warning("Skipped invalid probe types: %s", invalid_types)

        # Create agent for probing
        try:
            agent = create_agent(model_type, name=agent_name, role="proposer")
        except Exception as e:
            return json_response({
                "error": f"Failed to create agent: {str(e)}",
                "hint": f"model_type '{model_type}' may not be available"
            }, status=400)

        # Create prober with optional ELO integration
        elo_system = self.get_elo_system()
        prober = CapabilityProber(
            elo_system=elo_system,
            elo_penalty_multiplier=5.0
        )

        # Get stream hooks if available for real-time updates
        probe_hooks = self._get_probe_hooks(handler)

        report_id = f"probe-report-{uuid.uuid4().hex[:8]}"

        # Emit probe start event
        if probe_hooks and 'on_probe_start' in probe_hooks:
            probe_hooks['on_probe_start'](
                probe_id=report_id,
                target_agent=agent_name,
                probe_types=[pt.value for pt in probe_types],
                probes_per_type=probes_per_type
            )

        # Define run_agent_fn callback for prober
        async def run_agent_fn(target_agent, prompt: str) -> str:
            """Execute agent with probe prompt."""
            try:
                if asyncio.iscoroutinefunction(target_agent.generate):
                    raw_output = await target_agent.generate(prompt)
                else:
                    raw_output = target_agent.generate(prompt)
                return OutputSanitizer.sanitize_agent_output(raw_output, target_agent.name)
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
        # Use run_async() for safe sync/async bridging
        report = run_async(run_probes())

        # Transform results for frontend (vulnerability_found -> passed)
        by_type_transformed = self._transform_results(report, probe_hooks)

        # Record red team result in ELO system
        self._record_elo_result(elo_system, agent_name, report, report_id)

        # Save results to .nomic/probes/
        self._save_probe_report(agent_name, report)

        # Emit probe complete event
        if probe_hooks and 'on_probe_complete' in probe_hooks:
            probe_hooks['on_probe_complete'](
                report_id=report.report_id,
                target_agent=agent_name,
                probes_run=report.probes_run,
                vulnerabilities_found=report.vulnerabilities_found,
                vulnerability_rate=report.vulnerability_rate,
                elo_penalty=report.elo_penalty,
                by_severity={
                    "critical": report.critical_count,
                    "high": report.high_count,
                    "medium": report.medium_count,
                    "low": report.low_count,
                }
            )

        # Calculate summary for frontend
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

    def _get_probe_hooks(self, handler) -> Optional[dict]:
        """Get stream hooks for real-time probe updates if available."""
        try:
            server = getattr(handler, 'server', None)
            if server and hasattr(server, 'stream_server') and server.stream_server:
                from aragora.server.nomic_stream import create_nomic_hooks
                return create_nomic_hooks(server.stream_server.emitter)
        except ImportError:
            # nomic_stream module not available - this is expected in some deployments
            logger.debug("nomic_stream module not available for probe hooks")
        except Exception as e:
            logger.warning("Failed to get probe hooks: %s: %s", type(e).__name__, e)
        return None

    def _transform_results(self, report, probe_hooks) -> dict:
        """Transform probe results for frontend display."""
        by_type_transformed = {}
        for probe_type_key, results in report.by_type.items():
            transformed_results = []
            for r in results:
                result_dict = r.to_dict() if hasattr(r, 'to_dict') else r
                # Invert vulnerability_found to get passed
                passed = not result_dict.get('vulnerability_found', False)
                transformed_results.append({
                    "probe_id": result_dict.get('probe_id', ''),
                    "type": result_dict.get('probe_type', probe_type_key),
                    "passed": passed,
                    "severity": result_dict.get('severity', '').lower() if result_dict.get('severity') else None,
                    "description": result_dict.get('vulnerability_description', ''),
                    "details": result_dict.get('evidence', ''),
                    "response_time_ms": result_dict.get('response_time_ms', 0),
                })

                # Emit individual probe result event
                if probe_hooks and 'on_probe_result' in probe_hooks:
                    probe_hooks['on_probe_result'](
                        probe_id=result_dict.get('probe_id', ''),
                        probe_type=probe_type_key,
                        passed=passed,
                        severity=result_dict.get('severity', '').lower() if result_dict.get('severity') else None,
                        description=result_dict.get('vulnerability_description', ''),
                        response_time_ms=result_dict.get('response_time_ms', 0)
                    )

            by_type_transformed[probe_type_key] = transformed_results
        return by_type_transformed

    def _record_elo_result(self, elo_system, agent_name: str, report, report_id: str) -> None:
        """Record probe results in ELO system."""
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
                # Invalidate leaderboard cache after ELO update
                invalidate_leaderboard_cache()
            except Exception as e:
                logger.warning(
                    "ELO update failed for agent %s (non-fatal): %s: %s",
                    agent_name, type(e).__name__, e
                )

    def _save_probe_report(self, agent_name: str, report) -> None:
        """Save probe report to nomic directory."""
        nomic_dir = self.get_nomic_dir()
        if nomic_dir:
            try:
                probes_dir = nomic_dir / "probes" / agent_name
                probes_dir.mkdir(parents=True, exist_ok=True)
                date_str = datetime.now().strftime("%Y-%m-%d")
                probe_file = probes_dir / f"{date_str}_{report.report_id}.json"
                probe_file.write_text(json.dumps(report.to_dict(), indent=2, default=str))
            except Exception as e:
                logger.warning(
                    "Probe storage failed for %s (non-fatal): %s: %s",
                    agent_name, type(e).__name__, e
                )
