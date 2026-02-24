"""Harness step type for workflow engine.

Dispatches code analysis and implementation tasks to external harnesses
(Claude Code, Codex) through the CodeAnalysisHarness interface.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

from aragora.workflow.step import BaseStep, WorkflowContext

logger = logging.getLogger(__name__)

# Map harness type names to module paths and class names
_HARNESS_REGISTRY: dict[str, tuple[str, str]] = {
    "claude-code": ("aragora.harnesses.claude_code", "ClaudeCodeHarness"),
    "codex": ("aragora.harnesses.codex", "CodexHarness"),
}


class HarnessStep(BaseStep):
    """Step that delegates work to an external code analysis harness.

    Config keys:
        harness_type: str - Harness to use ("claude-code", "codex")
        analysis_type: str - Type of analysis (security, quality, architecture, etc.)
        repo_path: str - Repository path (defaults to cwd)
        files: list[str] - Specific files to analyze (optional)
        prompt: str - Custom analysis prompt (optional)
        adapt_to_audit: bool - Convert findings to AuditFinding format (default False)
        implementation_mode: bool - Use execute_implementation instead of analyze (default False)
    """

    def __init__(self, name: str, config: dict[str, Any] | None = None):
        super().__init__(name, config)

    async def execute(self, context: WorkflowContext) -> dict[str, Any]:
        """Execute analysis or implementation via harness."""
        config = {**self._config, **context.current_step_config}
        start_time = time.time()

        harness_type = config.get("harness_type", "claude-code")
        analysis_type_str = config.get("analysis_type", "general")
        repo_path = Path(config.get("repo_path", context.get_input("repo_path", ".")))
        files = config.get("files", [])
        prompt = config.get("prompt", context.get_input("task", ""))
        adapt_to_audit = config.get("adapt_to_audit", False)
        implementation_mode = config.get("implementation_mode", False)

        # Resolve harness class
        harness = await self._create_harness(harness_type, config)
        if harness is None:
            return {
                "harness": harness_type,
                "success": False,
                "error": f"Harness '{harness_type}' not available",
                "findings": [],
                "duration_seconds": time.time() - start_time,
            }

        # Initialize harness
        try:
            initialized = await harness.initialize()
            if not initialized:
                return {
                    "harness": harness_type,
                    "success": False,
                    "error": f"Harness '{harness_type}' failed to initialize",
                    "findings": [],
                    "duration_seconds": time.time() - start_time,
                }
        except (RuntimeError, OSError, ValueError, TypeError) as e:
            logger.debug("Harness initialization failed: %s", e)
            return {
                "harness": harness_type,
                "success": False,
                "error": f"Initialization failed: {type(e).__name__}",
                "findings": [],
                "duration_seconds": time.time() - start_time,
            }

        try:
            # Implementation mode: execute code changes
            if implementation_mode:
                stdout, stderr = await harness.execute_implementation(
                    repo_path=repo_path,
                    prompt=prompt,
                )
                elapsed = time.time() - start_time
                return {
                    "harness": harness_type,
                    "success": not stderr or "error" not in stderr.lower(),
                    "stdout": stdout,
                    "stderr": stderr,
                    "findings": [],
                    "duration_seconds": elapsed,
                }

            # Analysis mode: analyze repo or specific files
            analysis_type = self._resolve_analysis_type(analysis_type_str)

            if files:
                result = await harness.analyze_files(
                    files=[Path(f) for f in files],
                    analysis_type=analysis_type,
                    prompt=prompt,
                )
            else:
                result = await harness.analyze_repository(
                    repo_path=repo_path,
                    analysis_type=analysis_type,
                    prompt=prompt,
                )

            elapsed = time.time() - start_time
            output: dict[str, Any] = {
                "harness": harness_type,
                "success": result.success,
                "findings": [self._finding_to_dict(f) for f in result.findings],
                "stats": result.stats,
                "duration_seconds": elapsed,
            }

            # Optionally adapt to audit findings
            if adapt_to_audit and result.findings:
                try:
                    from aragora.harnesses.adapter import adapt_to_audit_findings

                    audit_findings = adapt_to_audit_findings(result)
                    output["audit_findings"] = [
                        {
                            "title": af.title,
                            "severity": af.severity,
                            "description": af.description,
                            "file_path": af.file_path,
                        }
                        for af in audit_findings
                    ]
                except (ImportError, AttributeError, TypeError) as e:
                    logger.debug("Audit adaptation failed: %s", e)

            # Emit event if callback available
            context.emit_event(
                "harness_analysis_complete",
                {
                    "harness": harness_type,
                    "findings_count": len(result.findings),
                    "success": result.success,
                },
            )

            return output

        except (RuntimeError, OSError, ValueError, TypeError, TimeoutError) as e:
            logger.debug("Harness execution failed: %s", e)
            return {
                "harness": harness_type,
                "success": False,
                "error": str(e),
                "findings": [],
                "duration_seconds": time.time() - start_time,
            }

    async def _create_harness(self, harness_type: str, config: dict[str, Any]) -> Any | None:
        """Create harness instance by type name."""
        registry_entry = _HARNESS_REGISTRY.get(harness_type)
        if registry_entry is None:
            logger.debug("Unknown harness type: %s", harness_type)
            return None

        module_path, class_name = registry_entry
        try:
            import importlib

            mod = importlib.import_module(module_path)
            harness_cls = getattr(mod, class_name)
            # Pass config if harness accepts it
            harness_config = config.get("harness_config")
            if harness_config is not None:
                return harness_cls(config=harness_config)
            return harness_cls()
        except (ImportError, AttributeError, TypeError, RuntimeError) as e:
            logger.debug("Failed to create harness '%s': %s", harness_type, e)
            return None

    @staticmethod
    def _resolve_analysis_type(type_str: str) -> Any:
        """Resolve analysis type string to AnalysisType enum."""
        try:
            from aragora.harnesses.base import AnalysisType

            upper = type_str.upper()
            if hasattr(AnalysisType, upper):
                return AnalysisType[upper]
            return AnalysisType.GENERAL
        except (ImportError, ValueError):
            return type_str

    @staticmethod
    def _finding_to_dict(finding: Any) -> dict[str, Any]:
        """Convert AnalysisFinding to dict."""
        return {
            "id": getattr(finding, "id", ""),
            "title": getattr(finding, "title", ""),
            "severity": getattr(finding, "severity", "info"),
            "description": getattr(finding, "description", ""),
            "file_path": getattr(finding, "file_path", ""),
            "line_start": getattr(finding, "line_start", None),
            "line_end": getattr(finding, "line_end", None),
            "recommendation": getattr(finding, "recommendation", ""),
        }
