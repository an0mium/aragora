"""MCP Self-Improvement Tools.

Provides tools for autonomous codebase assessment and self-improvement:
- assess_codebase: Run autonomous assessment, return health report
- generate_improvement_goals: Convert assessment to prioritized goals
- run_self_improvement: Execute a self-improvement cycle
- get_daemon_status: Query daemon state
- trigger_improvement_cycle: Trigger immediate daemon cycle
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Module-level daemon instance (singleton for status queries)
_daemon_instance: Any = None


def _get_daemon() -> Any:
    """Get or create the singleton daemon instance."""
    global _daemon_instance
    if _daemon_instance is None:
        from aragora.nomic.daemon import SelfImprovementDaemon
        _daemon_instance = SelfImprovementDaemon()
    return _daemon_instance


async def assess_codebase_tool(
    weights: str = "",
) -> dict[str, Any]:
    """Run autonomous codebase assessment and return a health report.

    Args:
        weights: Optional JSON string of source weights
            (e.g. '{"scanner": 0.4, "metrics": 0.3}')

    Returns:
        Dict with health_score, signal_sources, improvement_candidates
    """
    try:
        from aragora.nomic.assessment_engine import AutonomousAssessmentEngine

        weight_dict = None
        if weights:
            import json
            try:
                weight_dict = json.loads(weights)
            except (json.JSONDecodeError, TypeError):
                return {"error": "Invalid weights JSON"}

        engine = AutonomousAssessmentEngine(weights=weight_dict)
        report = await engine.assess()

        return report.to_dict()

    except ImportError:
        logger.warning("Assessment engine not available")
        return {"error": "Assessment engine module not available"}
    except (RuntimeError, ValueError, OSError) as e:
        logger.warning("Assessment failed: %s", e)
        return {"error": "Assessment failed"}


async def generate_improvement_goals_tool(
    max_goals: int = 5,
) -> dict[str, Any]:
    """Run assessment and convert results to prioritized improvement goals.

    Args:
        max_goals: Maximum number of goals to generate (default 5)

    Returns:
        Dict with goals list and assessment summary
    """
    try:
        from aragora.nomic.assessment_engine import AutonomousAssessmentEngine
        from aragora.nomic.goal_generator import GoalGenerator

        engine = AutonomousAssessmentEngine()
        report = await engine.assess()

        generator = GoalGenerator(max_goals=max_goals)
        goals = generator.generate_goals(report)

        return {
            "health_score": report.health_score,
            "goals": [
                {
                    "id": getattr(g, "id", ""),
                    "description": getattr(g, "description", str(g)),
                    "track": getattr(g, "track", None) and g.track.value or "core",
                    "estimated_impact": getattr(g, "estimated_impact", "medium"),
                    "priority": getattr(g, "priority", 0),
                    "file_hints": getattr(g, "file_hints", []),
                }
                for g in goals
            ],
            "goals_count": len(goals),
            "candidates_count": len(report.improvement_candidates),
        }

    except ImportError:
        logger.warning("Assessment/goal modules not available")
        return {"error": "Assessment or goal generator modules not available"}
    except (RuntimeError, ValueError, OSError) as e:
        logger.warning("Goal generation failed: %s", e)
        return {"error": "Goal generation failed"}


async def run_self_improvement_tool(
    objective: str = "",
    dry_run: bool = True,
) -> dict[str, Any]:
    """Execute a self-improvement cycle.

    Args:
        objective: High-level objective. Empty for self-directing mode.
        dry_run: If True (default), preview without executing changes.

    Returns:
        Dict with cycle results or dry-run preview
    """
    try:
        from aragora.nomic.self_improve import SelfImproveConfig, SelfImprovePipeline

        config = SelfImproveConfig(
            scan_mode=not bool(objective),
            autonomous=not dry_run,
            require_approval=dry_run,
        )
        pipeline = SelfImprovePipeline(config)

        effective_objective = objective or None

        if dry_run:
            preview = await pipeline.dry_run(effective_objective)
            return {
                "mode": "dry_run",
                "preview": preview,
            }

        result = await pipeline.run(effective_objective)
        return {
            "mode": "execute",
            "result": result.to_dict(),
        }

    except ImportError:
        logger.warning("Self-improvement pipeline not available")
        return {"error": "Self-improvement pipeline not available"}
    except (RuntimeError, ValueError, OSError) as e:
        logger.warning("Self-improvement failed: %s", e)
        return {"error": "Self-improvement execution failed"}


async def get_daemon_status_tool() -> dict[str, Any]:
    """Get the current self-improvement daemon status.

    Returns:
        Dict with daemon state, cycle counts, and recent history
    """
    try:
        daemon = _get_daemon()
        return daemon.get_status().to_dict()
    except ImportError:
        logger.warning("Daemon module not available")
        return {"error": "Daemon module not available"}
    except (RuntimeError, ValueError) as e:
        logger.warning("Daemon status check failed: %s", e)
        return {"error": "Daemon status check failed"}


async def trigger_improvement_cycle_tool(
    dry_run: bool = True,
) -> dict[str, Any]:
    """Trigger an immediate self-improvement cycle via the daemon.

    Args:
        dry_run: If True (default), assess only without executing changes.

    Returns:
        Dict with cycle result
    """
    try:
        from aragora.nomic.daemon import DaemonConfig, SelfImprovementDaemon

        config = DaemonConfig(dry_run=dry_run)
        daemon = SelfImprovementDaemon(config)

        result = await daemon.trigger_cycle()
        return result.to_dict()

    except ImportError:
        logger.warning("Daemon module not available")
        return {"error": "Daemon module not available"}
    except (RuntimeError, ValueError, OSError) as e:
        logger.warning("Daemon cycle trigger failed: %s", e)
        return {"error": "Daemon cycle trigger failed"}


__all__ = [
    "assess_codebase_tool",
    "generate_improvement_goals_tool",
    "run_self_improvement_tool",
    "get_daemon_status_tool",
    "trigger_improvement_cycle_tool",
]
