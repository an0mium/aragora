"""
Analysis operations handler mixin.

Extracted from handler.py for modularity. Provides meta-critique analysis
and argument graph statistics methods.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional, Protocol

from ..base import (
    HandlerResult,
    error_response,
    json_response,
)

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class _DebatesHandlerProtocol(Protocol):
    """Protocol defining the interface expected by AnalysisOperationsMixin.

    This protocol enables proper type checking for mixin classes that
    expect to be mixed into a class providing these methods/attributes.
    """

    ctx: Dict[str, Any]

    def get_storage(self) -> Optional[Any]:
        """Get debate storage instance."""
        ...

    def get_nomic_dir(self) -> Optional[Path]:
        """Get nomic directory path."""
        ...


class AnalysisOperationsMixin:
    """Mixin providing analysis operations for DebatesHandler."""

    def _get_meta_critique(self: _DebatesHandlerProtocol, debate_id: str) -> HandlerResult:
        """Get meta-level analysis of a debate (repetition, circular arguments, etc)."""
        from aragora.exceptions import (
            DatabaseError,
            RecordNotFoundError,
            StorageError,
        )

        try:
            from aragora.debate.meta import MetaCritiqueAnalyzer
            from aragora.debate.traces import DebateTrace
        except ImportError:
            return error_response("Meta critique module not available", 503)

        nomic_dir = self.get_nomic_dir()
        if not nomic_dir:
            return error_response("Nomic directory not configured", 503)

        try:
            trace_path = nomic_dir / "traces" / f"{debate_id}.json"
            if not trace_path.exists():
                return error_response("Debate trace not found", 404)

            trace = DebateTrace.load(trace_path)
            result = trace.to_debate_result()  # type: ignore[attr-defined]

            analyzer = MetaCritiqueAnalyzer()
            critique = analyzer.analyze(result)

            return json_response(
                {
                    "debate_id": debate_id,
                    "overall_quality": critique.overall_quality,
                    "productive_rounds": critique.productive_rounds,
                    "unproductive_rounds": critique.unproductive_rounds,
                    "observations": [
                        {
                            "type": o.observation_type,
                            "severity": o.severity,
                            "agent": getattr(o, "agent", None),
                            "round": getattr(o, "round_num", None),
                            "description": o.description,
                        }
                        for o in critique.observations
                    ],
                    "recommendations": critique.recommendations,
                }
            )
        except RecordNotFoundError:
            logger.info("Meta critique failed - debate not found: %s", debate_id)
            return error_response(f"Debate not found: {debate_id}", 404)
        except (StorageError, DatabaseError) as e:
            logger.error(
                "Failed to get meta critique for %s: %s: %s",
                debate_id,
                type(e).__name__,
                e,
                exc_info=True,
            )
            return error_response("Database error retrieving meta critique", 500)
        except ValueError as e:
            logger.warning("Invalid meta critique request for %s: %s", debate_id, e)
            return error_response(f"Invalid request: {e}", 400)

    def _get_graph_stats(self: _DebatesHandlerProtocol, debate_id: str) -> HandlerResult:
        """Get argument graph statistics for a debate.

        Returns node counts, edge counts, depth, branching factor, and complexity.
        """
        from aragora.exceptions import (
            DatabaseError,
            RecordNotFoundError,
            StorageError,
        )

        try:
            from aragora.debate.traces import DebateTrace
            from aragora.visualization.mapper import ArgumentCartographer
        except ImportError:
            return error_response("Graph analysis module not available", 503)

        nomic_dir = self.get_nomic_dir()
        if not nomic_dir:
            return error_response("Nomic directory not configured", 503)

        try:
            trace_path = nomic_dir / "traces" / f"{debate_id}.json"

            if not trace_path.exists():
                # Try replays directory as fallback
                replay_path = nomic_dir / "replays" / debate_id / "events.jsonl"
                if replay_path.exists():
                    return _build_graph_from_replay(debate_id, replay_path)
                return error_response("Debate not found", 404)

            # Load from trace file
            trace = DebateTrace.load(trace_path)
            result = trace.to_debate_result()  # type: ignore[attr-defined]

            # Build cartographer from debate result
            cartographer = ArgumentCartographer()
            cartographer.set_debate_context(debate_id, result.task or "")

            # Process messages from the debate
            for msg in result.messages:
                cartographer.update_from_message(
                    agent=msg.agent,
                    content=msg.content,
                    role=msg.role,
                    round_num=msg.round,
                )

            # Process critiques
            for critique in result.critiques:
                cartographer.update_from_critique(
                    critic_agent=critique.agent,
                    target_agent=critique.target or "",
                    severity=critique.severity,
                    round_num=getattr(critique, "round", 1),
                    critique_text=critique.reasoning,
                )

            stats = cartographer.get_statistics()
            return json_response(stats)

        except RecordNotFoundError:
            logger.info("Graph stats failed - debate not found: %s", debate_id)
            return error_response(f"Debate not found: {debate_id}", 404)
        except (StorageError, DatabaseError) as e:
            logger.error(
                "Failed to get graph stats for %s: %s: %s",
                debate_id,
                type(e).__name__,
                e,
                exc_info=True,
            )
            return error_response("Database error retrieving graph stats", 500)
        except ValueError as e:
            logger.warning("Invalid graph stats request for %s: %s", debate_id, e)
            return error_response(f"Invalid request: {e}", 400)


def _build_graph_from_replay(debate_id: str, replay_path: Path) -> HandlerResult:
    """Build graph stats from replay events file."""
    import json as json_mod

    from aragora.exceptions import (
        DatabaseError,
        StorageError,
    )

    try:
        from aragora.visualization.mapper import ArgumentCartographer
    except ImportError:
        return error_response("Graph analysis module not available", 503)

    try:
        cartographer = ArgumentCartographer()
        cartographer.set_debate_context(debate_id, "")

        with replay_path.open() as f:
            for line_num, line in enumerate(f, 1):
                if line.strip():
                    try:
                        event = json_mod.loads(line)
                    except json_mod.JSONDecodeError:
                        logger.warning(f"Skipping malformed JSONL line {line_num}")
                        continue

                    if event.get("type") == "agent_message":
                        cartographer.update_from_message(
                            agent=event.get("agent", "unknown"),
                            content=event.get("data", {}).get("content", ""),
                            role=event.get("data", {}).get("role", "proposer"),
                            round_num=event.get("round", 1),
                        )
                    elif event.get("type") == "critique":
                        cartographer.update_from_critique(
                            critic_agent=event.get("agent", "unknown"),
                            target_agent=event.get("data", {}).get("target", "unknown"),
                            severity=event.get("data", {}).get("severity", 0.5),
                            round_num=event.get("round", 1),
                            critique_text=event.get("data", {}).get("content", ""),
                        )

        stats = cartographer.get_statistics()
        return json_response(stats)
    except FileNotFoundError:
        logger.info("Build graph failed - replay file not found: %s", replay_path)
        return error_response(f"Replay file not found: {debate_id}", 404)
    except (StorageError, DatabaseError) as e:
        logger.error(
            "Failed to build graph from replay %s: %s: %s",
            debate_id,
            type(e).__name__,
            e,
            exc_info=True,
        )
        return error_response("Database error building graph", 500)
    except ValueError as e:
        logger.warning("Invalid replay data for %s: %s", debate_id, e)
        return error_response(f"Invalid replay data: {e}", 400)


__all__ = ["AnalysisOperationsMixin"]
