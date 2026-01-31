"""Hook tracking, bead management, and GUPP recovery helpers for Arena.

Extracted from orchestrator.py to reduce its size. These functions
implement the Gastown GUPP principle: "If there is work on your Hook,
YOU MUST RUN IT." They create, update, and complete beads and hook
queue entries so that interrupted debates can be recovered on restart.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING, Any

from aragora.logging_config import get_logger as get_structured_logger

if TYPE_CHECKING:
    from aragora.core import DebateResult

logger = get_structured_logger(__name__)


async def create_debate_bead(
    result: "DebateResult",
    protocol: Any,
    env: Any,
    bead_store_holder: Any,
) -> str | None:
    """Create a Bead to track a completed debate decision with git-backed audit trail.

    Args:
        result: The completed debate result.
        protocol: The debate protocol (checked for enable_bead_tracking, bead_min_confidence).
        env: The debate environment (provides context dict).
        bead_store_holder: Object that may hold ``_bead_store`` attribute (typically Arena).

    Returns:
        The bead ID if created, ``None`` otherwise.
    """
    enable_bead = getattr(protocol, "enable_bead_tracking", False)
    min_confidence = getattr(protocol, "bead_min_confidence", 0.5)

    if not enable_bead:
        return None

    if result.confidence < min_confidence:
        logger.debug(
            f"Skipping bead creation: confidence {result.confidence:.2f} < {min_confidence}"
        )
        return None

    try:
        from aragora.nomic.stores import Bead, BeadPriority, BeadStore, BeadType
        from aragora.nomic.stores.paths import resolve_store_dir

        bead_store = getattr(bead_store_holder, "_bead_store", None)
        if bead_store is None:
            bead_dir = Path(env.context.get("bead_dir")) if env.context else None
            bead_store = BeadStore(
                bead_dir=bead_dir or resolve_store_dir(),
                git_enabled=True,
                auto_commit=getattr(protocol, "bead_auto_commit", False),
            )
            await bead_store.initialize()
            bead_store_holder._bead_store = bead_store

        if result.confidence >= 0.9:
            priority = BeadPriority.HIGH
        elif result.confidence >= 0.7:
            priority = BeadPriority.NORMAL
        else:
            priority = BeadPriority.LOW

        bead = Bead.create(
            bead_type=BeadType.DEBATE_DECISION,
            title=f"Decision: {result.task[:50]}..."
            if len(result.task) > 50
            else f"Decision: {result.task}",
            description=result.final_answer[:500] if result.final_answer else "",
            priority=priority,
            tags=["debate", result.status, f"confidence:{result.confidence:.0%}"],
            metadata={
                "debate_id": result.debate_id,
                "consensus_reached": result.consensus_reached,
                "confidence": result.confidence,
                "rounds_used": result.rounds_used,
                "participants": result.participants,
                "winner": result.winner,
                "domain": "general",
            },
        )

        bead_id = await bead_store.create(bead)
        logger.info(f"Created debate bead: {bead_id[:8]} for debate {result.debate_id[:8]}")
        return bead_id

    except ImportError:
        logger.debug("Bead tracking unavailable: aragora.nomic.stores not found")
        return None
    except (OSError, ValueError, TypeError, AttributeError, KeyError, RuntimeError) as e:
        logger.warning(f"Failed to create debate bead: {e}")
        return None


async def create_pending_debate_bead(
    debate_id: str,
    task: str,
    protocol: Any,
    env: Any,
    agents: list,
    bead_store_holder: Any,
) -> str | None:
    """Create a pending bead to track debate work in progress.

    Called at debate START (unlike :func:`create_debate_bead` which is
    called at END). This enables GUPP recovery by creating durable work
    tracking before execution.

    Args:
        debate_id: The debate ID.
        task: The debate task/question.
        protocol: Debate protocol (checked for enable_hook_tracking).
        env: Debate environment.
        agents: List of participating agents.
        bead_store_holder: Object that may hold ``_bead_store`` attribute.

    Returns:
        The bead ID if created, ``None`` otherwise.
    """
    enable_hooks = getattr(protocol, "enable_hook_tracking", False)
    if not enable_hooks:
        return None

    try:
        from aragora.nomic.stores import Bead, BeadPriority, BeadStore, BeadType
        from aragora.nomic.stores.paths import resolve_store_dir

        bead_store = getattr(bead_store_holder, "_bead_store", None)
        if bead_store is None:
            bead_dir = Path(env.context.get("bead_dir")) if env.context else None
            bead_store = BeadStore(
                bead_dir=bead_dir or resolve_store_dir(),
                git_enabled=True,
                auto_commit=getattr(protocol, "bead_auto_commit", False),
            )
            await bead_store.initialize()
            bead_store_holder._bead_store = bead_store

        bead = Bead.create(
            bead_type=BeadType.DEBATE_DECISION,
            title=f"[Pending] Decision: {task[:50]}..."
            if len(task) > 50
            else f"[Pending] Decision: {task}",
            description="Debate in progress...",
            priority=BeadPriority.NORMAL,
            tags=["debate", "pending", "gupp-tracked"],
            metadata={
                "debate_id": debate_id,
                "participants": [a.name for a in agents],
                "status": "in_progress",
            },
        )

        bead_id = await bead_store.create(bead)
        logger.debug(f"Created pending debate bead: {bead_id[:8]} for debate {debate_id[:8]}")
        return bead_id

    except ImportError:
        logger.debug("Bead tracking unavailable")
        return None
    except (OSError, ValueError, TypeError, AttributeError, KeyError, RuntimeError) as e:
        logger.warning(f"Failed to create pending debate bead: {e}")
        return None


async def update_debate_bead(
    bead_id: str,
    result: "DebateResult",
    success: bool,
    bead_store_holder: Any,
) -> None:
    """Update a pending debate bead with final results.

    Args:
        bead_id: The bead ID to update.
        result: The final debate result.
        success: Whether the debate completed successfully.
        bead_store_holder: Object that may hold ``_bead_store`` attribute.
    """
    if not bead_id:
        return

    try:
        from aragora.nomic.stores import BeadPriority, BeadStatus

        bead_store = getattr(bead_store_holder, "_bead_store", None)
        if bead_store is None:
            return

        bead = await bead_store.get(bead_id)
        if not bead:
            return

        bead.title = (
            f"Decision: {result.task[:50]}..."
            if len(result.task) > 50
            else f"Decision: {result.task}"
        )
        bead.description = result.final_answer[:500] if result.final_answer else ""

        if result.confidence >= 0.9:
            bead.priority = BeadPriority.HIGH
        elif result.confidence >= 0.7:
            bead.priority = BeadPriority.NORMAL
        else:
            bead.priority = BeadPriority.LOW

        bead.metadata.update(
            {
                "consensus_reached": result.consensus_reached,
                "confidence": result.confidence,
                "rounds_used": result.rounds_used,
                "winner": result.winner,
                "status": "completed" if success else "failed",
            }
        )

        bead.tags = ["debate", result.status, f"confidence:{result.confidence:.0%}"]

        if success:
            await bead_store.update_status(bead_id, BeadStatus.COMPLETED)
        else:
            await bead_store.update_status(bead_id, BeadStatus.FAILED)

        logger.debug(
            f"Updated debate bead: {bead_id[:8]} status={'completed' if success else 'failed'}"
        )

    except ImportError:
        logger.debug("Bead tracking unavailable: aragora.nomic.stores not installed")
    except (OSError, ValueError, TypeError, AttributeError, KeyError, RuntimeError) as e:
        logger.warning(f"Failed to update debate bead: {e}")


async def init_hook_tracking(
    debate_id: str,
    bead_id: str,
    protocol: Any,
    agents: list,
    hook_registry_holder: Any,
) -> dict[str, str]:
    """Initialize GUPP hook tracking for debate work.

    Pushes the debate bead onto each participating agent's hook queue,
    ensuring work recovery on crash via the GUPP principle.

    Args:
        debate_id: The debate ID.
        bead_id: The bead ID tracking this debate.
        protocol: Debate protocol (checked for enable_hook_tracking).
        agents: List of participating agents.
        hook_registry_holder: Object that may hold ``_hook_registry`` and ``_bead_store``.

    Returns:
        Dict mapping agent_id to hook_entry_id.
    """
    enable_hooks = getattr(protocol, "enable_hook_tracking", False)
    if not enable_hooks or not bead_id:
        return {}

    try:
        from aragora.nomic.hook_queue import HookQueueRegistry

        hook_registry = getattr(hook_registry_holder, "_hook_registry", None)
        if hook_registry is None:
            bead_store = getattr(hook_registry_holder, "_bead_store", None)
            if bead_store is None:
                logger.debug("Hook tracking requires bead_store, skipping")
                return {}
            hook_registry = HookQueueRegistry(bead_store)
            hook_registry_holder._hook_registry = hook_registry

        hook_entries: dict[str, str] = {}
        for agent in agents:
            agent_id = getattr(agent, "name", str(agent))
            try:
                hook_queue = await hook_registry.get_queue(agent_id)
                entry = await hook_queue.push(
                    bead_id=bead_id,
                    priority=75,
                    max_attempts=3,
                )
                hook_entries[agent_id] = entry.id
                logger.debug(f"Pushed debate {debate_id[:8]} to hook for {agent_id}")
            except (OSError, ConnectionError, ValueError, TypeError, asyncio.TimeoutError) as e:
                logger.warning(f"Failed to push hook for {agent_id}: {e}")

        if hook_entries:
            logger.info(f"GUPP: Tracked debate {debate_id[:8]} on {len(hook_entries)} agent hooks")

        return hook_entries

    except ImportError:
        logger.debug("Hook tracking unavailable: aragora.nomic.hook_queue not found")
        return {}
    except (OSError, ValueError, TypeError, AttributeError) as e:
        logger.warning(f"Failed to initialize hook tracking: {e}")
        return {}


async def complete_hook_tracking(
    bead_id: str,
    hook_entries: dict[str, str],
    success: bool,
    hook_registry_holder: Any,
    error_msg: str = "",
) -> None:
    """Complete or fail hook entries for debate work.

    Args:
        bead_id: The bead ID tracking this debate.
        hook_entries: Dict mapping agent_id to hook_entry_id.
        success: Whether the debate completed successfully.
        hook_registry_holder: Object that may hold ``_hook_registry``.
        error_msg: Error message if failed.
    """
    if not hook_entries or not bead_id:
        return

    try:
        hook_registry = getattr(hook_registry_holder, "_hook_registry", None)
        if hook_registry is None:
            return

        for agent_id, entry_id in hook_entries.items():
            try:
                hook_queue = await hook_registry.get_queue(agent_id)
                if success:
                    await hook_queue.complete(bead_id)
                    logger.debug(f"Completed hook {entry_id[:8]} for {agent_id}")
                else:
                    await hook_queue.fail(bead_id, error_msg or "Debate failed")
                    logger.debug(f"Failed hook {entry_id[:8]} for {agent_id}")
            except (OSError, ConnectionError, ValueError, TypeError, asyncio.TimeoutError) as e:
                logger.warning(f"Failed to complete hook for {agent_id}: {e}")

    except ImportError:
        logger.debug("Hook tracking unavailable: aragora.nomic.hook_queue not installed")
    except (OSError, ValueError, TypeError, AttributeError) as e:
        logger.warning(f"Failed to complete hook tracking: {e}")


async def recover_pending_debates(
    bead_store: Any = None,
    max_age_hours: int = 24,
) -> list[dict]:
    """Recover pending debates from hook queues on startup.

    GUPP Principle: "If there is work on your Hook, YOU MUST RUN IT."

    This function should be called during server startup to identify
    debates that were interrupted by crashes and need to be resumed.

    Args:
        bead_store: Optional BeadStore instance (creates default if ``None``).
        max_age_hours: Maximum age of recoverable work in hours.

    Returns:
        List of dicts with debate recovery info.
    """
    try:
        from datetime import datetime, timedelta, timezone

        from aragora.nomic.stores import BeadStatus, BeadStore, BeadType
        from aragora.nomic.stores.paths import resolve_store_dir
        from aragora.nomic.hook_queue import HookQueueRegistry

        if bead_store is None:
            bead_store = BeadStore(bead_dir=resolve_store_dir(), git_enabled=False)
            await bead_store.initialize()

        registry = HookQueueRegistry(bead_store)
        recovered = await registry.recover_all()

        if not recovered:
            logger.info("GUPP recovery: No pending work found")
            return []

        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=max_age_hours)
        debates_to_resume: dict[str, dict] = {}

        for agent_id, beads in recovered.items():
            for bead in beads:
                if bead.bead_type != BeadType.DEBATE_DECISION:
                    continue
                if bead.created_at < cutoff_time:
                    logger.debug(f"Skipping stale bead {bead.id[:8]} (too old)")
                    continue
                if bead.status in (BeadStatus.COMPLETED, BeadStatus.FAILED):
                    continue

                debate_id = bead.metadata.get("debate_id", bead.id)
                if debate_id not in debates_to_resume:
                    debates_to_resume[debate_id] = {
                        "debate_id": debate_id,
                        "bead_id": bead.id,
                        "agents": [],
                        "bead": bead,
                    }
                debates_to_resume[debate_id]["agents"].append(agent_id)

        result = list(debates_to_resume.values())
        if result:
            logger.info(
                f"GUPP recovery: Found {len(result)} debates to resume "
                f"across {sum(len(d['agents']) for d in result)} agent hooks"
            )

        return result

    except ImportError as e:
        logger.debug(f"GUPP recovery unavailable: {e}")
        return []
    except (OSError, ValueError, TypeError, KeyError, RuntimeError) as e:
        logger.warning(f"GUPP recovery failed: {e}")
        return []
