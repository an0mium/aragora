"""
Server startup control plane initialization.

This module handles control plane coordinator, shared state,
witness patrol, mayor coordinator, and persistent task queue initialization.
"""

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


async def init_control_plane_coordinator() -> Optional[Any]:
    """Initialize the Control Plane coordinator.

    Creates and connects the ControlPlaneCoordinator which manages:
    - Agent registry (service discovery)
    - Task scheduler (distributed task execution)
    - Health monitor (agent health tracking)
    - Policy management (automatic policy sync from compliance store)

    Policy sync is enabled by default and controlled by ARAGORA_POLICY_SYNC_ON_STARTUP
    (or CP_ENABLE_POLICY_SYNC for backward compatibility). Set to "false" to disable.

    Returns:
        Connected ControlPlaneCoordinator, or None if initialization fails
    """
    try:
        from aragora.control_plane.coordinator import ControlPlaneCoordinator

        coordinator = await ControlPlaneCoordinator.create()

        # Log policy manager status
        if coordinator.policy_manager:
            policy_count = (
                len(coordinator.policy_manager._policies)
                if hasattr(coordinator.policy_manager, "_policies")
                else 0
            )
            logger.info(
                f"Control Plane coordinator initialized and connected "
                f"(policies_loaded={policy_count})"
            )
        else:
            logger.info("Control Plane coordinator initialized and connected (no policy manager)")

        return coordinator
    except ImportError as e:
        logger.debug(f"Control Plane not available: {e}")
        return None
    except Exception as e:
        # Redis may not be available - this is OK for local development
        logger.warning(f"Control Plane coordinator not started (Redis may be unavailable): {e}")
        return None


async def init_shared_control_plane_state() -> bool:
    """Initialize the shared control plane state for the AgentDashboardHandler.

    Connects to Redis for multi-instance state sharing. Falls back to in-memory
    for single-instance deployments.

    Returns:
        True if Redis connected, False if using in-memory fallback
    """
    try:
        from aragora.control_plane.shared_state import get_shared_state

        state = await get_shared_state(auto_connect=True)
        if state.is_persistent:
            logger.info("Shared control plane state connected to Redis (HA enabled)")
            return True
        else:
            logger.info("Shared control plane state using in-memory fallback (single-instance)")
            return False
    except ImportError as e:
        logger.debug(f"Shared control plane state not available: {e}")
        return False
    except Exception as e:
        logger.warning(f"Shared control plane state initialization failed: {e}")
        return False


# Global witness instance for handlers to access
_witness_behavior: Optional[Any] = None


def get_witness_behavior() -> Optional[Any]:
    """Get the global witness behavior instance.

    Returns:
        WitnessBehavior instance if initialized, None otherwise
    """
    return _witness_behavior


async def init_witness_patrol() -> bool:
    """Initialize and start the Witness Patrol for Gas Town monitoring.

    Creates the WitnessBehavior with an AgentHierarchy and starts the patrol
    loop which monitors:
    - Agent health and heartbeats
    - Bead progress and stuck detection
    - Convoy completion rates
    - Automatic escalation to MAYOR on critical issues

    The witness instance is stored globally and can be accessed via
    get_witness_behavior() for the status endpoint.

    Returns:
        True if patrol started successfully, False otherwise
    """
    global _witness_behavior

    try:
        from aragora.nomic.witness_behavior import WitnessBehavior, WitnessConfig
        from aragora.nomic.agent_roles import AgentHierarchy

        # Create hierarchy with default persistence directory
        hierarchy = AgentHierarchy()

        # Configure witness with reasonable defaults
        config = WitnessConfig(
            patrol_interval_seconds=30,
            heartbeat_timeout_seconds=120,
            notify_mayor_on_critical=True,
        )

        # Create witness behavior
        witness = WitnessBehavior(
            hierarchy=hierarchy,
            config=config,
        )

        # Start the patrol loop
        await witness.start_patrol()

        _witness_behavior = witness
        logger.info("Witness patrol started successfully")
        return True

    except ImportError as e:
        logger.debug(f"Witness behavior not available: {e}")
        return False
    except Exception as e:
        logger.warning(f"Witness patrol initialization failed: {e}")
        return False


# Global mayor coordinator instance
_mayor_coordinator: Optional[Any] = None


def get_mayor_coordinator() -> Optional[Any]:
    """Get the global mayor coordinator instance.

    Returns:
        MayorCoordinator instance if initialized, None otherwise
    """
    return _mayor_coordinator


async def init_mayor_coordinator() -> bool:
    """Initialize the Mayor Coordinator for distributed leadership.

    Creates the MayorCoordinator which bridges leader election with the
    Gas Town agent hierarchy:
    - When this node wins election, it becomes MAYOR
    - When it loses election, it demotes to WITNESS
    - Provides current mayor info via get_mayor_coordinator()

    Returns:
        True if coordinator started successfully, False otherwise
    """
    global _mayor_coordinator

    try:
        from aragora.nomic.mayor_coordinator import MayorCoordinator
        from aragora.nomic.agent_roles import AgentHierarchy
        import os

        # Get node ID from environment or generate one
        node_id = os.environ.get("ARAGORA_NODE_ID")
        region = os.environ.get("ARAGORA_REGION")

        # Create hierarchy (will be shared with witness if both are initialized)
        hierarchy = AgentHierarchy()

        # Create and start coordinator
        coordinator = MayorCoordinator(
            hierarchy=hierarchy,
            node_id=node_id,
            region=region,
        )

        if await coordinator.start():
            _mayor_coordinator = coordinator
            is_mayor = "yes" if coordinator.is_mayor else "no"
            logger.info(
                f"Mayor coordinator started (node={coordinator.node_id}, "
                f"is_mayor={is_mayor}, region={region or 'global'})"
            )
            return True

        return False

    except ImportError as e:
        logger.debug(f"Mayor coordinator not available: {e}")
        return False
    except Exception as e:
        logger.warning(f"Mayor coordinator initialization failed: {e}")
        return False


async def init_persistent_task_queue() -> int:
    """Initialize persistent task queue with recovery of pending tasks.

    Creates the singleton PersistentTaskQueue and recovers any tasks that
    were pending/running when the server was previously stopped.

    Returns:
        Number of tasks recovered
    """
    try:
        from aragora.workflow.queue import (
            get_persistent_task_queue,
            PersistentTaskQueue,  # noqa: F401
        )

        # Get or create the singleton queue
        queue = get_persistent_task_queue()

        # Start the queue processor
        await queue.start()

        # Recover pending tasks from previous runs
        recovered = await queue.recover_tasks()

        # Schedule cleanup of old completed tasks (24h retention)
        import asyncio

        async def cleanup_loop():
            while True:
                await asyncio.sleep(3600)  # Run hourly
                try:
                    deleted = queue.delete_completed_tasks(older_than_hours=24)
                    if deleted > 0:
                        logger.debug(f"Cleaned up {deleted} old completed tasks")
                except Exception as e:
                    logger.warning(f"Task cleanup failed: {e}")

        asyncio.create_task(cleanup_loop())

        if recovered > 0:
            logger.info(f"Persistent task queue started, recovered {recovered} tasks")
        else:
            logger.info("Persistent task queue started")

        return recovered

    except ImportError as e:
        logger.debug(f"Persistent task queue not available: {e}")
    except Exception as e:
        logger.warning(f"Failed to initialize persistent task queue: {e}")

    return 0
