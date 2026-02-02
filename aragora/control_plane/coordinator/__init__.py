"""
Control Plane Coordinator Package.

This package provides a modular implementation of the control plane coordinator,
split into focused submodules for better maintainability:

- state_manager: State tracking, agent registry, health monitoring, KM integration
- policy_enforcer: Policy evaluation, enforcement, and compliance checking
- scheduler_bridge: Task scheduling, lifecycle management, and result waiting
- core: Main ControlPlaneCoordinator facade that composes the submodules

Usage:
    from aragora.control_plane.coordinator import (
        ControlPlaneCoordinator,
        ControlPlaneConfig,
        create_control_plane,
    )

    # Create and connect
    coordinator = await ControlPlaneCoordinator.create()

    # Or use the convenience function
    coordinator = await create_control_plane()
"""

from aragora.control_plane.coordinator.state_manager import (
    ControlPlaneConfig,
    StateManager,
)
from aragora.control_plane.coordinator.policy_enforcer import PolicyEnforcer
from aragora.control_plane.coordinator.scheduler_bridge import SchedulerBridge
from aragora.control_plane.coordinator.core import (
    ControlPlaneCoordinator,
    create_control_plane,
)

__all__ = [
    # Main coordinator
    "ControlPlaneCoordinator",
    "create_control_plane",
    # Configuration
    "ControlPlaneConfig",
    # Submodules
    "StateManager",
    "PolicyEnforcer",
    "SchedulerBridge",
]
