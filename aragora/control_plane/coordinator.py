"""
Control Plane Coordinator for Aragora.

This module re-exports from the modular coordinator package for backward compatibility.
The implementation has been decomposed into focused submodules:

- coordinator/state_manager.py: State tracking, agent registry, health monitoring
- coordinator/policy_enforcer.py: Policy evaluation and enforcement
- coordinator/scheduler_bridge.py: Task scheduling integration
- coordinator/core.py: Main ControlPlaneCoordinator facade

This file maintains the original public API while delegating to the new package.
"""

# Re-export everything from the modular package
from aragora.control_plane.coordinator import (
    ControlPlaneCoordinator,
    ControlPlaneConfig,
    create_control_plane,
    StateManager,
    PolicyEnforcer,
    SchedulerBridge,
)

__all__ = [
    "ControlPlaneCoordinator",
    "ControlPlaneConfig",
    "create_control_plane",
    "StateManager",
    "PolicyEnforcer",
    "SchedulerBridge",
]
