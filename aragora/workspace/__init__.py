"""
Workspace Manager - Gastown parity extension for developer orchestration.

Provides per-repo project containers (Rigs) with isolated agent pools,
convoy-based work batch tracking, bead management, and merge queue
(Refinery) capabilities.

This module implements the Gastown workspace model on top of the
Agent Fabric and existing Aragora control plane infrastructure.

Key concepts:
- Rig: Per-repo project container with its own agent pool and config.
- Convoy: A batch of related work items (beads) tracked as a unit.
- Bead: An atomic unit of work with JSONL-backed state tracking.
- Refinery: Merge queue that gates merges with test + review status.

Usage:
    from aragora.workspace import WorkspaceManager

    ws = WorkspaceManager(workspace_root="/path/to/projects")
    rig = await ws.create_rig("my-repo", repo_url="https://...")
    convoy = await ws.create_convoy(rig.rig_id, work_items=[...])
    status = await ws.get_convoy_status(convoy.convoy_id)
"""

from aragora.workspace.manager import WorkspaceManager
from aragora.workspace.rig import Rig, RigConfig, RigStatus
from aragora.workspace.convoy import Convoy, ConvoyStatus, ConvoyTracker
from aragora.workspace.bead import Bead, BeadStatus, BeadManager

__all__ = [
    "WorkspaceManager",
    "Rig",
    "RigConfig",
    "RigStatus",
    "Convoy",
    "ConvoyStatus",
    "ConvoyTracker",
    "Bead",
    "BeadStatus",
    "BeadManager",
]
