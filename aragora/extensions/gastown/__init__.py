"""
Gastown Extension - Developer Orchestration Layer.

Provides workspace management for multi-agent software engineering:
- Workspace: Root container for projects and agents
- Rig: Per-repository container with agent context
- Convoy: Work tracking unit with artifacts and handoffs
- Hooks: Git worktree-based persistent storage
- Coordinator: Mayor-style orchestration interface

Inspired by steveyegge/gastown, adapted for Aragora's enterprise semantics.
"""

from .models import (
    Workspace,
    WorkspaceConfig,
    Rig,
    RigConfig,
    Convoy,
    ConvoyStatus,
    ConvoyArtifact,
    Hook,
    HookType,
)
from .workspace import WorkspaceManager
from .convoy import ConvoyTracker
from .hooks import HookRunner
from .coordinator import Coordinator
from .adapter import GastownConvoyAdapter

__all__ = [
    # Models
    "Workspace",
    "WorkspaceConfig",
    "Rig",
    "RigConfig",
    "Convoy",
    "ConvoyStatus",
    "ConvoyArtifact",
    "Hook",
    "HookType",
    # Managers
    "WorkspaceManager",
    "ConvoyTracker",
    "HookRunner",
    "Coordinator",
    "GastownConvoyAdapter",
]
