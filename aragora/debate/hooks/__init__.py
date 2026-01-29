"""Debate hooks for automated post-debate processing."""

from __future__ import annotations

from .manager import (
    AuditHooks,
    DebateHooks,
    HookCallback,
    HookManager,
    HookPriority,
    HookType,
    PropulsionHooks,
    create_hook_manager,
)
from .receipt_delivery_hook import ReceiptDeliveryHook, create_receipt_delivery_hook

__all__ = [
    "HookManager",
    "HookPriority",
    "HookType",
    "HookCallback",
    "DebateHooks",
    "AuditHooks",
    "PropulsionHooks",
    "create_hook_manager",
    "ReceiptDeliveryHook",
    "create_receipt_delivery_hook",
]
