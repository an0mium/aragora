"""Base types for the Decision Plan export adapter system.

Defines:
- TicketData: normalized ticket representation for any destination
- ExportReceipt: tracks export status per adapter
- ExportAdapter: abstract base class all adapters implement
"""

from __future__ import annotations

import hashlib
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


# ---------------------------------------------------------------------------
# Ticket priority mapping
# ---------------------------------------------------------------------------


class TicketPriority(str, Enum):
    """Normalized priority levels for exported tickets."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


# ---------------------------------------------------------------------------
# TicketData — the normalized representation of a ticket
# ---------------------------------------------------------------------------


@dataclass
class TicketData:
    """Normalized ticket produced from a DecisionPlan.

    Each adapter maps this to its platform-specific format.
    """

    title: str
    description: str
    priority: TicketPriority = TicketPriority.MEDIUM
    acceptance_criteria: list[str] = field(default_factory=list)
    labels: list[str] = field(default_factory=list)
    # Provenance
    plan_id: str = ""
    debate_id: str = ""
    task_id: str = ""
    # Extra metadata available to adapters
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "description": self.description,
            "priority": self.priority.value,
            "acceptance_criteria": self.acceptance_criteria,
            "labels": self.labels,
            "plan_id": self.plan_id,
            "debate_id": self.debate_id,
            "task_id": self.task_id,
            "metadata": self.metadata,
        }

    @property
    def content_hash(self) -> str:
        """SHA-256 hash of title + description for dedup."""
        raw = f"{self.title}:{self.description}".encode("utf-8")
        return hashlib.sha256(raw).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Export status / receipt
# ---------------------------------------------------------------------------


class ExportStatus(str, Enum):
    """Status of a single export operation."""

    PENDING = "pending"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class ExportReceipt:
    """Tracks the result of exporting tickets to a single adapter.

    One receipt is produced per adapter invocation.
    """

    id: str = field(default_factory=lambda: f"exp-{uuid.uuid4().hex[:12]}")
    adapter_name: str = ""
    status: ExportStatus = ExportStatus.PENDING
    tickets_exported: int = 0
    tickets_failed: int = 0
    created_at: float = field(default_factory=time.time)
    completed_at: float | None = None
    error: str | None = None
    # Per-ticket results keyed by task_id
    ticket_results: list[dict[str, Any]] = field(default_factory=list)
    # Plan provenance
    plan_id: str = ""
    debate_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "id": self.id,
            "adapter_name": self.adapter_name,
            "status": self.status.value,
            "tickets_exported": self.tickets_exported,
            "tickets_failed": self.tickets_failed,
            "created_at": self.created_at,
            "plan_id": self.plan_id,
            "debate_id": self.debate_id,
        }
        if self.completed_at is not None:
            result["completed_at"] = self.completed_at
            result["duration_s"] = round(self.completed_at - self.created_at, 3)
        if self.error is not None:
            result["error"] = self.error
        if self.ticket_results:
            result["ticket_results"] = self.ticket_results
        return result

    def mark_success(self) -> None:
        self.status = ExportStatus.SUCCESS
        self.completed_at = time.time()

    def mark_failed(self, error: str) -> None:
        self.status = ExportStatus.FAILED
        self.error = error
        self.completed_at = time.time()


# ---------------------------------------------------------------------------
# ExportAdapter — abstract base
# ---------------------------------------------------------------------------


class ExportAdapter(ABC):
    """Abstract base class for ticket export adapters.

    Each adapter converts normalized TicketData into the destination
    platform's format and pushes it via the platform's API.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable adapter name (e.g., 'jira', 'linear', 'webhook')."""
        ...

    @abstractmethod
    async def export_ticket(self, ticket: TicketData) -> dict[str, Any]:
        """Export a single ticket to the destination.

        Returns:
            Dict with at least ``{"success": bool}``.
            On success, should include platform-specific fields
            (e.g., ``issue_key``, ``issue_url``).
        """
        ...

    async def export_tickets(self, tickets: list[TicketData]) -> ExportReceipt:
        """Export multiple tickets and produce a receipt.

        Default implementation calls export_ticket() sequentially.
        Adapters may override for batch APIs.
        """
        receipt = ExportReceipt(adapter_name=self.name)
        if tickets:
            receipt.plan_id = tickets[0].plan_id
            receipt.debate_id = tickets[0].debate_id

        for ticket in tickets:
            try:
                result = await self.export_ticket(ticket)
                result["task_id"] = ticket.task_id
                result["title"] = ticket.title
                receipt.ticket_results.append(result)
                if result.get("success"):
                    receipt.tickets_exported += 1
                else:
                    receipt.tickets_failed += 1
            except (ConnectionError, TimeoutError, OSError, ValueError, RuntimeError) as exc:
                receipt.tickets_failed += 1
                receipt.ticket_results.append(
                    {
                        "task_id": ticket.task_id,
                        "title": ticket.title,
                        "success": False,
                        "error": str(exc),
                    }
                )

        if receipt.tickets_failed == 0:
            receipt.mark_success()
        elif receipt.tickets_exported > 0:
            # Partial success
            receipt.status = ExportStatus.SUCCESS
            receipt.completed_at = time.time()
        else:
            receipt.mark_failed("All ticket exports failed")

        return receipt
