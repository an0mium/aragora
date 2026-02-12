"""Canonical Decision Receipt contract.

This module provides a single import surface for receipt types:
- ``DecisionReceipt`` (canonical): ``aragora.gauntlet.receipt.DecisionReceipt``
- ``LegacyDecisionReceipt``: ``aragora.export.decision_receipt.DecisionReceipt``

Use ``DecisionReceipt`` for new integrations.
"""

from __future__ import annotations

from aragora.gauntlet.receipt import DecisionReceipt as DecisionReceipt
from aragora.export.decision_receipt import DecisionReceipt as LegacyDecisionReceipt

__all__ = ["DecisionReceipt", "LegacyDecisionReceipt"]
