"""Receipt import surface.

This package now coexists with the historical ``aragora.receipts`` module import
path used for decision-receipt classes. Re-export those classes here so imports
like ``from aragora.receipts import DecisionReceipt`` keep working while the
package also exposes operational receipt helpers.
"""

from aragora.export.decision_receipt import DecisionReceipt as LegacyDecisionReceipt
from aragora.gauntlet.receipt import DecisionReceipt

from .lane import LaneCompletionReceipt, emit_lane_receipt, validate_receipt
from .provenance import emit_operational_receipt

__all__ = [
    "DecisionReceipt",
    "LaneCompletionReceipt",
    "LegacyDecisionReceipt",
    "emit_lane_receipt",
    "emit_operational_receipt",
    "validate_receipt",
]
