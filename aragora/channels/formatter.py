"""
Base receipt formatter interface.

Defines the contract for channel-specific receipt formatters.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from aragora.export.decision_receipt import DecisionReceipt


class ReceiptFormatter(ABC):
    """Base class for channel-specific receipt formatters."""

    @property
    @abstractmethod
    def channel_type(self) -> str:
        """Return the channel type identifier (e.g., 'slack', 'teams')."""
        ...

    @abstractmethod
    def format(
        self,
        receipt: "DecisionReceipt",
        options: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Format a decision receipt for this channel.

        Args:
            receipt: The decision receipt to format
            options: Optional formatting options (e.g., compact mode, summary only)

        Returns:
            Channel-specific formatted payload
        """
        ...

    def format_summary(
        self,
        receipt: "DecisionReceipt",
        max_length: int = 280,
    ) -> str:
        """
        Format a short text summary of the receipt.

        Args:
            receipt: The decision receipt
            max_length: Maximum length of the summary

        Returns:
            Short text summary
        """
        decision = receipt.verdict or "No decision"
        confidence = receipt.confidence or 0

        summary = f"[{confidence:.0%} confidence] {decision}"
        if len(summary) > max_length:
            summary = summary[: max_length - 3] + "..."
        return summary


# Registry of formatters by channel type
_FORMATTERS: Dict[str, type[ReceiptFormatter]] = {}


def register_formatter(formatter_class: type[ReceiptFormatter]) -> type[ReceiptFormatter]:
    """Register a formatter class for its channel type."""
    instance = formatter_class()
    _FORMATTERS[instance.channel_type] = formatter_class
    return formatter_class


def get_formatter(channel_type: str) -> Optional[ReceiptFormatter]:
    """Get a formatter instance for the given channel type."""
    formatter_class = _FORMATTERS.get(channel_type)
    if formatter_class:
        return formatter_class()
    return None


def format_receipt_for_channel(
    receipt: "DecisionReceipt",
    channel_type: str,
    options: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Format a receipt for a specific channel.

    Args:
        receipt: The decision receipt to format
        channel_type: The target channel type
        options: Optional formatting options

    Returns:
        Channel-specific formatted payload

    Raises:
        ValueError: If no formatter is registered for the channel type
    """
    formatter = get_formatter(channel_type)
    if not formatter:
        raise ValueError(f"No formatter registered for channel: {channel_type}")
    return formatter.format(receipt, options)
